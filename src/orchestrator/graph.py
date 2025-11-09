# src/orchestrator/graph.py
from __future__ import annotations

import os
import asyncio
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, END

from ..bus import ConversationBus, DebateMessage
from ..agents.base import BaseAgent, AgentContext

# Optional MCP client
try:
    from ..mcp_tools.mcp_client import MCPToolRouter
except Exception:
    MCPToolRouter = None

PARALLEL_ANALYSES = os.getenv(
    "PARALLEL_ANALYSES", "true").lower() in ("1", "true", "yes", "on")
BARRIERED_ROUNDS = os.getenv(
    "BARRIERED_ROUNDS",  "true").lower() in ("1", "true", "yes", "on")


def _parse_mcp_servers() -> List[List[str]]:
    raw = os.getenv("MCP_SERVERS", "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return [p.split() for p in parts]


@dataclass
class DebateState:
    topic: str
    round_no: int
    agents: List[BaseAgent]
    bus: ConversationBus
    total_rounds: int
    mcp_router: Optional[Any] = None


def render_history(bus: ConversationBus) -> str:
    parts: List[str] = []
    for m in bus.history():
        header = f"[{m.type.upper()}] {m.agent} (round {m.round})"
        if getattr(m, "status", None):
            header += f" — status: {m.status}"
        parts.append(f"{header}:\n{m.content}\n")
    return "\n".join(parts)


def _round_recipients(all_agents: List[BaseAgent], sender: str) -> List[str]:
    s = (sender or "").strip().lower()
    return [
        a.name
        for a in all_agents
        if getattr(a, "participates", True)
        and a.name
        and a.name.lower() not in (s, "planner", "judge")
    ]


# ---------------------------- MCP bootstrap/shutdown ---------------------------- #

async def _mcp_bootstrap(state: DebateState):
    if state.mcp_router or MCPToolRouter is None:
        return state

    servers = _parse_mcp_servers()
    if not servers:
        state.bus.publish({
            "type": "system",
            "agent": "orchestrator",
            "round": 0,
            "content": "[MCP enabled but MCP_SERVERS not set; continuing without MCP]"
        })
        return state

    try:
        router = MCPToolRouter(servers)
        await router.start()
        state.mcp_router = router
        state.bus.publish({
            "type": "system",
            "agent": "orchestrator",
            "round": 0,
            "content": f"MCP initialised with servers: {[' '.join(s) for s in servers]}"
        })
    except Exception as e:
        state.bus.publish({
            "type": "system",
            "agent": "orchestrator",
            "round": 0,
            "content": f"[MCP initialisation failed: {e}]"
        })
    return state


async def _mcp_shutdown(state: DebateState):
    if state.mcp_router:
        try:
            await state.mcp_router.stop()
        except Exception:
            pass
        state.mcp_router = None
    return state


# ----------------------------------- steps ----------------------------------- #

async def _analysis_step(state: DebateState):
    async def run_for_agent(a: BaseAgent):
        # IMPORTANT: run Planner even if participates=False; others still respect participates flag
        if not (a.name and a.name.lower() == "planner") and not getattr(a, "participates", True):
            return

        ctx = AgentContext(
            state.topic,
            0,
            render_history(state.bus),
            a.tool_budget_per_round,
            state.mcp_router
        )
        try:
            res = await a.analysis(ctx)
        except Exception as e:
            res = {
                "text": f"[{a.name} error during analysis: {e}]",
                "citations": [], "tool_calls": [], "status": "error"
            }

        state.bus.publish({
            "type": "analysis",
            "agent": a.name,
            "round": 0,
            "content": res.get("text", ""),
            "citations": res.get("citations", []),
            "tool_calls": res.get("tool_calls", []),
            "status": res.get("status")
        })

    # Include all agents here so Planner can speak in the analysis phase.
    all_agents = list(state.agents)
    if PARALLEL_ANALYSES:
        await asyncio.gather(*[run_for_agent(a) for a in all_agents])
    else:
        for a in all_agents:
            await run_for_agent(a)
    return state


async def _debate_round(state: DebateState):
    r = state.round_no
    active_agents = [a for a in state.agents if getattr(a, "participates", True)]

    async def draft(a: BaseAgent):
        # ==== Exclude Planner/Judge from the debate context ====
        filtered_msgs = [
            m for m in state.bus.history()
            if (m.agent != a.name) and (m.agent.lower() not in ("planner", "judge"))
        ]

        history_text = "\n".join(
            f"[{m.type.upper()}] {m.agent} (round {m.round})"
            + (f" — status: {m.status}" if getattr(m, "status", None) else "")
            + f":\n{m.content}\n"
            for m in filtered_msgs
        )
        ctx = AgentContext(state.topic, r, history_text,
                           a.tool_budget_per_round, state.mcp_router)
        try:
            res = await a.debate_turn(ctx)
            text = res.get("text", "")
            cites = res.get("citations", [])
            calls = res.get("tool_calls", [])
            status = res.get("status")
            msg_type = "argument" if r == 1 else "rebuttal"
        except Exception as e:
            text, cites, calls, status = f"[{a.name} error during debate round {r}: {e}]", [], [], "error"
            msg_type = "rebuttal" if r > 1 else "argument"
        return a, msg_type, text, cites, calls, status

    # Barriered, parallel drafting
    drafts: List[Tuple[BaseAgent, str, str, List[Any], List[dict], Optional[str]]] = await asyncio.gather(
        *[draft(a) for a in active_agents]
    )

    # Publish in defined order
    for a, msg_type, text, cites, calls, status in drafts:
        if a is None:
            continue
        state.bus.publish({
            "type": msg_type,
            "agent": a.name,
            "round": r,
            "content": text,
            "citations": cites or [],
            "tool_calls": calls or [],
            "to": _round_recipients(active_agents, a.name),
            "status": status
        })

    state.round_no += 1
    return state


async def _consensus(state: DebateState):
    finaliser = next((a for a in state.agents if (
        a.name or "").lower() == "judge"), state.agents[0])
    ctx = AgentContext(
        state.topic,
        state.total_rounds,
        render_history(state.bus),
        finaliser.tool_budget_per_round,
        state.mcp_router
    )
    try:
        res = await finaliser.propose_consensus(ctx)
        text = res.get("text", "")
        cites = res.get("citations", [])
        calls = res.get("tool_calls", [])
        status = res.get("status")
    except Exception as e:
        text, cites, calls, status = f"[{finaliser.name} error during consensus proposal: {e}]", [], [], "error"

    state.bus.publish({
        "type": "final_conclusion",
        "agent": finaliser.name,
        "round": state.total_rounds,
        "content": text,
        "citations": cites,
        "tool_calls": calls,
        "status": status
    })
    return state


def build_graph():
    g = StateGraph(DebateState)
    g.add_node("mcp_bootstrap", _mcp_bootstrap)
    g.add_node("analyses", _analysis_step)
    g.add_node("debate_round", _debate_round)
    g.add_node("consensus", _consensus)
    g.add_node("mcp_shutdown", _mcp_shutdown)

    g.set_entry_point("mcp_bootstrap")
    g.add_edge("mcp_bootstrap", "analyses")
    g.add_edge("analyses", "debate_round")

    def continue_debate(state: DebateState) -> str:
        return "debate_round" if state.round_no <= state.total_rounds else "consensus"

    g.add_conditional_edges("debate_round", continue_debate, {
        "debate_round": "debate_round", "consensus": "consensus"})
    g.add_edge("consensus", "mcp_shutdown")
    g.add_edge("mcp_shutdown", END)

    return g.compile()
