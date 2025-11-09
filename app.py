# src/app.py
from __future__ import annotations
from src.orchestrator.graph import build_graph, DebateState
from src.agents.judge_agent import JudgeAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.model_agent import ModelAgent, LLMBackend
from src.mcp_tools.mcp_client import MCPToolRouter
from src.bus import ConversationBus
from src.config import settings
"""
Single merged file — Flask web app + debate engine, streaming via SSE.
No telemetry and no NDJSON file. The UI receives live events as they are produced.
"""

import os
import sys
import json
import asyncio
import threading
from queue import Queue
from typing import List, Dict, Any, Optional, Callable
from flask import Flask, request, jsonify, Response, stream_with_context, render_template

# --------------------------------------------------------------------------------------
# Make project root importable
# --------------------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------------------------
# Imports from your project
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Debate engine logic
# --------------------------------------------------------------------------------------


def make_backend_for(vendor: str, model: str) -> LLMBackend:
    v = vendor.lower()
    if v not in ("openai", "anthropic", "gemini"):
        raise ValueError(f"Unsupported vendor: {vendor}")
    return LLMBackend(vendor=v, model=model)


def make_model_agents(tool_router: MCPToolRouter | None, tool_budget: int) -> List[ModelAgent]:
    out: List[ModelAgent] = []
    for a in settings.agents:
        a_low = a.lower()
        if a_low == "openai":
            backend = make_backend_for("openai", settings.openai_model)
        elif a_low == "claude":
            backend = make_backend_for("anthropic", settings.claude_model)
        elif a_low == "gemini":
            backend = make_backend_for("gemini", settings.gemini_model)
        else:
            raise ValueError(f"Unknown agent '{a}'")
        out.append(
            ModelAgent(
                name=a,
                backend=backend,
                tool_router=tool_router,
                tool_allowlist=(tool_router.list_tools()
                                if tool_router else []),
                tool_budget_per_round=tool_budget,
            )
        )
    return out


def _role_for_name(name: str) -> str:
    n = name.lower()
    if n == "planner":
        return "Planner"
    if n == "judge":
        return "Judge"
    return "Agent"


async def run_debate_streaming(topic: str, total_rounds: int, emit: Callable[[Dict[str, Any]], None]) -> None:
    """Run the debate and call `emit(evt)` for each UI event (live)."""
    if not topic or topic.strip().lower() in {"your topic here", "topic here"}:
        raise ValueError("Please provide a real topic")

    # 1) Optional MCP tools (enabled by default if servers are provided via settings)
    router: Optional[MCPToolRouter] = None
    if settings.mcp_servers_cmds:
        router = MCPToolRouter(settings.mcp_servers_cmds)
        await router.start()
        print("[engine] MCP tools:", router.list_tools(), flush=True)

    # 2) Debating model agents
    agents = make_model_agents(router, settings.tool_budget_per_round)

    # 3) Planner
    if os.getenv("ENABLE_PLANNER", "true").lower() in ("1", "true", "yes", "on"):
        agents = [PlannerAgent("Planner", tool_router=router)] + agents

    # 4) Judge
    judge_vendor = os.getenv("JUDGE_VENDOR", "openai").lower()
    if judge_vendor == "openai":
        judge_model = os.getenv("JUDGE_MODEL", settings.openai_model)
    elif judge_vendor == "anthropic":
        judge_model = os.getenv("JUDGE_MODEL", settings.claude_model)
    elif judge_vendor == "gemini":
        judge_model = os.getenv("JUDGE_MODEL", settings.gemini_model)
    else:
        raise ValueError(
            "JUDGE_VENDOR must be one of: openai | anthropic | gemini")

    judge_backend = make_backend_for(judge_vendor, judge_model)
    judge = JudgeAgent(
        name="Judge",
        backend=judge_backend,
        tool_router=router,
        tool_allowlist=(router.list_tools() if router else []),
        tool_budget_per_round=settings.tool_budget_per_round,
    )
    agents.append(judge)

    # 5) Orchestrate debate (STREAMING)
    bus = ConversationBus()
    graph = build_graph()

    # ⬇️ IMPORTANT: pass the already-started router into DebateState
    state = DebateState(
        topic=topic,
        round_no=1,
        agents=agents,
        bus=bus,
        total_rounds=total_rounds,
        mcp_router=router,
    )

    roster = [{"name": a.name, "role": _role_for_name(a.name)} for a in agents]
    emit({"type": "run.start", "topic": topic,
         "agents": roster, "total_rounds": total_rounds})

    printed = 0
    last_round_banner = None
    print(
        f"[engine] Debate starting: {topic} (rounds={total_rounds})", flush=True)

    async for _ in graph.astream(state, stream_mode="values"):
        hist = bus.history()
        while printed < len(hist):
            m = hist[printed]

            if m.type == "analysis" and last_round_banner != "analyses":
                last_round_banner = "analyses"

            if m.type in ("argument", "rebuttal"):
                if last_round_banner != f"round-{m.round}":
                    last_round_banner = f"round-{m.round}"
                    emit({"type": "round.start", "round": m.round})

            if m.type == "analysis":
                emit({"type": "message", "agent": m.agent, "role": _role_for_name(m.agent),
                      "phase": "analysis", "content": m.content})
            elif m.type in ("argument", "rebuttal"):
                emit({"type": "message", "agent": m.agent, "role": _role_for_name(m.agent),
                      "phase": m.type, "round": m.round, "content": m.content})
            elif m.type == "final_conclusion":
                emit({"type": "conclusion", "agent": m.agent, "content": m.content})
            elif m.type == "system":
                emit({"type": "system", "content": m.content})

            if m.tool_calls:
                for call in m.tool_calls:
                    emit({"type": "tool", "agent": m.agent, "details": {
                        "tool": f"{call.get('server')}:{call.get('tool')}",
                        "args": call.get("args", {}),
                        "result": call.get("result")
                    }})

            printed += 1

    if router:
        await router.stop()

    emit({"type": "run.end"})
    print("[engine] Debate complete", flush=True)

# --------------------------------------------------------------------------------------
# Flask app (serves HTML and exposes streaming endpoint)
# --------------------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")


@app.get("/api/ping")
def ping():
    return jsonify({"ok": True})


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/debate/stream")
def api_debate_stream():
    """
    Server-Sent Events endpoint.
    Usage from browser:
      new EventSource(`/api/debate/stream?topic=...&rounds=3`)
    """
    topic = (request.args.get("topic") or "").strip()
    try:
        rounds = int(request.args.get("rounds") or 3)
    except Exception:
        rounds = 3

    q: Queue[Optional[Dict[str, Any]]] = Queue()

    def emit(evt: Dict[str, Any]) -> None:
        q.put(evt)

    def runner():
        """Run the async engine and push events into the queue."""
        try:
            asyncio.run(run_debate_streaming(topic, rounds, emit))
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=runner, daemon=True).start()

    @stream_with_context
    def event_stream():
        yield ": start\n\n"
        while True:
            evt = q.get()
            if evt is None:
                yield "event: done\ndata: {}\n\n"
                break
            try:
                payload = json.dumps(evt, ensure_ascii=False)
            except Exception as e:
                payload = json.dumps(
                    {"type": "error", "message": f"serialize failed: {e}"})
            yield f"data: {payload}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(event_stream(), mimetype="text/event-stream", headers=headers)


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True,
            use_reloader=False, threaded=True)
