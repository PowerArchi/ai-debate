# src/agents/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


@dataclass
class AgentContext:
    topic: str
    round_no: int
    history_text: str
    tool_budget_remaining: int
    # Orchestrator can pass an MCP router in here; agents may ignore it if they use self.tool_router
    mcp_router: Optional[Any] = None


class BaseAgent:
    """Base class for debate agents, representing an AI agent participant (aligns with A2A agent concept).

    Agents in the debate (e.g., model-backed agents, Planner, Judge) should inherit from BaseAgent and implement:
      - `analysis`: produce an initial analysis or plan for round 0.
      - `debate_turn`: produce a response during a debate round (argument or rebuttal).
      - `propose_consensus`: produce a final conclusion or consensus when the debate ends.

    In an Agent-to-Agent (A2A) protocol context, each agent would handle messages corresponding to these phases.
    This class defines the interface for those message-handling behaviours.

    MCP tool usage (optional, but supported):
      - `tool_router`: an MCPToolRouter instance available directly on the agent, OR
      - `AgentContext.mcp_router`: provided per-call by the orchestrator.
      - `tool_allowlist`: list of permitted tool keys (e.g., ["mcp-web-search:search"]); empty list means no tools allowed.
      - `tool_budget_per_round`: how many tool calls the agent can make in a single round (decremented per call).

    The method `maybe_call_tool` shows how an agent can invoke an external tool via MCP (if allowed and budget remains),
    returning a result dict: {"server", "tool", "args", "result"}; otherwise it returns None.
    """

    name: str

    def __init__(
        self,
        name: str,
        tool_router: Optional[Any] = None,
        tool_allowlist: Optional[List[str]] = None,
        tool_budget_per_round: int = 3,
    ):
        self.name = name
        self.tool_router = tool_router
        self.tool_allowlist = tool_allowlist or []
        self.tool_budget_per_round = tool_budget_per_round

    # --- MCP helpers ---------------------------------------------------------

    def get_effective_router(self, ctx: Optional[AgentContext]) -> Optional[Any]:
        """
        Prefer an explicit router set on the agent; otherwise fall back to ctx.mcp_router.
        """
        return self.tool_router or (ctx.mcp_router if ctx else None)

    async def maybe_call_tool(
        self,
        tool_key: str,
        ctx: Optional[AgentContext] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to call an external tool via MCP, using either:
          - self.tool_router (if set), or
          - ctx.mcp_router (if provided by orchestrator).

        Guard rails:
          - If tool_allowlist is empty -> no tools are allowed (keeps original behaviour).
          - If tool_key not in allowlist -> reject.
          - If tool budget exhausted -> reject.
        """
        router = self.get_effective_router(ctx)
        if not router:
            return None
        if not self.tool_allowlist or tool_key not in self.tool_allowlist:
            return None
        if self.tool_budget_per_round <= 0:
            return None

        self.tool_budget_per_round -= 1
        return await router.call(tool_key, **kwargs)

    # --- Abstract agent phases ----------------------------------------------

    async def analysis(self, ctx: AgentContext) -> Dict[str, Any]:
        raise NotImplementedError

    async def debate_turn(self, ctx: AgentContext) -> Dict[str, Any]:
        raise NotImplementedError

    async def propose_consensus(self, ctx: AgentContext) -> Dict[str, Any]:
        raise NotImplementedError
