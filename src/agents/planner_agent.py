# src/agents/planner_agent.py
from __future__ import annotations
from typing import Dict, Any
from .base import BaseAgent, AgentContext


PLAN_TEMPLATE = """Structured Debate Plan

Topic
- {topic}

1) Rounds & Turn Order (orchestrator-controlled)
- Total rounds: use the run's configured value.
- Turn order: use the orchestrator's configured order each round.
- If rounds are barriered, all agents respond to the same prior transcript; otherwise turns are sequential.

2) Role / Perspective Assignments
- Each agent adopts a distinct, contrasting perspective (e.g., optimistic futurist, cautious pragmatist, policy/ops skeptic).
- Agents must keep a stable stance across rounds. If a stance changes, the message MUST begin with:
  **STANCE CHANGE:** <new stance>  — and include at least one recent citation justifying the change.

3) Evidence Requirements (per round, per agent)
- Provide ≥1 **recent** source when asserting facts (prefer the last 12–18 months when applicable).
- Acceptable types: market/industry research, peer-reviewed or technical reports, enterprise case studies, benchmarks, or survey data.
- Embed citations inline using a consistent marker, e.g., [CITATION: Source, Year, Title/Note].
- Tool usage (if available) should prioritize retrieving up-to-date evidence rather than generic summaries.

4) Key Term Definitions (to be established early)
- Agents should explicitly define any ambiguous or critical terms in the topic (e.g., how “dominant,” “adoption,” or “production” are measured).
- Agree (or note disagreements) on metrics/thresholds so comparisons remain coherent across rounds.

5) Minimal Tool Strategy (if MCP/tools are available)
- Early rounds: pull trend/benchmark/context data.
- Mid rounds: seek counter-evidence or independent validation of contested claims.
- Late rounds: verify figures or timelines cited during rebuttals.
- Keep calls minimal and record them; prefer recent, high-quality sources.

6) Risks & Safeguards
- Common pitfalls: vague or dated claims, circular logic, ignoring opposing points, and undefined metrics.
- Safeguards: require specificity, favor quantifiable evidence, call out weak/dated sources, and surface uncertainties.

Planner Notes
- The Planner does not debate; it sets rules and nudges toward evidence-based discussion.
- The Judge must weight arguments by evidence density/recency, discount unsupported claims, and note any stance changes.
"""


class PlannerAgent(BaseAgent):
    """
    Lightweight, deterministic planner (no LLM).
    Compatible with:  PlannerAgent("Planner", tool_router=router)
    The planner does not participate in debate rounds.
    """
    participates = False

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    async def analysis(self, ctx: AgentContext) -> Dict[str, Any]:
        # Emit a general, reusable plan (NOT a topic-specific prompt).
        text = PLAN_TEMPLATE.format(topic=ctx.topic)
        return {
            "text": text,
            "citations": [],
            "tool_calls": [],
            "status": "reading"  # preserves the UI behavior you expect
        }

    async def debate_turn(self, ctx: AgentContext) -> Dict[str, Any]:
        # Gentle, generic reminder—does not inject topic- or agent-specific content.
        return {
            "text": "Planner: Reminder — cite recent evidence (last 12–18 months) and address all opposing points. Judge will discount unsupported claims.",
            "citations": [],
            "tool_calls": [],
            "status": "complete"
        }

    async def propose_consensus(self, ctx: AgentContext) -> Dict[str, Any]:
        # Planner doesn’t write conclusions; it nudges structure only.
        return {
            "text": "Planner: Recommend a joint conclusion that states common ground, remaining disagreements, explicit confidence (0–1), and caveats.",
            "citations": [],
            "tool_calls": [],
            "status": "complete"
        }
