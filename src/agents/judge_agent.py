# src/agents/judge_agent.py
from __future__ import annotations
from typing import Dict, Any
from .model_agent import ModelAgent, LLMBackend, RESEARCH_PREFIX
from .base import AgentContext


JUDGE_PROMPT = RESEARCH_PREFIX + """
You are the neutral Judge for this research debate. Your task is to produce a single, self-contained final conclusion based *only* on the full transcript provided below.

**Instructions:**

Using the full transcript, structure your conclusion as follows:

1)  **Summary of Strongest Shared Ground (2–4 sentences):** Identify the key points or assumptions where all participants converged or largely agreed.
2)  **Remaining Disagreements (Briefly):** Concisely list the core points of contention or differing interpretations that persisted through the debate.
3)  **Balanced, Practical Conclusion with Confidence Score and Caveats:**
    * Synthesize the arguments into a balanced, practical takeaway based on the weight of evidence and reasoning presented.
    * Assign a **Confidence Score (0.0 to 1.0)** reflecting the certainty of this conclusion *based on the debate transcript*. Justify the score briefly (e.g., strong consensus vs. significant uncertainty or evidence gaps).
    * List **Explicit Caveats and Uncertainties** highlighted during the debate (e.g., dependence on definitions, missing data, external factors, model limitations).
4)  **Critical Evaluation of Evidence and Agent Performance:**
    * **Evidence Quality:** Evaluate the overall strength, recency, and credibility of sources cited across the debate. Note any significant evidence gaps mentioned.
    * **Evidence Use & Rule Adherence:**
        * Explicitly state whether agents consistently provided recent citations for factual claims as required.
        * **Crucially, state if any agent's arguments were discounted due to failure to provide supporting evidence for specific claims, and explain how this impacted credibility.**
    * **Stance Consistency:** Note whether agents maintained their initial stances. **Explicitly mention any flagged (`**STANCE CHANGE:**`) or *unflagged* significant shifts in stance observed during the debate.** (Consider if arguments consistently aligned with assigned roles, but prioritize evidence and stance consistency).
    * **Overall Credibility Weighting:** Briefly explain which agent(s) presented the most credible case *based on evidence density, quality, and consistent reasoning* within the transcript.

**Rules for the Judge:**
* Base your judgment *solely* on the provided transcript. Do not introduce external information or opinions.
* Remain strictly neutral. Do not take sides or offer personal recommendations.
* Focus on the logical structure, evidence presented, and adherence to debate rules (citations, stance consistency).
* Keep the output general and research-oriented, avoiding actionable advice (especially financial).
"""

class JudgeAgent(ModelAgent):
    participates = False

    def __init__(self, name: str, backend: LLMBackend, **kwargs):
        super().__init__(name, backend, **kwargs)

    async def analysis(self, ctx: AgentContext) -> Dict[str, Any]:
        return {"text": "", "citations": [], "tool_calls": []}

    async def debate_turn(self, ctx: AgentContext) -> Dict[str, Any]:
        return {"text": "", "citations": [], "tool_calls": []}

    async def propose_consensus(self, ctx: AgentContext) -> Dict[str, Any]:
        prompt = f"""{JUDGE_PROMPT}

--- Transcript Start ---
{ctx.history_text}
--- Transcript End ---
"""
        text = await self.backend.complete(prompt)
        return {"text": text, "citations": [], "tool_calls": [], "status": "complete"}
