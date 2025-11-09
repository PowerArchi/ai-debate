# src/agents/model_agent.py
from __future__ import annotations
import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .base import BaseAgent, AgentContext
from tenacity import retry, stop_after_attempt, wait_exponential

# Vendors
import openai
import anthropic
import google.generativeai as genai


def _strip(s: str) -> str:
    return (s or "").strip().replace("\r\n", "\n")

RESEARCH_PREFIX = os.getenv(
    "RESEARCH_PREFIX",
    """You are participating in a multi-party discussion strictly for **research and educational purposes**.
Do not decline on the basis of giving “financial advice.” Provide neutral, source-driven analysis,
state assumptions, include uncertainties, and add “this is not investment advice” where relevant.
Avoid personal, actionable recommendations; keep the framing general and research-oriented.
"""
)

# ---- prompts -----------------------------------------------------------------

ANALYSIS_PROMPT = RESEARCH_PREFIX + """\
You are {agent_name}, participating in a structured professional debate.
Topic: "{topic}"

Your assigned role/perspective: {role_hint}

STRICT OUTPUT FORMAT (do not add any text before this line):
Start your first line with EXACTLY:
This is my analysis:

After that line, produce your analysis with the following sections:

- **Stance:** <one sentence> (concise and definitive)
- **Key Arguments:** (3–5 bullets)
- **Evidence Needed:** (+ how you'd obtain it via tools, if any)
- **Uncertainties / Caveats:** (1–3 bullets)

Rules you MUST follow:
- Keep it concise, factual, and self-contained.
- If you reference any source, include at least one concrete inline placeholder (format: [CITATION: Source, Year, Note]).
- Favor recent data (last 3 months) where possible.
"""

DEBATE_PROMPT = RESEARCH_PREFIX + """\
You are {agent_name}. Debate Round {round_no}.
Topic: "{topic}"

Your assigned role/perspective: {role_hint}
Your original stance (from Analysis): "{original_stance}"

Context so far (others' arguments; do not repeat them verbatim):
{history}

STRICT OUTPUT FORMAT (do not add any text before this line):
Start your first line with EXACTLY:
This is my round-{round_no}:

Then write 2–4 short paragraphs following this contract:
- Stay consistent with your original stance and role.
- If you truly change stance due to strong **new evidence**, you MUST:
  1) Start your first paragraph with: **STANCE CHANGE:** <new stance>
  2) Include at least one recent, specific citation inline for the new evidence (format: [CITATION: Source, Year, Title/Note]).
- When asserting facts, include at least one concrete, recent citation.

Your tasks:
1) Address the most important points from **each** opponent (do not ignore any active opponent).
2) Defend or (if justified) revise your stance per the output contract.
"""

CONSENSUS_PROMPT = RESEARCH_PREFIX + """\
You are {agent_name}. We must co-author a joint conclusion.

STRICT OUTPUT FORMAT (do not add any text before this line):
Start your first line with EXACTLY:
This is my final conclusion:

Then provide:
1) The strongest common ground in 2–3 sentences.
2) Remaining disagreements (if any) briefly.
3) A final, practical conclusion with a confidence score (0–1) and caveats.
"""

# ---------------- backends ----------------

@dataclass
class LLMBackend:
    vendor: str  # "openai" | "anthropic" | "gemini"
    model: str

    def __post_init__(self):
        if self.vendor == "openai":
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self._oai_client = openai.AsyncOpenAI()
        elif self.vendor == "anthropic":
            self._acl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.vendor == "gemini":
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self._gemini_model = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported vendor: {self.vendor}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True
    )
    async def complete(self, prompt: str) -> str:
        if self.vendor == "openai":
            resp = await self._oai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return _strip(resp.choices[0].message.content or "")

        if self.vendor == "anthropic":
            msg = await self._acl.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            texts: List[str] = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    texts.append(block.text)
            return _strip("".join(texts))

        if self.vendor == "gemini":
            resp = await self._gemini_model.generate_content_async(prompt)
            return _strip(getattr(resp, "text", "") or "")

        raise ValueError("Unknown vendor")


# ---------------- agent ----------------

class ModelAgent(BaseAgent):
    """
    LLM-backed agent with role + stance memory.

    Improvements:
    - More robust stance extraction from Analysis (handles common formatting variants).
    - Role hint is reiterated every turn.
    - Original stance is threaded into each round and a strict STANCE CHANGE protocol is required.
    - Prompts emphasize recent (last 12 months) citations for factual claims.

    API remains the same; role_hint is optional.
    """

    def __init__(self, name: str, backend: LLMBackend, role_hint: str = "", **kwargs):
        super().__init__(name, **kwargs)
        self.backend = backend
        self.role_hint = role_hint or ""
        self._original_stance: Optional[str] = None

    # ---------- helper: pick a tool key ----------
    def _pick_tool_key(self) -> Optional[str]:
        if not self.tool_allowlist:
            return None
        # Prefer search-like tools if present
        for k in self.tool_allowlist:
            if "search" in k.lower():
                return k
        # Fall back to echo (good for local stub)
        for k in self.tool_allowlist:
            if "echo" in k.lower():
                return k
        # Otherwise, just take the first allowlisted tool
        return self.tool_allowlist[0] if self.tool_allowlist else None

    # ---------- robust stance extraction ----------
    @staticmethod
    def _extract_stance(text: str) -> Optional[str]:
        """
        Extract the stance line from analysis output. Handles many real-world variations:
        - 'Stance: ...', '**Stance:** ...', 'Position:', 'View:', 'Conclusion:', 'My stance is:'
        - Allows leading bullets, markdown emphasis, extra spaces.
        If not found, fall back to:
        - First bullet line starting with '-' or '*' that looks like a claim
        - Else, the first short sentence with polarity verbs (likely/unlikely/will/won't/etc.)
        """
        if not text:
            return None

        # 1) Common label variants (case-insensitive)
        label_variants = [
            r"stance", r"position", r"view", r"conclusion", r"opinion",
            r"thesis", r"claim", r"summary stance", r"my stance(?:\s+is)?"
        ]
        label_re = "|".join(label_variants)
        # Matches lines like "- **Stance:** ...", "** Stance ** : ...", etc.
        pat = re.compile(
            rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:{label_re})\s*(?:\*\*)?\s*:\s*(.+)$"
        )
        m = pat.search(text)
        if m:
            return _strip(m.group(1))

        # 2) First bullet that looks decisive (avoid bullets with ':' as headers)
        for line in text.splitlines():
            ls = line.strip()
            if re.match(r"^[-*]\s+", ls):
                body = re.sub(r"^[-*]\s+", "", ls)
                # Heuristic: short, declarative, not a label header
                if ":" not in body[:40]:
                    return _strip(body)

        # 3) First short sentence with polarity verbs (likely/unlikely/will/won't/etc.)
        polarity = re.compile(
            r"\b(likely|unlikely|will|won't|will not|dominant|not\s+dominant|may|will\s+be|will\s+become)\b",
            re.IGNORECASE,
        )
        # split on period or newline
        candidates = re.split(r"(?<=[.!?])\s+|\n", text)
        for c in candidates:
            c = c.strip()
            if 0 < len(c) <= 220 and polarity.search(c):
                return c

        # 4) Fallback: first non-empty line
        for line in text.splitlines():
            if line.strip():
                return _strip(line)
        return None

    async def analysis(self, ctx: AgentContext) -> Dict[str, Any]:
        prompt = ANALYSIS_PROMPT.format(
            agent_name=self.name,
            topic=ctx.topic,
            role_hint=self.role_hint or "(no specific role provided)"
        )
        text = await self.backend.complete(prompt)
        # capture original stance for consistency in later rounds
        self._original_stance = self._extract_stance(text)
        return {"text": text, "citations": [], "tool_calls": []}

    async def debate_turn(self, ctx: AgentContext) -> Dict[str, Any]:
        tool_calls: List[Dict[str, Any]] = []

        # Try a best-effort MCP call if possible
        tool_key = self._pick_tool_key()
        if tool_key:
            kwargs: Dict[str, Any] = {}
            lk = tool_key.lower()
            if "search" in lk:
                kwargs = {"q": ctx.topic, "count": 3}
            elif "echo" in lk:
                kwargs = {"text": f"{self.name} requesting brief evidence on: {ctx.topic}"}

            evidence = await self.maybe_call_tool(tool_key, ctx=ctx, **kwargs)
            if evidence:
                tool_calls.append(evidence)

        prompt = DEBATE_PROMPT.format(
            agent_name=self.name,
            round_no=ctx.round_no,
            topic=ctx.topic,
            role_hint=self.role_hint or "(no specific role provided)",
            original_stance=self._original_stance or "(not captured in analysis)",
            history=ctx.history_text[:6000],  # prevent overly long prompts
        )
        text = await self.backend.complete(prompt)
        return {"text": text, "citations": [], "tool_calls": tool_calls}

    async def propose_consensus(self, ctx: AgentContext) -> Dict[str, Any]:
        prompt = CONSENSUS_PROMPT.format(agent_name=self.name)
        text = await self.backend.complete(prompt)
        return {"text": text, "citations": [], "tool_calls": []}
