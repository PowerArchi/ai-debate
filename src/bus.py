# src/bus.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DebateMessage:
    """A structured message exchanged between agents, aligning with Agent-to-Agent (A2A) concepts.

    Each DebateMessage represents one communication turn in the debate:
      - agent:     sender name
      - to:        optional list of recipients (A2A-style directed delivery)
      - content:   main text content
      - type:      "analysis" | "argument" | "rebuttal" | "final_conclusion" | "system"
      - citations: optional evidence metadata
      - tool_calls:optional MCP tool call records
      - status:    optional lifecycle marker from the agent ("complete", "error", etc.)
      - timestamp: ISO-8601 when created
    """
    # Core message fields
    type: str
    agent: str
    round: int
    content: str

    # Optional metadata
    citations: Optional[List[Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Recipients for directed messages (None = broadcast to relevant agents)
    to: Optional[List[str]] = None

    # Optional status reported by the agent ("complete", "error", ...)
    status: Optional[str] = None

    # Housekeeping
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Convenience
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationBus:
    """Very small in-memory message bus for storing the debate transcript."""
    def __init__(self) -> None:
        self._msgs: List[DebateMessage] = []

    # Publish either a DebateMessage or a dict with compatible keys
    def publish(self, msg: DebateMessage | Dict[str, Any]) -> None:
        if isinstance(msg, dict):
            dm = DebateMessage(
                type=msg.get("type", ""),
                agent=msg.get("agent", ""),
                round=int(msg.get("round", 0)),
                content=msg.get("content", "") or "",
                citations=msg.get("citations"),
                tool_calls=msg.get("tool_calls"),
                to=msg.get("to"),
                status=msg.get("status"),
                timestamp=msg.get("timestamp") or (datetime.utcnow().isoformat() + "Z"),
            )
            self._msgs.append(dm)
        else:
            self._msgs.append(msg)

    def history(self) -> List[DebateMessage]:
        # Return a shallow copy to avoid accidental mutation
        return list(self._msgs)

    def history_upto_round(self, r: int) -> List[DebateMessage]:
        """Snapshot up to round r. Analyses (round 0) always included."""
        if r <= 0:
            return [m for m in self._msgs if m.type in ("plan", "analysis", "system")]
        return [
            m for m in self._msgs
            if (m.type in ("plan", "analysis", "argument", "rebuttal", "final_conclusion", "system"))
            and (m.round <= r)
        ]

    def tail(self, n: int = 20) -> List[DebateMessage]:
        return self._msgs[-n:]
