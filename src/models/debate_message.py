# src/models/debate_message.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DebateMessage:
    """
    Represents one message in the debate, exchanged between agents.
    Each message may include optional tool call records if the agent
    used any MCP tools during its reasoning.
    """
    role: str
    content: str
    to: Optional[str] = None
    # Record any MCP tool calls made during this turn
    tool_calls: Optional[List[Dict[str, Any]]] = field(default=None)
