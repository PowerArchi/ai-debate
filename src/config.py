# src/config.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import List

load_dotenv()

def _csv(key: str, default: str) -> list[str]:
    raw = os.getenv(key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

def _env_bool(key: str, default: bool = True) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _parse_mcp_servers_env() -> list[list[str]]:
    """
    Parses MCP_SERVERS as a JSON list.
    Accepts either:
      - JSON strings that are themselves JSON-encoded argv arrays, or
      - direct JSON arrays.

    """
    raw = os.getenv("MCP_SERVERS", "[]").strip()
    try:
        parsed = json.loads(raw)
        cmds: list[list[str]] = []
        for item in parsed:
            if isinstance(item, str):
                # Item is a JSON-encoded argv array
                try:
                    argv = json.loads(item)
                    if isinstance(argv, list):
                        cmds.append([str(x) for x in argv])
                except Exception:
                    # ignore malformed items
                    pass
            elif isinstance(item, list):
                cmds.append([str(x) for x in item])
        return cmds
    except Exception:
        return []

@dataclass
class Settings:
    # API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Model names (defaults aligned with your .env)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Debate configuration
    agents: list[str] = None
    tool_budget_per_round: int = int(os.getenv("TOOL_BUDGET_PER_ROUND", "3"))

    # MCP
    mcp_servers_cmds: list[list[str]] = None

    def __post_init__(self):
        self.agents = _csv("AGENTS", "OpenAI,Claude,Gemini")
        self.mcp_servers_cmds = _parse_mcp_servers_env()

settings = Settings()
