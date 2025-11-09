# src/mcp_tools/mcp_client.py
from __future__ import annotations
import anyio
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack
from types import SimpleNamespace

# Optional: only present if the MCP SDK is installed
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
except Exception:
    ClientSession = None  # type: ignore
    stdio_client = None   # type: ignore
    StdioServerParameters = None  # type: ignore


class MCPToolHandle:
    """Represents a single tool exposed by an MCP server (client-side handle)."""
    def __init__(self, server_name: str, tool_name: str, session: Any):
        self.server_name = server_name
        self.tool_name = tool_name
        self.session = session  # must provide call_tool(name, kwargs)

    async def call(self, **kwargs) -> Dict[str, Any]:
        result = await self.session.call_tool(self.tool_name, kwargs)
        return {
            "server": self.server_name,
            "tool": self.tool_name,
            "args": kwargs,
            "result": result,
        }


# ---------- In-process local stub (no subprocess, no JSON-RPC, Python 3.11 OK) ----------

class _InProcessStubSession:
    """
    Minimal session that mimics the subset of the MCP client API we use:
      - initialize()
      - list_tools() -> object with .tools (list of objects with .name)
      - call_tool(name, kwargs)
      - close()
    """
    def __init__(self) -> None:
        self._tools = [SimpleNamespace(name="echo")]

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, kwargs: Dict[str, Any]) -> Any:
        if name != "echo":
            raise ValueError(f"Unknown tool '{name}' for local-stub")
        text = kwargs.get("text") or kwargs.get("q") or ""
        return {"message": "Echo (local-stub) active", "echo": text, "args": kwargs}

    async def close(self) -> None:
        return None


class MCPToolRouter:
    """
    Manages multiple tool endpoints and routes calls.
    Supports:
      1) real MCP servers via stdio (if MCP SDK present)
      2) a built-in 'local-stub' in-process server (no subprocess, 3.11-friendly)
    """
    def __init__(self, servers_argv: List[List[str]]):
        # Example inputs:
        #   [["local-stub"]]  -> in-process stub
        #   [["python","-u","src/mcp_tools/mcp_server_local_stub.py"]]  -> real stdio server (requires SDK & real server)
        self.servers_argv = servers_argv
        self.sessions: List[Any] = []
        self.tools: Dict[str, MCPToolHandle] = {}
        self._stack: Optional[AsyncExitStack] = None

    async def start(self) -> None:
        if self._stack is not None:
            return
        self._stack = AsyncExitStack()

        for argv in self.servers_argv:
            if not isinstance(argv, list) or not argv:
                raise ValueError(f"Invalid MCP server argv: {argv!r}")

            server_name = argv[0].strip().lower()

            # Case 1: in-process local stub (no subprocess, no SDK)
            if server_name == "local-stub":
                session = _InProcessStubSession()
                await session.initialize()
                tools = await session.list_tools()
                for t in tools.tools:
                    key = f"{server_name}:{t.name}"
                    self.tools[key] = MCPToolHandle(server_name, t.name, session)
                self.sessions.append(session)
                continue

            # Case 2: real stdio server (requires MCP SDK AND a real MCP server)
            if ClientSession is None or stdio_client is None or StdioServerParameters is None:
                raise RuntimeError(
                    "MCP SDK not available but non-stub server requested. "
                    "Either install MCP SDK or use ['local-stub']."
                )

            # Split into executable and args
            cmd = argv[0]
            args = argv[1:]
            params = StdioServerParameters(command=cmd, args=args, env=None, cwd=None)
            send, recv = await self._stack.enter_async_context(stdio_client(params))

            session = ClientSession(send, recv)
            await session.initialize()

            tools = await session.list_tools()
            for t in tools.tools:
                key = f"{cmd}:{t.name}"  # use executable name as server_name for display
                self.tools[key] = MCPToolHandle(cmd, t.name, session)

            self.sessions.append(session)

    async def stop(self) -> None:
        # Close protocol sessions politely
        for s in self.sessions:
            with anyio.move_on_after(1):
                try:
                    await s.close()
                except Exception:
                    pass
        self.sessions.clear()

        # Close stdio contexts if we opened any
        if self._stack is not None:
            with anyio.move_on_after(1):
                try:
                    await self._stack.aclose()
                except Exception:
                    pass
            self._stack = None

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    async def call(self, tool_key: str, **kwargs) -> Dict[str, Any]:
        if tool_key not in self.tools:
            raise ValueError(f"Unknown MCP tool '{tool_key}'. Known tools: {self.list_tools()}")
        return await self.tools[tool_key].call(**kwargs)
