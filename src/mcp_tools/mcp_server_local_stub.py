# src/mcp_tools/mcp_server_local_stub.py
"""
A minimal local MCP server stub for offline testing.

This server speaks just enough of the Model Context Protocol (MCP)
to satisfy the MCP client (via stdio). It advertises one fake tool
called "echo", which simply returns whatever arguments were sent.

It never leaves your machine and requires no network or API keys.
"""

import sys
import json
import asyncio

async def main():
    # Announce available tools (per MCP spec)
    tools_list = {
        "tools": [
            {
                "name": "echo",
                "description": "Echoes the provided arguments back as result.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    }

    # Write initial handshake response so the client knows we're alive
    sys.stdout.write(json.dumps({"type": "tools_list", "data": tools_list}) + "\n")
    sys.stdout.flush()

    # Wait for incoming requests (but we’ll just echo back a canned result)
    while True:
        line = sys.stdin.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        try:
            request = json.loads(line)
            response = {
                "type": "tool_result",
                "tool": "echo",
                "args": request.get("params", {}),
                "result": {"message": "Echo stub active", "received": request},
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
