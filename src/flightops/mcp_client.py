"""Thin adapter from the MCP client to the tool format the Anthropic API expects."""

from __future__ import annotations

import os
import sys

from mcp import Client, StdioServerParameters

# The MCP client starts servers with a minimal environment, so settings the server needs must be
# passed on explicitly. The API key is deliberately left out: the server never calls the model.
SERVER_ENV_VARS = ("PYTHONPATH", "FASTEMBED_CACHE_PATH", "FLIGHTOPS_RETRIEVAL_MODE")


def stdio_server() -> StdioServerParameters:
    """Launch the FlightOps MCP server as a subprocess of this Python interpreter."""
    env = {name: os.environ[name] for name in SERVER_ENV_VARS if name in os.environ}
    return StdioServerParameters(
        command=sys.executable, args=["-m", "flightops.mcp_server"], env=env
    )


class McpTools:
    """Wraps a connected mcp.Client. Use inside `async with Client(...) as client`."""

    def __init__(self, client: Client) -> None:
        self.client = client

    async def list_tools(self) -> list[dict]:
        listed = await self.client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in listed.tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = await self.client.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return text, bool(result.is_error)
