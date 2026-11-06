"""Thin adapter from the MCP client to the tool format the Anthropic API expects."""

from __future__ import annotations

import sys

from mcp import Client, StdioServerParameters


def stdio_server() -> StdioServerParameters:
    """Launch the FlightOps MCP server as a subprocess of this Python interpreter."""
    return StdioServerParameters(command=sys.executable, args=["-m", "flightops.mcp_server"])


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
