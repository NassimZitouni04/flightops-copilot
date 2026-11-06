"""MCP server for document search and live aviation lookups.

Run: python -m flightops.mcp_server
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from flightops import aviation
from flightops.retrieval import Retriever

server = MCPServer(
    "flightops",
    instructions="Search FAA Part 107 and PX4 docs, and look up live weather and airspace.",
)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


def _lookup(fetch: Callable[..., Any], *args: Any) -> Any:
    """Turn expected lookup failures into tool errors, so Claude sees the reason and can report it.

    The MCP server hides the message of any other exception type.
    """
    try:
        return fetch(*args)
    except aviation.LookupError_ as error:
        raise ToolError(str(error)) from error
    except httpx.HTTPError as error:
        raise ToolError(f"The aviation data service could not be reached: {error}") from error


@server.tool()
def search_docs(query: str, k: int = 5) -> dict:
    """Search FAA Part 107 regulations and PX4 flight documentation. Use this for any question
    about drone rules, flight modes, failsafes or PX4 configuration. Returns passages with URLs."""
    results = get_retriever().search(query, k=max(1, min(k, 10)))
    return {
        "results": [
            {"doc": r["doc"], "title": r["title"], "url": r["url"], "text": r["text"]}
            for r in results
        ]
    }


@server.tool()
def get_metar(station: str) -> dict:
    """Get the latest METAR weather observation for an ICAO airport code such as KPHX."""
    return _lookup(aviation.get_metar, station)


@server.tool()
def check_part107_weather(station: str, planned_altitude_ft_agl: int = 400) -> dict:
    """Check the latest METAR at an ICAO station against Part 107 visibility, cloud clearance and
    altitude limits for a planned altitude. Use this when asked whether weather allows a flight."""
    metar = _lookup(aviation.get_metar, station)
    return {"metar": metar, **aviation.evaluate_weather(metar, planned_altitude_ft_agl)}


@server.tool()
def lookup_airspace(latitude: float, longitude: float) -> dict:
    """List FAA controlled airspace layers at a coordinate and say whether a surface layer means
    ATC authorization is needed under 14 CFR 107.41. Screening only: not a LAANC authorization."""
    return _lookup(aviation.lookup_airspace, latitude, longitude)


if __name__ == "__main__":
    server.run("stdio")
