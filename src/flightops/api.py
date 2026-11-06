"""FastAPI app. Run: uvicorn flightops.api:app --port 8000"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mcp import Client
from pydantic import BaseModel, Field

from flightops.agent import Agent
from flightops.mcp_client import McpTools, stdio_server

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    if getattr(app.state, "tools", None) is not None:  # Tests inject their own tools.
        yield
        return
    async with Client(stdio_server()) as client:
        app.state.tools = McpTools(client)
        app.state.agent = Agent(app.state.tools) if has_credentials() else None
        yield


app = FastAPI(title="FlightOps Copilot", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "agent_enabled": getattr(app.state, "agent", None) is not None}


@app.post("/api/search")
async def search(request: SearchRequest) -> dict:
    """Document search only. Uses no model calls, so it works without an API key."""
    output, is_error = await app.state.tools.call_tool(
        "search_docs", {"query": request.query, "k": request.k}
    )
    if is_error:
        raise HTTPException(status_code=502, detail=output)
    return json.loads(output)


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict:
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="The agent is off because ANTHROPIC_API_KEY is not set. Search still works.",
        )
    try:
        return await agent.ask(request.question)
    except anthropic.AuthenticationError as error:
        raise HTTPException(
            status_code=503, detail="The Anthropic API key was rejected."
        ) from error
    except anthropic.RateLimitError as error:
        raise HTTPException(status_code=429, detail="Rate limited. Try again shortly.") from error
    except anthropic.APIStatusError as error:
        raise HTTPException(
            status_code=502, detail=f"Model API error {error.status_code}."
        ) from error
    except anthropic.APIConnectionError as error:
        raise HTTPException(status_code=502, detail="Could not reach the model API.") from error
