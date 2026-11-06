import json

import pytest
from evals_helpers import answer_eval
from fastapi.testclient import TestClient
from mcp import Client

from flightops import aviation, mcp_server
from flightops.api import app
from flightops.mcp_client import McpTools, stdio_server


class FakeRetriever:
    def search(self, query, k=5):
        return [{"doc": "14 CFR 107.51", "title": "t", "url": "u", "text": query}][:k]


@pytest.fixture
def fake_retriever(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_retriever", lambda: FakeRetriever())


EXPECTED_TOOLS = {"search_docs", "get_metar", "check_part107_weather", "lookup_airspace"}


async def test_tool_schemas_are_valid_for_the_anthropic_api():
    async with Client(mcp_server.server) as client:
        tools = await McpTools(client).list_tools()
    assert {t["name"] for t in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool["description"]
        assert tool["input_schema"]["type"] == "object"


async def test_search_tool_returns_json(fake_retriever):
    async with Client(mcp_server.server) as client:
        output, is_error = await McpTools(client).call_tool("search_docs", {"query": "altitude"})
    assert is_error is False
    assert json.loads(output)["results"][0]["doc"] == "14 CFR 107.51"


async def test_weather_tool_combines_metar_and_rules(monkeypatch):
    metar = {"station": "KPHX", "visib": "1", "clouds": []}
    monkeypatch.setattr(aviation, "get_metar", lambda station: metar)
    async with Client(mcp_server.server) as client:
        output, is_error = await McpTools(client).call_tool(
            "check_part107_weather", {"station": "KPHX", "planned_altitude_ft_agl": 200}
        )
    assert is_error is False
    assert json.loads(output)["overall"] == "fail"


async def test_tool_exception_becomes_an_mcp_error():
    async with Client(mcp_server.server) as client:
        output, is_error = await McpTools(client).call_tool(
            "lookup_airspace", {"latitude": 200, "longitude": 0}
        )
    assert is_error is True
    assert "out of range" in output


async def test_real_stdio_server_lists_tools():
    async with Client(stdio_server()) as client:
        tools = await McpTools(client).list_tools()
    assert {t["name"] for t in tools} == EXPECTED_TOOLS


class FakeAgent:
    async def ask(self, question):
        return {"answer": f"echo {question}", "sources": [], "tools_used": []}


class ApiTools:
    async def call_tool(self, name, arguments):
        return json.dumps({"results": [{"title": arguments["query"]}]}), False


@pytest.fixture
def api():
    app.state.tools = ApiTools()
    app.state.agent = FakeAgent()
    with TestClient(app) as client:
        yield client
    app.state.tools = None
    app.state.agent = None


def test_health_and_index(api):
    assert api.get("/api/health").json() == {"status": "ok", "agent_enabled": True}
    assert "FlightOps Copilot" in api.get("/").text


def test_ask(api):
    assert api.post("/api/ask", json={"question": "How high?"}).json()["answer"] == "echo How high?"


def test_ask_validation(api):
    assert api.post("/api/ask", json={"question": ""}).status_code == 422
    assert api.post("/api/ask", json={"question": "x" * 2001}).status_code == 422


def test_ask_without_key_explains_itself_and_search_still_works(api):
    app.state.agent = None
    response = api.post("/api/ask", json={"question": "How high?"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert (
        api.post("/api/search", json={"query": "altitude"}).json()["results"][0]["title"]
        == "altitude"
    )


def test_answer_grading_requires_facts_tools_and_sources():
    case = {
        "must_match": ["400"],
        "must_not_match": ["500 feet"],
        "required_tools": ["search_docs"],
        "required_doc": "14 CFR 107.51",
    }
    good = {
        "answer": "400 feet",
        "tools_used": [{"name": "search_docs"}],
        "sources": [{"doc": "14 CFR 107.51"}],
    }
    assert answer_eval.grade(case, good) == []
    assert answer_eval.grade(case, {**good, "answer": "500 feet"})
    assert answer_eval.grade(case, {**good, "tools_used": []})
    assert answer_eval.grade(case, {**good, "sources": []})
