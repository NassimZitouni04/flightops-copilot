"""Runs the graph through the real Anthropic SDK with only the network mocked.

This checks the real request serialization and response parsing, including replaying content
blocks in later turns, without needing an API key or spending money.
"""

import json

import anthropic
import httpx2
import pytest
from anthropic import DefaultAsyncHttpxClient

from flightops.agent import MAX_MODEL_TURNS, Agent, cost_usd, extract_sources

USAGE = {"input_tokens": 100, "output_tokens": 50}


def message(content, stop_reason):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": USAGE,
    }


def tool_use(name, tool_input, tool_id="toolu_1"):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def text(value):
    return {"type": "text", "text": value}


class FakeTools:
    def __init__(self, output='{"results": []}', is_error=False, raises=None):
        self.output, self.is_error, self.raises = output, is_error, raises
        self.calls = []

    async def list_tools(self):
        return [
            {"name": "z_tool", "description": "z", "input_schema": {"type": "object"}},
            {"name": "search_docs", "description": "s", "input_schema": {"type": "object"}},
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.raises:
            raise self.raises
        return self.output, self.is_error


def build(responses, tools=None, status=200):
    """Create an agent whose SDK client replays canned responses and records request bodies."""
    requests = []
    queue = list(responses)

    def handler(request):
        requests.append({"body": json.loads(request.content), "headers": request.headers})
        return httpx2.Response(
            status, json=queue.pop(0) if queue else message([text("x")], "end_turn")
        )

    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="https://api.test",
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return Agent(tools or FakeTools(), client=client, trace_path=None), requests


async def test_answers_directly_without_tools():
    agent, requests = build([message([text("Hello.")], "end_turn")])
    result = await agent.ask("hi")
    assert result["answer"] == "Hello."
    assert result["tools_used"] == []
    assert len(requests) == 1


async def test_request_shape():
    agent, requests = build([message([text("ok")], "end_turn")])
    await agent.ask("hi")
    body, headers = requests[0]["body"], requests[0]["headers"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]
    assert body["output_config"] == {"effort": "medium"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "tool_choice" not in body
    assert "thinking" not in body
    assert [t["name"] for t in body["tools"]] == ["search_docs", "z_tool"]


async def test_tool_loop_replays_history_correctly():
    tools = FakeTools(output='{"results": [{"doc": "14 CFR 107.51", "url": "u1"}]}')
    agent, requests = build(
        [
            message(
                [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    text("Checking."),
                    tool_use("search_docs", {"query": "altitude"}),
                ],
                "tool_use",
            ),
            message([text("400 feet, per 14 CFR 107.51.")], "end_turn"),
        ],
        tools,
    )
    result = await agent.ask("How high?")

    assert result["answer"] == "400 feet, per 14 CFR 107.51."
    assert tools.calls == [("search_docs", {"query": "altitude"})]
    assert result["sources"] == [{"doc": "14 CFR 107.51", "url": "u1"}]
    assert result["model_turns"] == 2

    assistant, tool_turn = requests[1]["body"]["messages"][1:3]
    assert [block["type"] for block in assistant["content"]] == ["thinking", "text", "tool_use"]
    assert assistant["content"][0]["signature"] == "sig"
    assert tool_turn["content"][0]["type"] == "tool_result"
    assert tool_turn["content"][0]["tool_use_id"] == "toolu_1"
    assert tool_turn["content"][0]["is_error"] is False


async def test_parallel_tool_results_go_in_one_user_message():
    tools = FakeTools()
    agent, requests = build(
        [
            message(
                [
                    tool_use("search_docs", {"query": "a"}, "t1"),
                    tool_use("search_docs", {"query": "b"}, "t2"),
                ],
                "tool_use",
            ),
            message([text("done")], "end_turn"),
        ],
        tools,
    )
    await agent.ask("q")
    results = requests[1]["body"]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]


@pytest.mark.parametrize(
    "tools",
    [FakeTools(output="METAR unavailable", is_error=True), FakeTools(raises=RuntimeError("boom"))],
)
async def test_tool_failures_are_reported_to_the_model(tools):
    agent, requests = build(
        [
            message([tool_use("search_docs", {"query": "x"})], "tool_use"),
            message([text("The lookup failed.")], "end_turn"),
        ],
        tools,
    )
    result = await agent.ask("q")
    assert result["answer"] == "The lookup failed."
    assert result["tools_used"][0]["is_error"] is True
    assert requests[1]["body"]["messages"][2]["content"][0]["is_error"] is True


async def test_turn_limit_stops_a_runaway_loop():
    looping = [message([tool_use("search_docs", {"query": "x"})], "tool_use")] * 20
    agent, requests = build(looping)
    result = await agent.ask("q")
    assert len(requests) == MAX_MODEL_TURNS
    assert "tool-call limit" in result["answer"]


async def test_refusal_is_not_shown_as_an_empty_answer():
    agent, _ = build([message([], "refusal")])
    result = await agent.ask("q")
    assert result["stop_reason"] == "refusal"
    assert "declined" in result["answer"]


async def test_max_tokens_is_flagged():
    agent, _ = build([message([text("partial")], "max_tokens")])
    assert "cut off" in (await agent.ask("q"))["answer"]


async def test_pause_turn_continues():
    agent, requests = build(
        [message([text("wait")], "pause_turn"), message([text("final")], "end_turn")]
    )
    assert (await agent.ask("q"))["answer"] == "final"
    assert len(requests) == 2


async def test_authentication_errors_propagate():
    error = {"type": "error", "error": {"type": "authentication_error", "message": "bad"}}
    agent, _ = build([error], status=401)
    with pytest.raises(anthropic.AuthenticationError):
        await agent.ask("q")


async def test_trace_is_written(tmp_path):
    agent, _ = build([message([text("ok")], "end_turn")])
    agent.trace_path = tmp_path / "runs.jsonl"
    await agent.ask("hello")
    row = json.loads(agent.trace_path.read_text())
    assert row["question"] == "hello"
    assert row["cost_usd"] > 0


def test_cost_calculation():
    usage = [
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
        }
    ]
    assert cost_usd(usage) == 29.2


def test_extract_sources_skips_errors_and_bad_json_and_deduplicates():
    log = [
        {"name": "search_docs", "is_error": True, "result": "{}"},
        {"name": "search_docs", "is_error": False, "result": "not json"},
        {"name": "get_metar", "is_error": False, "result": "{}"},
        {
            "name": "search_docs",
            "is_error": False,
            "result": '{"results":[{"doc":"a","url":"u"},{"doc":"a","url":"u"}]}',
        },
    ]
    assert extract_sources(log) == [{"doc": "a", "url": "u"}]
