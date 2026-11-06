"""End-to-end answer eval with deterministic grading. Makes real Claude API calls and costs money.

Run: python evals/answer_eval.py --yes
Each case passes only if the answer has the required facts, the agent used the required tools, and
the expected document appears in its sources. Live weather cases can't check values that change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
CASES = HERE / "answer_cases.jsonl"
RESULTS = HERE / "results" / "answers.json"


def grade(case: dict, result: dict) -> list[str]:
    """Return a list of failure reasons. An empty list means the case passed."""
    failures = []
    answer = result["answer"]
    for pattern in case.get("must_match", []):
        if not re.search(pattern, answer):
            failures.append(f"answer is missing /{pattern}/")
    for pattern in case.get("must_not_match", []):
        if re.search(pattern, answer):
            failures.append(f"answer contains /{pattern}/")
    used = {tool["name"] for tool in result["tools_used"]}
    for tool in case.get("required_tools", []):
        if tool not in used:
            failures.append(f"did not call {tool}")
    doc = case.get("required_doc")
    if doc and doc not in {source["doc"] for source in result["sources"]}:
        failures.append(f"{doc} not in sources")
    return failures


async def run(cases: list[dict]) -> list[dict]:
    from mcp import Client

    from flightops.agent import Agent
    from flightops.mcp_client import McpTools, stdio_server

    rows = []
    async with Client(stdio_server()) as client:
        agent = Agent(McpTools(client))
        for case in cases:
            result = await agent.ask(case["q"])
            failures = grade(case, result)
            rows.append(
                {
                    "id": case["id"],
                    "passed": not failures,
                    "failures": failures,
                    "seconds": result["seconds"],
                    "cost_usd": result["cost_usd"],
                    "tools": [tool["name"] for tool in result["tools_used"]],
                    "answer": result["answer"],
                }
            )
            print(f"{'PASS' if not failures else 'FAIL'} {case['id']} {failures or ''}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Confirm that this spends API credits.")
    args = parser.parse_args()
    load_dotenv()
    if not args.yes:
        print("This makes real API calls that cost money. Re-run with --yes to continue.")
        return 2
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Set ANTHROPIC_API_KEY first, for example in a .env file.")
        return 2

    cases = [json.loads(line) for line in CASES.open(encoding="utf-8")]
    rows = asyncio.run(run(cases))
    seconds = [row["seconds"] for row in rows]
    summary = {
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 4),
        "p50_seconds": percentile(seconds, 0.5),
        "p95_seconds": percentile(seconds, 0.95),
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 4),
        "mean_cost_usd": round(sum(row["cost_usd"] for row in rows) / len(rows), 4),
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
