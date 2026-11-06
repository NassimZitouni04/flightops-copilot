"""Compare search modes on hand-labeled questions. Uses no LLM calls and costs nothing.

Run: python evals/retrieval_eval.py [--gate]
A result counts as relevant when it comes from a document listed for that question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flightops.retrieval import MODES, Retriever

HERE = Path(__file__).parent
QUESTIONS = HERE / "retrieval_questions.jsonl"
RESULTS = HERE / "results" / "retrieval.json"
GATE = {"mode": "hybrid_rerank", "metric": "hit_at_5", "minimum": 0.80}
K = 10


def score_ranking(ranked_docs: list[str], expected: set[str]) -> dict:
    """Return hit@1, hit@5 and reciprocal rank for one question."""
    position = next((i for i, doc in enumerate(ranked_docs) if doc in expected), None)
    return {
        "hit_at_1": float(position == 0),
        "hit_at_5": float(position is not None and position < 5),
        "rr": 0.0 if position is None else 1.0 / (position + 1),
    }


def evaluate(retriever: Retriever, questions: list[dict], mode: str) -> dict:
    rows = []
    for question in questions:
        ranked = [r["doc"] for r in retriever.search(question["q"], k=K, mode=mode)]
        rows.append(
            {"q": question["q"], "top": ranked[:3], **score_ranking(ranked, set(question["docs"]))}
        )
    count = len(rows)
    return {
        "hit_at_1": round(sum(r["hit_at_1"] for r in rows) / count, 4),
        "hit_at_5": round(sum(r["hit_at_5"] for r in rows) / count, 4),
        "mrr_at_10": round(sum(r["rr"] for r in rows) / count, 4),
        "misses_at_5": [r for r in rows if not r["hit_at_5"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="Exit 1 if the quality gate fails.")
    args = parser.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    retriever = Retriever()
    known = {r["doc"] for r in retriever.records}
    unknown = sorted({d for q in questions for d in q["docs"]} - known)
    if unknown:
        print(f"Labels point at documents missing from the corpus: {unknown}")
        return 2

    results = {mode: evaluate(retriever, questions, mode) for mode in MODES}
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"questions": len(questions), "modes": results}, indent=2) + "\n")

    print(f"{len(questions)} questions")
    print(f"{'mode':<15}{'hit@1':>8}{'hit@5':>8}{'MRR@10':>9}")
    for mode, row in results.items():
        print(f"{mode:<15}{row['hit_at_1']:>8.3f}{row['hit_at_5']:>8.3f}{row['mrr_at_10']:>9.3f}")

    if args.gate:
        value = results[GATE["mode"]][GATE["metric"]]
        if value < GATE["minimum"]:
            print(f"GATE FAILED: {GATE['mode']} {GATE['metric']} {value} < {GATE['minimum']}")
            return 1
        print(f"Gate passed: {GATE['mode']} {GATE['metric']} {value} >= {GATE['minimum']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
