import json

from evals_helpers import score_ranking

from flightops.ingest import clean_markdown, parse_part107, parse_px4_markdown, split_long
from flightops.retrieval import DEFAULT_CORPUS, Retriever, reciprocal_rank_fusion, tokenize

PART_XML = """<ROOT>
<DIV8 N="§ 107.51" TYPE="SECTION"><HEAD>§ 107.51 Operating limitations.</HEAD>
<P>(a) The groundspeed may not exceed 87 knots (100 miles per hour) under this section.</P>
<P>(b) Altitude limit of 400 feet.</P></DIV8>
<DIV8 N="§ 107.2" TYPE="SECTION"><HEAD>§ 107.2 [Reserved]</HEAD><P>Reserved</P></DIV8>
<DIV5 TYPE="PART"><HEAD>Part 107</HEAD></DIV5>
</ROOT>"""

PX4_MD = """---
title: x
---
# Return Mode

Intro text describing how the vehicle returns to a safe location when commanded by the pilot.

![image](a.png)

## Return Types {#return_types}

Text about [return types](other.md) that is long enough to be retained as its own chunk here.

## Tiny

short
"""


def test_part107_parser_builds_section_records():
    records = parse_part107(PART_XML)
    assert [r["doc"] for r in records] == ["14 CFR 107.51"]
    assert records[0]["url"].endswith("section-107.51")
    assert "87 knots" in records[0]["text"]


def test_px4_parser_splits_by_heading_and_cleans_markup():
    records = parse_px4_markdown("docs/en/flight_modes/return.md", PX4_MD)
    assert [r["title"] for r in records] == ["Return Mode", "Return Mode: Return Types"]
    assert records[0]["url"] == "https://docs.px4.io/main/en/flight_modes/return.html"
    assert "other.md" not in records[1]["text"]
    assert "a.png" not in records[0]["text"]


def test_clean_markdown_removes_containers_and_comments():
    assert clean_markdown("::: tip\nhello\n:::\n<!-- x -->") == "hello"


def test_split_long_respects_limit_and_keeps_text():
    text = "\n\n".join(["word " * 50] * 10)
    chunks = split_long(text, 400)
    assert all(len(chunk) <= 400 for chunk in chunks)
    assert sum(chunk.count("word") for chunk in chunks) == 500


def test_tokenize_keeps_section_numbers():
    assert "107.51" in tokenize("See Section 107.51!")


def test_rrf_rewards_items_ranked_by_both_lists():
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_bm25_finds_exact_terms_in_real_corpus():
    results = Retriever().search("anti-collision lighting civil twilight", k=3, mode="bm25")
    assert any(r["doc"] == "14 CFR 107.29" for r in results)


def test_corpus_records_are_well_formed():
    with DEFAULT_CORPUS.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len({r["id"] for r in records}) == len(records)
    assert {r["source"] for r in records} == {"faa_part_107", "px4_docs"}
    assert not any("{#" in r["title"] for r in records)


def test_score_ranking():
    assert score_ranking(["a", "b"], {"a"}) == {"hit_at_1": 1.0, "hit_at_5": 1.0, "rr": 1.0}
    assert score_ranking(["x", "a"], {"a"})["rr"] == 0.5
    assert score_ranking(["x"], {"a"}) == {"hit_at_1": 0.0, "hit_at_5": 0.0, "rr": 0.0}
