"""Download FAA Part 107 and PX4 flight docs, split them into chunks, write data/corpus.jsonl.

Run with: python -m flightops.ingest
The output is committed so evals and tests are reproducible without network access.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"

ECFR_TITLES = "https://www.ecfr.gov/api/versioner/v1/titles.json"
ECFR_PART = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-14.xml?part=107"
PX4_BRANCH = "https://api.github.com/repos/PX4/PX4-Autopilot/branches/main"
PX4_TREE = "https://api.github.com/repos/PX4/PX4-Autopilot/git/trees/{sha}?recursive=1"
PX4_RAW = "https://raw.githubusercontent.com/PX4/PX4-Autopilot/{sha}/{path}"

PX4_PREFIXES = ("docs/en/flying/", "docs/en/flight_modes_mc/", "docs/en/flight_modes/")
PX4_FILES = (
    "docs/en/config/safety.md",
    "docs/en/config/safety_intro.md",
    "docs/en/config/battery.md",
)

MAX_CHARS = 1400
MIN_CHARS = 80


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text on paragraph boundaries so no chunk exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = ""
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def parse_part107(xml_text: str) -> list[dict]:
    """Turn eCFR XML into one or more chunks per regulation section."""
    root = ET.fromstring(xml_text)
    records: list[dict] = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        head = (div.findtext("HEAD") or "").strip()
        number = (div.get("N") or "").replace("§", "").strip()
        paragraphs = [re.sub(r"\s+", " ", "".join(p.itertext())).strip() for p in div.iter("P")]
        body = "\n\n".join(p for p in paragraphs if p)
        if not number or len(body) < MIN_CHARS:
            continue
        for i, chunk in enumerate(split_long(body)):
            records.append(
                {
                    "id": f"cfr-{number}-{i}",
                    "source": "faa_part_107",
                    "doc": f"14 CFR {number}",
                    "title": head,
                    "url": f"https://www.ecfr.gov/current/title-14/section-{number}",
                    "text": f"{head}\n\n{chunk}",
                }
            )
    return records


def clean_markdown(text: str) -> str:
    """Remove markup that adds noise to search: frontmatter, HTML, images, container fences."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<[^>\n]+>", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^:::.*$", "", text, flags=re.M)
    return text.strip()


def parse_px4_markdown(path: str, markdown: str) -> list[dict]:
    """Split one PX4 markdown page by its headings."""
    text = clean_markdown(markdown)
    page = path.removeprefix("docs/en/").removesuffix(".md")
    url = f"https://docs.px4.io/main/en/{page}.html"
    parts = re.split(r"^(#{1,3} .+)$", text, flags=re.M)

    page_title = page
    sections: list[tuple[str, str]] = []
    heading = ""
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            heading = re.sub(r"\s*\{#[^}]*\}", "", part.lstrip("# ")).strip()
            if part.startswith("# "):
                page_title = heading
        elif part.strip():
            sections.append((heading, part.strip()))

    records: list[dict] = []
    for s_idx, (section, body) in enumerate(sections):
        if len(body) < MIN_CHARS:
            continue
        title = page_title if section in ("", page_title) else f"{page_title}: {section}"
        for c_idx, chunk in enumerate(split_long(body)):
            records.append(
                {
                    "id": f"px4-{page.replace('/', '-')}-{s_idx}-{c_idx}",
                    "source": "px4_docs",
                    "doc": page,
                    "title": title,
                    "url": url,
                    "text": f"{title}\n\n{chunk}",
                }
            )
    return records


def main() -> None:
    client = httpx.Client(timeout=60, follow_redirects=True)

    titles = client.get(ECFR_TITLES).raise_for_status().json()["titles"]
    date = next(t["up_to_date_as_of"] for t in titles if t["number"] == 14)
    part = client.get(ECFR_PART.format(date=date)).raise_for_status().text
    records = parse_part107(part)
    cfr_count = len(records)

    sha = client.get(PX4_BRANCH).raise_for_status().json()["commit"]["sha"]
    tree = client.get(PX4_TREE.format(sha=sha)).raise_for_status().json()["tree"]
    paths = sorted(
        item["path"]
        for item in tree
        if item["path"].endswith(".md")
        and (item["path"].startswith(PX4_PREFIXES) or item["path"] in PX4_FILES)
    )
    for path in paths:
        response = client.get(PX4_RAW.format(sha=sha, path=path)).raise_for_status()
        records.extend(parse_px4_markdown(path, response.text))

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "ecfr_date": date,
        "px4_commit": sha,
        "px4_pages": len(paths),
        "part_107_chunks": cfr_count,
        "px4_chunks": len(records) - cfr_count,
    }
    (CORPUS_PATH.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
