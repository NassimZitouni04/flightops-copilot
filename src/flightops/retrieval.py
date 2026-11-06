"""Corpus search: BM25, dense embeddings, reciprocal rank fusion and cross-encoder reranking."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "corpus.jsonl"
CACHE_DIR = ROOT / ".cache"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # Standard constant from the original reciprocal rank fusion paper.
CANDIDATES = 30

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge ranked lists of document indexes. Items ranked high in any list rise to the top."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda index: (-scores[index], index))


class Retriever:
    def __init__(self, corpus_path: Path = DEFAULT_CORPUS) -> None:
        with corpus_path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle]
        self._bm25 = BM25Okapi([tokenize(r["text"]) for r in self.records])
        self._digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:16]
        self._embedder = None
        self._vectors: np.ndarray | None = None
        self._reranker = None

    def _rank_bm25(self, query: str, limit: int) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:limit]]

    def _ensure_dense(self) -> None:
        if self._vectors is not None:
            return
        from fastembed import TextEmbedding

        self._embedder = TextEmbedding(EMBED_MODEL)
        cache = CACHE_DIR / f"embeddings-{self._digest}.npy"
        if cache.exists():
            self._vectors = np.load(cache)
            return
        vectors = np.array(list(self._embedder.embed([r["text"] for r in self.records])))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache, vectors)
        self._vectors = vectors

    def _rank_dense(self, query: str, limit: int) -> list[int]:
        self._ensure_dense()
        vector = np.array(list(self._embedder.query_embed(query))[0])
        vector /= np.linalg.norm(vector)
        return [int(i) for i in np.argsort(-(self._vectors @ vector))[:limit]]

    def _rerank(self, query: str, indexes: list[int]) -> list[int]:
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        scores = list(self._reranker.rerank(query, [self.records[i]["text"] for i in indexes]))
        order = sorted(range(len(indexes)), key=lambda position: -scores[position])
        return [indexes[position] for position in order]

    def search(self, query: str, k: int = 5, mode: Mode | None = None) -> list[dict]:
        """Return the top k chunks. Mode defaults to FLIGHTOPS_RETRIEVAL_MODE or hybrid_rerank."""
        mode = mode or os.environ.get("FLIGHTOPS_RETRIEVAL_MODE", "hybrid_rerank")
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if mode == "bm25":
            ranked = self._rank_bm25(query, k)
        elif mode == "dense":
            ranked = self._rank_dense(query, k)
        else:
            fused = reciprocal_rank_fusion(
                [self._rank_bm25(query, CANDIDATES), self._rank_dense(query, CANDIDATES)]
            )
            ranked = fused[:CANDIDATES]
            if mode == "hybrid_rerank":
                ranked = self._rerank(query, ranked)
        return [self.records[i] for i in ranked[:k]]
