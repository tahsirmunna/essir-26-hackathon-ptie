"""A thin wrapper over qdrant-client.

Dense cosine search over a single vector, fused with an in-memory BM25 (lexical) pass so
exact terms an embedding can blur — names, numbers, acronyms — are not lost. No extra
service is needed: BM25 is built from the same payload text already stored in Qdrant.
"""

from __future__ import annotations

import re
from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import get_settings

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_RRF_K = 60  # standard Reciprocal Rank Fusion constant


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class VectorStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self._bm25_cache: tuple[object, dict[str, int]] | None = None
        self._sections_cache: list[str] | None = None

    # --- inspection ---------------------------------------------------------
    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection).count

    # --- write --------------------------------------------------------------
    def ensure_collection(self, dim: int, reset: bool = False) -> None:
        """Create the collection sized to the embedding dimension.

        The vector size is fixed at creation, so if you change embedding models you
        must re-ingest (or ingest into a differently named collection).
        """
        if reset and self.exists():
            self.client.delete_collection(self.collection)
        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # A payload index on `page` would let searches be scoped/filtered (only the
            # references section, only tables, ...) — see client.create_payload_index(...).
        self._bm25_cache = None
        self._sections_cache = None

    def upsert(self, points: list[models.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)
        self._bm25_cache = None  # payload text changed; rebuild the lexical index lazily
        self._sections_cache = None

    # --- read ---------------------------------------------------------------
    def _bm25(self) -> tuple[object | None, dict[str, int]]:
        """Lazily build (and cache) a BM25 index over every point's payload text."""
        if self._bm25_cache is not None:
            return self._bm25_cache

        from rank_bm25 import BM25Okapi

        ids: list[str] = []
        corpus: list[list[str]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                ids.append(str(p.id))
                corpus.append(_tokenize(str((p.payload or {}).get("text", ""))))
            if offset is None:
                break

        bm25 = BM25Okapi(corpus) if corpus else None
        index_of_id = {pid: i for i, pid in enumerate(ids)}
        self._bm25_cache = (bm25, index_of_id)
        return self._bm25_cache

    def search(
        self, vector: list[float], top_k: int, query_text: str | None = None
    ) -> list[models.ScoredPoint]:
        """Dense search, fused with a BM25 lexical pass when `query_text` is given.

        We over-fetch a larger dense candidate pool, then re-rank it with Reciprocal Rank
        Fusion between the dense cosine rank and the BM25 rank over the same candidates.
        RRF needs no score normalisation across the two systems, which is what makes
        combining a cosine similarity and a BM25 score simple to get right.
        A `query_filter` can be added here later to scope retrieval to part of the doc.
        """
        candidate_k = max(top_k * 4, top_k)
        dense_hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=candidate_k,
            with_payload=True,
        ).points
        if not query_text or len(dense_hits) <= top_k:
            return dense_hits[:top_k]

        bm25, index_of_id = self._bm25()
        if bm25 is None:
            return dense_hits[:top_k]

        scores = bm25.get_scores(_tokenize(query_text))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse_rank_of_index = {corpus_idx: rank for rank, corpus_idx in enumerate(order)}

        def fused_score(dense_rank: int, hit: models.ScoredPoint) -> float:
            score = 1.0 / (_RRF_K + dense_rank)
            corpus_idx = index_of_id.get(str(hit.id))
            if corpus_idx is not None and corpus_idx in sparse_rank_of_index:
                score += 1.0 / (_RRF_K + sparse_rank_of_index[corpus_idx])
            return score

        reranked = sorted(
            enumerate(dense_hits), key=lambda pair: fused_score(*pair), reverse=True
        )
        return [hit for _, hit in reranked[:top_k]]

    # --- structure (Level 3: whole-document coverage) ------------------------
    def list_sections(self) -> list[str]:
        """Every distinct section name in the collection, cached like `_bm25()`.

        Backs the level-3 "one chunk per section" coverage pass — a document-structure
        lookup, not a ranked search, so it lives next to `search()` rather than in it.
        """
        if self._sections_cache is not None:
            return self._sections_cache

        names: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                section = (p.payload or {}).get("section")
                if section:
                    names.add(str(section))
            if offset is None:
                break

        self._sections_cache = sorted(names)
        return self._sections_cache

    def points_for_section(self, section: str, limit: int = 1) -> list[dict]:
        """Payloads of the first `limit` chunks filed under an exact section name."""
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="section", match=models.MatchValue(value=section))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload or {} for p in points]


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.qdrant_url, s.qdrant_collection)
