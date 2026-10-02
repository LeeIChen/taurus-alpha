"""Hybrid search over a metadata-indexed document store.

Ranks chunks two ways, keyword (BM25) and dense embeddings, and merges the two
rankings with Reciprocal Rank Fusion. Exact metadata filters are applied before
ranking. Everything is held in memory; without an embedder (fastembed missing or
TAURUS_EMBEDDINGS=off) search falls back to BM25 only.

Each indexed chunk should carry `company`, `doc_type` and `page_number`
metadata so agents can filter by company and cite pages.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.tools.embeddings import Embedder, embed_with_cache, get_default_embedder, split_windows

_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60  # standard RRF damping constant
CANDIDATES_PER_RANKER = 50


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class _Entry:
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    terms: Counter = field(default_factory=Counter)


class InMemoryIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._entries: List[_Entry] = []
        self._df: Counter = Counter()
        self._embedder: Optional[Embedder] = None
        self._window_vectors: Optional[np.ndarray] = None  # (n_windows, dim)
        self._window_owner: Optional[np.ndarray] = None  # entry index per window

    def add(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        terms = Counter(_tokenize(content))
        self._entries.append(_Entry(doc_id, content, metadata or {}, terms))
        self._df.update(terms.keys())
        self._window_vectors = None  # embeddings are stale until build_embeddings() runs again

    def build_embeddings(self, embedder: Embedder, cache_path: Optional[Path] = None) -> int:
        """Embed every entry (in overlapping word windows). Returns the number of windows."""
        windows: List[str] = []
        owners: List[int] = []
        for i, entry in enumerate(self._entries):
            for window in split_windows(entry.content):
                windows.append(window)
                owners.append(i)
        self._embedder = embedder
        self._window_vectors = embed_with_cache(embedder, windows, cache_path)
        self._window_owner = np.asarray(owners)
        return len(windows)

    @property
    def has_embeddings(self) -> bool:
        return self._window_vectors is not None and len(self._window_vectors) > 0

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, List[Any]]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top_k chunks matching every metadata filter.

        `filters` maps a metadata key to the accepted values, e.g.
        {"company": ["Apple"], "doc_type": ["10-K", "10-Q"]}. Empty lists are ignored.
        Each result is {"doc_id", "content", "metadata", "score"}; `score` is the RRF
        score when embeddings are built, otherwise the BM25 score.
        """
        candidates = [i for i, e in enumerate(self._entries) if _matches(e.metadata, filters or {})]
        if not candidates:
            return []

        bm25 = self._bm25_scores(query, candidates)
        if not self.has_embeddings:
            ranked = sorted(((s, i) for i, s in bm25.items() if s > 0), reverse=True)
            return [self._result(i, s) for s, i in ranked[:top_k]]

        dense = self._dense_scores(query, candidates)
        fused: Dict[int, float] = {}
        for scores in (bm25, dense):
            ranking = sorted((i for i, s in scores.items() if s > 0), key=scores.get, reverse=True)
            for rank, i in enumerate(ranking[:CANDIDATES_PER_RANKER]):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        return [self._result(i, s) for i, s in ranked[:top_k]]

    def _bm25_scores(self, query: str, candidates: List[int]) -> Dict[int, float]:
        n = len(self._entries)
        avg_len = sum(sum(e.terms.values()) for e in self._entries) / n
        query_terms = _tokenize(query)
        scores = {}
        for i in candidates:
            entry = self._entries[i]
            length = sum(entry.terms.values())
            score = 0.0
            for term in query_terms:
                tf = entry.terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / avg_len))
            scores[i] = score
        return scores

    def _dense_scores(self, query: str, candidates: List[int]) -> Dict[int, float]:
        """Cosine similarity of the query to each candidate's best-matching window."""
        mask = np.isin(self._window_owner, candidates)
        sims = self._window_vectors[mask] @ self._embedder.embed_query(query)
        scores: Dict[int, float] = {}
        for owner, sim in zip(self._window_owner[mask].tolist(), sims.tolist()):
            if sim > scores.get(owner, -1.0):
                scores[owner] = sim
        return scores

    def _result(self, i: int, score: float) -> Dict[str, Any]:
        e = self._entries[i]
        return {"doc_id": e.doc_id, "content": e.content, "metadata": dict(e.metadata), "score": score}


def _matches(metadata: Dict[str, Any], filters: Dict[str, List[Any]]) -> bool:
    return all(not allowed or metadata.get(key) in allowed for key, allowed in filters.items())


# Process-wide index. Load documents into it at startup.
default_index = InMemoryIndex()

# Lower-cased alternative names -> canonical `company` metadata value.
# load_filings() also registers each filing's ticker and company name.
_COMPANY_ALIASES: Dict[str, str] = {
    "google": "Alphabet",
    "alphabet inc": "Alphabet",
    "facebook": "Meta",
    "meta platforms": "Meta",
    "nvidia corporation": "Nvidia",
    "amazon.com": "Amazon",
    "tesla motors": "Tesla",
}


def canonical_company(name: str) -> str:
    key = name.strip().lower()
    return _COMPANY_ALIASES.get(key, name.strip())


def available_companies(index: InMemoryIndex = default_index) -> List[str]:
    return sorted({e.metadata["company"] for e in index._entries if "company" in e.metadata})


def companies_mentioned(text: str, index: InMemoryIndex = default_index) -> List[str]:
    """Indexed companies named in `text` by name, ticker or alias (whole words only)."""
    indexed = set(available_companies(index))
    found = set()
    for alias, company in _COMPANY_ALIASES.items():
        if company in indexed and re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
            found.add(company)
    return sorted(found)


def load_filings(
    directory: Path,
    index: InMemoryIndex = default_index,
    embedder: Optional[Embedder] = None,
) -> int:
    """Load every *.jsonl chunk file under `directory` into `index` and embed it.

    Vectors are cached in `<directory>/.embeddings/` so restarts only embed new text.
    Pass `embedder` to override the default; dense search is skipped if none is available.
    Returns the chunk count.
    """
    directory = Path(directory)
    count = 0
    for path in sorted(directory.glob("**/*.jsonl")):
        with path.open() as f:
            for line in f:
                chunk = json.loads(line)
                metadata = chunk.get("metadata", {})
                index.add(chunk["doc_id"], chunk["content"], metadata)
                if "company" in metadata:
                    _COMPANY_ALIASES.setdefault(metadata["company"].lower(), metadata["company"])
                    if "ticker" in metadata:
                        _COMPANY_ALIASES.setdefault(metadata["ticker"].lower(), metadata["company"])
                count += 1

    embedder = embedder or get_default_embedder()
    if count and embedder is not None:
        cache = directory / ".embeddings" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', embedder.name)}.npz"
        index.build_embeddings(embedder, cache_path=cache)
    return count


def rag_search(
    query: str,
    company: Optional[str] = None,
    doc_types: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    filters = {"company": [canonical_company(company)] if company else [], "doc_type": doc_types or []}
    return default_index.search(query, filters=filters, top_k=top_k)
