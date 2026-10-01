"""Hybrid search over a metadata-indexed document store.

Scaffold: keyword (BM25) scoring plus exact metadata filters, held in memory.
Swap `InMemoryIndex` for a vector store and blend in embedding similarity in
`search()` when real data is wired up.

Each indexed chunk should carry `company`, `doc_type` and `page_number`
metadata so agents can filter by company and cite pages.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TOKEN = re.compile(r"[a-z0-9]+")


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

    def add(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        terms = Counter(_tokenize(content))
        self._entries.append(_Entry(doc_id, content, metadata or {}, terms))
        self._df.update(terms.keys())

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, List[Any]]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top_k chunks matching every metadata filter, ranked by BM25.

        `filters` maps a metadata key to the accepted values, e.g.
        {"company": ["Apple"], "doc_type": ["10-K", "10-Q"]}. Empty lists are ignored.
        Each result is {"doc_id", "content", "metadata", "score"}.
        """
        candidates = [e for e in self._entries if _matches(e.metadata, filters or {})]
        if not candidates:
            return []

        n = len(self._entries)
        avg_len = sum(sum(e.terms.values()) for e in self._entries) / n
        query_terms = _tokenize(query)

        scored = []
        for entry in candidates:
            length = sum(entry.terms.values())
            score = 0.0
            for term in query_terms:
                tf = entry.terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / avg_len))
            scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"doc_id": e.doc_id, "content": e.content, "metadata": dict(e.metadata), "score": s}
            for s, e in scored[:top_k]
            if s > 0
        ]


def _matches(metadata: Dict[str, Any], filters: Dict[str, List[Any]]) -> bool:
    return all(not allowed or metadata.get(key) in allowed for key, allowed in filters.items())


# Process-wide index. Load documents into it at startup.
default_index = InMemoryIndex()


def rag_search(
    query: str,
    company: Optional[str] = None,
    doc_types: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    filters = {"company": [company] if company else [], "doc_type": doc_types or []}
    return default_index.search(query, filters=filters, top_k=top_k)
