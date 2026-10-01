"""Text embeddings for the dense half of hybrid search.

Default: BAAI/bge-small-en-v1.5 via fastembed (local ONNX, no API key; the
model is downloaded once to ~/.cache). Swap in a hosted embedder (e.g. Voyage AI)
by implementing the `Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

logger = logging.getLogger("uvicorn.error")

DEFAULT_MODEL = os.environ.get("TAURUS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
WINDOW_WORDS = 250  # bge-small reads at most 512 tokens; pages run ~600 words
WINDOW_OVERLAP = 50
SAVE_EVERY = 256  # windows embedded between cache saves


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: List[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name)

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        # Small batches are ~2x faster on CPU: large ones pad every text to the longest.
        return np.asarray(list(self._model.passage_embed(texts, batch_size=16)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


def get_default_embedder() -> Optional[Embedder]:
    """Return the configured embedder, or None when dense search is disabled/unavailable."""
    if os.environ.get("TAURUS_EMBEDDINGS", "on").lower() in {"0", "off", "false"}:
        return None
    try:
        return FastEmbedEmbedder()
    except ImportError:
        return None


def split_windows(text: str) -> List[str]:
    words = text.split()
    if len(words) <= WINDOW_WORDS:
        return [" ".join(words)]
    step = WINDOW_WORDS - WINDOW_OVERLAP
    return [" ".join(words[i : i + WINDOW_WORDS]) for i in range(0, len(words) - WINDOW_OVERLAP, step)]


def embed_with_cache(embedder: Embedder, windows: List[str], cache_path: Optional[Path]) -> np.ndarray:
    """Embed `windows`, reusing vectors from `cache_path` for any text seen before."""
    keys = [hashlib.sha1(w.encode()).hexdigest() for w in windows]
    cached: dict = {}
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        cached = dict(zip(data["keys"].tolist(), data["vectors"]))

    missing = [i for i, k in enumerate(keys) if k not in cached]
    for start in range(0, len(missing), SAVE_EVERY):
        batch = missing[start : start + SAVE_EVERY]
        for i, vec in zip(batch, embedder.embed_passages([windows[i] for i in batch])):
            cached[keys[i]] = vec
        if cache_path:  # save as we go so an interrupted first build keeps its progress
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            done = list(cached)
            np.savez(cache_path, keys=np.array(done), vectors=np.stack([cached[k] for k in done]))
        logger.info("Embedded %d/%d new windows", min(start + SAVE_EVERY, len(missing)), len(missing))
    return np.stack([cached[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
