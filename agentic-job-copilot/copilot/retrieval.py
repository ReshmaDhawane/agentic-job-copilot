"""Hybrid retrieval: BM25 keyword search fused with vector search.

Vectors come from TF-IDF + truncated SVD (LSA) so the project runs fully
offline; swap `Embedder` for OpenAI / Azure embeddings in production.
Scores are combined with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = re.compile(r"[a-z0-9+#/.-]+")


def tokenize(text: str) -> list[str]:
    # keep inner punctuation (ci/cd, node.js, c++) but drop sentence punctuation
    return [t for t in (w.strip(".-/") for w in TOKEN.findall(text.lower())) if t]


def chunk_text(text: str, max_words: int = 60, overlap: int = 15) -> list[str]:
    """Split on paragraphs/bullets first, then window long blocks with overlap."""
    blocks = [b.strip(" -•\t") for b in re.split(r"\n\s*\n|\n[-•]", text) if b.strip()]
    chunks: list[str] = []
    for block in blocks:
        words = block.split()
        if len(words) <= max_words:
            chunks.append(block)
            continue
        step = max_words - overlap
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + max_words]))
            if i + max_words >= len(words):
                break
    return chunks


@dataclass
class Hit:
    chunk_id: int
    text: str
    score: float


class Embedder:
    def __init__(self, dims: int = 64):
        self.dims = dims
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        x = self.tfidf.fit_transform(texts)
        k = min(self.dims, x.shape[1] - 1, len(texts) - 1)
        if k >= 2:
            self.svd = TruncatedSVD(n_components=k, random_state=0)
            v = self.svd.fit_transform(x)
        else:
            v = x.toarray()
        return _normalize(v)

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self.tfidf.transform(texts)
        v = self.svd.transform(x) if self.svd else x.toarray()
        return _normalize(v)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


class HybridIndex:
    def __init__(self, chunks: list[str], rrf_k: int = 60):
        if not chunks:
            raise ValueError("cannot index an empty document")
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c) for c in chunks])
        self.embedder = Embedder()
        self.vectors = self.embedder.fit_transform(chunks)

    @classmethod
    def from_document(cls, text: str) -> "HybridIndex":
        return cls(chunk_text(text))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        bm = self.bm25.get_scores(tokenize(query))
        vec = self.vectors @ self.embedder.transform([query])[0]
        bm_rank = np.argsort(-bm)
        vec_rank = np.argsort(-vec)
        fused: dict[int, float] = {}
        for rank_list, raw in ((bm_rank, bm), (vec_rank, vec)):
            for r, idx in enumerate(rank_list):
                if raw[idx] <= 0:
                    continue  # ignore chunks with no signal from this retriever
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (self.rrf_k + r + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(i, self.chunks[i], s) for i, s in best]
