"""Étape 3 — Recherche sémantique avec FAISS.

Construit un index FAISS à partir des embeddings et fournit une recherche des
segments les plus pertinents pour une question. On utilise IndexFlatIP
(produit scalaire) ; comme les vecteurs sont normalisés L2, cela équivaut à la
similarité cosinus.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings, settings
from .embed import Embedder
from .ingest import Chunk, build_corpus, chunks_to_dicts


@dataclass
class SearchHit:
    score: float
    chunk: dict  # {id, text, source, category, chunk_index}


class VectorStore:
    def __init__(self, cfg: Settings | None = None, embedder: Embedder | None = None):
        self.cfg = cfg or settings
        self.embedder = embedder or Embedder(self.cfg)
        self.index = None            # faiss.Index
        self.chunks: list[dict] = []  # métadonnées alignées sur l'index
        self._vectors = None         # cache des embeddings reconstruits (n, dim)

    # ---- Construction -----------------------------------------------------
    def build(self, chunks: list[Chunk] | None = None, verbose: bool = True) -> "VectorStore":
        import faiss

        if chunks is None:
            if verbose:
                print("📥 Ingestion du corpus…")
            chunks = build_corpus(self.cfg, verbose=verbose)

        self.chunks = chunks_to_dicts(chunks)
        texts = [c["text"] for c in self.chunks]

        if verbose:
            print(f"🧠 Calcul des embeddings ({self.cfg.embedding_model})…")
        vecs = self.embedder.encode_passages(texts, show_progress=verbose)

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self.index = index
        if verbose:
            print(f"✅ Index FAISS prêt : {index.ntotal} vecteurs (dim={dim})")
        return self

    # ---- Ajout incrémental ------------------------------------------------
    def add_chunks(self, chunks: list[dict], verbose: bool = False) -> int:
        """Indexe de nouveaux chunks SANS tout recalculer : on embedde uniquement
        les nouveaux textes et on les ajoute à l'index FAISS existant.

        `chunks` : dicts avec au moins text/source/category/chunk_index. Les `id`
        sont attribués à la suite de l'index courant (= position FAISS).
        """
        if self.index is None:
            raise RuntimeError("Index non construit/chargé.")
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        vecs = self.embedder.encode_passages(texts, show_progress=verbose)
        start = len(self.chunks)
        for i, c in enumerate(chunks):
            c = dict(c)
            c.setdefault("chunk_index", i)
            c["id"] = start + i
            self.chunks.append(c)
        self.index.add(vecs)
        self._vectors = None  # invalide le cache d'embeddings reconstruits
        return len(chunks)

    # ---- Persistance ------------------------------------------------------
    def save(self) -> None:
        import faiss

        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.cfg.faiss_path))
        self.cfg.meta_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "embedding_model": self.cfg.embedding_model,
            "n_chunks": len(self.chunks),
            "dim": int(self.index.d),
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
        }
        self.cfg.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, cfg: Settings | None = None, embedder: Embedder | None = None) -> "VectorStore":
        import faiss

        cfg = cfg or settings
        store = cls(cfg, embedder)
        if not cfg.faiss_path.exists():
            raise FileNotFoundError(
                f"Index absent : {cfg.faiss_path}. Lancez d'abord la construction."
            )
        store.index = faiss.read_index(str(cfg.faiss_path))
        store.chunks = json.loads(cfg.meta_path.read_text(encoding="utf-8"))
        return store

    # ---- Recherche --------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        if self.index is None:
            raise RuntimeError("Index non construit/chargé.")
        k = top_k or self.cfg.top_k
        qvec = self.embedder.encode_queries([query])
        scores, idxs = self.index.search(qvec, k)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            hits.append(SearchHit(score=float(score), chunk=self.chunks[int(idx)]))
        return hits

    def search_idx(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Comme search(), mais renvoie (index_global, score) — utile au graphe."""
        if self.index is None:
            raise RuntimeError("Index non construit/chargé.")
        qvec = self.embedder.encode_queries([query])
        scores, idxs = self.index.search(qvec, top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    @property
    def vectors(self) -> "np.ndarray":
        """Matrice (n, dim) des embeddings normalisés, reconstruite depuis FAISS.

        IndexFlatIP conserve les vecteurs : on les reconstruit une fois (puis cache)
        pour construire le graphe et re-ranker en mode Graph-RAG.
        """
        if self._vectors is None:
            if self.index is None:
                raise RuntimeError("Index non construit/chargé.")
            n = self.index.ntotal
            self._vectors = self.index.reconstruct_n(0, n).astype("float32")
        return self._vectors
