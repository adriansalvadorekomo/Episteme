"""Étape 2 — Création des embeddings.

Transforme les segments textuels en vecteurs numériques via SentenceTransformers.
Les embeddings capturent le *sens* du texte : deux passages proches
sémantiquement auront des vecteurs proches (forte similarité cosinus).
"""
from __future__ import annotations

import numpy as np

from .config import Settings, settings

# Les modèles e5 attendent des préfixes explicites : "query:" pour une question,
# "passage:" pour un document indexé. Ils améliorent nettement la pertinence.
_E5_LIKE = ("e5", "multilingual-e5")


class Embedder:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.device = self.cfg.resolve_device()
        self._needs_prefix = any(tag in self.cfg.embedding_model.lower() for tag in _E5_LIKE)
        self._model = None  # chargement paresseux

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.cfg.embedding_model, device=self.device
            )
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _prep(self, texts: list[str], kind: str) -> list[str]:
        if not self._needs_prefix:
            return texts
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode les documents à indexer. Vecteurs normalisés L2 (=> cosinus)."""
        vecs = self.model.encode(
            self._prep(texts, "passage"),
            batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode des questions utilisateur."""
        vecs = self.model.encode(
            self._prep(texts, "query"),
            batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")
