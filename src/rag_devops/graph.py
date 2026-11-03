"""Graph-RAG — récupération fondée sur un graphe de connaissances.

Au lieu de ne renvoyer que les k plus proches voisins vectoriels, on construit un
GRAPHE sémantique entre les chunks (arêtes = forte similarité cosinus), puis on
récupère par **PageRank personnalisé (PPR)** : on « amorce » le graphe sur les
chunks qui matchent la question, puis on laisse la pertinence se diffuser le long
des arêtes (multi-saut). Un chunk qui ne matche pas directement la question mais
qui est fortement connecté à plusieurs chunks pertinents remonte quand même —
d'où une récupération plus intelligente, plus complète et mieux ancrée.

Le graphe est purement sémantique : il se reconstruit automatiquement pour
n'importe quel corpus (DevOps, Data Engineering, AI Engineering…), ce qui rend
l'assistant extensible à de nouveaux domaines sans code spécifique.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings, settings
from .index import SearchHit, VectorStore

# Mots-outils ignorés pour le recouvrement lexical (FR + EN, minimal).
_STOP = {
    "the", "and", "for", "with", "that", "this", "une", "des", "les", "dans",
    "pour", "avec", "est", "sont", "que", "qui", "quoi", "comment", "pourquoi",
    "sur", "par", "aux", "le", "la", "un", "de", "du", "en", "à", "and", "ou",
    "what", "how", "why", "does", "is", "are", "a", "an", "of", "to", "in", "on",
}
_WORD = re.compile(r"[a-zA-Zàâçéèêëîïôûùüÿñæœ0-9][a-zA-Zàâçéèêëîïôûùüÿñæœ0-9+.#-]{2,}")


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP}


# --------------------------------------------------------------------------
# Construction du graphe
# --------------------------------------------------------------------------
def build_graph(store: VectorStore, cfg: Settings | None = None, verbose: bool = True) -> "KnowledgeGraph":
    """Construit le graphe kNN sémantique à partir de l'index vectoriel."""
    cfg = cfg or settings
    vecs = store.vectors                      # (n, dim), normalisés L2
    n = vecs.shape[0]
    knn = min(cfg.graph_knn + 1, n)           # +1 car le plus proche voisin = soi-même

    if verbose:
        print(f"🕸️  Construction du graphe : {n} nœuds, kNN={cfg.graph_knn}, seuil={cfg.graph_sim_threshold}")

    # Recherche batchée des voisins de chaque nœud via l'index FAISS existant.
    scores, idxs = store.index.search(vecs, knn)

    src_l, dst_l, w_l = [], [], []
    thr = cfg.graph_sim_threshold
    for i in range(n):
        for s, j in zip(scores[i], idxs[i]):
            j = int(j)
            if j < 0 or j == i:
                continue
            if s < thr:
                continue
            # arête non orientée (on ajoute les deux sens ; dédup via symétrisation)
            src_l.append(i); dst_l.append(j); w_l.append(float(s))
            src_l.append(j); dst_l.append(i); w_l.append(float(s))

    src = np.asarray(src_l, dtype=np.int32)
    dst = np.asarray(dst_l, dtype=np.int32)
    w = np.asarray(w_l, dtype=np.float32)

    if verbose:
        deg = np.bincount(src, minlength=n) if len(src) else np.zeros(n)
        iso = int((deg == 0).sum())
        print(f"✅ Graphe : {len(w) // 2} arêtes uniques · degré moyen {deg.mean():.1f} · {iso} nœuds isolés")

    return KnowledgeGraph(n=n, src=src, dst=dst, w=w, cfg=cfg)


# --------------------------------------------------------------------------
# Le graphe + la récupération PPR
# --------------------------------------------------------------------------
@dataclass
class KnowledgeGraph:
    n: int
    src: np.ndarray          # (E,) source de chaque arête
    dst: np.ndarray          # (E,) destination
    w: np.ndarray            # (E,) poids (cosinus)
    cfg: Settings

    # ---- persistance ----
    def save(self, path: Path | None = None) -> None:
        path = Path(path or self.cfg.graph_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, n=self.n, src=self.src, dst=self.dst, w=self.w)

    @classmethod
    def load(cls, cfg: Settings | None = None, path: Path | None = None) -> "KnowledgeGraph | None":
        cfg = cfg or settings
        path = Path(path or cfg.graph_path)
        if not path.exists():
            return None
        d = np.load(path)
        return cls(
            n=int(d["n"]), src=d["src"], dst=d["dst"], w=d["w"].astype("float32"), cfg=cfg
        )

    # ---- PageRank personnalisé (power method, vectorisé) ----
    def _ppr(self, personalization: np.ndarray) -> np.ndarray:
        n, src, dst, w = self.n, self.src, self.dst, self.w
        p = personalization
        if len(src) == 0:
            return p  # graphe vide : on retombe sur les graines
        # normalisation par degré sortant (pondéré) -> probabilités de transition
        outw = np.bincount(src, weights=w, minlength=n)
        outw[outw == 0] = 1.0
        wn = w / outw[src]

        restart = self.cfg.graph_restart
        r = p.copy()
        for _ in range(self.cfg.graph_iters):
            # diffusion : r_v += Σ_u P(u→v) · r_u
            spread = np.bincount(dst, weights=wn * r[src], minlength=n)
            r_new = restart * p + (1.0 - restart) * spread
            if np.abs(r_new - r).sum() < 1e-6:
                r = r_new
                break
            r = r_new
        return r

    def retrieve(
        self,
        query: str,
        store: VectorStore,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Récupère les chunks par PPR amorcé sur les meilleurs hits vectoriels."""
        cfg = self.cfg
        k = top_k or cfg.top_k

        # 1) graines = meilleurs hits vectoriels (indice global + cosinus)
        seeds = store.search_idx(query, max(cfg.graph_seed_k, k))
        if not seeds:
            return []

        p = np.zeros(self.n, dtype=np.float32)
        for idx, s in seeds:
            p[idx] = max(p[idx], max(s, 0.0))   # personnalisation ∝ similarité
        if p.sum() == 0:
            p[seeds[0][0]] = 1.0
        p = p / p.sum()

        # 2) diffusion sur le graphe (multi-saut)
        r = self._ppr(p)

        # 3) bassin de candidats = meilleurs scores PPR (∪ graines, pour robustesse)
        pool_n = max(cfg.graph_pool, k)
        pool = set(int(i) for i in np.argsort(-r)[:pool_n])
        pool.update(idx for idx, _ in seeds)
        pool_idx = np.fromiter(pool, dtype=np.int64)

        # 4) re-ranking : PPR (graphe) · cosinus (vecteur) · recouvrement lexical
        qvec = store.embedder.encode_queries([query])[0]
        vsim = store.vectors[pool_idx] @ qvec                # cosinus requête↔chunk
        pscore = r[pool_idx]
        qterms = _terms(query)
        if qterms:
            lex = np.array(
                [len(qterms & _terms(store.chunks[i]["text"])) / len(qterms) for i in pool_idx],
                dtype=np.float32,
            )
        else:
            lex = np.zeros(len(pool_idx), dtype=np.float32)

        def _norm(a: np.ndarray) -> np.ndarray:
            m = a.max()
            return a / m if m > 0 else a

        combined = (
            cfg.graph_w_ppr * _norm(pscore)
            + cfg.graph_w_vec * _norm(np.clip(vsim, 0, None))
            + cfg.graph_w_lex * lex
        )
        order = np.argsort(-combined)[:k]

        hits: list[SearchHit] = []
        for o in order:
            gi = int(pool_idx[o])
            # score affiché = cosinus (interprétable), ordre = score combiné
            hits.append(SearchHit(score=float(vsim[o]), chunk=store.chunks[gi]))
        return hits
