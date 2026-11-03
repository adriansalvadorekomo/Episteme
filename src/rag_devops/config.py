"""Configuration centrale de l'assistant RAG DevOps/MLOps.

Toutes les valeurs sont surchargeables via variables d'environnement, ce qui
permet au notebook, à l'API FastAPI et à la CLI de partager les mêmes réglages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Racine du projet (…/LLM-RAG)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Borne le contrôle réseau de Hugging Face : sur une connexion lente/bloquée,
# `from_pretrained` ne doit jamais rester bloqué — il bascule vite sur le cache
# local. Surchargeable (ex. HF_HUB_OFFLINE=1 pour un fonctionnement 100 % hors-ligne).
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "12")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    # --- Corpus ---------------------------------------------------------
    # Dossier source contenant les PDF / Markdown / txt du cours DevOps-MLOps.
    corpus_dir: Path = field(
        default_factory=lambda: Path(
            _env("RAG_CORPUS_DIR", str(Path.home() / "Downloads" / "Devops - Lake"))
        )
    )
    # Dossier où sont persistés l'index FAISS + les métadonnées des chunks.
    index_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "index"
    )

    # --- Découpage (chunking) ------------------------------------------
    chunk_size: int = int(_env("RAG_CHUNK_SIZE", "1000"))      # caractères
    chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", "180"))  # caractères
    min_chunk_chars: int = int(_env("RAG_MIN_CHUNK", "80"))

    # --- Embeddings -----------------------------------------------------
    # Modèle multilingue (corpus FR + EN). e5 exige les préfixes query:/passage:
    embedding_model: str = _env(
        "RAG_EMBED_MODEL", "intfloat/multilingual-e5-small"
    )
    embedding_batch_size: int = int(_env("RAG_EMBED_BATCH", "64"))

    # --- LLM open-source (Transformers) --------------------------------
    # Qwen2.5-1.5B-Instruct : léger, excellent en français, tient sur 8 Go VRAM.
    # Alternatives : Qwen/Qwen2.5-3B-Instruct, microsoft/Phi-3-mini-4k-instruct.
    llm_model: str = _env("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens: int = int(_env("RAG_MAX_NEW_TOKENS", "512"))
    temperature: float = float(_env("RAG_TEMPERATURE", "0.2"))
    # Adaptateur LoRA optionnel (fine-tuning sur le lakehouse). Chemin d'un dossier
    # d'adaptateur PEFT, ou "auto" pour charger le plus récent de lakehouse/adapters.
    # Vide => modèle de base inchangé.
    llm_adapter: str = _env("RAG_LLM_ADAPTER", "")

    # --- Récupération ---------------------------------------------------
    top_k: int = int(_env("RAG_TOP_K", "5"))

    # --- Mémoire conversationnelle --------------------------------------
    # Nombre de messages passés (user+assistant) injectés comme contexte pour
    # que l'assistant soit conscient du fil de la discussion.
    history_turns: int = int(_env("RAG_HISTORY_TURNS", "8"))

    # --- Graph-RAG ------------------------------------------------------
    # Récupération fondée sur un graphe de connaissances (kNN sémantique entre
    # chunks) + PageRank personnalisé : multi-saut, plus « intelligent » et mieux
    # ancré que la simple recherche vectorielle. Désactivable via RAG_USE_GRAPH=0.
    use_graph: bool = _env("RAG_USE_GRAPH", "1") not in ("0", "false", "False", "")
    graph_knn: int = int(_env("RAG_GRAPH_KNN", "12"))          # arêtes/nœud
    graph_sim_threshold: float = float(_env("RAG_GRAPH_SIM", "0.55"))  # seuil cosinus
    graph_seed_k: int = int(_env("RAG_GRAPH_SEED_K", "10"))    # graines PPR (hits vectoriels)
    graph_pool: int = int(_env("RAG_GRAPH_POOL", "40"))        # candidats avant re-ranking
    graph_restart: float = float(_env("RAG_GRAPH_RESTART", "0.5"))  # proba de redémarrage PPR
    graph_iters: int = int(_env("RAG_GRAPH_ITERS", "40"))      # itérations power-method
    # Pondération du re-ranking final : PPR (graphe) · cosinus · recouvrement lexical.
    graph_w_ppr: float = float(_env("RAG_GRAPH_W_PPR", "0.5"))
    graph_w_vec: float = float(_env("RAG_GRAPH_W_VEC", "0.4"))
    graph_w_lex: float = float(_env("RAG_GRAPH_W_LEX", "0.1"))

    # --- Garde-fou de périmètre (anti-hallucination hors-sujet) ---------
    # On refuse de répondre si la question n'est pas de l'ingénierie logicielle.
    # Décision CONTRASTIVE : la question est-elle plus proche d'ancres « in-domaine »
    # que d'ancres « hors-sujet » ? La pénalité cross-lingue s'annule entre les deux
    # côtés, donc le même seuil marche en EN/ES/FR/DE. In-scope si (maxIN - maxOFF)
    # ≥ scope_margin.
    scope_guard: bool = _env("RAG_SCOPE_GUARD", "1") not in ("0", "false", "False", "")
    scope_margin: float = float(_env("RAG_SCOPE_MARGIN", "0.0"))

    # Ancrage minimal en mode RAG : si le meilleur extrait récupéré est sous ce
    # cosinus, le corpus ne couvre probablement pas la question → réponse honnête
    # « je ne sais pas » plutôt qu'une invention. (0 = désactivé.)
    rag_min_score: float = float(_env("RAG_MIN_SCORE", "0.78"))
    # Fréquence documentaire minimale d'une techno nommée pour être « couverte ».
    # Une techno citée au passage (df 1-2) ne compte pas comme couverte → IDK honnête.
    rag_min_df: int = int(_env("RAG_MIN_DF", "3"))

    # --- Périmètre d'expertise (scalable) -------------------------------
    # Domaines que l'assistant accepte d'enseigner. Ajouter un domaine ici (ou via
    # RAG_DOMAINS="a,b,c") élargit le périmètre du garde-fou anti-hors-sujet, en
    # phase avec l'ajout de nouveaux corpus (data-engineering, ai-engineering…).
    domains: list[str] = field(
        default_factory=lambda: [
            d.strip()
            for d in _env(
                "RAG_DOMAINS",
                "DevOps,MLOps,DataOps,Data Engineering,AI Engineering,"
                "Software Engineering,Cloud & SRE,Observabilité,CI/CD,"
                "Conteneurs & orchestration",
            ).split(",")
            if d.strip()
        ]
    )

    # --- Lakehouse (MinIO/S3 + Apache Iceberg) --------------------------
    # Source de vérité des documents que l'assistant apprend. MinIO fournit un
    # stockage objet S3, Iceberg la couche tabulaire (catalogue SQLite, sans Spark).
    s3_endpoint: str = _env("RAG_S3_ENDPOINT", "http://localhost:9000")
    s3_access_key: str = _env("RAG_S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = _env("RAG_S3_SECRET_KEY", "minioadmin")
    s3_region: str = _env("RAG_S3_REGION", "us-east-1")
    s3_bucket: str = _env("RAG_S3_BUCKET", "lakehouse")
    iceberg_namespace: str = _env("RAG_ICEBERG_NS", "lakehouse")
    lakehouse_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "lakehouse")
    # Backend de stockage : "s3" (MinIO/S3), "local" (entrepôt Iceberg sur disque),
    # ou "auto" (= s3 si l'endpoint répond, sinon local). Permet de fonctionner même
    # sans conteneur S3, puis de basculer sur MinIO quand il tourne.
    lakehouse_backend: str = _env("RAG_LAKEHOUSE_BACKEND", "auto")

    # --- Matériel -------------------------------------------------------
    # "auto" => CUDA si dispo, sinon CPU. Forçable via RAG_DEVICE=cpu|cuda.
    device: str = _env("RAG_DEVICE", "auto")

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "corpus.faiss"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def graph_path(self) -> Path:
        return self.index_dir / "graph.npz"

    @property
    def iceberg_catalog_db(self) -> Path:
        return self.lakehouse_dir / "catalog.db"

    @property
    def adapters_dir(self) -> Path:
        return self.lakehouse_dir / "adapters"

    @property
    def s3_warehouse(self) -> str:
        return f"s3://{self.s3_bucket}/warehouse"

    def domains_str(self) -> str:
        return ", ".join(self.domains)


# Instance partagée par défaut.
settings = Settings()
