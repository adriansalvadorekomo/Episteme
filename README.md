# 🧠 Assistant LLM / RAG — DevOps & MLOps

Assistant pédagogique (**professeur virtuel DevOps/MLOps**) qui répond aux questions en
s'appuyant **uniquement** sur un corpus de cours, via une architecture
**RAG** (Retrieval-Augmented Generation).

> TP — *Développement d'un assistant LLM basé sur le RAG.* Livrable principal :
> le notebook [`notebooks/rag_devops_assistant.ipynb`](notebooks/rag_devops_assistant.ipynb).

## Architecture

```
PDF / Markdown / txt  ─►  nettoyage + chunking  ─►  embeddings (SentenceTransformers)
                                                              │
                                                              ▼
question  ─►  embedding requête  ─►  FAISS top-k  ─►  contexte  ─►  prompt
                                                              │
                                                              ▼
                                   LLM open-source (Transformers)  ─►  réponse fondée
```

| Rôle | Outil |
|------|-------|
| Environnement / packaging | **uv** |
| Extraction PDF | PyMuPDF |
| Embeddings (FR + EN) | SentenceTransformers — `intfloat/multilingual-e5-small` |
| Index & recherche sémantique | FAISS (`IndexFlatIP`, cosinus) |
| LLM open-source | Transformers — `Qwen/Qwen2.5-1.5B-Instruct` |
| API | FastAPI + Uvicorn |
| Interface web | Astro (pnpm) |

## Prérequis

- [uv](https://docs.astral.sh/uv/) · Python ≥ 3.12
- [pnpm](https://pnpm.io/) · Node ≥ 20 (pour l'interface web)
- GPU NVIDIA optionnel (CUDA) — sinon exécution CPU automatique.

## Installation

```bash
uv sync --extra notebook      # dépendances Python (+ Jupyter)
```

Le corpus par défaut est `~/Downloads/Devops - Lake`. Pour en utiliser un autre :

```bash
export RAG_CORPUS_DIR="/chemin/vers/mon/corpus"
```

## Utilisation

### 0. Lakehouse (stockage des documents)

> 📘 Step-by-step guide (formats, PDFs, domains, troubleshooting): [docs/ingesting-documents.md](docs/ingesting-documents.md)

Le lakehouse (stockage objet S3 + tables Apache Iceberg) est la source de vérité des
documents dont l'assistant apprend. Sans conteneur S3 démarré, il bascule
automatiquement sur un entrepôt Iceberg **local** (`data/lakehouse/`).

```bash
docker compose up -d                  # stockage objet S3 (SeaweedFS) sur :9000  (optionnel)
uv run rag-lakehouse init             # crée le bucket/namespace + tables Iceberg
uv run rag-lakehouse backfill         # migre le corpus (RAG_CORPUS_DIR) dans le lakehouse
uv run rag-lakehouse status           # backend + nb de documents/chunks

uv run rag-ingest mon_doc.pdf --domain data-engineering   # ingestion incrémentale
uv run rag-ingest --sync ~/notes                          # ingestion d'un dossier
```

> Le backend est choisi par `RAG_LAKEHOUSE_BACKEND` (`auto` | `s3` | `local`). MinIO
> n'étant plus distribué librement, `docker compose` démarre SeaweedFS (S3-compatible)
> sur le port 9000 ; remplacer l'image par `minio/minio` ne change rien au code.

### 1. Construire l'index FAISS (depuis le corpus brut)

```bash
uv run rag-build                      # lit le corpus, embeddings, index -> data/index/
# (ou : uv run rag-lakehouse rebuild  — reconstruit l'index depuis la table SILVER)
```

### 2. Poser une question en ligne de commande

```bash
uv run rag-ask "C'est quoi un Pod Kubernetes ?"
```

### 3. Notebook (livrable)

```bash
uv run jupyter lab                    # ouvrir notebooks/rag_devops_assistant.ipynb
```

### 4. API FastAPI

```bash
uv run uvicorn api.main:app --port 8000
# Documentation interactive : http://localhost:8000/docs
```

Endpoints : `GET /info`, `POST /ask`, `POST /search`, `POST /summarize`,
`POST /ingest` (upload), `GET /documents`, `POST /finetune` + `GET /finetune/status`.

### 5. Interface web (Astro)

```bash
cd web
pnpm install
pnpm dev                              # http://localhost:4321  (proxy /api -> :8000)
```

Démarrer l'API **avant** le front. L'app est un produit de chat (modes LLM / RAG,
historique, panneau **Knowledge base** pour téléverser des documents dans le lakehouse).

### 6. Fine-tuning QLoRA (optionnel, extra `finetune`)

```bash
uv sync --extra finetune
uv run rag-finetune --max-examples 300        # entraîne un adaptateur LoRA sur le lakehouse
RAG_LLM_ADAPTER=auto uv run uvicorn api.main:app --port 8000   # sert le modèle fine-tuné
```

> Lourd : l'entraînement et le service ne tiennent pas ensemble en 8 Go — arrêter l'API
> pendant `rag-finetune`. `RAG_LLM_ADAPTER=auto` charge le dernier adaptateur ; le mode
> RAG reste ancré sur le corpus.

## Configuration (variables d'environnement)

| Variable | Défaut | Rôle |
|----------|--------|------|
| `RAG_CORPUS_DIR` | `~/Downloads/Devops - Lake` | dossier source du corpus |
| `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-small` | modèle d'embeddings |
| `RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | LLM de génération |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `180` | découpage |
| `RAG_TOP_K` | `5` | nb d'extraits récupérés |
| `RAG_DEVICE` | `auto` | `cpu` \| `cuda` \| `auto` |
| `RAG_USE_GRAPH` | `1` | activer le **Graph-RAG** (récupération par graphe) |
| `RAG_GRAPH_KNN` | `12` | arêtes par nœud dans le graphe de connaissances |
| `RAG_GRAPH_RESTART` | `0.5` | proba. de redémarrage du PageRank personnalisé |
| `RAG_SCOPE_GUARD` | `1` | garde-fou anti-hors-sujet (refuse hors ingénierie logicielle) |
| `RAG_SCOPE_MARGIN` | `0.0` | seuil contrastif in-domaine / hors-sujet |
| `RAG_DOMAINS` | *(liste)* | périmètre d'expertise (CSV) — élargit le garde-fou |
| `RAG_ANALOGIES` | `1` | analogie à la Feynman, étiquetée, dans les réponses ancrées |
| `RAG_OCR` | `auto` | OCR des pages scannées : `auto` \| `force` \| `off` |

## Le professeur, le Graph-RAG et le garde-fou

- **Persona** — en modes LLM et RAG, l'assistant est le *Professeur DevMLops*, de
  niveau doctorat et style Feynman : premiers principes, analogie, exemple concret.
- **Graph-RAG** — au lieu d'un simple top-k vectoriel, on construit un **graphe
  sémantique** entre les chunks (arêtes = forte similarité cosinus) et on récupère
  par **PageRank personnalisé** amorcé sur les meilleurs hits : récupération
  multi-saut, plus complète et mieux ancrée (moins d'hallucinations). Re-ranking
  final = graphe · cosinus · recouvrement lexical.
- **Garde-fou de périmètre** — décision *contrastive* (plus proche des ancres
  « in-domaine » que « hors-sujet ») : refuse poliment cuisine/politique/etc.,
  et marche identiquement en **EN / ES / FR / DE**.
- **Multilingue** — la langue de la question (EN/ES/FR/DE) est détectée et imposée
  à la génération, même quand le contexte récupéré est dans une autre langue.

- **Mode étude** — « résume le chapitre 2 de Inference Engineering » : la partie demandée
  est lue en entier, section par section (map → reduce), avec pages et citations.
- **Diagrammes ancrés** — « explique visuellement… » : figure construite à partir de la
  structure réelle du chapitre, ou de relations dont chaque composant figure dans les sources.
- **Vérification phrase par phrase** — badge ✓/◐ sous chaque réponse ancrée ; les phrases
  non soutenues par les sources sont soulignées.
- **Ce qu'il sait** — `mise run knowledge` (domaines, sujets, livres à chapitres).

Détails : [docs/asking-and-trusting-answers.md](docs/asking-and-trusting-answers.md).

### Ajouter un nouveau domaine (scalabilité)

L'assistant est conçu pour grandir vers le data engineering, l'AI engineering, etc. :

1. Déposez les documents du domaine dans un sous-dossier du corpus
   (p. ex. `data-engineering/`, `ai-engineering/`).
2. Ajoutez le domaine au périmètre : `export RAG_DOMAINS="DevOps,MLOps,…,Data Engineering,AI Engineering"`.
3. Reconstruisez l'index **et** le graphe : `uv run rag-build`.

Aucun code à modifier : embeddings, graphe et garde-fou s'adaptent automatiquement.

## Structure

```
.
├── notebooks/rag_devops_assistant.ipynb   # ← livrable : toutes les étapes + démo
├── src/rag_devops/                        # paquet réutilisable
│   ├── config.py      # configuration centrale
│   ├── ingest.py      # étape 1 — chargement, nettoyage, chunking
│   ├── embed.py       # étape 2 — embeddings SentenceTransformers
│   ├── index.py       # étape 3 — index & recherche FAISS
│   ├── llm.py         # étape 4 — génération Transformers (+ adaptateur LoRA)
│   ├── graph.py       # Graph-RAG — graphe de connaissances + PageRank personnalisé
│   ├── pipeline.py    # orchestration + professeur Feynman + garde-fou + langues
│   ├── lake/          # lakehouse : stockage S3 (storage) + tables Iceberg (catalog)
│   │   └── lakehouse.py   #   ingestion, migration, reconstruction de l'index
│   ├── finetune.py    # fine-tuning QLoRA sur le corpus du lakehouse
│   ├── summarize.py   # bonus — résumé de document
│   └── cli.py         # rag-build / rag-ask / rag-lakehouse / rag-ingest / rag-finetune
├── api/main.py                            # API FastAPI (+ /ingest, /documents, /finetune)
├── web/                                   # interface Astro (chat + Knowledge base)
├── docker-compose.yml                     # stockage objet S3 (SeaweedFS)
├── data/index/                            # index FAISS persisté (généré)
└── data/lakehouse/                        # entrepôt Iceberg local + adaptateurs LoRA (généré)
```
