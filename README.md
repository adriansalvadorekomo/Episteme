# Episteme — grounded RAG study assistant

A study assistant that answers **only from its course material**, cites the passages it used,
and **checks every sentence of its answer against those passages**, so you can see which
statements your sources back and which they don't.

> **Personal project**, designed and built alone (November 2026). Everything runs on one
> workstation with a local model (Qwen2.5-1.5B-Instruct on an 8 GB RTX 4060); nothing is sent
> to a hosted LLM.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture/c4-container-dark.svg">
  <img src="docs/architecture/c4-container-light.svg" width="100%" alt="C4 container diagram of Episteme. A learner uses the Astro web app, which calls the FastAPI API and RAG engine; the engine runs Qwen2.5-1.5B on CUDA in-process, retrieves from a FAISS index and a kNN graph, and reads and writes a lakehouse made of SeaweedFS object storage (Bronze) and Apache Iceberg tables in a SQLite catalog (Silver). A Python CLI ingests course material with OCR, rebuilds the index and fine-tunes the model. Model weights come from the Hugging Face Hub; a Cloudflare Quick Tunnel exposes demos.">
</picture>

## Situation · Task · Action · Result · Learning

- **Situation.** A small local LLM answers fluently even when its sources don't support it,
  and top-k retrieval reads only a few passages of a chapter, so chapter summaries skip
  things.
- **Task.** Build a study assistant that answers only from its course material, cites it,
  and shows which sentences are not backed.
- **Action.** Documents land in S3-compatible storage and Apache Iceberg tables
  (Bronze / Silver), from which a FAISS index is rebuilt. Retrieval walks a similarity graph
  (Graph-RAG), and a chapter mode reads whole chapters section by section. A local Qwen2.5
  model is served behind FastAPI, and every answer is checked sentence by sentence against
  its sources (96 pytest tests).
- **Result.** On a 10-question test, **162 of 185 statements (88%) were backed** by the
  sources, up from 78% before the verification-driven fixes.
- **Learning.** A small model sounds sure either way, so the check has to be visible to the
  reader, not hidden in a log.

## What it does

| Feature | How |
|---|---|
| **Grounded answers** | Retrieves passages from the lakehouse and answers with numbered citations `[n]`. An "LLM" mode answers from general knowledge instead, and says so. |
| **Graph-RAG retrieval** | FAISS top-k seeds a personalized PageRank over a kNN graph of the chunks (12 neighbours per chunk), then re-ranks by graph score, cosine and term overlap. Multi-hop context, not just the 5 nearest passages. |
| **Chapter mode** | *"Summarize chapter 3 of Inference Engineering"* reads every section of the chapter (PDF bookmarks or "Chapter N" headings), summarises each one, then writes the overview from those summaries only. Page ranges and one citation per section. |
| **Sentence-level verification** | Each sentence is compared with the closest passages (multilingual-e5 similarity), its numbers and product names must appear in the sources, and "always / never / only" must be in the sources too. The UI shows **✓ VERIFIED 13/13** or **◐ 34/38** and underlines the sentences that aren't backed. |
| **Grounded diagrams** | *"Explain visually…"* draws a Mermaid diagram whose every component appears in the answer or its sources. For a chapter, the figure is the chapter's real structure. When too little can be backed, there is no diagram, and a note says so. |
| **Feynman analogies** | One everyday analogy per answer, in a labelled box ("an analogy, not from the source"), excluded from verification. |
| **Four languages** | Questions and answers in English, Spanish, French and German, even when the sources are in another language. |
| **Lakehouse ingestion** | PDF, Markdown, text and images. Scanned pages go through Tesseract OCR (eng/fra/spa/deu), and files are deduplicated by sha256. A dry run reports what an ingest would do without writing anything. |
| **Web app** | Chat with history, citations, verification details, rendered diagrams, document uploads that are indexed live, and a live GPU / pipeline-step panel. |
| **Fine-tuning (optional)** | QLoRA on the lakehouse content (PEFT + bitsandbytes), served as a LoRA adapter. |

## Architecture

The diagram above is the **C4 container view**: what runs, where the data lives, and who
talks to whom.

| Container | Technology | Responsibility |
|---|---|---|
| Web app | Astro 5, GSAP, Mermaid, marked + DOMPurify | Chat UI, citations, verification badge, diagrams, uploads, GPU panel |
| API + RAG engine | FastAPI, Uvicorn, PyTorch, Transformers, sentence-transformers | Routing, Graph-RAG, chapter mode, generation, diagrams, verification. Qwen2.5-1.5B runs **in-process**, in fp16 on CUDA. |
| CLI & jobs | Python, uv, mise | `rag-ingest` (OCR, dry run), `rag-lakehouse` (rebuild, status, knowledge), `rag-finetune` |
| Serving index | FAISS `IndexFlatIP` (384-d), NumPy | Vectors plus the kNN graph, rebuilt from Silver and loaded by the API at start-up |
| Lakehouse · Bronze | SeaweedFS (S3 API) in Docker | Raw files, sha256-deduplicated, and the Iceberg data files (Parquet) |
| Lakehouse · Silver | Apache Iceberg (pyiceberg), SQLite catalog | `documents` and `chunks` tables: the source of truth for the index |

The **component view** zooms into the API and shows how a question becomes a checked
answer:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture/c4-component-dark.svg">
  <img src="docs/architecture/c4-component-light.svg" width="100%" alt="C4 component diagram of the Episteme API. A question from the web app goes to the router and scope guard, which sends chapter requests to chapter mode (reading Bronze pages section by section) and other questions to the Graph-RAG retriever (FAISS top-k plus personalized PageRank). The generator, Qwen2.5-1.5B, writes the answer with citations; the grounded-diagram builder and the sentence verifier then check it against the same sources using the multilingual-e5 embedder before the answer returns to the web app.">
</picture>

## Results

Measured through the live API on 10 questions across Inference Engineering, Kubernetes,
MLflow, SonarQube, Prometheus and HDFS, in English and French (method and per-question
numbers in [docs/asking-and-trusting-answers.md](docs/asking-and-trusting-answers.md)):

| Question type | Time | Statements backed |
|---|---|---|
| 6 regular questions (Graph-RAG) | 10–27 s | 36 of 38 |
| Visual question with a diagram | 20–46 s | 12 of 17 |
| Chapter summary, English | ~60 s | 38 of 42 |
| Chapter summary + figure, French | ~2 min | 38 of 48 |
| A topic inside a set of slides | ~66 s | 38 of 40 |
| **Overall** | | **162 of 185 (88%), up from 78%** |

The flagged sentences were mostly filler ("this is crucial for reliability"), plus, in the
French chapter, details the model added that weren't in that chapter. Verification doesn't
make a 1.5B model right; it makes the sentences it couldn't back **visible**.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) with Python ≥ 3.12, [pnpm](https://pnpm.io/)
with Node ≥ 20, Docker for the S3 store, and Tesseract for OCR. An NVIDIA GPU is optional;
without one everything runs on the CPU, more slowly.

```bash
mise run install       # Python (uv) + web (pnpm) dependencies
mise run ocr:setup     # OCR models for French, Spanish and German (English comes with Tesseract)
mise run lake:up       # S3 object store (SeaweedFS) on :9000
mise run lake:init     # bucket + Iceberg tables

mise run inspect -- "notes/Big Data 0.pdf"                  # what an ingest would do (writes nothing)
mise run ingest  -- "notes/Big Data 0.pdf" --domain big-data
mise run ingest  -- --sync notes/kubernetes/ --domain devops

mise run dev           # API on :8000 + web app on http://localhost:4321
```

`mise run help` lists every command, grouped by step. `mise run knowledge` shows what
Episteme knows: domains, distinctive topics, and books you can ask about chapter by chapter.

Try these questions in the chat:

- *What is a ReplicaSet in Kubernetes and why is it used?*
- *Summarize chapter 1 of Inference Engineering*
- *Explique visuellement le chapitre 3 de Inference Engineering*
- *Summarize the part about pods in the Kubernetes Notes*

## Commands

| Command | What it does |
|---|---|
| `rag-ingest FILE [FILE…] --domain NAME` | Ingests files into Bronze + Silver and adds their chunks to the index incrementally |
| `rag-ingest --sync FOLDER [--domain NAME] [--with-images]` | Ingests a folder; without `--domain`, each top-level subfolder is a domain |
| `rag-ingest --dry-run …` | Text, scanned or mixed PDF, pages to OCR, chunk estimate, duplicates; writes nothing |
| `rag-lakehouse status \| rebuild \| knowledge` | Counts, rebuild of the index + graph from Silver, knowledge report |
| `rag-ask "QUESTION"` | One question from the terminal |
| `rag-finetune --max-examples 300` | QLoRA adapter trained on the lakehouse (stop the API first: 8 GB isn't enough for both) |

**Safeguards.** If the S3 store has been used but is down, ingest refuses to write to the
local fallback store. A rebuild refuses to shrink the index by more than half unless you
pass `--force`.

## API

FastAPI on `:8000`, with interactive docs at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Answer + sources + `verification` (claims, supported, flagged sentences with reasons) + `metrics` |
| `POST /search` | Retrieval only |
| `POST /summarize` | Summary of one document |
| `POST /ingest` · `GET /documents` | Upload into the lakehouse (indexed live) · list the documents |
| `GET /gpu` | GPU memory and utilisation (NVML) plus the current pipeline step |
| `POST /diagram/repair` | Fixes a Mermaid diagram the browser couldn't parse |
| `POST /finetune` · `GET /finetune/status` | Start and follow a QLoRA run |
| `GET /health` · `GET /info` | Liveness · models, index size, settings |

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Generation model |
| `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model (retrieval and verification) |
| `RAG_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `180` | Chunking, in characters |
| `RAG_TOP_K` | `5` | Passages per answer |
| `RAG_USE_GRAPH` · `RAG_GRAPH_KNN` · `RAG_GRAPH_RESTART` | `1` · `12` · `0.5` | Graph-RAG on/off, neighbours per chunk, PageRank restart probability |
| `RAG_SCOPE_GUARD` · `RAG_DOMAINS` | `1` · *(list)* | Polite refusal outside software engineering; the domains it covers |
| `RAG_ANALOGIES` | `1` | Labelled Feynman analogy in grounded answers |
| `RAG_OCR` · `RAG_OCR_DPI` | `auto` · `300` | OCR scanned pages only, every page (`force`), or never (`off`) |
| `RAG_LAKEHOUSE_BACKEND` | `auto` | `s3`, `local` or `auto` (S3 when it answers) |
| `RAG_S3_ENDPOINT` | `http://localhost:9000` | SeaweedFS / any S3-compatible store |
| `RAG_LLM_ADAPTER` | *(empty)* | `auto` serves the latest LoRA adapter |

## Project structure

```
.
├── src/rag_devops/
│   ├── pipeline.py      # routing, scope guard, languages, prompts, analogies
│   ├── graph.py         # kNN graph + personalized PageRank (Graph-RAG)
│   ├── index.py         # FAISS index      · embed.py  multilingual-e5
│   ├── llm.py           # Qwen2.5 via Transformers (+ LoRA adapter)
│   ├── study.py         # chapter mode: document structure, map → reduce
│   ├── diagrams.py      # grounded diagrams: extracted relations, checked, built as Mermaid
│   ├── verify.py        # sentence-level verification
│   ├── ocr.py           # Tesseract OCR for scanned pages and images
│   ├── preflight.py     # ingest dry run      · knowledge.py  "what it knows" report
│   ├── lake/            # S3 storage, Iceberg catalog, ingestion and index rebuild
│   ├── finetune.py      # QLoRA fine-tuning
│   └── cli.py           # rag-ingest · rag-lakehouse · rag-ask · rag-build · rag-finetune
├── api/main.py          # FastAPI app
├── web/                 # Astro + GSAP + Mermaid chat app
├── tests/               # 96 pytest tests: OCR, dry run, chapter mode, diagrams, verification
├── docs/                # guides + architecture diagrams (and the script that draws them)
├── notebooks/           # end-to-end walkthrough of the pipeline
├── docker-compose.yml   # SeaweedFS (S3 API)
├── mise.toml            # every command: mise run help
└── tunnel.sh            # Cloudflare Quick Tunnels for remote demos
```

## Documentation

- [Ingesting documents](docs/ingesting-documents.md): formats, OCR, dry run, domains, troubleshooting
- [Asking Episteme, and knowing when to trust the answer](docs/asking-and-trusting-answers.md): chapter mode, diagrams, the verification badge, measured results
- [Architecture diagrams](docs/architecture/): C4 container and component views, light and dark, with the script that draws them

## Limitations

- The model is small. It sometimes adds filler or a detail from outside the passage;
  verification flags these sentences but doesn't remove them.
- Answers take 10–27 s, and a chapter summary 1–2 minutes, on an 8 GB laptop-class GPU.
- Cross-lingual verification (a French answer from English sources) uses a separately
  calibrated threshold with a thinner margin than same-language checks.
- Serving and fine-tuning don't fit in 8 GB together.

## Credits

Models: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and
[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small). Diagram logos
come from [Simple Icons](https://simpleicons.org) (CC0) and the Apache Iceberg and SeaweedFS
projects; each trademark belongs to its owner.
