"""API FastAPI exposant l'assistant RAG DevOps/MLOps.

Endpoints :
  GET  /health        — état du service
  GET  /info          — infos sur l'index chargé
  POST /search        — recherche sémantique seule (FAISS)
  POST /ask           — réponse RAG complète (FAISS + LLM)
  POST /summarize     — résumé d'un document/thème (bonus)

Lancement :  uv run uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from rag_devops import lake
from rag_devops.config import settings
from rag_devops.graph import KnowledgeGraph
from rag_devops.index import VectorStore
from rag_devops.llm import LLM
from rag_devops.pipeline import RAGPipeline
from rag_devops.summarize import summarize_document

# État partagé (chargé une seule fois au démarrage).
state: dict = {}
_ingest_lock = asyncio.Lock()
_finetune_state: dict = {"status": "idle"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement de l'index au démarrage ; le LLM est chargé paresseusement
    # à la première requête /ask pour un démarrage rapide.
    try:
        store = VectorStore.load(settings)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Index FAISS absent. Exécutez d'abord : uv run rag-build"
        ) from exc
    llm = LLM(settings)
    graph = KnowledgeGraph.load(settings) if settings.use_graph else None
    state["store"] = store
    state["graph"] = graph
    state["pipeline"] = pipeline = RAGPipeline(store, llm, settings, graph=graph)

    # Préchauffage en tâche de fond : l'API répond tout de suite, et la première
    # question n'attend ni le chargement des modèles ni le calcul des caches.
    def _warm() -> None:
        try:
            store.embedder.model           # modèle d'embeddings
            pipeline._ensure_scope_vecs()  # ancres du garde-fou de périmètre
            pipeline.corpus_df             # fréquences documentaires (garde-fou IDK)
            store.vectors                  # matrice d'embeddings (graphe / re-ranking)
            llm.load()                     # LLM (verrou : jamais chargé deux fois)
        except Exception as exc:  # noqa: BLE001
            print(f"(préchauffage incomplet : {exc})")

    import threading

    threading.Thread(target=_warm, name="warmup", daemon=True).start()
    yield
    state.clear()


app = FastAPI(
    title="Assistant RAG DevOps/MLOps",
    description="Assistant pédagogique fondé sur un corpus de cours DevOps & MLOps.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # démo locale ; restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schémas -----------------------------------------------------------
class Turn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["C'est quoi un Pod Kubernetes ?"])
    mode: str = Field(default="rag", pattern="^(rag|llm)$")  # rag = ancré+cité, llm = intuition
    top_k: int | None = Field(default=None, ge=1, le=15)
    # Mémoire conversationnelle : tours précédents (du plus ancien au plus récent).
    history: list[Turn] = Field(default_factory=list)
    # Force un diagramme (toggle « Visual »). Sinon détecté depuis la formulation.
    visual: bool = False


class RepairRequest(BaseModel):
    code: str = Field(..., min_length=3, max_length=6000)
    error: str = Field(default="", max_length=2000)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int | None = Field(default=None, ge=1, le=15)


class SummarizeRequest(BaseModel):
    topic: str = Field(..., min_length=2, examples=["FastAPI", "Kubernetes", "MLflow"])


# --- Endpoints ---------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect bare root to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": "store" in state}


@app.get("/info")
def info():
    store: VectorStore = state["store"]
    pipeline: RAGPipeline = state["pipeline"]
    graph: KnowledgeGraph | None = state.get("graph")
    return {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "device": settings.resolve_device(),
        "n_chunks": len(store.chunks),
        "top_k": settings.top_k,
        "categories": sorted({c["category"] for c in store.chunks}),
        "retrieval": "graph" if pipeline.uses_graph else "vector",
        "graph_edges": (int(len(graph.w) // 2) if graph else 0),
        "domains": settings.domains,
        "lakehouse": lake.lakehouse_status(settings),
        "adapter": bool(settings.llm_adapter),
    }


@app.post("/search")
def search(req: SearchRequest):
    store: VectorStore = state["store"]
    hits = store.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "score": round(h.score, 4),
                "source": h.chunk["source"],
                "category": h.chunk["category"],
                "text": h.chunk["text"],
            }
            for h in hits
        ],
    }


@app.post("/ask")
def ask(req: AskRequest):
    pipeline: RAGPipeline = state["pipeline"]
    history = [{"role": t.role, "content": t.content} for t in req.history]
    result = pipeline.ask(req.question, mode=req.mode, top_k=req.top_k, history=history, visual=req.visual)
    return {
        "question": result.question,
        "answer": result.answer,
        "mode": result.mode,
        "grounded": result.grounded,
        "method": result.method,  # "graph" | "vector" | "llm"
        "sources": result.sources,  # numérotées [n] : n, source, category, score, text
        "metrics": result.metrics,  # tokens, tok/s, gen_ms, retrieval_ms, peak VRAM…
        "visual": result.visual,    # la réponse contient un diagramme Mermaid
        "verification": result.verification,  # {claims, supported, score, unsupported:[{text, reason}]}
    }


@app.post("/diagram/repair")
def diagram_repair(req: RepairRequest):
    """Le navigateur n'a pas pu analyser un diagramme : le LLM le corrige (un essai)."""
    from rag_devops.diagrams import repair_mermaid

    pipeline: RAGPipeline = state["pipeline"]
    fixed = repair_mermaid(req.code, req.error, pipeline.llm)
    if not fixed:
        raise HTTPException(status_code=422, detail="Diagram could not be repaired.")
    return {"code": fixed}


@app.get("/gpu")
def gpu():
    """Télémétrie GPU en direct (utilisation, VRAM, température, puissance)."""
    from rag_devops.gpu import gpu_stats

    pipeline: RAGPipeline | None = state.get("pipeline")
    return {**gpu_stats(), "model_loaded": bool(pipeline and pipeline.llm.loaded),
            "progress": pipeline.progress if pipeline else None}  # étape réelle (mode étude)


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    store: VectorStore = state["store"]
    pipeline: RAGPipeline = state["pipeline"]
    text = summarize_document(store, req.topic, llm=pipeline.llm, cfg=settings)
    if text.startswith("Aucun document"):
        raise HTTPException(status_code=404, detail=text)
    return {"topic": req.topic, "summary": text}


# --- Lakehouse ---------------------------------------------------------
def _do_ingest(data: bytes, domain: str, filename: str) -> dict:
    """Ingestion incrémentale sur l'index VIVANT (exécuté hors boucle asyncio)."""
    store: VectorStore = state["store"]
    res = lake.ingest_bytes(data, domain, filename, settings, store=store)
    if res.get("status") == "ingested":
        store.save()
        new_graph = res.get("graph")
        if new_graph is not None:
            state["graph"] = new_graph
            state["pipeline"].graph = new_graph
    res.pop("graph", None)
    res.pop("chunks", None)
    return res


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), domain: str = Form("auto")):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide.")
    async with _ingest_lock:  # une ingestion à la fois (index partagé)
        return await run_in_threadpool(_do_ingest, data, domain, file.filename)


@app.get("/documents")
def documents():
    return {"documents": lake.list_documents(settings)}


# --- Fine-tuning (lourd : exécuté en tâche de fond) --------------------
def _run_finetune(max_examples: int, epochs: float) -> None:
    _finetune_state.clear()
    _finetune_state["status"] = "running"
    try:
        from rag_devops.finetune import run_finetune

        res = run_finetune(settings, max_examples=max_examples, epochs=epochs)
        _finetune_state.update(status="done", **res)
    except Exception as exc:  # noqa: BLE001
        _finetune_state.update(status="error", error=str(exc))


@app.post("/finetune")
def finetune(background: BackgroundTasks, max_examples: int = 200, epochs: float = 1.0):
    if _finetune_state.get("status") == "running":
        return {"status": "running", "note": "un entraînement est déjà en cours"}
    background.add_task(_run_finetune, max_examples, epochs)
    return {"status": "started", "max_examples": max_examples, "epochs": epochs}


@app.get("/finetune/status")
def finetune_status():
    return _finetune_state
