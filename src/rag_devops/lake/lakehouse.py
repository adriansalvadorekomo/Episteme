"""Orchestration du lakehouse : ingestion, migration, reconstruction de l'index.

Réutilise le pipeline d'ingestion existant (`ingest.py`) pour le nettoyage et le
découpage, l'`Embedder` pour les vecteurs, et `VectorStore`/`graph` pour la couche
de service. Ajouter un document = écrire BRONZE + SILVER puis indexer de façon
incrémentale (on n'embedde que les nouveaux chunks).
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from ..config import Settings, settings
from ..ingest import Chunk, PDF_EXTS, TEXT_EXTS, clean_text, discover_files, split_into_chunks
from . import catalog as cat
from . import storage


# --------------------------------------------------------------------------
# Extraction de texte depuis des octets (sans passer par un fichier temporaire)
# --------------------------------------------------------------------------
def _extract(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTS:
        import fitz

        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)
    if ext in TEXT_EXTS:
        return data.decode("utf-8", errors="ignore")
    # type inconnu : tentative de décodage texte
    return data.decode("utf-8", errors="ignore")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classify_domain(text: str, store, cfg: Settings | None = None) -> str:
    """Devine le domaine d'un document : catégorie existante dont le centroïde
    d'embeddings est le plus proche du document. Permet au système d'assigner
    automatiquement le domaine d'un fichier attaché (aucun choix pour l'utilisateur).
    """
    import numpy as np

    cfg = cfg or settings
    if store is None or not getattr(store, "chunks", None):
        return "uploads"
    vecs = store.vectors
    by: dict[str, list[int]] = {}
    for i, c in enumerate(store.chunks):
        by.setdefault(c["category"], []).append(i)
    names, cents = [], []
    for cat, idxs in by.items():
        m = vecs[idxs].mean(axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            names.append(cat)
            cents.append(m / n)
    if not cents:
        return "uploads"
    qv = store.embedder.encode_passages([text[:2000]], show_progress=False)[0]
    sims = np.vstack(cents) @ qv
    return names[int(sims.argmax())]


def _doc_exists(sha: str, cfg: Settings) -> bool:
    from pyiceberg.expressions import EqualTo

    try:
        tbl = storage.table(cat.DOCUMENTS, cfg)
        return tbl.scan(row_filter=EqualTo("sha256", sha)).to_arrow().num_rows > 0
    except Exception:
        return False


def _append(name: str, rows: list[dict], cfg: Settings) -> None:
    tbl = storage.table(name, cfg)
    tbl.append(pa.Table.from_pylist(rows, schema=cat.arrow_for(name)))


# --------------------------------------------------------------------------
# Ingestion d'un document
# --------------------------------------------------------------------------
def ingest_bytes(
    data: bytes,
    domain: str,
    filename: str,
    cfg: Settings | None = None,
    source: str | None = None,
    store=None,
    do_index: bool = True,
) -> dict:
    """Ingère un fichier (octets) dans le lakehouse.

    - dédoublonnage par sha256 (un ré-upload identique est ignoré) ;
    - BRONZE (fichier brut) + SILVER (chunks) écrits dans Iceberg ;
    - si `store` fourni et `do_index`, indexation incrémentale (embeddings des seuls
      nouveaux chunks + ajout FAISS + reconstruction du graphe).

    Renvoie un résumé ; `chunks` contient les dicts des nouveaux chunks.
    """
    cfg = cfg or settings
    sha = _sha256(data)

    if _doc_exists(sha, cfg):
        return {"status": "duplicate", "doc_id": sha,
                "source": source or f"{domain or 'uploads'}/{filename}", "chunks_added": 0}

    text = clean_text(_extract(data, filename))
    pieces = split_into_chunks(text, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_chars)
    if not pieces:
        return {"status": "empty", "doc_id": sha,
                "source": source or f"{domain or 'uploads'}/{filename}", "chunks_added": 0}

    # Domaine assigné automatiquement par le système quand il n'est pas précisé.
    if domain in (None, "", "auto"):
        domain = classify_domain(text, store, cfg)
    source = source or f"{domain}/{filename}"

    s3_uri = storage.put_bronze(data, source, cfg)
    now = datetime.utcnow()

    # SILVER — chunks
    chunk_rows = [
        {
            "chunk_id": f"{sha}:{i}", "doc_id": sha, "domain": domain, "source": source,
            "chunk_index": i, "text": piece, "char_len": len(piece), "created_at": now,
        }
        for i, piece in enumerate(pieces)
    ]
    _append(cat.CHUNKS, chunk_rows, cfg)

    # BRONZE — documents
    _append(cat.DOCUMENTS, [{
        "doc_id": sha, "domain": domain, "filename": filename, "source": source,
        "s3_uri": s3_uri, "sha256": sha, "bytes": len(data),
        "mime": Path(filename).suffix.lower().lstrip(".") or "txt",
        "n_chunks": len(pieces), "status": "indexed", "ingested_at": now,
    }], cfg)

    # dicts pour la couche de service (mêmes clés que VectorStore.chunks)
    new_chunks = [
        {"text": p, "source": source, "category": domain, "chunk_index": i}
        for i, p in enumerate(pieces)
    ]
    result = {
        "status": "ingested", "doc_id": sha, "source": source, "domain": domain,
        "chunks_added": len(pieces), "chunks": new_chunks,
    }

    if do_index and store is not None:
        from ..graph import build_graph

        store.add_chunks(new_chunks)
        graph = build_graph(store, cfg, verbose=False)
        graph.save()
        result["graph"] = graph
    return result


def ingest_path(path, domain: str | None = None, cfg: Settings | None = None, store=None) -> dict:
    """Ingère un fichier du disque. Domaine par défaut = dossier parent."""
    cfg = cfg or settings
    p = Path(path)
    data = p.read_bytes()
    domain = domain or p.parent.name or "(racine)"
    return ingest_bytes(data, domain, p.name, cfg=cfg, store=store)


# --------------------------------------------------------------------------
# Migration du corpus existant + reconstruction de l'index depuis SILVER
# --------------------------------------------------------------------------
def backfill_from_folder(
    folder=None,
    cfg: Settings | None = None,
    verbose: bool = True,
    domain: str | None = None,
    do_rebuild: bool = True,
) -> dict:
    """Charge un dossier de corpus dans le lakehouse (BRONZE+SILVER), sans indexer.

    - `domain` fixé => tous les fichiers de ce dossier vont dans ce domaine
      (source = "<domain>/<chemin relatif>") ; sinon le domaine = dossier de 1er niveau.
    - `do_rebuild` => reconstruit l'index de service à la fin (mettre False pour
      enchaîner plusieurs dossiers puis reconstruire une seule fois).
    """
    cfg = cfg or settings
    folder = Path(folder) if folder else cfg.corpus_dir
    files = discover_files(folder)
    ingested = dup = empty = 0
    for fp in files:
        rel = fp.relative_to(folder)
        if domain:
            dom, source = domain, f"{domain}/{rel}"
        else:
            dom = rel.parts[0] if len(rel.parts) > 1 else "(racine)"
            source = str(rel)
        res = ingest_bytes(fp.read_bytes(), dom, fp.name, cfg=cfg, source=source, do_index=False)
        ingested += res["status"] == "ingested"
        dup += res["status"] == "duplicate"
        empty += res["status"] == "empty"
        if verbose and res["status"] != "ingested":
            print(f"  {res['status']:<9} {source}")
    out = {"files": len(files), "ingested": ingested, "duplicates": dup, "empty": empty}
    if verbose:
        print(f"  → {folder.name}: {ingested} ingérés · {dup} doublons · {empty} vides")
    if do_rebuild:
        out.update(rebuild_serving_index(cfg, verbose=verbose))
    return out


def _silver_chunks(cfg: Settings) -> list[Chunk]:
    rows = storage.table(cat.CHUNKS, cfg).scan().to_arrow().to_pylist()
    rows.sort(key=lambda r: (r["source"], r["chunk_index"]))
    return [
        Chunk(id=i, text=r["text"], source=r["source"], category=r["domain"],
              chunk_index=r["chunk_index"])
        for i, r in enumerate(rows)
    ]


def rebuild_serving_index(cfg: Settings | None = None, verbose: bool = True) -> dict:
    """Reconstruit l'index FAISS + le graphe depuis la table SILVER (chunks)."""
    from ..graph import build_graph
    from ..index import VectorStore

    cfg = cfg or settings
    chunks = _silver_chunks(cfg)
    if not chunks:
        return {"chunks": 0, "note": "SILVER vide — rien à indexer"}
    store = VectorStore(cfg).build(chunks=chunks, verbose=verbose)
    store.save()
    graph = build_graph(store, cfg, verbose=verbose)
    graph.save()
    return {"chunks": len(chunks)}


# --------------------------------------------------------------------------
# Lecture
# --------------------------------------------------------------------------
def list_documents(cfg: Settings | None = None) -> list[dict]:
    cfg = cfg or settings
    try:
        rows = storage.table(cat.DOCUMENTS, cfg).scan().to_arrow().to_pylist()
    except Exception:
        return []
    for r in rows:
        ts = r.get("ingested_at")
        r["ingested_at"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    rows.sort(key=lambda r: r["ingested_at"], reverse=True)
    return rows


def lakehouse_status(cfg: Settings | None = None) -> dict:
    cfg = cfg or settings
    backend = storage.resolve_backend(cfg)
    try:
        docs = storage.table(cat.DOCUMENTS, cfg).scan().to_arrow().num_rows
        chunks = storage.table(cat.CHUNKS, cfg).scan().to_arrow().num_rows
        return {"connected": True, "backend": backend, "documents": docs, "chunks": chunks}
    except Exception as exc:
        return {"connected": False, "backend": backend, "error": str(exc)}
