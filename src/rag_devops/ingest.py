"""Étape 1 — Préparation et nettoyage du corpus.

Charge les documents (PDF, Markdown, txt), extrait et nettoie le texte, puis
le découpe en segments (chunks) adaptés à la recherche sémantique.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import Settings, settings

# Extensions prises en charge.
TEXT_EXTS = {".md", ".txt", ".sample"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}  # via OCR


@dataclass
class Chunk:
    """Un segment de texte prêt pour l'embedding, avec sa provenance."""

    id: int
    text: str
    source: str       # chemin relatif au corpus
    category: str     # dossier de 1er niveau (BASIC, Mlops, K8's, …)
    chunk_index: int  # position du chunk dans son document


# --------------------------------------------------------------------------
# Extraction du texte
# --------------------------------------------------------------------------
def extract_pdf(path: Path) -> str:
    """Extrait le texte d'un PDF via PyMuPDF ; les pages scannées passent par l'OCR."""
    import pymupdf  # PyMuPDF (ancien nom du module : fitz)

    from .ocr import pdf_text

    with pymupdf.open(path) as doc:
        return pdf_text(doc)[0]


def extract_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_raw(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return extract_pdf(path)
    if ext in TEXT_EXTS:
        return extract_text_file(path)
    if ext in IMAGE_EXTS:
        from .ocr import image_text

        return image_text(path.read_bytes(), path.name)[0]
    return ""


# --------------------------------------------------------------------------
# Nettoyage
# --------------------------------------------------------------------------
_WS_RUN = re.compile(r"[ \t ]+")
_MULTI_NL = re.compile(r"\n{3,}")
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")        # mots coupés en fin de ligne
_BULLET = re.compile(r"^\s*[•▪◦·]\s*", re.MULTILINE)
_PAGE_NUM = re.compile(r"\n\s*\d{1,3}\s*\n")     # numéros de page isolés


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _BULLET.sub("- ", text)
    text = _PAGE_NUM.sub("\n", text)
    # normalise espaces par ligne, retire lignes vides superflues
    lines = [_WS_RUN.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Découpage (chunking) — frontières de paragraphes, avec recouvrement
# --------------------------------------------------------------------------
def split_into_chunks(
    text: str, chunk_size: int, overlap: int, min_chars: int
) -> list[str]:
    """Découpe en respectant au mieux les paragraphes.

    On agrège les paragraphes jusqu'à ~chunk_size caractères, puis on applique
    un recouvrement (overlap) pour ne pas perdre le contexte aux frontières.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        # paragraphe trop long -> on le coupe durement
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i : i + chunk_size])
            continue

        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            # démarre le nouveau buffer avec un recouvrement textuel
            tail = buf[-overlap:] if buf else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para

    if buf:
        chunks.append(buf)

    # filtre les fragments trop courts (bruit)
    return [c.strip() for c in chunks if len(c.strip()) >= min_chars]


# --------------------------------------------------------------------------
# Pipeline d'ingestion complet
# --------------------------------------------------------------------------
def discover_files(corpus_dir: Path, include_images: bool = False) -> list[Path]:
    """Fichiers ingérables d'un dossier. Les images (OCR) sont opt-in : un coffre Obsidian
    contient souvent des centaines de captures sans texte utile."""
    exts = TEXT_EXTS | PDF_EXTS | (IMAGE_EXTS if include_images else set())
    files = [
        p
        for p in sorted(corpus_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in exts
        and ".git" not in p.parts
    ]
    return files


def build_corpus(cfg: Settings | None = None, verbose: bool = True) -> list[Chunk]:
    """Charge, nettoie et découpe tout le corpus en une liste de Chunk."""
    cfg = cfg or settings
    corpus_dir = cfg.corpus_dir
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus introuvable : {corpus_dir}")

    files = discover_files(corpus_dir)
    chunks: list[Chunk] = []
    cid = 0

    for path in files:
        rel = path.relative_to(corpus_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "(racine)"
        try:
            raw = load_raw(path)
        except Exception as exc:  # pragma: no cover - robustesse ingestion
            if verbose:
                print(f"  ⚠️  Échec extraction {rel}: {exc}")
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            continue

        pieces = split_into_chunks(
            cleaned, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_chars
        )
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    source=str(rel),
                    category=category,
                    chunk_index=i,
                )
            )
            cid += 1

        if verbose:
            print(f"  ✓ {rel} → {len(pieces)} chunks")

    if verbose:
        print(f"\n{len(files)} fichiers → {len(chunks)} chunks au total")
    return chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
