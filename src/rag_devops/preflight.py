"""Pré-vol d'ingestion : que se passerait-il si on ingérait ce document ? (lecture seule)

Classe chaque fichier AVANT ingestion — PDF texte, PDF scanné, PDF mixte, image,
texte — et indique la voie d'extraction (normale ou OCR, et sur quelles pages), le volume
attendu, le domaine qui serait attribué et si le fichier est déjà dans le lakehouse.
Rien n'est écrit : ni lakehouse, ni index.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .config import Settings, settings
from .ingest import IMAGE_EXTS, PDF_EXTS, TEXT_EXTS
from .ocr import _MIN_TEXT, available, is_scanned, ocr_languages

OCR_SECONDS_PER_PAGE = (0.3, 1.0)  # mesuré sur RTX 4060 / CPU, 300 dpi


def _pages_list(pages: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(p) for p in pages[:limit])
    return shown + (f" … (+{len(pages) - limit})" if len(pages) > limit else "")


def inspect_file(path, cfg: Settings | None = None, domain: str | None = None) -> dict:
    """Diagnostic d'un fichier (aucune écriture)."""
    cfg = cfg or settings
    p = Path(path)
    info: dict = {"file": str(p), "name": p.name, "ext": p.suffix.lower(), "ok": True}
    if not p.is_file():
        return {**info, "ok": False, "kind": "missing", "verdict": "file not found"}
    data = p.read_bytes()
    info["bytes"] = len(data)
    info["sha256"] = hashlib.sha256(data).hexdigest()
    info["domain"] = domain or p.parent.name or "(racine)"
    step = max(1, cfg.chunk_size - cfg.chunk_overlap)

    if info["ext"] in PDF_EXTS:
        import pymupdf

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            return {**info, "ok": False, "kind": "unreadable PDF", "verdict": f"cannot open: {exc}"}
        with doc:
            if doc.needs_pass:
                return {**info, "ok": False, "kind": "encrypted PDF",
                        "verdict": "password-protected — cannot be ingested"}
            text_pages, scanned, blank, chars = [], [], [], 0
            for i, page in enumerate(doc, 1):
                t = page.get_text("text").strip()
                if len(t) >= _MIN_TEXT:
                    text_pages.append(i)
                    chars += len(t)
                elif is_scanned(page, t):
                    scanned.append(i)
                else:
                    blank.append(i)
            info.update(pages=doc.page_count, text_pages=len(text_pages), scanned_pages=scanned,
                        blank_pages=len(blank), chars=chars)
        if scanned and not text_pages:
            info["kind"] = "scanned PDF"
        elif scanned:
            info["kind"] = "mixed PDF"
        elif text_pages:
            info["kind"] = "text PDF"
        else:
            info["kind"] = "empty PDF"
        if scanned:
            lo, hi = (round(len(scanned) * s) for s in OCR_SECONDS_PER_PAGE)
            eta = "<1s" if hi <= 1 else (f"≈{hi}s" if lo == hi else f"≈{max(lo, 1)}–{hi}s")
            info["route"] = (f"OCR on {len(scanned)} page(s) [{_pages_list(scanned)}], {eta}"
                             + ("" if not text_pages else f" + normal extraction on {len(text_pages)}"))
        else:
            info["route"] = "normal extraction — no OCR needed" if text_pages else "nothing to extract"
        if scanned and not available(cfg):
            info["ok"] = False
            info["route"] += "  ⚠ OCR unavailable (install Tesseract, or RAG_OCR is off)"

    elif info["ext"] in IMAGE_EXTS:
        info.update(kind="image", pages=1, chars=0, scanned_pages=[1])
        info["route"] = "OCR (whole image)"
        if not available(cfg):
            info["ok"] = False
            info["route"] += "  ⚠ OCR unavailable"
    elif info["ext"] in TEXT_EXTS:
        text = data.decode("utf-8", errors="ignore").strip()
        info.update(kind="markdown" if info["ext"] == ".md" else "text", chars=len(text))
        info["route"] = "normal extraction"
    else:
        return {**info, "ok": False, "kind": "unsupported",
                "verdict": f"unsupported type {info['ext'] or '(none)'} — use .pdf .md .txt or an image"}

    # volume attendu (la partie OCR n'est connue qu'après reconnaissance)
    info["chunks_est"] = max(1, round(info["chars"] / step)) if info.get("chars") else 0

    # déjà dans le lakehouse ? Lecture seule : on n'ouvre le catalogue que s'il existe déjà
    # (l'ouvrir le créerait, avec son dossier — un pré-vol ne doit rien écrire).
    info["duplicate"] = False
    try:
        from .lake import storage
        from .lake.lakehouse import _doc_exists

        backend = storage.resolve_backend(cfg)
        db = cfg.lakehouse_dir / ("catalog-s3.db" if backend == "s3" else "catalog.db")
        if db.exists():
            info["duplicate"] = _doc_exists(info["sha256"], cfg)
    except Exception:
        pass

    if info["duplicate"]:
        info["verdict"] = "already in the lakehouse — ingest would skip it (duplicate)"
    elif not info["ok"]:
        info["verdict"] = "cannot be fully ingested — see route"
    elif info.get("scanned_pages"):
        info["verdict"] = "OCR ingest" if info["kind"] in ("scanned PDF", "image") else "normal + OCR ingest"
    else:
        info["verdict"] = "normal ingest"
    return info


def format_report(info: dict, cfg: Settings | None = None) -> str:
    cfg = cfg or settings
    if info.get("kind") in ("missing", "unsupported", "encrypted PDF", "unreadable PDF"):
        return f"{info['name']}\n  ✗ {info['verdict']}"
    lines = [info["name"]]
    if "pages" in info and info["kind"] != "image":
        detail = (f"{info['pages']} pages · {info['text_pages']} with text · "
                  f"{len(info['scanned_pages'])} scanned · {info['blank_pages']} blank")
    else:
        detail = f"{info['bytes'] / 1024:.0f} KB"
    lines.append(f"  type      {info['kind']}  ({detail})")
    lines.append(f"  route     {info['route']}")
    if info.get("scanned_pages"):
        lines.append(f"  ocr       languages {ocr_languages(cfg)} · {cfg.ocr_dpi} dpi · mode {cfg.ocr_mode}")
    if info.get("chars"):
        extra = " (+ OCR text, known after recognition)" if info.get("scanned_pages") else ""
        lines.append(f"  content   ≈{info['chars']:,} characters → ≈{info['chunks_est']} chunks{extra}")
    lines.append(f"  domain    {info['domain']}")
    lines.append(f"  verdict   {'✓' if info['ok'] and not info['duplicate'] else '•'} {info['verdict']}")
    return "\n".join(lines)
