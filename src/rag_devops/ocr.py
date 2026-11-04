"""Couche OCR : pages scannées de PDF et images → texte (Tesseract via PyMuPDF).

- Une page de PDF n'est reconnue que si elle n'a pas de couche texte mais contient des
  images : les pages normales restent extraites telles quelles (PDF mixtes gérés page à
  page, et pas de coût OCR inutile).
- Les images (.png, .jpg, .webp…) sont ouvertes comme des documents d'une page.
- Langues : anglais, français, espagnol, allemand quand leurs modèles sont présents —
  `data/tessdata` (voir `mise run ocr:setup`), sinon le tessdata du système.

Modes (RAG_OCR) : "auto" (défaut), "force" (OCR de chaque page), "off".
"""
from __future__ import annotations

import functools
from pathlib import Path

from .config import PROJECT_ROOT, Settings, settings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_WANTED = ("eng", "fra", "spa", "deu")
_MIN_TEXT = 20  # en dessous, une page « vide » mais illustrée est considérée scannée


@functools.lru_cache(maxsize=1)
def tessdata_dir() -> str | None:
    """Dossier de modèles Tesseract : celui du projet s'il existe, sinon celui du système."""
    local = PROJECT_ROOT / "data" / "tessdata"
    if any(local.glob("*.traineddata")):
        return str(local)
    try:
        import pymupdf

        return pymupdf.get_tessdata()
    except Exception:
        return None


def ocr_languages(cfg: Settings | None = None) -> str:
    """Langues Tesseract disponibles parmi eng/fra/spa/deu (ex. "eng+fra+spa+deu")."""
    cfg = cfg or settings
    if cfg.ocr_langs:
        return cfg.ocr_langs
    td = tessdata_dir()
    have = {p.stem for p in Path(td).glob("*.traineddata")} if td else set()
    return "+".join(lang for lang in _WANTED if lang in have) or "eng"


def available(cfg: Settings | None = None) -> bool:
    cfg = cfg or settings
    return cfg.ocr_mode != "off" and tessdata_dir() is not None


def _ocr_page(page, cfg: Settings) -> str:
    tp = page.get_textpage_ocr(language=ocr_languages(cfg), dpi=cfg.ocr_dpi, full=True,
                               tessdata=tessdata_dir())
    return page.get_text(textpage=tp)


_MIN_IMAGE_COVER = 0.20  # une page « scannée » est couverte par une vraie image


def is_scanned(page, text: str | None = None) -> bool:
    """Page sans couche texte mais couverte (≥ 20 %) par une image. Un filet décoratif
    de 1 px ou un logo ne suffit pas : inutile (et bruyant) d'envoyer ça à Tesseract."""
    text = page.get_text("text") if text is None else text
    if len(text.strip()) >= _MIN_TEXT:
        return False
    area = abs(page.rect) or 1.0
    try:
        covered = sum(abs(page.rect & pymupdf_rect(info["bbox"])) for info in page.get_image_info())
    except Exception:
        return bool(page.get_images(full=False))
    return covered / area >= _MIN_IMAGE_COVER


def pymupdf_rect(bbox):
    import pymupdf

    return pymupdf.Rect(bbox)


def page_text(page, cfg: Settings | None = None) -> tuple[str, bool]:
    """Texte d'une page ; (texte, ocr_utilisé)."""
    cfg = cfg or settings
    text = page.get_text("text")
    if not available(cfg):
        return text, False
    scanned = is_scanned(page, text)
    if cfg.ocr_mode == "force" or scanned:
        try:
            ocr = _ocr_page(page, cfg)
        except Exception:
            return text, False  # Tesseract indisponible ou page illisible : on garde l'extraction
        if len(ocr.strip()) > len(text.strip()):
            return ocr, True
    return text, False


def pdf_text(doc, cfg: Settings | None = None) -> tuple[str, int]:
    """Texte d'un document PyMuPDF ouvert ; (texte, nb de pages passées à l'OCR)."""
    parts, ocr_pages = [], 0
    for page in doc:
        text, used = page_text(page, cfg)
        parts.append(text)
        ocr_pages += used
    return "\n".join(parts), ocr_pages


def image_text(data: bytes, filename: str, cfg: Settings | None = None) -> tuple[str, int]:
    """OCR d'une image ; (texte, 1 si reconnue)."""
    import pymupdf

    cfg = cfg or settings
    if not available(cfg):
        return "", 0
    ext = Path(filename).suffix.lower().lstrip(".") or "png"
    with pymupdf.open(stream=data, filetype=ext) as img:
        pdf = pymupdf.open("pdf", img.convert_to_pdf())  # une image = un document d'une page
    with pdf:
        page = pdf[0]
        try:
            text = _ocr_page(page, cfg)
        except Exception:
            return "", 0
    return text, int(bool(text.strip()))
