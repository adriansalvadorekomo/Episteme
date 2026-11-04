"""Couche OCR : pages scannées, PDF mixtes, images, multilingue, mode off."""
import dataclasses

import pymupdf
import pytest

from rag_devops import ocr
from rag_devops.config import settings
from rag_devops.lake.lakehouse import _extract

pytestmark = pytest.mark.skipif(ocr.tessdata_dir() is None, reason="Tesseract non installé")

FR = ("Le système HDFS répartit les blocs de données sur plusieurs nœuds. "
      "La réplication garantit la tolérance aux pannes. Hadoop exécute MapReduce avec YARN.")


def _render(text: str) -> pymupdf.Pixmap:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=300)
    page.insert_textbox(pymupdf.Rect(40, 40, 555, 260), text, fontsize=14)
    return page.get_pixmap(dpi=200)


def _scanned_pdf(*pages: str, text_pages: tuple[str, ...] = ()) -> bytes:
    """PDF dont les pages `pages` ne sont QUE des images (pas de couche texte)."""
    doc = pymupdf.open()
    for t in text_pages:
        doc.new_page(width=595, height=300).insert_textbox(pymupdf.Rect(40, 40, 555, 260), t, fontsize=14)
    for t in pages:
        p = doc.new_page(width=595, height=300)
        p.insert_image(p.rect, pixmap=_render(t))
    return doc.tobytes()


def test_languages_cover_en_fr_es_de():
    langs = ocr.ocr_languages().split("+")
    assert "eng" in langs
    if len(langs) < 4:
        pytest.skip("run `mise run ocr:setup` for fra/spa/deu")
    assert set(langs) == {"eng", "fra", "spa", "deu"}


def test_scanned_pdf_is_ocrd_with_accents():
    text, pages = _extract(_scanned_pdf(FR), "scan.pdf")
    assert pages == 1
    for word in ("HDFS", "Hadoop", "MapReduce", "YARN", "réplication", "tolérance"):
        assert word in text, word
    if "fra" in ocr.ocr_languages():
        assert "système" in text  # l'accent survit avec le modèle français


def test_text_pdf_is_not_ocrd():
    text, pages = _extract(_scanned_pdf(text_pages=("Kubernetes schedules pods onto nodes.",)), "t.pdf")
    assert pages == 0 and "Kubernetes schedules pods" in text


def test_mixed_pdf_only_ocrs_scanned_pages():
    data = _scanned_pdf(FR, text_pages=("Docker builds images from a Dockerfile.",))
    text, pages = _extract(data, "mixed.pdf")
    assert pages == 1
    assert "Docker builds images" in text and "HDFS" in text


def test_image_is_ocrd():
    png = _render("Kafka topics are partitioned across brokers for scalability.").tobytes("png")
    text, pages = _extract(png, "whiteboard.png")
    assert pages == 1 and "Kafka" in text and "partitioned" in text


def test_ocr_off_leaves_scans_empty(monkeypatch):
    monkeypatch.setattr("rag_devops.ocr.settings", dataclasses.replace(settings, ocr_mode="off"))
    text, pages = _extract(_scanned_pdf(FR), "scan.pdf")
    assert pages == 0 and not text.strip()


def test_short_ocr_document_is_kept(tmp_path, monkeypatch):
    """A whiteboard photo with ~75 chars of text must not be dropped as noise."""
    from rag_devops.lake import lakehouse

    calls = {}
    monkeypatch.setattr(lakehouse, "_doc_exists", lambda sha, cfg: False)
    monkeypatch.setattr(lakehouse.storage, "put_bronze", lambda data, source, cfg: "file:///x")
    monkeypatch.setattr(lakehouse, "_append", lambda name, rows, cfg: calls.setdefault(name, rows))
    png = _render("Kafka topics are partitioned across brokers. Consumers form groups.").tobytes("png")
    res = lakehouse.ingest_bytes(png, "big-data", "board.png", settings, do_index=False)
    assert res["status"] == "ingested" and res["chunks_added"] == 1 and res["ocr_pages"] == 1
    assert "Kafka" in calls["chunks"][0]["text"]


def test_decorative_image_does_not_trigger_ocr():
    """A blank page with a thin decorative rule (tiny image) is not 'scanned'."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    rule = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 2), False)
    rule.clear_with(0)
    page.insert_image(pymupdf.Rect(90, 400, 500, 402), pixmap=rule)
    assert not ocr.is_scanned(page)
    text, pages = _extract(doc.tobytes(), "deco.pdf")
    assert pages == 0
