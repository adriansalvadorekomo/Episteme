"""Pré-vol : classification des documents avant ingestion, sans aucune écriture."""
import dataclasses

import pymupdf
import pytest

from rag_devops.config import settings
from rag_devops.preflight import format_report, inspect_file
from test_ocr import FR, _render, _scanned_pdf


@pytest.fixture
def cfg(tmp_path):
    # lakehouse/index isolés : le pré-vol ne doit jamais toucher les vraies données
    return dataclasses.replace(settings, lakehouse_dir=tmp_path / "lh", index_dir=tmp_path / "idx",
                               lakehouse_backend="local")


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def test_text_pdf_normal_ingest(tmp_path, cfg):
    r = inspect_file(_write(tmp_path, "t.pdf", _scanned_pdf(text_pages=("Kubernetes schedules pods. " * 6,) * 3)), cfg)
    assert r["kind"] == "text PDF" and r["pages"] == 3 and r["scanned_pages"] == []
    assert r["verdict"] == "normal ingest" and "no OCR" in r["route"] and r["chunks_est"] >= 1


def test_scanned_pdf_ocr_ingest(tmp_path, cfg):
    r = inspect_file(_write(tmp_path, "s.pdf", _scanned_pdf(FR, FR)), cfg)
    assert r["kind"] == "scanned PDF" and r["scanned_pages"] == [1, 2] and r["verdict"] == "OCR ingest"


def test_mixed_pdf_lists_ocr_pages(tmp_path, cfg):
    r = inspect_file(_write(tmp_path, "m.pdf", _scanned_pdf(FR, text_pages=("Docker builds images. " * 5,))), cfg)
    assert r["kind"] == "mixed PDF" and r["scanned_pages"] == [2]
    assert r["verdict"] == "normal + OCR ingest" and "OCR on 1 page(s) [2]" in r["route"]


def test_image_and_markdown(tmp_path, cfg):
    img = inspect_file(_write(tmp_path, "b.png", _render("Kafka topics").tobytes("png")), cfg)
    md = inspect_file(_write(tmp_path, "n.md", "# Spark\n\nSpark caches DataFrames."), cfg, domain="big-data")
    assert img["kind"] == "image" and img["verdict"] == "OCR ingest"
    assert md["kind"] == "markdown" and md["verdict"] == "normal ingest" and md["domain"] == "big-data"


def test_unsupported_and_encrypted(tmp_path, cfg):
    assert inspect_file(_write(tmp_path, "x.docx", b"PK"), cfg)["kind"] == "unsupported"
    doc = pymupdf.open(); doc.new_page().insert_text((72, 72), "secret")
    enc = tmp_path / "e.pdf"
    doc.save(enc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    r = inspect_file(enc, cfg)
    assert r["kind"] == "encrypted PDF" and not r["ok"]
    assert "password" in format_report(r, cfg)


def test_default_domain_is_parent_folder(tmp_path, cfg):
    (tmp_path / "0 - Inbox").mkdir()
    r = inspect_file(_write(tmp_path / "0 - Inbox", "n.md", "# x\n\nsome text here"), cfg)
    assert r["domain"] == "0 - Inbox"


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from rag_devops import cli

    lh, idx = tmp_path / "lh", tmp_path / "idx"
    monkeypatch.setattr(cli, "settings", dataclasses.replace(settings, lakehouse_dir=lh, index_dir=idx,
                                                             lakehouse_backend="local"))
    f = _write(tmp_path, "s.pdf", _scanned_pdf(FR))
    cli.ingest_cmd([str(f), "--domain", "big-data", "--dry-run"])
    out = capsys.readouterr().out
    assert "scanned PDF" in out and "OCR ingest" in out and "nothing was written" in out
    assert not lh.exists() and not idx.exists()
