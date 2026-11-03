"""Schémas et helpers des tables Iceberg du lakehouse.

Trois tables dans le namespace `lakehouse` :
  - documents     (BRONZE) : un enregistrement par fichier ingéré
  - chunks        (SILVER) : un enregistrement par segment de texte
  - finetune_runs : journal des exécutions de fine-tuning LoRA
"""
from __future__ import annotations

import pyarrow as pa
from pyiceberg.schema import Schema
from pyiceberg.types import (
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
)

DOCUMENTS = "documents"
CHUNKS = "chunks"
FINETUNE_RUNS = "finetune_runs"

# --- Schémas Iceberg ------------------------------------------------------
DOCUMENTS_SCHEMA = Schema(
    NestedField(1, "doc_id", StringType(), required=True),
    NestedField(2, "domain", StringType(), required=True),
    NestedField(3, "filename", StringType(), required=True),
    NestedField(4, "source", StringType(), required=True),
    NestedField(5, "s3_uri", StringType(), required=True),
    NestedField(6, "sha256", StringType(), required=True),
    NestedField(7, "bytes", LongType(), required=True),
    NestedField(8, "mime", StringType(), required=False),
    NestedField(9, "n_chunks", IntegerType(), required=True),
    NestedField(10, "status", StringType(), required=True),
    NestedField(11, "ingested_at", TimestampType(), required=True),
)

CHUNKS_SCHEMA = Schema(
    NestedField(1, "chunk_id", StringType(), required=True),
    NestedField(2, "doc_id", StringType(), required=True),
    NestedField(3, "domain", StringType(), required=True),
    NestedField(4, "source", StringType(), required=True),
    NestedField(5, "chunk_index", IntegerType(), required=True),
    NestedField(6, "text", StringType(), required=True),
    NestedField(7, "char_len", IntegerType(), required=True),
    NestedField(8, "created_at", TimestampType(), required=True),
)

FINETUNE_RUNS_SCHEMA = Schema(
    NestedField(1, "run_id", StringType(), required=True),
    NestedField(2, "created_at", TimestampType(), required=True),
    NestedField(3, "base_model", StringType(), required=True),
    NestedField(4, "adapter_path", StringType(), required=True),
    NestedField(5, "n_examples", IntegerType(), required=True),
    NestedField(6, "status", StringType(), required=True),
)

# --- Schémas PyArrow correspondants (pour écrire) -------------------------
# nullable=False pour coller aux champs Iceberg `required` (sinon mismatch à l'append).
def _f(name: str, typ, *, nullable: bool = False) -> pa.Field:
    return pa.field(name, typ, nullable=nullable)


DOCUMENTS_ARROW = pa.schema([
    _f("doc_id", pa.string()), _f("domain", pa.string()), _f("filename", pa.string()),
    _f("source", pa.string()), _f("s3_uri", pa.string()), _f("sha256", pa.string()),
    _f("bytes", pa.int64()), _f("mime", pa.string(), nullable=True), _f("n_chunks", pa.int32()),
    _f("status", pa.string()), _f("ingested_at", pa.timestamp("us")),
])
CHUNKS_ARROW = pa.schema([
    _f("chunk_id", pa.string()), _f("doc_id", pa.string()), _f("domain", pa.string()),
    _f("source", pa.string()), _f("chunk_index", pa.int32()), _f("text", pa.string()),
    _f("char_len", pa.int32()), _f("created_at", pa.timestamp("us")),
])
FINETUNE_RUNS_ARROW = pa.schema([
    _f("run_id", pa.string()), _f("created_at", pa.timestamp("us")),
    _f("base_model", pa.string()), _f("adapter_path", pa.string()),
    _f("n_examples", pa.int32()), _f("status", pa.string()),
])

_TABLES = {
    DOCUMENTS: (DOCUMENTS_SCHEMA, DOCUMENTS_ARROW),
    CHUNKS: (CHUNKS_SCHEMA, CHUNKS_ARROW),
    FINETUNE_RUNS: (FINETUNE_RUNS_SCHEMA, FINETUNE_RUNS_ARROW),
}


def ensure_tables(catalog, namespace: str) -> None:
    """Crée le namespace et les tables manquantes (idempotent)."""
    try:
        catalog.create_namespace_if_not_exists(namespace)
    except AttributeError:  # pyiceberg plus ancien
        if (namespace,) not in catalog.list_namespaces():
            catalog.create_namespace(namespace)
    existing = {t[-1] for t in catalog.list_tables(namespace)}
    for name, (schema, _arrow) in _TABLES.items():
        if name not in existing:
            catalog.create_table((namespace, name), schema=schema)


def arrow_for(name: str) -> pa.Schema:
    return _TABLES[name][1]
