"""Lakehouse — stockage et catalogue des documents dont l'assistant apprend.

Couches (medallion) :
  BRONZE  documents  — fichiers bruts (MinIO) + métadonnées (table Iceberg)
  SILVER  chunks     — texte nettoyé et découpé (table Iceberg)
  GOLD    index FAISS + graphe (servis depuis data/index/), reconstruits depuis SILVER

MinIO fournit le stockage objet S3 ; Apache Iceberg (pyiceberg + catalogue SQLite)
la couche tabulaire, sans Spark.
"""
from .lakehouse import (  # noqa: F401
    ingest_bytes,
    ingest_path,
    list_documents,
    backfill_from_folder,
    rebuild_serving_index,
    lakehouse_status,
)
from .storage import init_lakehouse  # noqa: F401
