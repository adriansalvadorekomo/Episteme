"""Accès au stockage : fichiers bruts (BRONZE) + catalogue Iceberg.

Deux backends, choisis par `cfg.lakehouse_backend` :
  - "s3"    : MinIO / S3 (objets via boto3, entrepôt Iceberg sur S3) ;
  - "local" : entrepôt Iceberg sur disque + fichiers bruts dans data/lakehouse/bronze ;
  - "auto"  : s3 si l'endpoint répond, sinon local.

Ainsi le lakehouse fonctionne même sans conteneur S3 disponible, et bascule sur
MinIO (sans changement de code) dès qu'il tourne.
"""
from __future__ import annotations

import functools
import socket
from urllib.parse import urlparse

from ..config import Settings, settings
from . import catalog as cat

BRONZE_PREFIX = "bronze"


# --- Choix du backend -----------------------------------------------------
def _s3_reachable(cfg: Settings) -> bool:
    try:
        u = urlparse(cfg.s3_endpoint)
        host, port = u.hostname or "localhost", u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except Exception:
        return False


def resolve_backend(cfg: Settings | None = None) -> str:
    cfg = cfg or settings
    if cfg.lakehouse_backend in ("s3", "local"):
        return cfg.lakehouse_backend
    return "s3" if _s3_reachable(cfg) else "local"


# --- MinIO / S3 (boto3) ---------------------------------------------------
def s3_client(cfg: Settings | None = None):
    import boto3

    cfg = cfg or settings
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint,
        aws_access_key_id=cfg.s3_access_key,
        aws_secret_access_key=cfg.s3_secret_key,
        region_name=cfg.s3_region,
    )


def ensure_bucket(cfg: Settings | None = None) -> None:
    cfg = cfg or settings
    s3 = s3_client(cfg)
    try:
        s3.create_bucket(Bucket=cfg.s3_bucket)
    except Exception as exc:  # bucket déjà présent => idempotent
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise


def put_bronze(data: bytes, source: str, cfg: Settings | None = None) -> str:
    """Stocke un fichier brut dans BRONZE (clé = source), renvoie son URI."""
    cfg = cfg or settings
    rel = source.lstrip("/")
    if resolve_backend(cfg) == "s3":
        key = f"{BRONZE_PREFIX}/{rel}"
        s3_client(cfg).put_object(Bucket=cfg.s3_bucket, Key=key, Body=data)
        return f"s3://{cfg.s3_bucket}/{key}"
    # local
    from pathlib import Path

    dest = cfg.lakehouse_dir / BRONZE_PREFIX / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest.as_uri()


# --- Catalogue Iceberg ----------------------------------------------------
@functools.lru_cache(maxsize=4)
def _catalog_cached(sig: tuple):
    from pyiceberg.catalog.sql import SqlCatalog

    backend = sig[0]
    if backend == "s3":
        (_b, db_uri, warehouse, endpoint, ak, sk, region) = sig
        props = {
            "uri": db_uri, "warehouse": warehouse,
            "s3.endpoint": endpoint, "s3.access-key-id": ak,
            "s3.secret-access-key": sk, "s3.region": region,
            "s3.path-style-access": "true",
        }
    else:
        (_b, db_uri, warehouse) = sig
        props = {"uri": db_uri, "warehouse": warehouse}
    return SqlCatalog("lakehouse", **props)


def get_catalog(cfg: Settings | None = None):
    cfg = cfg or settings
    cfg.lakehouse_dir.mkdir(parents=True, exist_ok=True)
    backend = resolve_backend(cfg)
    # Catalogue SQLite distinct par backend : les tables pointent vers des entrepôts
    # différents (s3:// vs file://), on ne les mélange donc pas.
    db_file = cfg.lakehouse_dir / ("catalog-s3.db" if backend == "s3" else "catalog.db")
    db_uri = f"sqlite:///{db_file}"
    if backend == "s3":
        sig = ("s3", db_uri, cfg.s3_warehouse, cfg.s3_endpoint,
               cfg.s3_access_key, cfg.s3_secret_key, cfg.s3_region)
    else:
        warehouse_dir = cfg.lakehouse_dir / "warehouse"
        warehouse_dir.mkdir(parents=True, exist_ok=True)
        sig = ("local", db_uri, warehouse_dir.as_uri())
    return _catalog_cached(sig)


def table(name: str, cfg: Settings | None = None):
    cfg = cfg or settings
    return get_catalog(cfg).load_table((cfg.iceberg_namespace, name))


def init_lakehouse(cfg: Settings | None = None) -> dict:
    """Crée le stockage, le namespace et les tables Iceberg (idempotent)."""
    cfg = cfg or settings
    backend = resolve_backend(cfg)
    if backend == "s3":
        ensure_bucket(cfg)
    catalog = get_catalog(cfg)
    cat.ensure_tables(catalog, cfg.iceberg_namespace)
    return {
        "backend": backend,
        "warehouse": cfg.s3_warehouse if backend == "s3" else str(cfg.lakehouse_dir / "warehouse"),
        "namespace": cfg.iceberg_namespace,
        "tables": sorted(t[-1] for t in catalog.list_tables(cfg.iceberg_namespace)),
    }
