"""Interface en ligne de commande : construction de l'index et questions."""
from __future__ import annotations

import argparse
import sys

from .config import settings
from .index import VectorStore
from .pipeline import RAGPipeline


def build_cmd(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construit l'index FAISS + le graphe du corpus.")
    parser.add_argument("--corpus", help="Dossier du corpus (surcharge la config).")
    parser.add_argument("--no-graph", action="store_true", help="Ne pas construire le graphe Graph-RAG.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.corpus:
        from pathlib import Path

        settings.corpus_dir = Path(args.corpus)

    print(f"Corpus : {settings.corpus_dir}")
    store = VectorStore(settings).build(verbose=True)
    store.save()
    print(f"💾 Index sauvegardé dans {settings.index_dir}")

    if not args.no_graph and settings.use_graph:
        from .graph import build_graph

        graph = build_graph(store, settings, verbose=True)
        graph.save()
        print(f"💾 Graphe Graph-RAG sauvegardé dans {settings.graph_path}")


def ask_cmd(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pose une question à l'assistant RAG.")
    parser.add_argument("question", nargs="+", help="La question à poser.")
    parser.add_argument("--k", type=int, default=settings.top_k, help="Nb d'extraits.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    question = " ".join(args.question)
    store = VectorStore.load(settings)
    rag = RAGPipeline(store)
    result = rag.ask(question, top_k=args.k)

    print("\n" + "=" * 70)
    print("QUESTION :", question)
    print("=" * 70)
    print(result.answer)
    print("\n— Sources —")
    for s in result.sources:
        print(f"  • {s['source']}  (score {s['score']})")


def lakehouse_cmd(argv: list[str] | None = None) -> None:
    """Gestion du lakehouse : init | backfill | rebuild | status."""
    parser = argparse.ArgumentParser(description="Lakehouse (MinIO + Iceberg).")
    parser.add_argument("action", choices=["init", "backfill", "rebuild", "status"])
    parser.add_argument("--folder", help="Dossier à migrer (backfill). Défaut : corpus_dir.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    from . import lake

    if args.action == "init":
        print("Initialisation du lakehouse…")
        print(lake.init_lakehouse(settings))
    elif args.action == "backfill":
        lake.init_lakehouse(settings)
        print(f"Migration de {args.folder or settings.corpus_dir} vers le lakehouse…")
        print(lake.backfill_from_folder(args.folder, settings, verbose=True))
    elif args.action == "rebuild":
        print("Reconstruction de l'index de service depuis SILVER…")
        print(lake.rebuild_serving_index(settings, verbose=True))
    elif args.action == "status":
        print(lake.lakehouse_status(settings))


def ingest_cmd(argv: list[str] | None = None) -> None:
    """Ingère des documents dans le lakehouse et met à jour l'index."""
    parser = argparse.ArgumentParser(description="Ingestion de documents.")
    parser.add_argument("paths", nargs="*", help="Fichiers à ingérer.")
    parser.add_argument("--domain", help="Domaine (sinon dossier parent).")
    parser.add_argument("--sync", help="Ingère récursivement un dossier.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    from pathlib import Path

    from . import lake
    from .ingest import discover_files

    lake.init_lakehouse(settings)
    files = [Path(p) for p in args.paths]
    if args.sync:
        files += discover_files(Path(args.sync))
    if not files:
        print("Aucun fichier. Donne des chemins ou --sync <dossier>.")
        return

    if len(files) == 1 and not args.sync:
        # incrémental : embeddings des seuls nouveaux chunks + ajout FAISS
        from .graph import KnowledgeGraph
        from .index import VectorStore

        store = VectorStore.load(settings)
        res = lake.ingest_path(files[0], args.domain, settings, store=store)
        if res["status"] == "ingested":
            store.save()
        print(res["status"], res["source"], f"(+{res['chunks_added']} chunks)")
    else:
        for fp in files:
            res = lake.ingest_path(fp, args.domain, settings, store=None)  # do_index via rebuild
            print(f"  {res['status']:<9} {res['source']} (+{res['chunks_added']} chunks)")
        print("Reconstruction de l'index…")
        print(lake.rebuild_serving_index(settings, verbose=True))


def finetune_cmd(argv: list[str] | None = None) -> None:
    """Lance un fine-tuning QLoRA sur le corpus du lakehouse (extra `finetune`)."""
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA sur le lakehouse.")
    parser.add_argument("--max-examples", type=int, default=300, help="Nb de paires Q/R.")
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    from .finetune import run_finetune

    print(run_finetune(settings, max_examples=args.max_examples, epochs=args.epochs))


if __name__ == "__main__":  # pragma: no cover
    build_cmd()
