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
    parser.add_argument("action", choices=["init", "backfill", "rebuild", "status", "knowledge"])
    parser.add_argument("--no-books", action="store_true", help="knowledge : ne pas lister les livres à chapitres.")
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
    elif args.action == "knowledge":
        from .knowledge import report

        print(report(settings, books=not args.no_books))


def ingest_cmd(argv: list[str] | None = None) -> None:
    """Ingère des documents dans le lakehouse et met à jour l'index."""
    parser = argparse.ArgumentParser(description="Ingestion de documents.")
    parser.add_argument("paths", nargs="*", help="Fichiers à ingérer.")
    parser.add_argument("--domain", help="Domaine pour tous les fichiers (sinon : dossier parent "
                                         "pour un fichier, sous-dossier de 1er niveau pour --sync).")
    parser.add_argument("--sync", help="Ingère récursivement un dossier (chemins relatifs conservés).")
    parser.add_argument("--with-images", action="store_true",
                        help="Avec --sync : inclure aussi les images (.png, .jpg…) via OCR.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Diagnostic sans rien écrire : type de document, extraction normale "
                             "ou OCR (et sur quelles pages), volume, domaine, doublon.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    from pathlib import Path

    from . import lake

    files = [Path(p) for p in args.paths]
    if not files and not args.sync:
        print("Aucun fichier. Donne des chemins ou --sync <dossier>.")
        return

    if args.dry_run:
        _dry_run(files, args)
        return

    lake.init_lakehouse(settings)

    if args.sync:
        # Dossier : chemins relatifs conservés (source = "<domaine>/<chemin relatif>"), donc
        # deux README.md dans des sous-dossiers différents ne se marchent pas dessus.
        # --domain s'applique à tout le dossier ; sinon domaine = sous-dossier de 1er niveau.
        folder = Path(args.sync)
        if not folder.is_dir():
            print(f"Dossier introuvable : {folder}")
            return
        res = lake.backfill_from_folder(folder, settings, verbose=True, domain=args.domain,
                                        do_rebuild=False, include_images=args.with_images)
        ingested = res["ingested"]
        for fp in files:  # fichiers passés en plus du dossier
            r = lake.ingest_path(fp, args.domain, settings, store=None)
            ingested += r["status"] == "ingested"
            print(f"  {r['status']:<9} {r['source']} (+{r['chunks_added']} chunks)")
        if not ingested:
            print("Rien de nouveau (tous les fichiers sont déjà dans le lakehouse) : index inchangé.")
            return
        print(f"{ingested} nouveau(x) document(s) — reconstruction de l'index…")
        print(lake.rebuild_serving_index(settings, verbose=True))
        return

    if len(files) == 1:
        # incrémental : embeddings des seuls nouveaux chunks + ajout FAISS
        from .graph import KnowledgeGraph
        from .index import VectorStore

        store = VectorStore.load(settings)
        res = lake.ingest_path(files[0], args.domain, settings, store=store)
        if res["status"] == "ingested":
            store.save()
        ocr = f", OCR {res['ocr_pages']} page(s)" if res.get("ocr_pages") else ""
        print(res["status"], res["source"], f"(+{res['chunks_added']} chunks{ocr})")
    else:
        ingested = 0
        for fp in files:
            res = lake.ingest_path(fp, args.domain, settings, store=None)  # indexé par la reconstruction
            ingested += res["status"] == "ingested"
            print(f"  {res['status']:<9} {res['source']} (+{res['chunks_added']} chunks)")
        if not ingested:
            print("Rien de nouveau : index inchangé.")
            return
        print("Reconstruction de l'index…")
        print(lake.rebuild_serving_index(settings, verbose=True))


def _dry_run(files, args) -> None:
    """Pré-vol : diagnostic de chaque document, aucune écriture (ni lakehouse, ni index)."""
    from collections import Counter
    from pathlib import Path

    from .ingest import discover_files
    from .preflight import format_report, inspect_file

    targets = [(f, args.domain) for f in files]
    if args.sync:
        folder = Path(args.sync)
        for fp in discover_files(folder, include_images=args.with_images):
            rel = fp.relative_to(folder)
            dom = args.domain or (rel.parts[0] if len(rel.parts) > 1 else "(racine)")
            targets.append((fp, dom))
    reports = [inspect_file(f, settings, domain=d) for f, d in targets]
    for r in reports:
        print(format_report(r, settings) + "\n")
    if len(reports) > 1:
        kinds = Counter(r.get("kind", "?") for r in reports)
        ocr_pages = sum(len(r.get("scanned_pages") or []) for r in reports)
        dup = sum(bool(r.get("duplicate")) for r in reports)
        print(f"{len(reports)} files · " + " · ".join(f"{n} {k}" for k, n in kinds.most_common())
              + f" · {ocr_pages} page(s) to OCR · {dup} already ingested")
    print("(dry run — nothing was written)")


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
