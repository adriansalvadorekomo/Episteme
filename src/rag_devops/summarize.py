"""Bonus — Résumé de document.

Produit un résumé pédagogique d'un document (ou d'un thème) du corpus, en
agrégeant ses chunks puis en demandant au LLM un résumé structuré.
"""
from __future__ import annotations

from .config import Settings, settings
from .index import VectorStore
from .llm import LLM

SUMMARY_SYSTEM = (
    "Tu es un professeur DevOps/MLOps. Tu résumes un support de cours de façon "
    "claire et structurée, en français, en te basant UNIQUEMENT sur le texte fourni."
)

SUMMARY_TEMPLATE = (
    "Voici des extraits d'un document du cours « {name} » :\n\n{content}\n\n"
    "Rédige un résumé pédagogique structuré :\n"
    "- 2 à 4 phrases d'introduction sur le sujet\n"
    "- les points clés sous forme de liste à puces\n"
    "- les commandes ou outils importants s'il y en a.\n"
    "Ne mentionne que ce qui figure dans les extraits."
)


def _collect_text(store: VectorStore, match: str, max_chars: int) -> tuple[str, str]:
    """Rassemble le texte des chunks dont la source contient `match`."""
    selected = [
        c for c in store.chunks if match.lower() in c["source"].lower()
    ]
    selected.sort(key=lambda c: (c["source"], c["chunk_index"]))
    name = selected[0]["source"] if selected else match
    buf = []
    total = 0
    for c in selected:
        if total + len(c["text"]) > max_chars:
            break
        buf.append(c["text"])
        total += len(c["text"])
    return name, "\n\n".join(buf)


def summarize_document(
    store: VectorStore,
    match: str,
    llm: LLM | None = None,
    cfg: Settings | None = None,
    max_chars: int = 6000,
) -> str:
    """Résume le(s) document(s) dont la source contient `match`.

    `match` peut être un nom de fichier partiel ou un thème (ex: 'FastAPI').
    """
    cfg = cfg or settings
    llm = llm or LLM(cfg)
    name, content = _collect_text(store, match, max_chars)
    if not content:
        return f"Aucun document trouvé pour « {match} »."
    prompt = SUMMARY_TEMPLATE.format(name=name, content=content)
    return llm.generate(SUMMARY_SYSTEM, prompt)
