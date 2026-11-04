"""« Que sait Episteme ? » — carte des connaissances du lakehouse (mise run knowledge).

Par domaine : documents, chunks, part du corpus et les sujets qui le DISTINGUENT
(TF-IDF au niveau des domaines : un mot présent partout — « the », « data » — pèse peu).
Plus : le périmètre du garde-fou, et les documents utilisables en mode chapitre.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

from .config import Settings, settings
from .diagrams import _fold

_JUNK = set("""that this with from have will your which their there they then than were been into more most
also some such when what where while about other only over each them these those using used uses use
pour avec dans sont cette tout tous plus mais elle comme fait faire leur leurs nous vous etre avoir
para como esta este estos estas pero tambien sobre entre cuando donde tiene puede
eine einer eines nicht auch oder werden wird sind kann dass sich wenn noch""".split())


_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|<[^>]+>|\[[^\]]*\]\([^)]*\)")
_MARKUP = set("href https http html github githubusercontent png jpg svg img src alt width height".split())


def _terms(text: str) -> list[str]:
    text = _URL.sub(" ", text)  # liens, e-mails, balises : pas des sujets
    return [t for t in re.findall(r"[a-z][a-z0-9+#-]{3,}", _fold(text)) if t not in _JUNK and t not in _MARKUP]


def domain_stats(chunks: list[dict], top: int = 7) -> dict[str, dict]:
    tf: dict[str, Counter] = defaultdict(Counter)
    n_chunks: Counter = Counter()
    for c in chunks:
        d = c.get("category") or "(root)"
        n_chunks[d] += 1
        tf[d].update(_terms(c["text"]))
    D = len(tf) or 1
    df: Counter = Counter()
    for counts in tf.values():
        df.update({t for t, n in counts.items() if n >= 2})
    out = {}
    for d, counts in tf.items():
        total = sum(counts.values()) or 1
        scored = sorted(((n / total) * math.log((1 + D) / (1 + df[t])), t) for t, n in counts.items() if n >= 3)
        out[d] = {"chunks": n_chunks[d], "top": [t for _, t in reversed(scored[-top:])]}
    return out


def _chapter_count(uri: str, filename: str, sha: str, cfg: Settings) -> tuple[int, str]:
    """(nombre, « chapters » | « sections ») d'après les signets du PDF, mis en cache par
    empreinte. « chapters » seulement si les signets s'intitulent vraiment « Chapter N »."""
    cache_path = cfg.lakehouse_dir / "structure_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if isinstance(cache.get(sha), list):
        return tuple(cache[sha])
    from .lake.storage import get_bronze
    from .study import _CHAPTER_TITLE

    import pymupdf

    try:
        with pymupdf.open(stream=get_bronze(uri, cfg), filetype="pdf") as doc:
            toc = doc.get_toc()
        chapters = sum(1 for _, t, _ in toc if _CHAPTER_TITLE.match(t))
        n, kind = (chapters, "chapters") if chapters else (len([1 for lvl, _, _ in toc if lvl <= 2]), "sections")
    except Exception:
        n, kind = 0, "sections"
    cache[sha] = [n, kind]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return n, kind


def report(cfg: Settings | None = None, books: bool = True) -> str:
    from . import lake
    from .ocr import ocr_languages

    cfg = cfg or settings
    chunks = json.loads(cfg.meta_path.read_text()) if cfg.meta_path.exists() else []
    docs = lake.list_documents(cfg)
    status = lake.lakehouse_status(cfg)
    stats = domain_stats(chunks)
    docs_by_domain = Counter(d["domain"] for d in docs)
    total = len(chunks) or 1

    lines = [
        "",
        "  EPISTEME · what it knows",
        "  ────────────────────────",
        f"  Knowledge base   {len(docs):,} documents · {len(chunks):,} indexed chunks · backend {status.get('backend', '?')}",
        f"  Retrieval        Graph-RAG over the whole index · models {cfg.embedding_model.split('/')[-1]} + {cfg.llm_model.split('/')[-1]}",
        f"  Scope guardrail  {', '.join(cfg.domains)}",
        f"  OCR languages    {ocr_languages(cfg)}",
        "",
        f"  {'DOMAIN':<24}{'DOCS':>6}{'CHUNKS':>8}{'SHARE':>7}   DISTINCTIVE TOPICS",
    ]
    for d, st in sorted(stats.items(), key=lambda kv: -kv[1]["chunks"]):
        share = f"{st['chunks'] / total:>6.0%}"
        lines.append(f"  {d[:23]:<24}{docs_by_domain.get(d, 0):>6}{st['chunks']:>8,}{share:>7}   {', '.join(st['top'])}")
    if books:
        pdfs = [d for d in docs if d["filename"].lower().endswith(".pdf")]
        withc = []
        for d in pdfs:
            n, kind = _chapter_count(d["s3_uri"], d["filename"], d["sha256"], cfg)
            if n >= 2:
                withc.append((d["filename"], n, kind, d["domain"]))
        books_n = sum(1 for w in withc if w[2] == "chapters")
        lines += ["", f"  CHAPTER MODE — {len(withc)} of {len(pdfs)} PDFs are structured ({books_n} books with chapters, "
                      f"{len(withc) - books_n} with bookmarked sections); the rest are summarized by topic / page groups"]
        for name, n, kind, dom in sorted(withc, key=lambda x: (x[2] != "chapters", -x[1])):
            lines.append(f"    {name[:48]:<50}{n:>3} {kind:<9} [{dom}]")
    lines += ["",
              "  Try: “summarize chapter 2 of <book>”, “explain the part about <topic> in <document>”,",
              "       “explain visually <concept>”. Ask in RAG mode to get cited, verifiable answers.", ""]
    return "\n".join(lines)
