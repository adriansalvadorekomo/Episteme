"""Mode étude : résumer / expliquer un CHAPITRE (ou une section, ou un document) entier.

La RAG classique ne voit que les 5 fragments les plus proches : pour un chapitre de
dizaines de pages, c'est ~5 % du contenu, d'où des réponses décousues. Ici :

  1. on reconnaît la demande (« résume le chapitre 2 de Inference Engineering »…) ;
  2. on retrouve le document (nom cité, sinon la meilleure source) et sa STRUCTURE
     (signets du PDF, sinon titres « Chapter N » ; titres Markdown pour les notes) ;
  3. MAP : chaque section du chapitre est lue en entier (OCR si page scannée) et résumée
     séparément, consignes strictes « uniquement ce qui est écrit » ;
  4. REDUCE : vue d'ensemble + points clés écrits à partir des seuls résumés de sections ;
  5. l'assemblage est déterministe (titres, pages, citations [n] par section).

Chaque section devient une source citée « document · pages · titre » : on peut vérifier.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

from .config import Settings, settings
from .ingest import clean_text

# ── 1. Intention ────────────────────────────────────────────────────────────
_SUMMARY_VERB = re.compile(
    r"\b(summari[sz]e|summary|sum up|overview|walk me through|explain|teach me|break down|"
    r"resume|resumen|res[uú]meme|explica|expl[ií]came|ens[eé][ñn]ame|"
    r"r[ée]sum[ée]|r[ée]sume|explique|expliquez|pr[ée]sente|"
    r"zusammenfass\w*|fasse\w*|erkl[äa]re?\w*|[üu]berblick)\b", re.I)
_UNIT = re.compile(
    # mots entiers (« chap » ne doit pas avaler « chapitre » et perdre le numéro)
    r"\b(chapter|chapitre|cap[ií]tulo|kapitel|chap\.|chap|ch\.|section|secci[oó]n|abschnitt|"
    r"partie|parte|part|teil)(?![a-zà-ÿ])\s*(?P<num>\d+(?:\.\d+)*)?", re.I)
_SECTION_NUM = re.compile(r"(?<![\d.])(?P<num>\d+\.\d+(?:\.\d+)?)(?![\d.])")
_WHOLE_DOC = re.compile(r"\b(book|document|pdf|notes?|slides|deck|course|libro|documento|apuntes|notas?|"
                        r"livre|cours|buch|dokument|notizen|folien)\b", re.I)
_CHAPTER_TITLE = re.compile(r"^\s*(chapter|chapitre|cap[ií]tulo|kapitel)\s*(\d+)\b", re.I)


def wants_study(question: str) -> bool:
    """« résume / explique le chapitre 2 », « summarize section 1.3 », « summarize the book X »."""
    q = question or ""
    return bool(_SUMMARY_VERB.search(q)) and bool(_UNIT.search(q) or _SECTION_NUM.search(q) or _WHOLE_DOC.search(q))


@dataclass
class Target:
    chapter: int | None = None
    section: str | None = None
    whole: bool = False


def parse_target(question: str) -> Target:
    """« chapter 2 » → chapitre 2 ; « section 1.3 » ou « 1.3 » → section ; sinon document entier."""
    m = next((u for u in _UNIT.finditer(question) if u.group("num")), None)
    if m:
        num = m.group("num")
    else:
        s = _SECTION_NUM.search(question)
        num = s.group("num") if s else None
    if num and "." in num:
        return Target(section=num)
    if num:
        return Target(chapter=int(num))
    return Target(whole=bool(_WHOLE_DOC.search(question)))


# ── 2. Document + structure ─────────────────────────────────────────────────
_STOP = set("the a an of to and in on for with from about this that chapter section summarize summary "
            "explain book document pdf please me de la le les du des el los del un una en por para "
            "resume resumen explica explique chapitre capitulo kapitel zusammenfassung erklare von der die das "
            # mots de structure / de type de document : jamais des sujets
            "part parte partie teil section seccion abschnitt about sobre regarding concerning "
            "notes note slides deck course cours libro livre buch dokument documento apuntes notas notiz "
            "folien whole entire complete todo tout ganze".split())


def _words(s: str) -> set[str]:
    from .diagrams import _fold

    return {w for w in re.findall(r"[a-z0-9]+", _fold(s)) if len(w) >= 3 and w not in _STOP}


def resolve_document(question: str, documents: list[dict]) -> dict | None:
    """Le document dont le NOM est cité dans la question (meilleur recouvrement de mots)."""
    q = _words(question)
    best, score = None, 0.0
    for d in documents:
        name = re.sub(r"\.\w+$", "", d.get("filename") or "")
        w = _words(name)
        if not w:
            continue
        hit = len(q & w) / len(w)
        if hit > score:
            best, score = d, hit
    return best if score >= 0.5 else None


@dataclass
class Node:
    level: int
    title: str
    start: int            # page de début (1-based), ou index de ligne pour le texte
    end: int              # page de fin incluse
    children: list["Node"] = field(default_factory=list)


@dataclass
class Book:
    name: str
    kind: str             # "pdf" | "text"
    pages: list[str]      # texte nettoyé par page (lignes pour le texte)
    nodes: list[Node]     # arbre des sections
    synthetic: bool = False  # pas de signets ni de titres : découpage en groupes de pages


def _heading_score(line: str) -> int:
    """Ressemble-t-il à un titre ? court, majuscule initiale, pas de ponctuation finale,
    pas de code (`batch/v1`, `kubectl get …`)."""
    words = line.split()
    if not (1 <= len(words) <= 7) or not (4 <= len(line) <= 60):
        return 0
    score = 0
    score += line[0].isupper()
    score += not line.endswith((".", ",", ";", ":"))
    score += not re.search(r"[/=<>{}$`]|\b(kubectl|docker|sudo|apiVersion)\b", line)
    score += sum(w[0].isupper() for w in words if w[0].isalpha()) >= max(1, len(words) // 2)
    return score


def _first_heading(pages: list[str]) -> str:
    """Meilleur titre d'un groupe de pages (les diapositives commencent par leur titre)."""
    best, best_score = "", 2
    for p in pages[:2]:
        for line in p.splitlines()[:12]:
            line = line.strip(" #*-•\t")
            sc = _heading_score(line)
            if sc > best_score:
                best, best_score = line, sc
                if sc == 4:
                    return best
    return best


def _window_nodes(pages: list[str]) -> list[Node]:
    """Document sans structure : ~10 groupes de pages, titrés par leur premier titre."""
    n = len(pages)
    size = max(3, -(-n // 10))
    out = []
    for start in range(1, n + 1, size):
        end = min(n, start + size - 1)
        title = _first_heading(pages[start - 1:end]) or f"Pages {start}–{end}"
        out.append(Node(1, title, start, end))
    return out


def _tree(entries: list[tuple[int, str, int]], last_page: int) -> list[Node]:
    """Signets (niveau, titre, page) → arbre avec plages de pages."""
    flat = [Node(l, t.strip(), max(1, p), last_page) for l, t, p in entries if p and p > 0]
    for i, n in enumerate(flat):  # fin = début de la prochaine entrée de niveau ≤
        for m in flat[i + 1:]:
            if m.level <= n.level:
                n.end = max(n.start, m.start - 1)
                break
    roots: list[Node] = []
    stack: list[Node] = []
    for n in flat:
        while stack and stack[-1].level >= n.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(n)
        stack.append(n)
    return roots


def _walk(nodes):
    for n in nodes:
        yield n
        yield from _walk(n.children)


@functools.lru_cache(maxsize=4)
def load_book(uri: str, filename: str) -> Book:
    from .lake.storage import get_bronze

    data = get_bronze(uri)
    if filename.lower().endswith(".pdf"):
        import pymupdf

        from .ocr import page_text

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = [clean_text(page_text(p)[0]) for p in doc]
            toc = doc.get_toc()
        if not toc:  # pas de signets : titres « Chapter N » en tête de page
            toc = [(1, re.sub(r"\s+", " ", p[:80]).strip(), i + 1)
                   for i, p in enumerate(pages) if _CHAPTER_TITLE.match(p[:40])]
        if not toc:  # ni signets ni chapitres (diapositives, notes) : groupes de pages
            return Book(filename, "pdf", pages, _window_nodes(pages), synthetic=True)
        return Book(filename, "pdf", pages, _tree(toc, len(pages)))
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    heads = [(len(m.group(1)), m.group(2), i + 1) for i, ln in enumerate(lines)
             if (m := re.match(r"^(#{1,3})\s+(.+)", ln))]
    return Book(filename, "text", lines, _tree(heads, len(lines)))


def find_target(book: Book, target: Target, question: str) -> Node | None:
    nodes = list(_walk(book.nodes))
    if target.section:
        return next((n for n in nodes if re.match(rf"^{re.escape(target.section)}\b", n.title)), None)
    if target.chapter is not None:
        for n in nodes:
            m = _CHAPTER_TITLE.match(n.title)
            if m and int(m.group(2)) == target.chapter:
                return n
        return next((n for n in nodes if re.match(rf"^{target.chapter}[.:\s]", n.title)), None)
    # titre cité : « explain the chapter about Hardware » — sans compter les mots du NOM du
    # document (sinon « Inference Engineering » ferait gagner « Chapter 0: Inference »)
    q = _words(question) - _words(re.sub(r"\.\w+$", "", book.name))
    best, score = None, 0
    for n in nodes:
        hit = len(q & _words(n.title))
        if hit > score:
            best, score = n, hit
    if best or not q:
        return best
    # pas de titre correspondant : les pages qui PARLENT du sujet (« la partie sur les pods »)
    from .diagrams import _fold

    scored = []
    for n in book.nodes if book.synthetic else [m for m in nodes if not m.children]:
        toks = re.findall(r"[a-z0-9]+", _fold(_text(book, n)))
        # combien la section PARLE du sujet (fréquence, singulier/pluriel confondus),
        # rapportée à sa longueur : une simple mention ne suffit pas
        tf = sum(1 for t in toks if t in q or t.rstrip("s") in q or t + "s" in q)
        if tf:
            scored.append((tf / max(200, len(toks)) * 1000, n))
    if not scored:
        return None
    top = max(h for h, _ in scored)
    best4 = sorted(scored, key=lambda x: -x[0])[:4]
    picked = sorted((n for h, n in best4 if h >= top * 0.5), key=lambda n: n.start)
    topic = " ".join(sorted(q))[:40]
    return Node(1, f"{book.name.rsplit('.', 1)[0]} — {topic}", picked[0].start, picked[-1].end, picked)


def chapter_titles(book: Book) -> list[str]:
    nodes = list(_walk(book.nodes))
    chaps = [n.title for n in nodes if _CHAPTER_TITLE.match(n.title)]
    return chaps or [n.title for n in book.nodes]


# ── 3-5. Map → reduce → assemblage ──────────────────────────────────────────
_MAP_SYSTEM = (
    "You summarize ONE section of a technical book for a student. Use ONLY the TEXT below. "
    "Write 3 to 5 bullet points, one sentence each, starting with '- '. Keep key terms, names "
    "and numbers exactly as written. No introduction, no conclusion, nothing that is not in the TEXT."
)
_REDUCE_SYSTEM = (
    "You write the overview of a book chapter for a student, using ONLY the section summaries "
    "given. Answer in exactly this format:\n"
    "OVERVIEW: <3 to 4 sentences: what the chapter covers and why it matters>\n"
    "TAKEAWAYS:\n- <takeaway>\n- <takeaway>\n- <takeaway>\n"
    "Do not add facts that are not in the summaries."
)
_WINDOW, _MAX_WINDOWS, _MAX_UNITS = 3500, 2, 10


def _units(node: Node) -> list[Node]:
    """Sections à résumer séparément : les enfants du nœud (fusionnés s'ils sont trop
    nombreux), sinon le nœud lui-même découpé en tranches de pages."""
    kids = node.children
    if not kids:
        return [node]
    units = list(kids)
    if units[0].start > node.start:  # introduction avant la 1re section
        units.insert(0, Node(node.level + 1, "Introduction", node.start, units[0].start - 1))
    while len(units) > _MAX_UNITS:  # fusionne les deux plus petites voisines
        i = min(range(len(units) - 1), key=lambda k: (units[k].end - units[k].start) + (units[k + 1].end - units[k + 1].start))
        a, b = units[i], units.pop(i + 1)
        units[i] = Node(a.level, f"{a.title} · {b.title}", a.start, b.end)
    return units


def _text(book: Book, node: Node) -> str:
    """Texte d'une section : ses pages (PDF) ou ses lignes (Markdown/texte)."""
    return "\n".join(book.pages[node.start - 1: node.end])


def _pages(node: Node, book: Book) -> str:
    if book.kind != "pdf":
        return ""
    return f"p. {node.start}" if node.end <= node.start else f"pp. {node.start}–{node.end}"


def _bullets(text: str) -> list[str]:
    out = []
    for ln in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)", ln)
        item = m.group(1).strip() if m else ""
        if m and len(item.split()) >= 3 and not re.sub(r"[*_\s]", "", item).endswith(":"):
            out.append(item)  # (une puce qui n'est qu'un intitulé « X: » est ignorée)
    return out[:5]


def summarize_target(book: Book, node: Node, llm, lang_final: str, progress=None, fix=None) -> dict:
    """Map → reduce. Renvoie {sections:[(node, bullets, excerpt)], overview, takeaways}.
    `fix(lines) -> lines` (facultatif) remet dans la langue demandée ce que le petit modèle
    aurait écrit dans la langue des sources."""
    fix = fix or (lambda lines: lines)
    units = _units(node)
    sections = []
    for i, u in enumerate(units, 1):
        if progress:
            progress(f"Reading {u.title[:48]}", i, len(units))
        text = _text(book, u).strip()
        if len(text) < 80:
            continue
        windows = [text[k:k + _WINDOW] for k in range(0, len(text), _WINDOW)][:_MAX_WINDOWS]
        bullets: list[str] = []
        for w in windows:
            raw = llm.generate(_MAP_SYSTEM, f"SECTION: {u.title}\nTEXT:\n{w}\n\n{lang_final}",
                               max_new_tokens=220, repetition_penalty=1.08)
            bullets += _bullets(raw)
        if bullets:
            sections.append((u, fix(bullets[:6]), text[:600]))
    if progress:
        progress("Connecting the sections", len(units), len(units))
    digest = "\n".join(f"[{u.title}]\n" + "\n".join(f"- {b}" for b in bl) for u, bl, _ in sections)
    raw = llm.generate(_REDUCE_SYSTEM, f"CHAPTER: {node.title}\nSECTION SUMMARIES:\n{digest[:6000]}\n\n{lang_final}",
                       max_new_tokens=320, repetition_penalty=1.08) if sections else ""
    m = re.search(r"OVERVIEW\s*:\s*(.+?)(?:\**\s*TAKEAWAYS\s*\**\s*:|$)", raw, re.S | re.I)
    overview = (m.group(1).strip() if m else raw.split("\n")[0].strip()) if raw else ""
    # le modèle recopie parfois le titre du chapitre en tête de la vue d'ensemble
    overview = re.sub(r"^\s*" + re.escape(node.title) + r"\s*[\n:.—-]*", "", overview).strip()
    overview = re.sub(r"\s+", " ", overview)
    t = re.search(r"TAKEAWAYS\s*:\s*(.+)$", raw, re.S | re.I)
    takeaways = _bullets(t.group(1)) if t else []
    if overview:
        overview = fix([overview])[0]
    takeaways = fix(takeaways) if takeaways else takeaways
    return {"sections": sections, "overview": overview, "takeaways": takeaways}


_LABELS = {
    "en": ("Overview", "Key takeaways", "Section by section"),
    "es": ("Visión general", "Ideas clave", "Sección por sección"),
    "fr": ("Vue d'ensemble", "À retenir", "Section par section"),
    "de": ("Überblick", "Kernaussagen", "Abschnitt für Abschnitt"),
}


def assemble(book: Book, node: Node, result: dict, lang: str) -> str:
    """Mise en forme déterministe : la structure ne dépend pas du modèle."""
    ov, kt, sbs = _LABELS.get(lang, _LABELS["en"])
    head = f"## {node.title}\n\n*{book.name}" + (f" · {_pages(node, book)}" if _pages(node, book) else "") + "*"
    out = [head]
    if result["overview"]:
        out.append(f"**{ov}.** {result['overview']}")
    out.append(f"### {sbs}")
    for n, (u, bullets, _) in enumerate(result["sections"], 1):
        pages = f" · {_pages(u, book)}" if _pages(u, book) else ""
        out.append(f"#### {u.title}{pages} [{n}]\n" + "\n".join(f"- {b}" for b in bullets))
    if result["takeaways"]:
        out.append(f"### {kt}\n" + "\n".join(f"- {t}" for t in result["takeaways"]))
    return "\n\n".join(out)


def _key_term(bullets: list[str]) -> str | None:
    """Notion phare d'une section : son premier terme en gras ou intitulé « X : … »."""
    for b in bullets:
        m = re.search(r"\*\*([^*]{2,40})\*\*", b) or re.match(r"^([^:]{3,32})\s:", b)
        if m:
            return m.group(1).strip(" :")
    return None


def structure_diagram(node: Node, result: dict, book: Book | None = None) -> str | None:
    """Figure du mode étude, construite à partir de la STRUCTURE réelle du document
    (chapitre → sections → notion phare de chaque section) : rien n'est deviné."""
    from .diagrams import _content_tokens, is_grounded, sanitize_mermaid

    secs = result["sections"]
    if len(secs) < 2:
        return None
    q = lambda t: t.replace('"', "'").replace(";", ",")[:60]
    lines = ["flowchart TD", f'  root["{q(node.title)}"]']
    for i, (u, bullets, _) in enumerate(secs, 1):
        lines.append(f'  root --> s{i}["{q(u.title)}"]')
        term = _key_term(bullets)
        # la notion doit figurer dans les pages de la section (sinon : mot du modèle, pas du livre)
        src = _content_tokens(_text(book, u)) if book else []
        if term and term.lower() not in u.title.lower() and (not book or is_grounded(term, set(src))):
            lines.append(f'  s{i} -.-> k{i}("{q(term)}")')
    return sanitize_mermaid("\n".join(lines))


_NOT_FOUND = {
    "en": "I couldn't find that part in *{doc}*. Its chapters are:",
    "es": "No encontré esa parte en *{doc}*. Sus capítulos son:",
    "fr": "Je n'ai pas trouvé cette partie dans *{doc}*. Ses chapitres sont :",
    "de": "Diesen Teil habe ich in *{doc}* nicht gefunden. Die Kapitel sind:",
}
_NO_DOC = {
    "en": "Which document do you mean? Name it in the question (e.g. “summarize chapter 2 of Inference Engineering”). Documents with chapters:",
    "es": "¿De qué documento hablas? Nómbralo en la pregunta (p. ej. «resume el capítulo 2 de Inference Engineering»). Documentos con capítulos:",
    "fr": "De quel document parles-tu ? Cite-le dans la question (ex. « résume le chapitre 2 de Inference Engineering »). Documents avec chapitres :",
    "de": "Welches Dokument meinst du? Nenne es in der Frage (z. B. „fasse Kapitel 2 von Inference Engineering zusammen“). Dokumente mit Kapiteln:",
}
