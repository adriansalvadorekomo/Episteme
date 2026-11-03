"""Explications visuelles : détection d'intention, consigne Mermaid, nettoyage, réparation.

Le petit LLM sait écrire du Mermaid, mais seulement si on le lui demande explicitement,
et avec des défauts prévisibles (lignes `style … label="…"` invalides, couleurs
aléatoires, nœuds dupliqués). Ce module :
  1. détecte quand l'utilisateur veut une explication visuelle (EN/ES/FR/DE) ;
  2. ajoute une consigne courte + un exemple en fin de prompt ;
  3. nettoie le code produit (en-tête, labels, lignes de style, doublons) ;
  4. sait demander au LLM de réparer un diagramme qui ne s'analyse pas.

Types pris en charge : `flowchart` (architectures) et `sequenceDiagram` (flux de requêtes).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# 1. Intention visuelle
# --------------------------------------------------------------------------
_VISUAL = re.compile(
    r"\b("
    # EN
    r"visual(ly|ize|ise|ization|isation)?|diagram(s|med)?|draw|sketch|illustrat(e|ion)|"
    r"flow\s?chart|sequence\s+diagram|map\s+out|"
    # ES
    r"visualmente|visualiza(r|lo|la)?|diagrama(s)?|esquema(s|tiza(r)?)?|dibuja(r|lo|la)?|"
    # FR
    r"visuel(le)?(lement)?|sch[ée]ma(s|tise(r)?)?|diagramme(s)?|dessine(r)?|illustre(r)?|"
    # DE
    r"visuell|diagramm(e)?|schaubild|zeichne(n)?|skizziere(n)?|veranschauliche(n)?"
    r")\b",
    re.IGNORECASE,
)
# faux positifs : produits ou sens non visuels
_NOT_VISUAL = re.compile(r"\bvisual\s+(studio|basic|regression|c\+\+)\b", re.IGNORECASE)

_SEQUENCE = re.compile(
    r"\b(sequence|handshake|log\s?-?in|sign\s?-?in|authenticat\w*|oauth\w*|jwt|"
    r"request\s+(flow|lifecycle|life\s?cycle)|step[\s-]by[\s-]step|interaction|round\s?trip|"
    r"secuencia|paso\s+a\s+paso|flujo\s+de\s+(la\s+)?petici[oó]n|"
    r"s[ée]quence|[ée]tape\s+par\s+[ée]tape|cycle\s+de\s+vie\s+d'une\s+requ[eê]te|"
    r"sequenz|ablauf|schritt\s+f[uü]r\s+schritt)\b",
    re.IGNORECASE,
)


def wants_visual(text: str) -> bool:
    """La question demande-t-elle une explication visuelle ?"""
    t = text or ""
    return bool(_VISUAL.search(t)) and not _NOT_VISUAL.search(t)


def diagram_kind(text: str) -> str:
    """'sequence' pour un flux d'interactions (login, OAuth, requête…), sinon 'flowchart'."""
    return "sequence" if _SEQUENCE.search(text or "") else "flowchart"


# --------------------------------------------------------------------------
# 2. Consigne ajoutée au prompt (en fin, pour la récence)
# --------------------------------------------------------------------------
_EXAMPLE = {
    "flowchart": (
        "```mermaid\nflowchart LR\n"
        "  subgraph Clients\n"
        '    web["Web App"]\n'
        "  end\n"
        "  subgraph Backend\n"
        '    gw["API Gateway"]\n'
        '    svc["Orders Service"]\n'
        "  end\n"
        "  subgraph Data\n"
        '    db[("Orders DB")]\n'
        "  end\n"
        "  web -->|HTTPS| gw\n"
        "  gw -->|REST| svc\n"
        "  svc -->|SQL| db\n```"
    ),
    "sequence": (
        "```mermaid\nsequenceDiagram\n"
        "  participant C as Client\n  participant A as API\n"
        "  C->>A: POST /login\n  A-->>C: 200 + token\n```"
    ),
}


_RULES = {
    "flowchart": (
        "Diagram rules: use the REAL components of the topic (not the example's); 5 to 12 "
        "nodes; group them in 2-4 subgraphs by layer (clients, services, data…); label each "
        "arrow with its protocol or action (HTTPS, gRPC, SQL, publish, deploy…); short ids; "
        "every node label in double quotes; one arrow per line; NO style, classDef, linkStyle, "
        "click or comment lines; no colors."
    ),
    "sequence": (
        "Diagram rules: use the REAL participants of the topic (not the example's); 3 to 6 "
        "participants; 4 to 12 messages, each labeled with the concrete action (POST /token, "
        "302 redirect, verify signature…); `->>` for requests, `-->>` for responses; optionally "
        "one `Note over X: …`; never repeat a message; NO style or comment lines."
    ),
}


def visual_directive(kind: str = "flowchart", grounded: bool = False) -> str:
    kind = kind if kind in _EXAMPLE else "flowchart"
    header = "flowchart LR" if kind == "flowchart" else "sequenceDiagram"
    rules = (
        f"VISUAL: the user wants a visual explanation. Include exactly ONE diagram in a "
        f"```mermaid code block using `{header}`, then explain it briefly. Example of the format:\n"
        f"{_EXAMPLE[kind]}\n" + _RULES[kind]
    )
    if grounded:
        rules += (" Only use components that appear in the CONTEXT; put citations [n] in the "
                  "text, never inside the diagram.")
    return rules


# --------------------------------------------------------------------------
# 3. Nettoyage du code Mermaid
# --------------------------------------------------------------------------
_HEADER = re.compile(r"^\s*(graph|flowchart)\b\s*(TD|TB|BT|LR|RL)?\s*;?\s*$", re.IGNORECASE)
_DROP = re.compile(r"^\s*(style|classDef|class|linkStyle|click)\b|^\s*%%|^\s*//", re.IGNORECASE)
# nœud : id + forme (délimiteurs les plus longs d'abord) + label
_NODE = re.compile(
    r"(?P<id>\b[A-Za-z_][\w-]*)\s*"
    r"(?P<open>\[\(|\(\(|\[\[|\{\{|\(\[|\[|\(|\{)"
    r"(?P<label>[^\[\](){}\n]*?|\"[^\"\n]*\")"
    r"(?P<close>\)\]|\)\)|\]\]|\}\}|\]\)|\]|\)|\})"
)
_MAX_SEQ_MESSAGES = 14
_PAIRS = {"[(": ")]", "((": "))", "[[": "]]", "{{": "}}", "([": "])", "[": "]", "(": ")", "{": "}"}
_QUOTED = re.compile(r'"[^"\n]*"')


_EDGE_LABEL = re.compile(r"(\|[^|\n]*\|)")


def _quote_labels(line: str) -> str:
    """Met les labels de nœuds ET d'arêtes entre guillemets. Les libellés d'arête `|…|`
    sont protégés : `|HTTP(S)|` ne doit jamais être pris pour un nœud `HTTP(S)`."""
    def fix(m: re.Match) -> str:
        op, cl, label = m["open"], m["close"], m["label"].strip()
        if _PAIRS.get(op) != cl:
            return m.group(0)
        if not (label.startswith('"') and label.endswith('"')):
            label = '"' + label.replace('"', "'") + '"'
        return f"{m['id']}{op}{label}{cl}"

    parts = _EDGE_LABEL.split(line)
    for k, seg in enumerate(parts):
        if k % 2:  # libellé d'arête
            inner = seg[1:-1].strip()
            if inner and not (inner.startswith('"') and inner.endswith('"')):
                inner = '"' + inner.replace('"', "'") + '"'
            parts[k] = f"|{inner}|"
        else:
            parts[k] = _NODE.sub(fix, seg)
    return "".join(parts)


def _outside_quotes(line: str, fn) -> str:
    """Applique fn aux segments hors guillemets (ne touche jamais au texte des labels)."""
    out, last = [], 0
    for q in _QUOTED.finditer(line):
        out.append(fn(line[last:q.start()]))
        out.append(q.group(0))
        last = q.end()
    out.append(fn(line[last:]))
    return "".join(out)


def _merge_duplicates(lines: list[str]) -> list[str]:
    """Fusionne les nœuds de même label (le petit modèle répète « Client Response »…)."""
    canon: dict[str, str] = {}   # label normalisé -> premier id
    alias: dict[str, str] = {}   # id dupliqué -> id canonique
    for ln in lines:
        for m in _NODE.finditer(ln):
            nid, label = m["id"], m["label"].strip().strip('"').strip().lower()
            if not label or nid in alias:
                continue
            if label in canon and canon[label] != nid:
                alias[nid] = canon[label]
            else:
                canon.setdefault(label, nid)
    if not alias:
        return lines

    def rename(seg: str) -> str:
        for a, c in alias.items():
            seg = re.sub(rf"\b{re.escape(a)}\b", c, seg)
        return seg

    out, seen = [], set()
    for ln in lines:
        # définition d'un alias → simple référence à l'id canonique
        ln = _NODE.sub(lambda m: alias[m["id"]] if m["id"] in alias else m.group(0), ln)
        ln = _outside_quotes(ln, rename)
        # clé d'arête sans les définitions de labels : `G --> J["x"]` ≡ `G --> J`
        key = re.sub(r"\s+", " ", _NODE.sub(lambda m: m["id"], ln).strip())
        if key in seen and "-->" in key:
            continue  # arête devenue identique après fusion
        seen.add(key)
        out.append(ln)
    return out



# ── réparations déterministes des habitudes du petit modèle ──────────────────
_NOTE_POS = re.compile(r"^(\s*Note)\s+(below|above|under|on\s+top\s+of)\s+", re.IGNORECASE)
_ARROW = r"(?:-->|-\.->|==>|---|-\.-)"
_EDGE = re.compile(rf"^(?P<pre>\s*.+?)\s*(?P<arrow>{_ARROW})\s*(?P<post>.+?)\s*$")
_BARE_STRING = re.compile(r'^\s*"[^"\n]*"\s*$')
_NARRATION = re.compile(rf'{_ARROW}\s*"[^"\n]*"\s*$')  # « a --> "phrase" » : pas une arête


_KEYWORD = re.compile(r"^\s*(subgraph|end|direction|class|classDef|style|click|linkStyle)\b", re.IGNORECASE)
_LINK_TOKEN = re.compile(r"--|==|-\.|~~~|&")
_LONE_ID = re.compile(r"^\s*[A-Za-z_][\w-]*\s*$")


def _is_statement(line: str) -> bool:
    """Vrai pour une instruction Mermaid ; faux pour une phrase d'explication."""
    if _KEYWORD.match(line) or _LONE_ID.match(line):
        return True
    if _LINK_TOKEN.search(line):
        return True
    m = _NODE.match(line.strip())  # définition de nœud seule : `db[("Orders DB")]`
    return bool(m and m.end() == len(line.strip()))


_MAX_EDGES = 30


def _normalize_syntax(line: str) -> str:
    """Mots-clés et flèches empruntés à d'autres types de diagrammes → flowchart."""
    line = re.sub(r"^(\s*)subgraph\b", r"\1subgraph", line, flags=re.IGNORECASE)
    line = re.sub(r"^(\s*)end\s*$", r"\1end", line, flags=re.IGNORECASE)
    line = line.replace("--|>", "-->").replace("<|--", "---").replace("..>", "-.->")
    return re.sub(r"(?<![-=.<])->(?![>])", "-->", line)  # `A->B` → `A-->B`


def _drop_loops(lines: list[str]) -> list[str]:
    """Le modèle peut boucler (`db --> db` × 90) : arêtes identiques dédoublonnées,
    auto-boucles supprimées, nombre d'arêtes plafonné (les `end` sont conservés)."""
    out, seen, edges = [], set(), 0
    for ln in lines:
        if _LINK_TOKEN.search(ln) and not _KEYWORD.match(ln):
            key = re.sub(r"\s+", " ", _NODE.sub(lambda m: m["id"], ln).strip())
            ends = re.findall(r"\b[A-Za-z_][\w-]*\b", _EDGE_LABEL.sub(" ", _QUOTED.sub(" ", key)))
            if key in seen or (len(ends) == 2 and ends[0] == ends[1]):
                continue
            seen.add(key)
            edges += 1
            if edges > _MAX_EDGES:
                continue
        out.append(ln)
    return out


_MAX_LR_CHAIN = 5


def _auto_direction(direction: str, lines: list[str]) -> str:
    """Une longue chaîne de gauche à droite devient illisible une fois réduite à la largeur
    du chat : au-delà de 5 étapes on passe en haut → bas."""
    if direction not in ("LR", "RL"):
        return direction
    graph: dict[str, set[str]] = {}
    for ln in lines:
        if _KEYWORD.match(ln) or not _LINK_TOKEN.search(ln):
            continue
        flat = _EDGE_LABEL.sub(" ", _QUOTED.sub(" ", _NODE.sub(lambda m: m["id"], ln)))
        ids = re.findall(r"\b[A-Za-z_][\w-]*\b", flat)
        for a, b in zip(ids, ids[1:]):
            graph.setdefault(a, set()).add(b)

    memo: dict[str, int] = {}

    def depth(n: str, stack: frozenset) -> int:
        if n in memo:
            return memo[n]
        d = 1 + max((depth(m, stack | {n}) for m in graph.get(n, ()) if m not in stack), default=0)
        memo[n] = d
        return d

    longest = max((depth(n, frozenset()) for n in graph), default=0)
    return "TD" if longest > _MAX_LR_CHAIN else direction


def _fold_edge_labels(lines: list[str]) -> list[str]:
    """`a --> b` suivi d'une ligne `"texte"` ⇒ `a -->|"texte"| b` ; arêtes vers une phrase
    entre guillemets (narration) supprimées."""
    out: list[str] = []
    for ln in lines:
        if _NARRATION.search(ln):
            continue
        if _BARE_STRING.match(ln):
            prev = out[-1] if out else ""
            m = _EDGE.match(prev)
            if m and "|" not in prev:
                out[-1] = f"{m['pre']} {m['arrow']}|{ln.strip()}| {m['post']}"
            continue  # libellé orphelin : ignoré
        out.append(ln)
    return out


# Formes sémantiques : le design ne dépend plus du modèle. Une base de données devient un
# cylindre, une passerelle un hexagone, une file un « sous-programme », un client une
# pilule — et chaque rôle reçoit une classe CSS stylée côté front (role-store, role-edge…).
_ROLES = [
    ("store", re.compile(r"\b(db|dbs|database(s)?|datastore|data\s*store|storage|s3|bucket|postgres\w*|mysql|"
                         r"mongo\w*|redis|cache|warehouse|lake(house)?|elasticsearch|vector\s*(db|store)|bd|"
                         r"base\s+de\s+(datos|donn[ée]es)|datenbank|speicher|registry)\b", re.I), "[(", ")]"),
    ("queue", re.compile(r"\b(queue(s)?|kafka|rabbitmq|topic(s)?|broker|event\s*bus|bus|pub/?sub|sqs|"
                         r"cola|file\s+d'attente|warteschlange)\b", re.I), "[[", "]]"),
    ("edge", re.compile(r"\b(gateway|load\s*balancer|balancer|proxy|ingress|router|cdn|nginx|firewall|"
                        r"passerelle|equilibrador|r[ée]partiteur)\b", re.I), "{{", "}}"),
    ("service", re.compile(r"\b(service(s)?|microservice(s)?|server|api|worker(s)?|servicio|dienst|app)\b", re.I),
     None, None),
    ("actor", re.compile(r"\b(user(s)?|client(s)?|browser|customer(s)?|mobile|front-?end|usuario(s)?|"
                         r"utilisateur(s)?|cliente(s)?|nutzer|benutzer|developer(s)?|d[ée]veloppeur(s)?)\b", re.I),
     "([", "])"),
]


def _shape_by_role(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    roles: dict[str, list[str]] = {r[0]: [] for r in _ROLES}
    assigned: set[str] = set()

    def fix(m: re.Match) -> str:
        nid, op, cl, label = m["id"], m["open"], m["close"], m["label"]
        if _PAIRS.get(op) != cl or nid in assigned:
            return m.group(0)
        for role, rx, o, c in _ROLES:
            if rx.search(label.strip('"')):
                assigned.add(nid)
                roles[role].append(nid)
                if o and op in ("[", "("):  # forme par défaut → forme sémantique
                    return f"{nid}{o}{label}{c}"
                return m.group(0)
        return m.group(0)

    out = [ln if ln.lstrip().lower().startswith("subgraph") else _NODE.sub(fix, ln) for ln in lines]
    return out, roles


def sanitize_mermaid(code: str) -> str | None:
    """Rend le code du modèle analysable et conforme au thème. None si non pris en charge."""
    code = re.sub(r"^```\w*\s*|```\s*$", "", (code or "").strip()).strip()
    lines = [ln.rstrip() for ln in code.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0].strip()

    if first.lower().startswith("sequencediagram"):
        # le petit modèle peut boucler (mêmes messages répétés) : on garde la 1re occurrence
        # de chaque ligne et on plafonne le nombre de messages.
        body, seen, msgs = [], set(), 0
        for ln in lines[1:]:
            if _DROP.match(ln):
                continue
            if ln.count('"') % 2:
                continue  # ligne tronquée
            ln = _NOTE_POS.sub(r"\1 over ", ln)  # « Note below X » n'existe pas : → « Note over X »
            key = re.sub(r"\s+", " ", ln.strip()).lower()
            if key in seen:
                continue
            seen.add(key)
            if "->" in ln or "-x" in ln or "-)" in ln:
                msgs += 1
                if msgs > _MAX_SEQ_MESSAGES:
                    continue
            body.append(ln)
        return "\n".join(["sequenceDiagram", *body]) if body else None

    m = _HEADER.match(first)
    if m:
        direction = (m.group(2) or "TD").upper()
        body = lines[1:]
    elif any("-->" in ln or "---" in ln for ln in lines):
        direction, body = "TD", lines  # en-tête oublié
    else:
        return None

    body = [_normalize_syntax(ln.rstrip().rstrip(";")) for ln in body if not _DROP.match(ln)]
    # ligne tronquée (budget épuisé) : guillemets ou barres de libellé non refermés
    body = [ln for ln in body if ln.count('"') % 2 == 0 and ln.count("|") % 2 == 0]
    body = _fold_edge_labels(body)
    body = [ln for ln in body if _is_statement(ln)]       # prose glissée dans le diagramme
    body = _drop_loops(body)
    body = [_quote_labels(ln) for ln in body]
    body = _merge_duplicates(body)
    body = [ln for ln in body if ln.strip()]
    if not body:
        return None
    direction = _auto_direction(direction, body)
    body, roles = _shape_by_role(body)
    classes = [f"  class {','.join(ids)} role-{role}" for role, ids in roles.items() if ids]
    return "\n".join([f"flowchart {direction}", *body, *classes])


_FENCE = re.compile(r"```[ \t]*(?P<lang>[\w-]*)[ \t]*\n(?P<body>.*?)```", re.DOTALL)
_BARE = re.compile(
    r"^(?:graph|flowchart)[ \t]+(?:TD|TB|BT|LR|RL)\b.*?(?=\n[ \t]*\n|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_DIAGRAM_START = re.compile(r"^\s*(graph|flowchart|sequenceDiagram)\b", re.IGNORECASE)


def normalize_answer(answer: str) -> str:
    """Garde un seul diagramme Mermaid, nettoyé, dans un bloc ```mermaid."""
    if not answer:
        return answer
    if answer.count("```") % 2 == 1:
        answer = answer.rstrip() + "\n```"  # bloc jamais refermé (budget de tokens épuisé)
    out, pos, kept = [], 0, False
    for m in _FENCE.finditer(answer):
        lang, body = m["lang"].lower(), m["body"]
        is_diagram = lang == "mermaid" or (lang in ("", "graph", "flowchart") and _DIAGRAM_START.match(body))
        if not is_diagram:
            continue
        out.append(answer[pos:m.start()])
        if not kept:
            clean = sanitize_mermaid(body)
            if clean:
                out.append(f"```mermaid\n{clean}\n```")
                kept = True
            else:
                out.append(m.group(0))  # non pris en charge : laissé tel quel (bloc de code)
        pos = m.end()  # diagrammes supplémentaires supprimés
    out.append(answer[pos:])
    result = "".join(out)

    if not kept:  # diagramme écrit sans clôture ```
        bare = _BARE.search(result)
        if bare:
            clean = sanitize_mermaid(bare.group(0))
            if clean:
                result = result[:bare.start()] + f"```mermaid\n{clean}\n```" + result[bare.end():]
    return result


# --------------------------------------------------------------------------
# 4. Réparation par le LLM (un seul essai, appelé quand le navigateur échoue à l'analyse)
# --------------------------------------------------------------------------
_REPAIR_SYSTEM = (
    "You fix Mermaid diagram syntax. Reply with ONLY the corrected Mermaid code: no code "
    "fences, no explanations. Keep the same meaning. Use `flowchart LR` or `sequenceDiagram`, "
    "put every node label in double quotes, and never write style, classDef, linkStyle or "
    "click lines."
)


def repair_mermaid(code: str, error: str, llm) -> str | None:
    user = f"Mermaid parse error:\n{(error or 'unknown error')[:800]}\n\nCode to fix:\n{code[:4000]}"
    fixed = llm.generate(_REPAIR_SYSTEM, user, max_new_tokens=400)
    m = _FENCE.search(fixed)
    return sanitize_mermaid(m["body"] if m else fixed)
