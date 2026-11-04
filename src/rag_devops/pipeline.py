"""Orchestration de l'assistant : deux modes qui sont la leçon MLOps.

- MODE LLM : intuition. Le modèle explique avec ses seules connaissances,
  sans récupération ni source (réponse « non ancrée »).
- MODE RAG : preuve. Même modèle, mais la réponse est fondée sur le corpus et
  porte des citations [n] renvoyant aux extraits du cours.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import Settings, settings
from .diagrams import attach_diagram, diagram_kind, extract_diagram, normalize_answer, visual_hint, wants_visual
from .graph import KnowledgeGraph
from .index import SearchHit, VectorStore
from .llm import LLM
from .study import wants_study

_ANALOGY_LABEL = {
    "en": "In plain words — an analogy, not from the source:",
    "es": "En palabras sencillas — una analogía, no viene de la fuente:",
    "fr": "En mots simples — une analogie, pas issue de la source :",
    "de": "Einfach gesagt — eine Analogie, nicht aus der Quelle:",
}

_CITE_RE = re.compile(r"\[\d+\]")  # marqueurs de citation [n]

# Consigne de format : réponses toujours en Markdown bien structuré.
_MD_RULE = (
    "FORMAT : rédige TOUJOURS ta réponse en **Markdown** bien structuré — "
    "titres courts (`##`/`###`), **gras** pour les termes clés, listes à puces "
    "ou numérotées, `code` en ligne pour les commandes/identifiants, et blocs "
    "``` ``` ``` pour les extraits de code. Aère le texte en paragraphes."
)

# Mots d'amorce (début de phrase, non des noms propres) — EN/ES/FR/DE.
_STARTERS = {
    "what", "how", "why", "when", "which", "who", "where", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "would", "should", "explain",
    "give", "tell", "define", "describe", "list", "a", "an", "the", "this",
    "that", "these", "those", "in", "on", "of", "to", "and", "or", "with",
    "for", "i", "you", "it", "please",
    "qué", "cómo", "cuál", "cuáles", "cuándo", "quién", "dónde", "es", "son",
    "era", "para", "por", "la", "el", "los", "las", "un", "una", "y", "o", "de",
    "en", "qu", "est", "quoi", "comment", "quelle", "quel", "quelles", "quels",
    "pourquoi", "le", "les", "une", "des", "et", "ce", "c", "was", "wie",
    "warum", "welche", "welcher", "wann", "wer", "wo", "ist", "sind", "ein",
    "eine", "der", "die", "das", "und", "oder", "mit", "für", "von", "zu",
    # verbes de requête / impératifs (souvent capitalisés en début de phrase) — EN/ES/FR/DE
    "show", "draw", "sketch", "illustrate", "visualize", "visualise", "help", "compare",
    "walk", "me", "us",
    "explica", "explícame", "explicame", "explicar", "muestra", "muéstrame", "muestrame",
    "dibuja", "dibújame", "dame", "hazme", "haz", "dime", "compara", "enséñame", "ensename",
    "puedes", "podrías", "podrias", "cómo",
    "explique", "expliquez", "explique-moi", "montre", "montrez", "dessine", "décris",
    "décrivez", "donne", "donnez", "peux", "pourrais", "moi",
    "erkläre", "erklär", "erklären", "zeig", "zeige", "zeigen", "zeichne", "beschreibe",
    "gib", "vergleiche", "kannst", "könntest", "mir", "bitte",
}
_TOK_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9]+")


def _proper_tokens(question: str) -> set[str]:
    """Noms propres / termes techniques d'une question (Istio, Terraform, mTLS, K8s…).

    Détecte les tokens capitalisés (hors mots d'amorce) ou à casse interne/chiffre.
    Ces tokens sont invariants d'une langue à l'autre → base d'un garde-fou
    d'ancrage robuste au multilingue.
    """
    out: set[str] = set()
    # En allemand TOUS les noms prennent une majuscule (« Architektur ») : une simple
    # majuscule initiale n'y signale pas une techno. On n'y retient que les formes sans
    # ambiguïté : casse interne (MLOps, mTLS), chiffres (K8s), sigles (AWS, CI).
    german = detect_lang(question) == "de"
    for t in _TOK_RE.findall(question):
        low = t.lower()
        if low in _STARTERS or len(low) < 2 or wants_visual(t):
            continue  # mot d'amorce, verbe de requête ou mot « visuel » (Diagramm, Schéma…)
        inner_upper = any(c.isupper() for c in t[1:])
        has_digit = any(c.isdigit() for c in t)
        if inner_upper or has_digit or (t[0].isupper() and not german):
            out.add(low)
    return out

# Réponse honnête quand l'assistant ne sait pas (corpus insuffisant), par langue.
_IDK = {
    "en": "I don't know. I couldn't find a reliable answer to this in the course corpus, and I won't guess.",
    "es": "No lo sé. No encontré una respuesta fiable a esto en el corpus del curso y prefiero no inventar.",
    "fr": "Je ne sais pas. Je n'ai pas trouvé de réponse fiable à cela dans le corpus du cours, et je préfère ne pas inventer.",
    "de": "Ich weiß es nicht. Ich habe dazu keine verlässliche Antwort im Kurskorpus gefunden und rate nicht.",
}

# Langues prises en charge : réponds TOUJOURS dans la langue de la question.
_LANG_RULE = (
    "LANGUE : réponds IMPÉRATIVEMENT dans la langue de la question, parmi "
    "l'anglais, l'espagnol, le français ou l'allemand. Si la question est dans "
    "une autre langue, réponds en anglais. N'explique pas ton choix de langue."
)

# Ancres « hors-sujet » (le pendant des domaines in-scope). La décision est
# contrastive, donc robuste d'une langue à l'autre.
_OFF_ANCHORS = [
    "cooking recipes and food", "politics, elections and government",
    "sports and football results", "geography, countries and capitals",
    "weather forecast", "poetry, love and feelings",
    "health, medicine and fitness", "history and famous people",
    "music, movies and entertainment", "travel and tourism",
]
# Phrases « in-domaine » génériques, ajoutées aux domaines de la config pour
# donner des ancres descriptives (meilleurs embeddings que les simples noms).
_IN_ANCHOR_EXTRA = [
    "software engineering and programming",
    "CI/CD pipelines, testing and automation",
    "cloud infrastructure, containers and orchestration",
    "data and machine learning pipelines",
]

# Détection de langue (EN/ES/FR/DE) — légère, sans dépendance.
_LANG_CHARS = {"es": "¿¡ñ", "de": "ßäöü", "fr": "çàâêîôûëïœù"}
_LANG_STOP = {
    "en": {"the", "is", "what", "how", "does", "a", "of", "to", "and", "why",
           "which", "with", "for", "in", "on", "are", "do", "can", "explain"},
    "es": {"que", "qué", "es", "cómo", "como", "cuál", "cual", "de", "la", "el",
           "los", "las", "un", "una", "para", "por", "y", "en", "qué"},
    "fr": {"que", "qu", "est", "quoi", "comment", "le", "la", "les", "un", "une",
           "de", "des", "pour", "et", "en", "ce", "c", "quelle", "quel"},
    "de": {"was", "ist", "wie", "der", "die", "das", "und", "ein", "eine", "für",
           "mit", "von", "zu", "warum", "welche", "sind", "bei"},
}
_LANG_DIRECTIVE = {
    "en": "Answer entirely in English.",
    "es": "Responde ÍNTEGRAMENTE en español.",
    "fr": "Réponds INTÉGRALEMENT en français.",
    "de": "Antworte VOLLSTÄNDIG auf Deutsch.",
}
# Version forte, placée en FIN de prompt (récence) : impose la langue même quand
# le contexte récupéré est dans une autre langue.
_LANG_FINAL = {
    "en": "IMPORTANT: Write your ENTIRE answer in English, even if the context above is in another language.",
    "es": "IMPORTANTE: Redacta TODA tu respuesta en español, aunque el contexto anterior esté en otro idioma.",
    "fr": "IMPORTANT : rédige TOUTE ta réponse en français, même si le contexte ci-dessus est dans une autre langue.",
    "de": "WICHTIG: Verfasse deine GESAMTE Antwort auf Deutsch, auch wenn der obige Kontext in einer anderen Sprache ist.",
}


def detect_lang(text: str) -> str:
    import re as _re

    t = text.lower()
    scores = {lang: 0.0 for lang in _LANG_STOP}
    for lang, chars in _LANG_CHARS.items():
        scores[lang] += 2.0 * sum(t.count(c) for c in chars)  # indices forts
    words = _re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ¿¡ß]+", t)
    for w in words:
        for lang, stop in _LANG_STOP.items():
            if w in stop:
                scores[lang] += 1.0
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"

# Refus de périmètre (hors ingénierie logicielle), dans les 4 langues.
_SCOPE_DECLINE = (
    "I'm **Episteme**, a software-engineering knowledge platform, and I only cover software engineering "
    "(DevOps, MLOps, DataOps, Data & AI Engineering…). Please rephrase your "
    "question around one of these topics.\n\n"
    "Je n'enseigne que l'ingénierie logicielle (DevOps, MLOps, DataOps, Data & "
    "AI Engineering…) — reformule ta question sur l'un de ces sujets.\n\n"
    "Solo enseño ingeniería de software (DevOps, MLOps, DataOps, Data & AI "
    "Engineering…); reformula tu pregunta sobre uno de estos temas.\n\n"
    "Ich unterrichte nur Software-Engineering (DevOps, MLOps, DataOps, Data & "
    "AI Engineering…); bitte formuliere deine Frage zu einem dieser Themen."
)


# --------------------------------------------------------------------------
# Persona partagée — un professeur de niveau doctorat, style Feynman.
# Le périmètre d'expertise vient de cfg.domains : ajouter un domaine élargit
# automatiquement le garde-fou anti-hors-sujet (scalabilité).
# --------------------------------------------------------------------------
def rag_system_prompt(cfg: Settings) -> str:
    return (
        "Tu es Episteme, une plateforme de connaissances pour l'ingénierie logicielle, "
        "qui enseigne au niveau doctorat. Devise : comprendre, relier, concevoir. "
        f"Domaines : {cfg.domains_str()}.\n\n"
        "RÈGLES IMPÉRATIVES (mode ANCRÉ) :\n"
        "1. Réponds UNIQUEMENT à partir du CONTEXTE fourni (extraits numérotés). "
        "N'ajoute jamais un fait, un chiffre, une commande ou un nom absent du contexte.\n"
        "2. Appuie CHAQUE affirmation sur une citation [n] renvoyant à l'extrait utilisé "
        "(ex. « Un Pod regroupe un ou plusieurs conteneurs [2]. »).\n"
        "3. Pédagogie Feynman : définition claire d'abord, puis explication par les "
        "principes, puis UN exemple concret (commande, extrait, scénario) tiré du contexte. "
        "N'écris PAS d'analogie dans le texte : une analogie clairement étiquetée est ajoutée "
        "séparément, le texte ne doit contenir que ce que disent les sources.\n"
        "4. Si le contexte ne contient pas la réponse, sois HONNÊTE : dis simplement "
        "« je ne sais pas » / « le corpus ne couvre pas ce point » — n'invente RIEN, "
        "ne cite aucun extrait inexistant.\n"
        "5. Si la question est hors du périmètre d'ingénierie logicielle ci-dessus, "
        "décline poliment en une phrase.\n\n"
        + _MD_RULE + "\n\n"
        + _LANG_RULE
        + " Sois rigoureux mais accessible."
    )


def llm_system_prompt(cfg: Settings) -> str:
    return (
        "Tu es Episteme, une plateforme de connaissances pour l'ingénierie logicielle, "
        "qui enseigne au niveau doctorat. Devise : comprendre, relier, concevoir. "
        f"Domaines : {cfg.domains_str()}.\n\n"
        "STYLE (à la Feynman) : explique à partir des premiers principes, avec une "
        "analogie simple et AU MOINS un exemple concret (commande, extrait de code ou "
        "scénario réel). La clarté prime sur le jargon ; définis les termes que tu "
        "introduis ; va à l'essentiel.\n\n"
        "PÉRIMÈTRE : tu n'enseignes QUE l'ingénierie logicielle et ses spécialités "
        "(les domaines ci-dessus : CI/CD, conteneurs, orchestration, cloud, pipelines "
        "de données et de ML, observabilité, etc.). Si la question sort de ce périmètre "
        "(cuisine, politique, géographie, santé…), décline poliment en une phrase et "
        "propose de la reformuler côté ingénierie. Ne réponds pas hors sujet.\n\n"
        "HONNÊTETÉ : c'est le mode « intuition », tu n'as PAS de corpus sous les yeux. "
        "Si tu n'es pas certain d'un fait précis (version, chiffre, syntaxe exacte), "
        "dis-le explicitement plutôt que de l'inventer. Si tu ignores la réponse, "
        "dis simplement et honnêtement « je ne sais pas » — n'invente JAMAIS de "
        "sources, de citations, de noms d'API ni de chiffres.\n\n"
        + _MD_RULE + "\n\n"
        + _LANG_RULE
    )


RAG_USER_TEMPLATE = (
    "CONTEXTE (extraits numérotés du support de cours) :\n{context}\n\n"
    "QUESTION DE L'ÉTUDIANT : {question}\n\n"
    "Réponds en t'appuyant exclusivement sur le contexte, avec des citations [n]."
)

# Valeurs par défaut (back-compat : importées par le notebook / l'ancien code).
RAG_SYSTEM_PROMPT = rag_system_prompt(settings)
LLM_SYSTEM_PROMPT = llm_system_prompt(settings)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    mode: str                       # "rag" | "llm"
    grounded: bool
    hits: list[SearchHit] = field(default_factory=list)
    method: str = "llm"             # "graph" | "vector" | "llm" | "scope" | "idk"
    metrics: dict = field(default_factory=dict)  # métriques d'inférence (tokens, tok/s, VRAM…)
    visual: bool = False            # réponse avec diagramme Mermaid
    verification: dict | None = None  # vérification phrase par phrase contre les sources

    @property
    def sources(self) -> list[dict]:
        """Sources numérotées [n], alignées sur les extraits du contexte."""
        out = []
        for i, h in enumerate(self.hits, 1):
            c = h.chunk
            out.append(
                {
                    "n": i,
                    "source": c["source"],
                    "category": c["category"],
                    "score": round(h.score, 4),
                    "text": c["text"],
                }
            )
        return out


def format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(f"[{i}] (source: {c['source']} · thème: {c['category']})\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        llm: LLM | None = None,
        cfg: Settings | None = None,
        graph: KnowledgeGraph | None = None,
    ):
        self.cfg = cfg or settings
        self.store = store
        self.llm = llm or LLM(self.cfg)
        # Graphe de connaissances : fourni, sinon chargé depuis le disque s'il existe.
        self.graph = graph
        if self.graph is None and self.cfg.use_graph:
            self.graph = KnowledgeGraph.load(self.cfg)
        self._rag_sys = rag_system_prompt(self.cfg)
        self._llm_sys = llm_system_prompt(self.cfg)
        self._in_vecs = None   # ancres in-domaine (lazy)
        self._off_vecs = None  # ancres hors-sujet (lazy)
        self._vocab = None     # vocabulaire du corpus (lazy)
        self.progress: dict | None = None  # étape en cours (mode étude), lue par /gpu

    @property
    def uses_graph(self) -> bool:
        return bool(self.cfg.use_graph and self.graph is not None)

    def _ensure_scope_vecs(self) -> None:
        if self._in_vecs is None:
            in_anchors = list(self.cfg.domains) + _IN_ANCHOR_EXTRA
            self._in_vecs = self.store.embedder.encode_passages(in_anchors, show_progress=False)
            self._off_vecs = self.store.embedder.encode_passages(_OFF_ANCHORS, show_progress=False)

    def in_scope(self, question: str) -> bool:
        """Garde-fou CONTRASTIF : la question relève-t-elle de l'ingénierie logicielle ?

        On compare la proximité de la question aux ancres « in-domaine » vs aux
        ancres « hors-sujet ». La pénalité cross-lingue s'annulant entre les deux
        côtés, le même seuil fonctionne en EN/ES/FR/DE. Évite que le modèle parte
        halluciner sur un sujet hors périmètre (cuisine, politique…).
        """
        if not self.cfg.scope_guard:
            return True
        self._ensure_scope_vecs()
        qv = self.store.embedder.encode_queries([question])[0]
        max_in = float((self._in_vecs @ qv).max())
        max_off = float((self._off_vecs @ qv).max())
        return (max_in - max_off) >= self.cfg.scope_margin

    def _decline(self, question: str, mode: str) -> RAGAnswer:
        return RAGAnswer(
            question=question, answer=_SCOPE_DECLINE, mode=mode, grounded=False,
            hits=[], method="scope",
        )

    @property
    def corpus_df(self) -> "dict[str, int]":
        """Fréquence documentaire (nb de chunks contenant le terme) de chaque token."""
        if self._vocab is None:
            from collections import Counter

            df: Counter = Counter()
            for c in self.store.chunks:
                for tok in set(_TOK_RE.findall(c["text"].lower())):
                    df[tok] += 1
            self._vocab = dict(df)
        return self._vocab

    def corpus_covers(self, question: str) -> bool:
        """Le corpus couvre-t-il VRAIMENT la/les techno(s) nommée(s) dans la question ?

        Chaque terme technique (Istio, Terraform, mTLS, Kafka…) doit apparaître
        dans assez de chunks (df ≥ rag_min_df). On prend le plus rare : si la techno
        la plus spécifique n'est citée qu'au passage (ou absente), le corpus ne la
        couvre pas → réponse honnête « je ne sais pas » plutôt qu'une invention.
        Une question sans terme propre n'est pas tranchée ici (True).
        """
        proper = _proper_tokens(question)
        if not proper:
            return True
        df = self.corpus_df
        return all(df.get(tok, 0) >= self.cfg.rag_min_df for tok in proper)

    # ---- Contexte conversationnel ----------------------------------------
    def _prep_history(self, history: list[dict] | None) -> list[dict]:
        """Garde les N derniers tours, nettoie les citations [n] des réponses."""
        if not history:
            return []
        turns = history[-self.cfg.history_turns:]
        out = []
        for h in turns:
            role, content = h.get("role"), (h.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            if role == "assistant":
                content = _CITE_RE.sub("", content).strip()  # évite de confondre la numérotation
            out.append({"role": role, "content": content})
        return out

    def _augment(self, history: list[dict] | None, question: str) -> str:
        """Pour une question de suivi courte (« et pourquoi ? »), on la rattache au
        dernier tour utilisateur afin que la recherche et le garde-fou aient le
        référent. Une question autonome est laissée telle quelle."""
        if not history or len(question.split()) > 7:
            return question
        prev_user = next(
            (h.get("content") for h in reversed(history) if h.get("role") == "user" and h.get("content")),
            None,
        )
        return f"{prev_user}\n{question}" if prev_user else question

    def retrieve(self, query_text: str, top_k: int | None = None) -> tuple[list[SearchHit], str]:
        """Récupère les extraits ; Graph-RAG si disponible, sinon vectoriel."""
        if self.uses_graph:
            hits = self.graph.retrieve(query_text, self.store, top_k=top_k)
            if hits:
                return hits, "graph"
        return self.store.search(query_text, top_k=top_k), "vector"

    def build_prompt(self, question: str, hits: list[SearchHit]) -> tuple[str, str]:
        context = format_context(hits)
        return self._rag_sys, RAG_USER_TEMPLATE.format(context=context, question=question)

    def _visual(self, question: str, visual: bool) -> bool:
        """Visuel si demandé par le toggle OU par la formulation (« explique visuellement… »,
        « diagrama », « schéma »…) — par défaut."""
        return bool(visual or wants_visual(question))

    def _verify(self, answer: str, evidence: list[str], question: str) -> tuple[dict | None, dict]:
        """Chaque affirmation de la réponse est confrontée aux sources (voir verify.py)."""
        from .verify import verify

        t0 = time.perf_counter()
        try:
            lang = detect_lang(question)
            src_lang = detect_lang(" ".join(evidence)[:4000])
            v = verify(answer, evidence, self.store.embedder, lang, cross_lingual=src_lang != lang)
        except Exception:  # la vérification ne doit jamais faire échouer une réponse
            v = None
        if v:
            v.pop("best_sims", None)
        return v, {"verify_ms": round((time.perf_counter() - t0) * 1000)}

    def _with_diagram(self, answer: str, evidence: str, question: str) -> tuple[str, dict]:
        """Étape 2 du mode visuel : le diagramme est EXTRAIT de la réponse écrite, chaque
        composant vérifié contre la réponse + les sources, puis construit par le code.
        Rien de vérifiable → pas de diagramme (et une note honnête)."""
        t0 = time.perf_counter()
        res = extract_diagram(answer, evidence, self.llm, diagram_kind(question))
        return attach_diagram(answer, res["code"], detect_lang(question)), {
            "diagram_ms": round((time.perf_counter() - t0) * 1000),
            "diagram_edges": res["kept"], "diagram_rejected": res["rejected"],
        }

    def ask_llm(
        self, question: str, history: list[dict] | None = None, visual: bool = False
    ) -> RAGAnswer:
        """MODE LLM : intuition pure (professeur), consciente du fil de discussion."""
        if not self.in_scope(self._augment(history, question)):
            return self._decline(question, "llm")
        lang = detect_lang(question)
        vis = self._visual(question, visual)
        extra = f"\n\n{visual_hint(diagram_kind(question))}" if vis else ""
        user = f"{_LANG_DIRECTIVE[lang]}\n\n{question}{extra}\n\n{_LANG_FINAL[lang]}"
        answer = self.llm.generate(self._llm_sys, user, history=self._prep_history(history))
        metrics = dict(self.llm.last_stats or {})
        if vis:
            answer, dm = self._with_diagram(answer, "", question)
            metrics.update(dm)
        else:
            answer = normalize_answer(answer)
        return RAGAnswer(
            question=question, answer=answer, mode="llm", grounded=False,
            hits=[], method="llm", metrics=metrics, visual=vis,
        )

    def ask_rag(
        self, question: str, top_k: int | None = None, history: list[dict] | None = None,
        visual: bool = False,
    ) -> RAGAnswer:
        """MODE RAG : réponse ancrée + citations, consciente du fil de discussion."""
        search_query = self._augment(history, question)
        if not self.in_scope(search_query):
            return self._decline(question, "rag")
        # Garde-fou d'ancrage : techno nommée mais absente du corpus → honnête IDK.
        if not self.corpus_covers(question):
            return RAGAnswer(
                question=question, answer=_IDK[detect_lang(question)], mode="rag",
                grounded=False, hits=[], method="idk",
            )
        t_ret = time.perf_counter()
        hits, method = self.retrieve(search_query, top_k=top_k)
        retrieval_ms = round((time.perf_counter() - t_ret) * 1000)
        # Ancrage insuffisant → réponse honnête « je ne sais pas » (pas d'invention).
        best = max((h.score for h in hits), default=0.0)
        if not hits or best < self.cfg.rag_min_score:
            return RAGAnswer(
                question=question,
                answer=_IDK[detect_lang(question)],
                mode="rag",
                grounded=False,
                hits=[],
                method="idk",
            )
        system_prompt, user_prompt = self.build_prompt(question, hits)
        vis = self._visual(question, visual)
        if vis:
            user_prompt += f"\n\n{visual_hint(diagram_kind(question))}"
        user_prompt = f"{user_prompt}\n\n{_LANG_FINAL[detect_lang(question)]}"
        answer = self.llm.generate(system_prompt, user_prompt, history=self._prep_history(history))
        metrics = {**(self.llm.last_stats or {}), "retrieval_ms": retrieval_ms, "chunks": len(hits)}
        if self.cfg.analogies:
            answer = self._with_answer_analogy(normalize_answer(answer), detect_lang(question))
        if vis:
            evidence = "\n".join(h.chunk["text"] for h in hits)
            answer, dm = self._with_diagram(answer, evidence, question)
            metrics.update(dm)
        else:
            answer = normalize_answer(answer)
        verification, vm = self._verify(answer, [h.chunk["text"] for h in hits], question)
        metrics.update(vm)
        return RAGAnswer(
            question=question, answer=answer, mode="rag", grounded=True,
            hits=hits, method=method, visual=vis, metrics=metrics, verification=verification,
        )

    def ask(
        self,
        question: str,
        mode: str = "rag",
        top_k: int | None = None,
        history: list[dict] | None = None,
        visual: bool = False,
    ) -> RAGAnswer:
        t0 = time.perf_counter()
        res = None
        if wants_study(question):  # « résume le chapitre 2 de X » : lecture complète, pas top-k
            res = self.ask_study(question, visual=visual, strict=(mode != "llm"))
        if res is None and mode == "llm":
            res = self.ask_llm(question, history=history, visual=visual)
        elif res is None:
            res = self.ask_rag(question, top_k=top_k, history=history, visual=visual)
        res.metrics["total_ms"] = round((time.perf_counter() - t0) * 1000)
        return res

    # ── Mode étude : un chapitre / une section / un document entier ─────────
    def _progress(self, step: str, i: int, n: int) -> None:
        self.progress = {"step": step, "i": i, "n": n}

    def ask_study(self, question: str, visual: bool = False, strict: bool = True) -> RAGAnswer | None:
        """Résume / explique une partie d'un document de façon systématique (voir study.py).
        Renvoie None si aucun document n'est identifiable et `strict` est faux (mode LLM :
        on retombe sur une réponse normale)."""
        from . import lake
        from .study import (_NO_DOC, _NOT_FOUND, Node, assemble, chapter_titles, find_target,
                            load_book, parse_target, resolve_document, summarize_target)

        lang = detect_lang(question)
        docs = lake.list_documents(self.cfg)
        doc = resolve_document(question, docs)
        if doc is None:
            if not strict:
                return None
            pdfs = sorted({d["filename"] for d in docs if d["filename"].lower().endswith(".pdf")})[:15]
            msg = _NO_DOC.get(lang, _NO_DOC["en"]) + "\n\n" + "\n".join(f"- {n}" for n in pdfs)
            return RAGAnswer(question=question, answer=msg, mode="rag", grounded=False, method="study")
        try:
            self._progress(f"Opening {doc['filename'][:48]}", 0, 1)
            book = load_book(doc["s3_uri"], doc["filename"])
            # chapitre n° > section n° > sujet cité (titre, sinon pages qui en parlent) > tout
            target = parse_target(question)
            node = find_target(book, target, question)
            if node is None and target.chapter is None and not target.section:
                roots = book.nodes
                while len(roots) == 1 and roots[0].children:  # « Table of Contents » englobant
                    roots = roots[0].children
                node = Node(0, re.sub(r"\.\w+$", "", book.name), 1, len(book.pages), roots)
            if node is None:
                titles = chapter_titles(book)
                msg = _NOT_FOUND.get(lang, _NOT_FOUND["en"]).format(doc=book.name) + "\n\n" + \
                      "\n".join(f"- {t}" for t in titles[:20])
                return RAGAnswer(question=question, answer=msg, mode="rag", grounded=False, method="study")

            final = _LANG_FINAL[lang]
            result = summarize_target(book, node, self.llm, final, self._progress,
                                      fix=lambda lines: self._ensure_lang(lines, lang))
            answer = assemble(book, node, result, lang)
            answer = self._with_analogy(answer, result, lang)

            # chaque section résumée devient une source citée « document · pages · titre »
            from .study import _pages
            hits = []
            for u, _, excerpt in result["sections"]:
                pages = _pages(u, book)
                hits.append(SearchHit(score=1.0, chunk={
                    "source": f"{book.name} · {pages} · {u.title}" if pages else f"{book.name} · {u.title}",
                    "category": doc.get("domain", ""), "text": excerpt, "chunk_index": u.start,
                }))
            metrics = {"sections": len(result["sections"]), "pages": node.end - node.start + 1}
            texts = ["\n".join(book.pages[u.start - 1: u.end]) for u, _, _ in result["sections"]]
            if visual or wants_visual(question):
                # la figure d'un chapitre = sa structure réelle (chapitre → sections →
                # notion phare), pas une extraction du modèle : fiable par construction
                from .study import structure_diagram
                code = structure_diagram(node, result, book)
                head, sep, rest = answer.partition("\n\n### ")  # vue d'ensemble → figure → sections
                answer = (f"{head}\n\n```mermaid\n{code}\n```\n\n### {rest}" if code and sep
                          else attach_diagram(answer, code, lang))
                metrics.update({"diagram_edges": len(result["sections"])})
            self._progress("Checking every claim against the pages", 1, 1)
            verification, vm = self._verify(answer, texts, question)
            metrics.update(vm)
            return RAGAnswer(question=question, answer=answer, mode="rag", grounded=True, hits=hits,
                             method="study", visual=bool(visual or wants_visual(question)), metrics=metrics,
                             verification=verification)
        finally:
            self.progress = None

    def _with_analogy(self, answer: str, result: dict, lang: str) -> str:
        """Mode étude : l'analogie va après la vue d'ensemble."""
        if not result.get("overview"):
            return answer
        basis = result["overview"] + "\n" + "\n".join(f"- {t}" for t in result.get("takeaways", []))
        block = self._analogy_block(basis, lang)
        if not block:
            return answer
        head, sep, rest = answer.partition("\n\n### ")  # après la vue d'ensemble
        return f"{head}\n\n{block}\n\n### {rest}" if sep else f"{answer}\n\n{block}"

    def _with_answer_analogy(self, answer: str, lang: str) -> str:
        """Réponse RAG : l'analogie va après le premier paragraphe (la définition)."""
        basis = re.sub(r"```.*?```", "", answer, flags=re.S)[:1500]
        block = self._analogy_block(basis, lang)
        if not block:
            return answer
        first, sep, rest = answer.partition("\n\n")
        return f"{first}\n\n{block}\n\n{rest}" if sep else f"{answer}\n\n{block}"

    def _ensure_lang(self, lines: list[str], lang: str) -> list[str]:
        """Le petit modèle répond parfois dans la langue des SOURCES (anglais) malgré la
        consigne : on retraduit alors, ligne à ligne, sans rien ajouter."""
        if not lines or detect_lang(" ".join(lines)) == lang:
            return lines
        names = {"en": "English", "es": "Spanish", "fr": "French", "de": "German"}
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1))
        raw = self.llm.generate(
            f"Translate each numbered line into {names[lang]}. Keep the numbering, technical terms, "
            "names and numbers unchanged. Output only the translated numbered lines.",
            f"{numbered}\n\n{_LANG_FINAL[lang]}", max_new_tokens=60 * len(lines) + 60, repetition_penalty=1.05,
        )
        got = dict((int(m.group(1)), m.group(2).strip())
                   for m in re.finditer(r"^\s*(\d+)[.)]\s+(.+)$", raw or "", re.M))
        out = [got.get(i, t) for i, t in enumerate(lines, 1)]
        return out if detect_lang(" ".join(out)) == lang else lines

    def _analogy_block(self, basis: str, lang: str) -> str | None:
        """Une analogie à la Feynman, CLAIREMENT étiquetée : c'est une aide à l'intuition,
        pas un fait du document (la vérification ne la compte pas comme une affirmation)."""
        self._progress("Finding an analogy", 1, 1)
        raw = self.llm.generate(
            "You are a teacher in the style of Richard Feynman. Explain the core idea below to a "
            "beginner with ONE everyday analogy (cooking, traffic, a library, a restaurant kitchen…), "
            "in 2 to 4 sentences. Start directly with the analogy. Do not add new technical facts.",
            f"CORE IDEA:\n{basis}\n\n{_LANG_FINAL[lang]}", max_new_tokens=160, repetition_penalty=1.08,
        ).strip()
        if not raw:
            return None
        # UN seul paragraphe, entièrement dans l'encadré étiqueté (sinon la suite de
        # l'analogie se lirait comme un fait du document), 4 phrases au plus
        flat = re.sub(r"\s+", " ", raw.replace("\n", " ")).strip()
        flat = " ".join(re.split(r"(?<=[.!?])\s+", flat)[:4])
        flat = self._ensure_lang([flat], lang)[0]
        label = _ANALOGY_LABEL.get(lang, _ANALOGY_LABEL["en"])
        return f"> 💡 **{label}** {flat}"


# Alias de compatibilité (ancien nom du prompt système RAG).
SYSTEM_PROMPT = RAG_SYSTEM_PROMPT
