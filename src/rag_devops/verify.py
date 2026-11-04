"""Vérification phrase par phrase : chaque affirmation d'une réponse ancrée est-elle
réellement soutenue par les sources qu'elle cite ?

Deux signaux indépendants :
  1. SENS — similarité sémantique (multilingual-e5) avec la meilleure fenêtre de 2 phrases
     des sources ;
  2. FAITS DURS — chaque nombre et chaque nom propre de la phrase (80 GB, 2015, vLLM,
     NVIDIA…) doit figurer tel quel dans les sources. C'est là que se logent les
     hallucinations les plus graves, et ce contrôle est déterministe.

Ne sont pas des affirmations : titres, analogie étiquetée, notes, blocs de code/diagrammes.
Résultat : {claims, supported, score, unsupported:[{text, reason}]} — affiché dans l'UI.
"""
from __future__ import annotations

import re

import numpy as np

from .diagrams import _content_tokens, _fold

SIM_SUPPORTED = 0.86      # calibré sur multilingual-e5-small (voir tests/test_verify.py)
# Réponse en français/espagnol/allemand, sources en anglais : les traductions fidèles
# tombent à 0.82–0.89 et les inventions plafonnent vers 0.81 (calibrage tests/test_verify.py)
SIM_CROSS_LINGUAL = 0.835
LEX_SUPPORTED = 0.6       # ou ≥ 60 % des mots significatifs présents dans la meilleure fenêtre
_MIN_WORDS = 6

_BLOCK = re.compile(r"```.*?```", re.S)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(¿¡A-ZÀ-Ý0-9])")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?:\s?(?:%|x|×|gb|mb|tb|ms|s|k|m|b|gbps|tflops))?", re.I)
_PROPER = re.compile(r"\b(?:[A-Z][a-z]+[A-Z]\w*|[A-Z]{2,}[a-z]*\d*|[a-z]+[A-Z]\w+|[A-Z][a-z]{2,}(?:\.[A-Za-z]+)?)\b")


def _plain(s: str) -> str:
    s = re.sub(r"\[(\d+)\]", " ", s)              # citations
    s = re.sub(r"[*_`#>]", " ", s)                # markdown
    return re.sub(r"\s+", " ", s).strip()


def claims_of(answer: str) -> list[str]:
    """Phrases factuelles d'une réponse (texte d'origine, pour pouvoir les surligner)."""
    out = []
    for block in _BLOCK.sub("\n\n", answer or "").split("\n"):
        line = block.strip()
        if not line or line.startswith(("#", "> 💡", ">💡")) or re.match(r"^\*[^*].*\*$", line):
            continue  # titres, analogie étiquetée, notes / sous-titres en italique
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        for sent in _SENT_SPLIT.split(line):
            if len(_plain(sent).split()) >= _MIN_WORDS:
                out.append(sent.strip())
    return out


def _windows(evidence: list[str], limit: int = 1500) -> list[str]:
    sents = []
    for text in evidence:
        sents += [s.strip() for s in _SENT_SPLIT.split(re.sub(r"\s+", " ", text)) if len(s.split()) >= 4]
    wins = [" ".join(sents[i:i + 2]) for i in range(max(1, len(sents) - 1))] if sents else []
    if len(wins) > limit:  # échantillonnage régulier si les sources sont énormes
        step = len(wins) / limit
        wins = [wins[int(i * step)] for i in range(limit)]
    return wins


_TECH = re.compile(r"\b(?:[A-Za-z]*[a-z][A-Z]\w*|[A-Z]{2,}\w*|[A-Za-z]+\d\w*)\b")  # vLLM, MLflow, GPU, H100


_UNITS = set("gb mb tb kb gib mib ms gbps tflops fps".split())  # une unité seule n'est pas un fait


def _norm_fact(s: str) -> str:
    return re.sub(r"[\s\-]+", "", _fold(s)).replace(",", ".")  # 16-bit ~ 16 bit ~ 16bit


def hard_facts(sentence: str, lang: str = "en") -> set[str]:
    """Nombres (≥ 2 chiffres ou décimaux, avec unité) et noms propres/techniques.
    En allemand, tous les noms prennent une majuscule : seules les formes techniques
    sans ambiguïté comptent (sigles, casse mixte, chiffres)."""
    plain = _plain(sentence)
    # intitulé de puce en tête (« Unit Economics: … », « Model Requirements: … ») : un titre,
    # pas une affirmation — ses majuscules ne sont pas des noms propres
    plain = re.sub(r"^\s*[A-Z][^:.!?]{0,40}:\s+", "", plain)
    facts = set()
    for m in _NUM.finditer(plain):
        f = _norm_fact(m.group(0))
        if len(re.sub(r"\D", "", f)) >= 2 or "." in f:
            facts.add(f)
    facts |= {_norm_fact(m.group(0)) for m in _TECH.finditer(plain)} - _UNITS
    if lang != "de":
        for m in _PROPER.finditer(plain):
            tok = m.group(0)
            before = plain[:m.start()].rstrip()
            if not before or before[-1] in ".:;!?¿¡(\"“–—-":
                continue  # majuscule de début de phrase, après « : » ou un intitulé
            if tok[0].isupper() and tok[1:].islower():
                facts.add(_norm_fact(tok))
    return facts


# Généralisations absolues et négations : « toujours », « jamais », « ne … pas »… Si la
# phrase en affirme une et que le passage qui la soutient le mieux n'en contient aucune,
# elle exagère ou contredit probablement la source.
_ABSOLUTE = re.compile(
    r"\b(always|never|all|every|none|only|not|no|cannot|can't|doesn't|don't|isn't|aren't|guaranteed?|"
    r"siempre|nunca|todos?|ningún|ninguna|solo|no|"
    r"toujours|jamais|tous|toutes|aucun|aucune|seulement|pas|"
    r"immer|nie|niemals|alle|kein|keine|nur|nicht)\b", re.I)


_FAMILY = {
    "neg": set("not no never cannot can't doesn't don't isn't aren't none nunca ningún ninguna jamais pas aucun aucune "
               "nie niemals kein keine nicht".split()),
    "all": set("always all every guaranteed guarantee siempre todos todo toujours tous toutes immer alle".split()),
    "only": set("only solo seulement nur".split()),
}


def _families(text: str) -> set[str]:
    text = text.replace("’", "'")  # PDF : apostrophe typographique (isn’t)
    words = {m.group(0).lower() for m in _ABSOLUTE.finditer(text)}
    return {f for f, ws in _FAMILY.items() if words & ws}


def verify(answer: str, evidence: list[str], embedder, lang: str = "en",
           cross_lingual: bool = False) -> dict | None:
    claims = claims_of(answer)
    wins = _windows(evidence)
    if not claims or not wins:
        return None
    ev_plain = _fold(" ".join(evidence))
    ev_compact = re.sub(r"[\s\-]+", "", ev_plain).replace(",", ".")
    ev_tokens = set(_content_tokens(" ".join(evidence)))

    C = embedder.encode_queries([_plain(c) for c in claims])
    W = embedder.encode_queries(wins)
    sims = C @ W.T
    best = sims.max(axis=1)
    arg = sims.argmax(axis=1)
    top3 = np.argsort(-sims, axis=1)[:, :3]

    threshold = SIM_CROSS_LINGUAL if cross_lingual else SIM_SUPPORTED
    unsupported = []
    for i, claim in enumerate(claims):
        # pluriel toléré : « TPUs » est soutenu par « TPU », « SSD » par « SSDs »
        missing = sorted(f for f in hard_facts(claim, lang)
                         if not any(v in ev_compact or v in ev_tokens for v in {f, f.removesuffix("s"), f + "s"}))
        toks = _content_tokens(claim)
        win_tokens = set(_content_tokens(wins[int(arg[i])]))
        win_stems = {t[:5] for t in win_tokens if len(t) >= 6}
        # mot présent, au singulier, ou même radical (reductions ~ reducing, optimize ~ optimization)
        lex = sum(t in win_tokens or t.rstrip("s") in win_tokens or (len(t) >= 6 and t[:5] in win_stems)
                  for t in toks) / max(1, len(toks))
        claim_plain = re.sub(r"\b(or|ou|o|oder)\s+(not|pas|no|nicht)\b", " ", _plain(claim), flags=re.I)  # « … or not »
        absolutes = {m.group(0).lower() for m in _ABSOLUTE.finditer(claim_plain)}
        # même famille (négation / « tous » / « seulement ») dans l'un des 3 meilleurs passages ?
        src_fams = set().union(*(_families(wins[int(j)]) for j in top3[i]))
        overstated = _families(claim_plain) - src_fams
        if missing:
            unsupported.append({"text": claim, "reason": f"not in the sources: {', '.join(missing[:3])}"})
        elif overstated and best[i] < 0.90:
            # (une reformulation très proche de la source, ≥ 0.90, n'est pas jugée sur un mot)
            unsupported.append({"text": claim, "reason": f"the sources don't say “{sorted(a for a in absolutes if _families(a) & overstated)[0]}” — may overstate them"})
        elif not (best[i] >= threshold or lex >= LEX_SUPPORTED):
            unsupported.append({"text": claim, "reason": "no passage in the sources says this"})
    n = len(claims)
    return {"claims": n, "supported": n - len(unsupported), "score": round((n - len(unsupported)) / n, 3),
            "unsupported": unsupported, "best_sims": [round(float(b), 3) for b in np.asarray(best)]}
