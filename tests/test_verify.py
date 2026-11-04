"""Vérification phrase par phrase — calibrée sur le chapitre 1 d'Inference Engineering.

Les preuves sont de vraies phrases du livre ; les affirmations à rejeter sont des
inventions plausibles (nom/nombre faux, généralisation absolue, contresens).
"""
import pytest

from rag_devops.config import settings
from rag_devops.verify import claims_of, hard_facts, verify

EVIDENCE = [
    "Establish a baseline: Some performance optimization techniques risk reducing model quality, "
    "requiring a baseline to compare against. Unlike standard intelligence benchmarks, evals are domain specific.",
    "Most AI products start with pay-per-token APIs because the tradeoffs make sense while looking for "
    "product-market fit or in the early stages of growth. Shared inference has zero overhead: you only pay for consumption.",
    "Two primary performance metrics for LLMs are time to first token (TTFT) and tokens per second (TPS). "
    "Latency is reported as percentiles such as P50, P90 and P99.",
]


@pytest.fixture(scope="module")
def embedder():
    from rag_devops.embed import Embedder

    return Embedder(settings)


def _flagged(answer, embedder):
    v = verify(answer, EVIDENCE, embedder, "en")
    return v, {u["text"].lstrip("- ") for u in v["unsupported"]}


def test_paraphrases_are_supported(embedder):
    ans = ("- Baselines are necessary to prevent reductions in model quality due to optimization techniques.\n"
           "- Many AI products begin with pay-per-token APIs while they search for product-market fit.\n"
           "- Time to first token and tokens per second are the two main performance metrics for LLMs.")
    v, flagged = _flagged(ans, embedder)
    assert v["supported"] == 3 and not flagged


@pytest.mark.parametrize("claim", [
    "Inference engineering was invented by Google in 2015 as part of the TensorFlow project.",  # noms/nombre
    "The book recommends Kubernetes CronJobs to reduce token latency by 90%.",                   # nombre + nom
    "Shared inference always requires a dedicated GPU cluster for every customer.",              # absolu
    "Latency percentiles are never used to report LLM performance.",                             # contresens
])
def test_fabrications_are_flagged(embedder, claim):
    v, flagged = _flagged(f"- {claim}", embedder)
    assert v["supported"] == 0, claim


def test_non_claims_are_ignored():
    ans = ("## Chapter 1\n\n*Inference Engineering.pdf · pp. 25–40*\n\n> 💡 **In plain words — an analogy:** "
           "It is like a restaurant kitchen.\n\n```mermaid\nflowchart LR\n a --> b\n```\n\n"
           "*◇ No diagram this time.*\n\n- Shared inference has zero overhead for most products.")
    assert claims_of(ans) == ["Shared inference has zero overhead for most products."]


def test_hard_facts_ignore_sentence_initial_and_label_capitals():
    assert hard_facts("Unit Economics: Decide on the cost per request.") == set()
    assert "h100" in hard_facts("The H100 has 80 GB of HBM3 memory.")
    assert "architektur" not in hard_facts("Die Architektur nutzt die GPU.", "de")


def test_or_not_and_sourced_universals_are_not_overstatements(embedder):
    ev = ["A quality gate determines whether a project is ready for release. If all conditions are met, "
          "the quality gate passes; if any condition fails, it fails."]
    ans = ("- The quality gate determines whether a project is ready for release or not.\n"
           "- If all conditions are met, the quality gate passes.")
    v = verify(ans, ev, embedder, "en")
    assert v["supported"] == 2, v["unsupported"]


FR_FAITHFUL = [
    "Les bases de référence sont nécessaires pour éviter une baisse de qualité du modèle due aux optimisations.",
    "Le temps jusqu'au premier token et les tokens par seconde sont les deux principales métriques de performance.",
]
FR_FABRICATED = [
    "L'ingénierie de l'inférence a été inventée par Google pour le projet TensorFlow.",
    "La sécurité des données est la priorité principale de l'inférence partagée.",
]


def test_cross_lingual_answers_use_the_calibrated_threshold(embedder):
    v = verify("\n".join(f"- {c}" for c in FR_FAITHFUL), EVIDENCE, embedder, "fr", cross_lingual=True)
    assert v["supported"] == 2, v["unsupported"]
    for c in FR_FABRICATED:
        assert verify(f"- {c}", EVIDENCE, embedder, "fr", cross_lingual=True)["supported"] == 0, c


def test_curly_apostrophe_negation_in_pdf_text_counts(embedder):
    ev = ["If there isn’t enough memory for the weights, loading the model fails with an out-of-memory error."]
    v = verify("- If there isn't enough memory for the weights, loading the model fails with an out-of-memory error.",
               ev, embedder, "en")
    assert v["supported"] == 1


def test_hard_facts_tolerate_plurals_and_hyphens(embedder):
    ev = ["Other options include the AMD MI350, ASIC chips and the Google TPU. Hopper adds 16-bit and FP8 tensor cores."]
    ans = "- Other options include the AMD MI350, ASICs and TPUs.\n- Hopper adds 16b and FP8 tensor cores."
    v = verify(ans, ev, embedder, "en")
    assert not any(u["reason"].startswith("not in the sources") for u in v["unsupported"]), v["unsupported"]
    v = verify("- Other options include the AMD MI400 and the Google TPU.", ev, embedder, "en")
    assert v["unsupported"] and "mi400" in v["unsupported"][0]["reason"]
