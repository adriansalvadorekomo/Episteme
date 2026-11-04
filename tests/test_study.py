"""Mode étude : intention, résolution du document, structure, ciblage, map→reduce, rendu."""
import pymupdf
import pytest

from rag_devops import study
from rag_devops.study import (Book, Node, assemble, find_target, parse_target, resolve_document,
                              summarize_target, wants_study, _heading_score, _tree, _window_nodes)


@pytest.mark.parametrize("q", [
    "Summarize chapter 1 of Inference Engineering", "Resume el capítulo 2 de Inference Engineering",
    "Explique la section 2.2 de Inference Engineering", "Fasse Kapitel 3 von Inference Engineering zusammen",
    "Summarize the Kubernetes Notes", "explain the part about pods in the Kubernetes Notes",
])
def test_study_intent(q):
    assert wants_study(q)


@pytest.mark.parametrize("q", ["What is a Pod in Kubernetes?", "Explain visually a microservices architecture",
                               "How does HDFS replicate data blocks?"])
def test_regular_questions_are_not_study(q):
    assert not wants_study(q)


def test_parse_target():
    assert parse_target("Summarize chapter 3 of X").chapter == 3
    assert parse_target("explique la section 2.2 de X").section == "2.2"
    assert parse_target("summarize 1.3 in the book").section == "1.3"


DOCS = [{"filename": "Inference Engineering.pdf"}, {"filename": "Kubernetes Notes.pdf"}, {"filename": "Big Data 0.pdf"}]


def test_resolve_document_by_name():
    assert resolve_document("resume el capítulo 2 de inference engineering", DOCS)["filename"] == "Inference Engineering.pdf"
    assert resolve_document("summarize the kubernetes notes", DOCS)["filename"] == "Kubernetes Notes.pdf"
    assert resolve_document("summarize chapter 2", DOCS) is None


def _book():
    toc = [(1, "Table of Contents", 1), (2, "Chapter 1: Prerequisites", 2), (3, "1.1 Scale", 3), (3, "1.2 Apps", 4),
           (2, "Chapter 2: Hardware", 6), (3, "2.1 GPUs", 6), (3, "2.2 Memory", 7)]
    pages = [f"page {i} " + ("GPU memory bandwidth HBM " * 20 if i >= 6 else "latency budget product " * 20) for i in range(1, 9)]
    return Book("Inference Engineering.pdf", "pdf", pages, _tree(toc, len(pages)))


def test_tree_page_ranges_and_targets():
    b = _book()
    c1 = find_target(b, parse_target("chapter 1"), "chapter 1")
    assert (c1.title, c1.start, c1.end, len(c1.children)) == ("Chapter 1: Prerequisites", 2, 5, 2)
    assert find_target(b, parse_target("section 2.2"), "section 2.2").title == "2.2 Memory"
    # titre cité, sans être trompé par le nom du document
    q = "explain the chapter about hardware in Inference Engineering"
    assert find_target(b, parse_target(q), q).title == "Chapter 2: Hardware"
    assert find_target(b, parse_target("chapter 9"), "chapter 9") is None


class FakeLLM:
    def __init__(self):
        self.calls = []
    def generate(self, system, user, **k):
        self.calls.append(system[:20])
        if "OVERVIEW" in system:
            return "OVERVIEW: The chapter covers latency budgets.\nTAKEAWAYS:\n- Measure latency first.\n- Budget per request."
        return "- Latency budgets define acceptable response time.\n- Product needs drive the budget.\n- Short"


def test_map_reduce_and_deterministic_layout():
    b = _book()
    node = find_target(b, parse_target("chapter 1"), "chapter 1")
    llm = FakeLLM()
    res = summarize_target(b, node, llm, "Answer in English.")
    # page 2 précède la 1re section (p. 3) : elle devient « Introduction », rien n'est sauté
    assert [u.title for u, _, _ in res["sections"]] == ["Introduction", "1.1 Scale", "1.2 Apps"]
    assert res["overview"].startswith("The chapter covers")
    assert all(len(bl) == 2 for _, bl, _ in res["sections"])  # puce trop courte ignorée
    md = assemble(b, node, res, "en")
    assert md.startswith("## Chapter 1: Prerequisites\n\n*Inference Engineering.pdf · pp. 2–5*")
    assert "#### Introduction · p. 2 [1]" in md and "#### 1.1 Scale · p. 3 [2]" in md and "#### 1.2 Apps · pp. 4–5 [3]" in md
    assert "### Key takeaways" in md


def test_documents_without_structure_get_titled_page_groups():
    pages = ["Kubernetes From Scratch\nintro text"] + ["using the following command.\nkubectl get pods"] * 9 + ["ConfigMaps\nconfig"] * 10
    nodes = _window_nodes(pages)
    assert nodes[0].title == "Kubernetes From Scratch" and nodes[0].start == 1
    assert any(n.title == "ConfigMaps" for n in nodes)
    assert _heading_score("using the following command.") < _heading_score("Scaling the Deployment")


def test_topic_targeting_prefers_pages_that_discuss_it():
    pages = ["Intro\nWelcome to the notes " * 5] * 3 + ["Pods\n" + "a pod runs containers pods share network " * 30] * 3 + ["Volumes\nstorage " * 50] * 3
    b = Book("Kubernetes Notes.pdf", "pdf", pages, _window_nodes(pages), synthetic=True)
    q = "summarize the part about pods in the Kubernetes Notes"
    node = find_target(b, parse_target(q), q)
    assert node is not None and all(3 < c.start <= 6 for c in node.children)


def test_french_chapter_word_is_not_swallowed_by_chap():
    # « chapitre 3 » ne doit pas être lu comme « chap » sans numéro (→ livre entier)
    assert parse_target("Explique visuellement le chapitre 3 de Inference Engineering").chapter == 3
    assert parse_target("summarize chap. 4").chapter == 4
    assert parse_target("the section about GPUs in chapter 5").chapter == 5


def test_structure_diagram_is_built_from_the_real_sections():
    b = _book()
    node = find_target(b, parse_target("chapter 1"), "chapter 1")
    res = summarize_target(b, node, FakeLLM(), "Answer in English.")
    res["sections"][1] = (res["sections"][1][0], ["**Latency budget** defines the response time."], "")
    code = study.structure_diagram(node, res)
    assert code.startswith("flowchart TD")
    for title in ("Chapter 1: Prerequisites", "Introduction", "1.1 Scale", "1.2 Apps", "Latency budget"):
        assert title in code


def test_fix_hook_rewrites_language():
    b = _book()
    node = find_target(b, parse_target("chapter 1"), "chapter 1")
    res = summarize_target(b, node, FakeLLM(), "x", fix=lambda lines: [f"FR {t}" for t in lines])
    assert res["overview"].startswith("FR ") and all(t.startswith("FR ") for t in res["takeaways"])
    assert all(bl[0].startswith("FR ") for _, bl, _ in res["sections"])


def test_key_notion_must_appear_in_the_section_pages():
    b = _book()
    node = find_target(b, parse_target("chapter 1"), "chapter 1")
    res = summarize_target(b, node, FakeLLM(), "x")
    u0, _, ex0 = res["sections"][1]
    res["sections"][1] = (u0, ["**Le code source** est important."], ex0)
    assert "code source" not in study.structure_diagram(node, res, b)
    res["sections"][1] = (u0, ["**Latency budget** matters."], ex0)
    assert "Latency budget" in study.structure_diagram(node, res, b)


def test_overview_stops_at_inline_takeaways_label():
    class InlineLLM(FakeLLM):
        def generate(self, system, user, **k):
            if "OVERVIEW" in system:
                return "OVERVIEW: The chapter covers budgets. **TAKEAWAYS:**\n- Measure latency first."
            return super().generate(system, user, **k)
    b = _book()
    node = find_target(b, parse_target("chapter 1"), "chapter 1")
    res = summarize_target(b, node, InlineLLM(), "x")
    assert "TAKEAWAYS" not in res["overview"] and res["takeaways"] == ["Measure latency first."]
