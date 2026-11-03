"""Tests du module diagrams : intention visuelle, nettoyage Mermaid, normalisation."""
import pytest

from rag_devops.diagrams import (
    diagram_kind,
    normalize_answer,
    sanitize_mermaid,
    visual_directive,
    wants_visual,
)

# Sortie réelle du modèle (sonde du 2026-11-03) : lignes `style` invalides + nœuds dupliqués.
REAL_MODEL_OUTPUT = """graph TD
    A[Client] --> B(API Gateway)
    B --> C1[Service 1]
    B --> C2[Service 2]
    C1 --> D[Response from Service 1]
    C2 --> E[Response from Service 2]
    D --> G[Forward to Client]
    E --> H[Forward to Client]
    G --> J[Client Response]
    H --> K[Client Response]
    style A fill:#ADD8E6,stroke:none; label="Client"
    style B fill:#FFD700,stroke:none; label="API Gateway"
"""


# ── intention ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", [
    "Explain visually a microservices architecture",
    "Can you draw the Kafka architecture?",
    "show me a diagram of a CI/CD pipeline",
    "Explícame visualmente la arquitectura de Kubernetes",
    "hazme un diagrama del flujo de OAuth",
    "Montre-moi un schéma de l'architecture RAG",
    "Explique visuellement le pipeline MLOps",
    "Zeig mir ein Diagramm der Microservices-Architektur",
    "Erkläre visuell, wie Docker funktioniert",
])
def test_wants_visual_positive(q):
    assert wants_visual(q)


@pytest.mark.parametrize("q", [
    "What is CI/CD?",
    "¿Qué es un Pod en Kubernetes?",
    "Comment configurer Visual Studio Code pour Python ?",
    "How do I install Visual Studio?",
    "Was ist ein Container?",
])
def test_wants_visual_negative(q):
    assert not wants_visual(q)


def test_diagram_kind():
    assert diagram_kind("Explain visually how OAuth login works") == "sequence"
    assert diagram_kind("diagrama del flujo de la petición HTTP paso a paso") == "sequence"
    assert diagram_kind("Explain visually a microservices architecture") == "flowchart"


def test_directive_mentions_format_and_grounding():
    d = visual_directive("flowchart", grounded=True)
    assert "```mermaid" in d and "flowchart LR" in d and "CONTEXT" in d
    assert "sequenceDiagram" in visual_directive("sequence")


# ── nettoyage ────────────────────────────────────────────────────────────────
def test_sanitize_real_model_output():
    out = sanitize_mermaid(REAL_MODEL_OUTPUT)
    assert out.startswith("flowchart TD\n")
    assert "style" not in out and "fill:" not in out          # lignes de style supprimées
    assert 'A(["Client"])' in out and 'B{{"API Gateway"}}' in out  # guillemets + formes sémantiques
    assert out.count('"Client Response"') == 1                 # doublons fusionnés…
    assert out.count('"Forward to Client"') == 1
    assert "K" not in [tok.strip() for tok in out.replace("-->", " ").split()]  # …et alias disparus
    edges = [ln.split('[')[0].split('(')[0].strip() for ln in out.splitlines()[1:]]
    assert len(edges) == len(set(edges))                       # aucune arête en double (G --> J)


def test_sanitize_header_and_semicolons():
    out = sanitize_mermaid("graph LR;\n  a[Build] --> b[Test];\n  b --> c[Deploy];")
    assert out.splitlines()[0] == "flowchart LR"
    assert not any(ln.rstrip().endswith(";") for ln in out.splitlines())


def test_sanitize_quotes_special_chars_and_keeps_quoted():
    out = sanitize_mermaid('flowchart TD\n  a[Client: browser] --> b["Already quoted"]')
    assert 'a(["Client: browser"])' in out and 'b["Already quoted"]' in out


def test_sanitize_missing_header_and_database_shape():
    out = sanitize_mermaid("app[App] --> db[(Postgres)]")
    assert out.startswith("flowchart TD") and 'db[("Postgres")]' in out


def test_sanitize_sequence_kept():
    out = sanitize_mermaid("sequenceDiagram\n  C->>A: login\n  A-->>C: token\n  %% comment")
    assert out.splitlines()[0] == "sequenceDiagram" and "%%" not in out


def test_sanitize_unsupported_returns_none():
    assert sanitize_mermaid("pie title Pets\n  \"Dogs\" : 3") is None


# ── normalisation de la réponse ──────────────────────────────────────────────
def test_normalize_keeps_one_clean_diagram():
    ans = ("Intro.\n\n```mermaid\n" + REAL_MODEL_OUTPUT + "```\n\nMore text.\n\n"
           "```mermaid\nflowchart LR\n a --> b\n```\nEnd.")
    out = normalize_answer(ans)
    assert out.count("```mermaid") == 1 and "style A" not in out
    assert out.startswith("Intro.") and "More text." in out and out.rstrip().endswith("End.")


def test_normalize_unlabeled_fence_and_bare_graph():
    assert "```mermaid\nflowchart TD" in normalize_answer("See:\n```\ngraph TD\n a[x] --> b[y]\n```")
    bare = normalize_answer("Here it is:\n\ngraph LR\n  a[Git] --> b[CI]\n\nDone.")
    assert "```mermaid\nflowchart LR" in bare and bare.rstrip().endswith("Done.")


def test_normalize_leaves_other_code_alone():
    ans = "```python\nprint('graph TD')\n```"
    assert normalize_answer(ans) == ans


# ── régressions issues de la sonde live ─────────────────────────────────────
LOOPING_SEQUENCE = "```mermaid\nsequenceDiagram\n  participant U as User\n  participant P as Provider\n" + (
    "  P-->>U: Redirect with access token\n  U->>P: Provide credentials\n" * 12)  # jamais refermé


def test_unterminated_looping_sequence_is_recovered():
    out = normalize_answer("OAuth:\n\n" + LOOPING_SEQUENCE)
    assert out.count("```mermaid") == 1 and out.rstrip().endswith("```")
    body = out.split("```mermaid\n")[1].split("```")[0]
    assert body.count("Redirect with access token") == 1   # boucle repliée
    assert body.count("Provide credentials") == 1


def test_sequence_message_cap():
    code = "sequenceDiagram\n" + "".join(f"  A->>B: step {i}\n" for i in range(30))
    assert sanitize_mermaid(code).count("->>") == 14


def test_request_verbs_are_not_technologies():
    from rag_devops.pipeline import _proper_tokens

    assert _proper_tokens("Explícame visualmente la arquitectura de Kubernetes") == {"kubernetes"}
    assert _proper_tokens("Hazme un diagrama de un pipeline de MLOps") == {"mlops"}
    de = _proper_tokens("Zeig mir ein Diagramm der Docker-Architektur")
    assert "architektur" not in de and "diagramm" not in de       # noms communs allemands
    assert _proper_tokens("Erkläre mir die MLOps-Pipeline mit K8s") == {"mlops", "k8s"}
    assert "istio" in _proper_tokens("Explain visually Istio mTLS")  # vraies technos conservées


# ── échecs réels de rendu (rejoués dans le navigateur) ───────────────────────
def test_comments_and_narration_edges_removed():
    out = sanitize_mermaid('flowchart LR\n  u["User"] --> gw["API Gateway"]\n  // Labels for clarity\n'
                           '  u --> "The user initiates a request."\n  gw --> s["Service A"]')
    assert "//" not in out and "initiates" not in out and 's["Service A"]' in out


def test_edge_label_on_next_line_is_folded():
    out = sanitize_mermaid('flowchart LR\n  user["Utilisateur"]\n  prom["Prometheus"]\n'
                           '  user --> prom\n    "Démarrez Prometheus"')
    assert 'user -->|"Démarrez Prometheus"| prom' in out
    assert not any(l.strip() == '"Démarrez Prometheus"' for l in out.splitlines())


def test_note_below_becomes_note_over():
    out = sanitize_mermaid("sequenceDiagram\n  participant U as User\n  U->>U: x\n  Note below U: after success")
    assert "Note over U: after success" in out


def test_semantic_shapes_and_roles():
    out = sanitize_mermaid('flowchart LR\n  web["Web App users"] --> lb["Load Balancer"]\n'
                           '  lb --> svc["Orders Service"]\n  svc --> q["Kafka topic"]\n  svc --> db["Postgres DB"]')
    assert 'lb{{"Load Balancer"}}' in out and 'q[["Kafka topic"]]' in out and 'db[("Postgres DB")]' in out
    assert 'svc["Orders Service"]' in out                       # un service garde sa forme
    assert "class db role-store" in out and "class lb role-edge" in out
    svc_line = next(l for l in out.splitlines() if l.strip().endswith("role-service"))
    assert "svc" in svc_line.split()[1].split(",")


def test_subgraph_titles_untouched_by_shapes():
    out = sanitize_mermaid('flowchart LR\n  subgraph Clients\n    u["User"]\n  end\n  u --> api["API"]')
    assert "subgraph Clients" in out and 'u(["User"])' in out


def test_prose_inside_diagram_dropped():
    out = sanitize_mermaid('flowchart LR\n  a["Client"] --> b["API"]\n  db[("Orders DB")]\n'
                           '  In this example, we have three services(A, B, and C) each with a database.\n  b --> db')
    assert "In this example" not in out and 'db[("Orders DB")]' in out and "b --> db" in out


def test_truncated_line_dropped():
    out = sanitize_mermaid('flowchart LR\n  a["Prometheus"] --> b["Grafana"]\n  b --> a\n    "Gardez un œ')
    assert "Gardez" not in out and out.count("-->") == 2


def test_edge_label_with_parentheses_is_protected_and_quoted():
    out = sanitize_mermaid('flowchart LR\n  c(["Clients"]) -->|HTTP(S)| lb["Load Balancer"]')
    assert '-->|"HTTP(S)"| lb' in out and 'HTTP("S")' not in out


def test_capitalized_keywords_and_foreign_arrows():
    out = sanitize_mermaid('flowchart LR\n  Subgraph Frontend\n    u["User"]\n  End\n  a["API"] --|> b["Svc"]\n  b->c["DB"]')
    assert "  subgraph Frontend" in out and "\n  end" in out
    assert "--|>" not in out and "b-->c" in out.replace(" ", "")


def test_flowchart_loop_and_truncation():
    code = 'flowchart LR\n  a["API"] -->|gRPC| db[("DB")]\n' + '  db -->|HTTP| db\n' * 90 + '  db -->|'
    out = sanitize_mermaid(code)
    assert out.count("-->") == 1 and "db -->|" not in out


def test_long_lr_chain_turns_top_down():
    chain = "flowchart LR\n" + "".join(f'  n{i}["Step {i}"] --> n{i+1}["Step {i+1}"]\n' for i in range(7))
    assert sanitize_mermaid(chain).startswith("flowchart TD")
    short = 'flowchart LR\n  a["A"] --> b["B"]\n  b --> c["C"]'
    assert sanitize_mermaid(short).startswith("flowchart LR")
