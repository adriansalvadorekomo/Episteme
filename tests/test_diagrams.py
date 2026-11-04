"""Tests du module diagrams : intention visuelle, nettoyage Mermaid, normalisation."""
import pytest

from rag_devops.diagrams import (
    attach_diagram,
    build_mermaid,
    diagram_kind,
    extract_diagram,
    ground_edges,
    normalize_answer,
    parse_edges,
    sanitize_mermaid,
    visual_hint,
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



# ── diagrammes ancrés : le cas réel « Inference Engineering » ────────────────
IE_ANSWER = (
    "Inference engineering focuses on optimizing the production serving of generative models. "
    "Requests from the application reach the inference server, which batches them on the GPU. "
    "The runtime (for example vLLM or TensorRT-LLM) executes the model and streams tokens back. "
    "Autoscaling adds GPU replicas when the request queue grows, and monitoring tracks latency "
    "and throughput."
)


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
    def generate(self, *a, **k):
        return self.reply


def test_hint_contains_no_copyable_example():
    for kind in ("flowchart", "sequence"):
        h = visual_hint(kind)
        assert "```" not in h and "Orders" not in h and "API Gateway" not in h


def test_parse_edges_tolerates_bullets_arrows_and_markup():
    edges = parse_edges("1. **Application** -> Inference Server : sends requests\n"
                        "- Inference Server → GPU : batches\nnoise line\nRuntime => Model\nA -> A : self")
    assert edges == [("Application", "sends requests", "Inference Server"),
                     ("Inference Server", "batches", "GPU"), ("Runtime", "", "Model")]


def test_copied_example_is_rejected_for_inference_engineering():
    copied = "Web App -> API Gateway : HTTPS\nAPI Gateway -> Orders Service : REST\nOrders Service -> Orders DB : SQL"
    res = extract_diagram(IE_ANSWER, "", FakeLLM(copied))
    assert res["code"] is None and res["kept"] == 0 and res["rejected"] == 3


def test_grounded_relations_build_a_valid_diagram():
    reply = ("Application -> Inference Server : sends requests\n"
             "Inference Server -> GPU : batches requests\n"
             "Runtime -> Model : executes\n"
             "Autoscaling -> GPU replicas : adds\n"
             "Monitoring -> Latency : tracks\n"
             "Kafka Cluster -> Orders DB : invented")          # inventé → rejeté
    res = extract_diagram(IE_ANSWER, "", FakeLLM(reply))
    assert res["kept"] == 5 and res["rejected"] == 1
    code = res["code"]
    assert code.startswith("flowchart") and '"Inference Server"' in code and "Kafka" not in code
    assert code.count("-->") == 5


def test_sequence_build():
    code = build_mermaid([("Client", "POST /token", "Auth Server"), ("Auth Server", "access token", "Client")], "sequence")
    assert code.startswith("sequenceDiagram") and "participant n1 as Client" in code and "n1->>n2: POST /token" in code


def test_partial_grounding_threshold():
    kept, rejected = ground_edges([("Inference Server", "", "GPU"), ("Quantum Router", "", "GPU")], IE_ANSWER)
    assert kept == [("Inference Server", "", "GPU")] and rejected == 1


def test_attach_places_figure_after_overview_or_notes_absence():
    ans = "Overview paragraph.\n\n- detail one\n- detail two"
    out = attach_diagram(ans, "flowchart LR\n  a --> b", "en")
    assert out.index("Overview") < out.index("```mermaid") < out.index("detail one")
    none = attach_diagram(ans + "\n\n```mermaid\nflowchart LR\n x --> y\n```", None, "fr")
    assert "```mermaid" not in none and "Pas de diagramme" in none


def test_copied_template_and_files_are_not_components():
    from rag_devops.diagrams import ground_edges, parse_edges
    assert parse_edges("Source -> Target : relation\nRuntime Optimization : Source -> Target : x") == []
    ev = "The router sends requests to the GPU workers. Inference Engineering.pdf"
    kept, _ = ground_edges([("router", "sends requests", "GPU workers"),
                            ("GPU workers", "", "Inference Engineering.pdf")], ev)
    assert kept == [("router", "sends requests", "GPU workers")]


def test_ungrounded_relation_label_is_dropped_but_edge_kept():
    from rag_devops.diagrams import ground_edges
    ev = "The router sends requests to the GPU workers and the cache."
    kept, _ = ground_edges([("router", "scales across clouds and regions", "GPU workers")], ev)
    assert kept == [("router", "", "GPU workers")]


def test_candidate_names_come_from_the_answer():
    from rag_devops.diagrams import candidate_names
    ans = "## Serving\n\n- **KV cache**: stores keys.\n- Batching: groups requests on the GPU.\n"
    names = candidate_names(ans)
    assert {"KV cache", "Batching", "GPU", "Serving"} <= set(names)


def test_parallel_edges_are_merged_into_one_arrow():
    from rag_devops.diagrams import ground_edges
    ev = "Runtime optimization of a single model improves execution speed and memory usage."
    kept, rejected = ground_edges([("Runtime optimization", "execution speed", "single model"),
                                   ("Runtime optimization", "memory usage", "single model")], ev)
    assert kept == [("Runtime optimization", "execution speed · memory usage", "single model")]
    assert rejected == 0
