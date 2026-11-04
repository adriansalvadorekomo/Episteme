# Asking Episteme, and knowing when to trust the answer

Episteme answers from the documents you ingested, and it shows you which parts of an answer
it can back with them. This page covers the kinds of questions it handles well, how chapter
summaries and diagrams are built, and how to read the verification badge.

## See what it knows first

```bash
mise run knowledge              # domains, distinctive topics, books with chapters
mise run knowledge -- --no-books   # faster: skip opening the PDFs to count chapters
```

It lists every domain with its number of documents and chunks and the topics that set it
apart from the others. It also lists the books whose chapters you can ask for by number.
Ask about something that isn't listed and the answer comes from the model's general
knowledge (LLM mode), not your sources.

## Three kinds of questions

| You ask | What happens | Example |
|---|---|---|
| A question | Graph-RAG retrieves the closest passages, and the answer cites them `[n]` | *What is a ReplicaSet in Kubernetes?* |
| A chapter, section or part of a named document | **Study mode** reads the whole part, section by section | *Summarize chapter 1 of Inference Engineering* |
| Either one, "visually" | Same as above, plus a diagram | *Explique visuellement le chapitre 3 de Inference Engineering* |

Study mode starts when the question asks to summarize or explain (in EN/ES/FR/DE) **and**
names a part of a document:

- `chapter 3`, `capítulo 2`, `chapitre 3`, `Kapitel 3`: a chapter, read from the PDF's bookmarks
  or from its "Chapter N" headings;
- `section 2.2` or just `2.2`: a numbered section;
- `the chapter about hardware`: matched against the chapter titles;
- `the part about pods in the Kubernetes Notes`: the pages that discuss that topic most;
- `summarize the Kubernetes Notes`: the whole document, in up to 10 parts.

Name the document in the question. Two words of its file name are enough (`Inference
Engineering`, `Kubernetes Notes`). If the part doesn't exist, Episteme lists the chapters it
does have.

### Why study mode exists

Normal retrieval sees the 5 closest passages, roughly 5% of a 30-page chapter, which is why a
chapter summary built that way skips things. Study mode instead:

1. **reads every section** of the requested part (OCR on scanned pages) and summarises each
   one separately in 3 to 5 bullets, with a strict "only what is written" instruction;
2. **writes the overview and the key takeaways** from those section summaries only;
3. **lays out the answer itself**, in code: titles, page ranges and one citation per
   section, so the structure never depends on the model.

A 16-page chapter takes about 30 to 45 s on an RTX 4060. The progress line in the chat shows
the real step ("Reading 1.2 Latency budget · 3/5…").

## Diagrams you can trust

A diagram is drawn when the question says *visually / diagram / visualmente / schéma /
Diagramm…*, or when the ◇ Visual toggle is on.

- **Chapter or section:** the figure is the part's real structure (chapter → its sections →
  each section's key notion). Nothing in it is guessed.
- **Regular question:** the answer is written first. The model then lists the relations it
  contains ("router → GPU workers : sends requests"), and every component must appear in the
  answer or its sources, or it is dropped. A relation label that isn't in them is removed and
  the arrow stays unlabelled. Episteme then builds the Mermaid code itself, so it always
  parses.
- When too few relations survive, there is no diagram, and a note
  *"◇ No diagram this time…"* says so. A missing diagram is better than a made-up one.

## Analogies

Every grounded answer gets one everyday analogy, in the style of Feynman's lectures. It sits
in its own labelled box:

> 💡 **In plain words — an analogy, not from the source:** …

It is there to build intuition, so it is **not** a fact from your documents and the
verification ignores it. Turn it off with `RAG_ANALOGIES=0`.

## Reading the verification badge

Every grounded answer is checked **sentence by sentence** against the passages it was written
from:

| Check | What it catches |
|---|---|
| **Meaning**: semantic similarity with the closest 2-sentence window of the sources | statements the sources don't make |
| **Hard facts**: every number, product name and acronym must appear in the sources | wrong versions, dates, figures, invented tools ("invented by Google in 2015") |
| **Overstatement**: "always / never / all / only" (in 4 languages) must also be in the sources | "X always requires Y" when the book says "usually" |

The result is shown under the answer:

- **✓ VERIFIED 13/13** (green): every statement is backed by a passage;
- **◐ VERIFIED 34/38** (amber): 4 statements weren't found in the sources. Open *checks*
  to see each one with its reason. A click jumps to the sentence, which is underlined with
  dots in the answer.

An amber sentence isn't necessarily wrong. Often it's a generic linking sentence ("the choice
depends on the use case") or a correct fact the passage puts differently. It means **this
sentence isn't backed by your documents, so check it before you rely on it**. The green ones
are.

### How reliable is it?

These are measured on the live system, 10 questions across Inference Engineering, Kubernetes,
MLflow, SonarQube, Prometheus and HDFS, in English and French:

| Question | Mode | Time | Statements backed by the sources |
|---|---|---|---|
| Explain visually what inference engineering is (EN) | Graph-RAG + diagram | 20 s | 13/15 |
| Summarize chapter 1 of Inference Engineering (EN) | study, 5 sections | 60 s | 32/34 to 38/42 |
| Explique visuellement le chapitre 3 (FR) | study + structure figure | 2 min | 37/48 to 38/48 |
| ReplicaSet · MLflow · Prometheus · HDFS | Graph-RAG | 10–16 s | 25/25 |
| SonarQube quality gate · TTFT vs TPS | Graph-RAG | 16–27 s | 11/13 |
| The part about pods in the Kubernetes Notes | study, 9 page groups | 66 s | 38/40 |

**Overall: 162 of 185 statements (88%) backed by the sources**, up from 78% before the
fixes described here. Flagged statements were mostly filler ("this is crucial for
reliability"), and in the French chapter, details the model added that aren't in that chapter
(VLANs, SSDs).

- Fabricated statements (wrong name, number, absolute or meaning) planted in answers: all
  flagged in the test suite (`tests/test_verify.py`).
- A French or Spanish answer drawn from English sources is checked with a calibrated
  cross-lingual threshold, because faithful translations score lower than same-language
  paraphrases.

What it can't promise: the model is small (Qwen2.5-1.5B), so it sometimes compresses or
misattributes an idea. Verification doesn't make it right 99% of the time. It makes the
statements it couldn't back **visible**, so you know which ones to check.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `RAG_ANALOGIES` | `1` | add the labelled Feynman analogy |
| `RAG_TOP_K` | `5` | passages retrieved for a regular question |
| `RAG_USE_GRAPH` | `1` | Graph-RAG retrieval (multi-hop) instead of plain top-k |
