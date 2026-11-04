# Ingesting documents into Episteme

How to go from your own files (Markdown notes, PDFs, text) to something Episteme can
answer from, with citations back to the source. Everything below uses the `mise` tasks
defined in [`mise.toml`](../mise.toml).

## What you can ingest

| Format | Extensions | Notes |
|---|---|---|
| Markdown | `.md` | Obsidian notes work as-is. |
| PDF | `.pdf` | Text is extracted with PyMuPDF. **Scanned pages go through OCR automatically** (see [OCR](#ocr-scanned-pdfs-and-images)). |
| Plain text | `.txt` | |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.tif` `.bmp` | Read with OCR: slides, whiteboard photos, screenshots of docs. |

Example: a 29-page revision sheet (`Big Data 0.pdf`) gave 51,858 characters, which were
split into 73 chunks.

## The pipeline

```mermaid
flowchart TD
  files["Your .md / .pdf / .txt files"] -->|mise run ingest| bronze[("BRONZE · raw file<br/>lakehouse, deduplicated by sha256")]
  bronze -->|clean + split| silver[("SILVER · ~1000-character chunks<br/>Apache Iceberg table")]
  silver -->|multilingual-e5| faiss[("Embeddings → FAISS index")]
  faiss --> rag["RAG · vector search"]
  faiss -->|kNN links between chunks| graph[("Knowledge graph")]
  graph --> grag["Graph-RAG · multi-hop retrieval"]
  silver --> guard["Guardrails update automatically<br/>technology-name check · honest 'I don't know'"]
```

- **BRONZE.** The original file, stored in the lakehouse: S3 (SeaweedFS) when it's running,
  local storage otherwise. A file is identified by its sha256, so ingesting the same file
  twice is a no-op (`duplicate`).
- **SILVER.** Cleaned text split into ~1000-character chunks that overlap by 180
  characters, so context isn't lost at the boundaries.
- **Index + graph.** Only the *new* chunks are embedded (incremental). The knowledge graph
  is then rebuilt from the existing vectors, which takes about a second.
- **Guardrails.** The "is this technology really in the corpus?" check counts terms across
  the chunks, so new topics stop triggering "I don't know" once they're ingested.

## OCR: scanned PDFs and images

Episteme reads text that only exists as pixels, using Tesseract driven by PyMuPDF.

- **Only where needed.** A PDF page with no text layer but with images is recognized.
  Normal pages are extracted as usual, so a mixed PDF is handled page by page and costs
  nothing extra. It takes about 0.3–1 s per scanned page.
- **Four languages at once:** English, French, Spanish and German, with accents
  preserved (*système, réplication, tolérance*).
- **Images** are ingested as one-page documents, even when they hold only a short line
  of text.
- **You can see it happen.** The CLI prints `(OCR 2 p.)`, and the web chip says
  *added to big-data · OCR 2p*.

One-time setup for the French, Spanish and German models (about 5 MB, no sudo; English
comes from the system Tesseract):

```bash
mise run ocr:setup      # → data/tessdata/{eng,fra,spa,deu}.traineddata
```

| Setting | Default | Effect |
|---|---|---|
| `RAG_OCR` | `auto` | `auto`: scanned pages only · `force`: OCR every page · `off`: never |
| `RAG_OCR_DPI` | `300` | Render resolution for OCR (higher = sharper, slower) |
| `RAG_OCR_LANGS` | *(auto)* | e.g. `eng+fra`; by default every installed one of eng/fra/spa/deu |

Folders: images are **opt-in** with `--sync`, because a vault usually holds hundreds of
screenshots with no useful text:

```bash
mise run ingest -- --sync path/to/vault/ --domain big-data --with-images
```

A single image (`mise run ingest -- board.png --domain big-data`) or a web upload is
always read.

## Check a document before ingesting it

`mise run inspect` tells you what an ingest *would* do — normal extraction or OCR, and on
which pages — without writing anything (no lakehouse, no index):

```bash
mise run inspect -- "$HOME/Obsedian Vaults/outsmarted/0 - Inbox/Inference Engineering.pdf" --domain inference-engineering
```
```text
Inference Engineering.pdf
  type      mixed PDF  (259 pages · 235 with text · 1 scanned · 23 blank)
  route     OCR on 1 page(s) [1], <1s + normal extraction on 235
  ocr       languages eng+fra+spa+deu · 300 dpi · mode auto
  content   ≈331,088 characters → ≈404 chunks (+ OCR text, known after recognition)
  domain    inference-engineering
  verdict   ✓ normal + OCR ingest
```

| Type | Meaning | What ingest does |
|---|---|---|
| `text PDF` | every page has a text layer | normal extraction |
| `scanned PDF` | pages are images only | OCR on every page |
| `mixed PDF` | some of each (e.g. an image-only cover) | normal extraction + OCR on the listed pages |
| `image` | `.png`, `.jpg`… | OCR |
| `markdown` / `text` | `.md`, `.txt` | read as-is |
| `encrypted PDF` | password-protected | cannot be ingested |

It also reports the domain the file would get and whether it's **already in the
lakehouse** (an ingest would skip it as a duplicate). It works on folders too:
`mise run inspect -- --sync FOLDER/ [--with-images]` prints one report per file plus totals.

> `mise run help` prints every command, grouped by step.

## Commands, in order

```bash
# 1. (optional) start the S3 object store; without it the lakehouse uses local storage
mise run lake:up

# 2. one time only: create the bucket and the Iceberg tables
mise run lake:init

# 3. ingest your documents
mise run ingest -- path/to/note.md --domain data-engineering   # a single file
mise run ingest -- "path/to/Big Data 0.pdf" --domain big-data   # PDFs too
mise run ingest -- --sync path/to/my-notes/ --domain big-data  # a whole folder, one domain
mise run ingest -- --sync path/to/my-vault/                     # a whole folder, domain = subfolder

# 4. check what's in the lakehouse
mise run lake:status

# 5. serve it and ask questions
mise run dev          # API + web on http://localhost:4321
```

> **Restart the API after ingesting from the command line.** The API loads the index once
> at startup, so documents added with `mise run ingest` only become answerable after
> `mise run dev` (or `mise run api`) is restarted. Uploads made *in the web app* don't
> need this: they're indexed live.

### Single file vs. folder

| | `ingest -- file` | `ingest -- --sync folder/` |
|---|---|---|
| Indexing | Incremental: only the new chunks are embedded | One full rebuild once all files are in (re-embeds everything), **skipped when nothing new was added** |
| Paths | `<domain>/<filename>` | `<domain>/<relative path>`, so `a/README.md` and `b/README.md` stay distinct |
| Use it for | Adding a note or a PDF | First import of a vault, or many files at once |

## Domains

A domain is the topic a document is filed under (`big-data`, `data-engineering`,
`K8's`…). Sources and citations show it, and the knowledge-base panel groups by it.

- **`--domain X`** sets it explicitly, for a single file or for a whole `--sync` folder.
  This is recommended for the CLI.
- **Without `--domain`**:
  - a single file uses its **parent folder's name** (`0 - Inbox/Big Data 0.pdf` →
    `0 - Inbox`);
  - a `--sync` folder uses each file's **first-level subfolder**, so a vault groups by
    its top-level sections (`streaming/flink/notes.md` → `streaming`).
- **Uploads in the web app** (📎 in the composer, or Settings → *Manage documents*) get
  their domain **detected automatically**: each document is assigned to the closest
  existing domain.

## Adding documents from the web app

1. Click 📎 in the composer, or drop files onto it. A chip shows *reading… → added to
   ‹domain›*.
2. Or open **Settings → Manage documents** to import several files and see everything the
   assistant knows.

The new content can be asked about right away; no restart is needed.

## Checking that it worked

```bash
mise run lake:status
# e.g. {'connected': True, 'backend': 's3', 'documents': 537, 'chunks': 6981}
```

Then ask a question that only your document can answer, in **RAG** mode. The answer is
marked **GROUNDED · GRAPH-RAG**, and its *Sources* list cites your file
(e.g. `big-data/Big Data 0.pdf`).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `empty` on a scanned PDF or an image | OCR found no text, or Tesseract isn't installed | Check `tesseract --version`; run `mise run ocr:setup`; try `RAG_OCR_DPI=400` for small print |
| OCR text without accents (`systeme`) | Only the English model is installed | `mise run ocr:setup` |
| `duplicate` | This exact file is already in the lakehouse | Nothing to do. An *edited* file has a new sha256 and is ingested as a new document |
| New content isn't answered | API started before a CLI ingest | Restart the API (`mise run dev`) |
| RAG says "I don't know" about a tool that *is* in your notes | The tool is mentioned in fewer than 3 chunks, so the guardrail treats it as not covered | Ingest more material on it, or lower `RAG_MIN_DF` |
| Documents disappeared after starting/stopping S3 | The `auto` backend uses S3 when it's reachable and local storage otherwise; each has its own catalog | Pin one: `export RAG_LAKEHOUSE_BACKEND=s3` (or `local`) |

## Current limitations

- OCR is tuned for printed text. Handwriting and very low-resolution photos read poorly.
- Re-ingesting an edited file adds a new document; the old version's chunks stay. There
  is no delete or replace yet.
- `--sync` re-embeds the whole corpus. That's fine for thousands of chunks, but slower as
  the lakehouse grows.
