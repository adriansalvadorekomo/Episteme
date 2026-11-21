# Architecture diagrams

Two [C4 model](https://c4model.com) views of Episteme, each in a light and a dark theme.

| View | Shows | Files |
|---|---|---|
| **Level 2 · containers** | What runs, where the data lives, who calls whom | `c4-container-light.svg` · `c4-container-dark.svg` |
| **Level 3 · API components** | Inside the API: how a question becomes a checked answer | `c4-component-light.svg` · `c4-component-dark.svg` |

The SVGs are what the main README embeds. Light or dark follows the reader's GitHub theme
through `<picture>`. The `@2x.png` files are the same images at 2560 px wide, for places that
don't take SVG (LinkedIn, slides).

## How they're drawn

`build_c4.py` writes every element from code. It uses the same visual system as the author's
GitHub profile and portfolio: a labelled panel, hairlines, no shadows, and one orange signal
on the element that matters most. Labels are set in IBM Plex Mono and names in Schibsted
Grotesk. Logos sit on small black chips, like the portfolio's dot-matrix mark.

- **Text is converted to vector paths**, because an SVG shown through `<img>` (as on GitHub)
  can't load web fonts. The images therefore look the same in every browser, in the GitHub
  app and once exported.
- **Width checks.** The script fails if a label is wider than its box or if a box's content
  overflows, so a wording change can't silently break the layout.
- **Accessibility.** Each SVG has `role="img"` with a `<title>` and a one-sentence `<desc>`.

Rebuild (the fonts are SIL Open Font License; the mark is the portfolio's `public/brand/mark.svg`):

```bash
uv run --no-project --with fonttools --with brotli python docs/architecture/build_c4.py \
    --mono IBMPlexMono-Regular.woff2 --mono-medium IBMPlexMono-Medium.woff2 \
    --grotesk SchibstedGrotesk-variable.woff2 --mark mark.svg

# PNG exports
for f in docs/architecture/c4-*.svg; do rsvg-convert -z 2 "$f" -o "${f%.svg}@2x.png"; done
```

## Logos

`icons/` holds the logo sources, so the build runs offline:

- [Simple Icons](https://simpleicons.org) (CC0): Python, FastAPI, Astro, GreenSock, Mermaid,
  PyTorch, Hugging Face, NVIDIA, Docker, SQLite, Cloudflare, Obsidian, Qwen, uv, Meta (for
  FAISS), Apache Parquet, Markdown;
- the [Apache Iceberg](https://github.com/apache/iceberg) logo (icon part only) and the
  [SeaweedFS](https://github.com/seaweedfs/seaweedfs) icon, from their repositories.

Each trademark belongs to its owner. The logos identify the technology each element uses and
imply no endorsement.
