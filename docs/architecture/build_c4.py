"""Draw Episteme's C4 architecture diagrams (docs/architecture/*.svg).

Two views, each in a light and a dark theme:
  c4-container-{light,dark}.svg   C4 level 2: what runs, where the data lives, who calls whom
  c4-component-{light,dark}.svg   C4 level 3: inside the API, how a question becomes a checked answer

Design: the same system as the author's GitHub profile and portfolio
(adriansalvadorekomodev.vercel.app): a labelled panel, hairlines, no shadows, one orange
signal, IBM Plex Mono for labels and Schibsted Grotesk for names. Technology logos sit on
small black "display" chips, like the dot-matrix mark.

Text is converted to vector paths, so the images look the same in every browser, on GitHub
(an SVG shown through <img> cannot load web fonts) and once exported to PNG for LinkedIn.

usage:
  uv run --no-project --with fonttools --with brotli python docs/architecture/build_c4.py \\
      --mono <IBM Plex Mono Regular .woff2/.ttf> --mono-medium <IBM Plex Mono Medium> \\
      --grotesk <Schibsted Grotesk variable> --mark <portfolio mark.svg>

Logos (docs/architecture/icons/): Simple Icons (CC0) plus the Apache Iceberg and SeaweedFS
project logos; each trademark belongs to its owner.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

HERE = Path(__file__).resolve().parent
ICONS = HERE / "icons"

ORANGE = "#ff4f00"
DISPLAY = "#121212"  # logo chips: the mark's black display, in both themes
THEMES = {
    # "aluminium edition" — same tokens as the profile images
    "light": dict(panel="#ebe9e4", ink="#121212", ink2="#5c5952", hair="#cdc9c0", node="#f4f2ee",
                  ext="#e3e0d9", zone="rgba(18,18,18,0.025)", chip_line="#121212"),
    # "black edition"
    "dark": dict(panel="#121212", ink="#ebe9e4", ink2="#9b978f", hair="#2e2e2e", node="#1b1b1b",
                 ext="#161616", zone="rgba(235,233,228,0.025)", chip_line="#3a3a3a"),
}

# Simple Icons brand colours (https://simpleicons.org); dark ones fall back to light ink on the chip.
BRAND = {
    "python": "3776AB", "fastapi": "009688", "astro": "BC52EE", "greensock": "88CE02",
    "mermaid": "FF3670", "pytorch": "EE4C2C", "huggingface": "FFD21E", "nvidia": "76B900",
    "docker": "2496ED", "sqlite": "003B57", "cloudflare": "F38020", "obsidian": "7C3AED",
    "qwen": "6950EF", "uv": "DE5FE9", "meta": "0467DF", "apacheparquet": "50ABF1",
    "markdown": "000000",
}

# Tabler Icons (MIT), stroked glyphs for things that have no logo.
GLYPHS = {
    "user": ["M8 7a4 4 0 1 0 8 0a4 4 0 0 0 -8 0", "M6 21v-2a4 4 0 0 1 4 -4h4a4 4 0 0 1 4 4v2"],
    "terminal": ["M5 7l5 5l-5 5", "M12 19l7 0"],
}


def num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


# ── text as vector paths ───────────────────────────────────────────────────
class Face:
    def __init__(self, path: str, tag: str, defs: dict[str, str], wght: float | None = None):
        f = TTFont(path)
        if wght is not None and "fvar" in f:
            f = instancer.instantiateVariableFont(f, {"wght": wght})
        self.f, self.tag, self.defs = f, tag, defs
        self.gs = f.getGlyphSet()
        self.cmap = f.getBestCmap()
        self.upm = f["head"].unitsPerEm
        self.hmtx = f["hmtx"]

    def _g(self, ch: str) -> str:
        g = self.cmap.get(ord(ch))
        if g is None:
            raise KeyError(f"{self.tag}: no glyph for {ch!r}")
        return g

    def width(self, text: str, size: float, track: float = 0) -> float:
        adv = sum(self.hmtx[self._g(c)][0] for c in text) * size / self.upm
        return adv + track * size * max(0, len(text) - 1)

    def text(self, s: str, x: float, y: float, size: float, fill: str,
             anchor: str = "start", track: float = 0) -> str:
        w = self.width(s, size, track)
        if anchor == "middle":
            x -= w / 2
        elif anchor == "end":
            x -= w
        k = size / self.upm
        uses = []
        for ch in s:
            g = self._g(ch)
            gid = f"{self.tag}{self.f.getGlyphID(g)}"
            if gid not in self.defs:
                pen = SVGPathPen(self.gs)
                self.gs[g].draw(pen)
                self.defs[gid] = pen.getCommands()
            if self.defs[gid]:
                uses.append(f'<use href="#{gid}" x="{num(x / k)}" y="{num(-y / k)}"/>')
            x += self.hmtx[g][0] * k + track * size
        return f'<g fill="{fill}" transform="scale({k:.7g} {-k:.7g})">{"".join(uses)}</g>'


# ── logos ──────────────────────────────────────────────────────────────────
def _luminance(hex6: str) -> float:
    def ch(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex6[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _iceberg(x: float, y: float, s: float) -> str:
    src = (ICONS / "apache-iceberg.svg").read_text()
    paths = []
    for m in re.finditer(r'<path ([^>]*?)d="M\s*([\d.]+)[^"]*"[^>]*/>', src):
        if float(m.group(2)) >= 595:  # the round icon; the wordmark sits left of it
            paths.append(m.group(0))
    return (f'<svg x="{num(x)}" y="{num(y)}" width="{num(s)}" height="{num(s)}" viewBox="601 19 180 180" '
            f'style="fill-rule:evenodd">{"".join(paths)}</svg>')


def _seaweedfs(x: float, y: float, s: float) -> str:
    src = (ICONS / "seaweedfs.svg").read_text()
    styles = {c: v.strip() for c, v in re.findall(r"\.(cls-\d+)\s*\{\s*fill:\s*([^;}]+);?\s*\}", src)}
    styles = {c: re.sub(r"url\(#([^)]+)\)", r"url(#sw-\1)", v) for c, v in styles.items()}
    body = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S).group(1)
    body = re.sub(r"<style>.*?</style>", "", body, flags=re.S)
    body = re.sub(r'class="(cls-\d+)"', lambda m: f'fill="{styles[m.group(1)]}"', body)
    body = re.sub(r'id="([^"]+)"', r'id="sw-\1"', body)
    body = re.sub(r'(xlink:href|href)="#([^"]+)"', r'\1="#sw-\2"', body)
    return f'<svg x="{num(x)}" y="{num(y)}" width="{num(s)}" height="{num(s)}" viewBox="0 0 512 512">{body}</svg>'


def logo(name: str, x: float, y: float, s: float) -> str:
    """Logo of side `s` at (x, y), drawn for a dark chip."""
    if name == "iceberg":
        return _iceberg(x, y, s)
    if name == "seaweedfs":
        return _seaweedfs(x, y, s)
    if name in GLYPHS:
        k = s / 24
        d = "".join(f'<path d="{p}"/>' for p in GLYPHS[name])
        return (f'<g transform="translate({num(x)} {num(y)}) scale({k:.6g})" fill="none" stroke="#ebe9e4" '
                f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{d}</g>')
    src = (ICONS / f"{name}.svg").read_text()
    d = re.search(r'<path d="([^"]+)"', src).group(1)
    color = "#" + BRAND[name]
    if _contrast(BRAND[name], DISPLAY[1:]) < 2.6:
        color = "#ebe9e4"
    return f'<path transform="translate({num(x)} {num(y)}) scale({s / 24:.6g})" fill="{color}" d="{d}"/>'


# ── drawing context ────────────────────────────────────────────────────────
@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def r(self) -> float:
        return self.x + self.w

    @property
    def b(self) -> float:
        return self.y + self.h


@dataclass
class Canvas:
    theme: str
    fonts: dict
    W: int
    H: int
    t: dict = field(init=False)
    zones: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    top: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.t = THEMES[self.theme]

    # text helpers ---------------------------------------------------------
    def mono(self, s, x, y, size=12, fill=None, anchor="start", track=0.0):
        return self.fonts["mono"].text(s, x, y, size, fill or self.t["ink2"], anchor, track)

    def monom(self, s, x, y, size=12, fill=None, anchor="start", track=0.0):
        return self.fonts["mono_md"].text(s, x, y, size, fill or self.t["ink"], anchor, track)

    def name(self, s, x, y, size=16, fill=None, anchor="start"):
        return self.fonts["grot_sb"].text(s, x, y, size, fill or self.t["ink"], anchor, -0.005)

    def body(self, s, x, y, size=13, fill=None, anchor="start"):
        return self.fonts["grot"].text(s, x, y, size, fill or self.t["ink2"], anchor)

    def fits(self, face: str, s: str, size: float, width: float, track: float = 0):
        w = self.fonts[face].width(s, size, track)
        assert w <= width, f"text too wide ({w:.0f} > {width:.0f}px): {s!r}"

    # elements -------------------------------------------------------------
    def chips(self, names: list[str], right: float, y: float, size: int = 28) -> str:
        out, x = [], right - size
        for n in reversed(names):
            out.append(f'<rect x="{num(x)}" y="{num(y)}" width="{size}" height="{size}" rx="7" '
                       f'fill="{DISPLAY}" stroke="{self.t["chip_line"]}" stroke-width="1"/>')
            pad = 6 if n not in ("seaweedfs", "iceberg") else 4
            out.append(logo(n, x + pad, y + pad, size - 2 * pad))
            x -= size + 6
        return "".join(out)

    def node(self, b: Box, kind: str, eyebrow: str, title: str, tech: str | None, desc: list[str],
             logos: list[str] = (), focal: bool = False, dashed: bool = False):
        t = self.t
        fill = t["ext"] if kind == "external" else t["node"]
        stroke, sw = (ORANGE, 2) if focal else (t["ink2"] if kind != "external" else t["hair"], 1.2)
        if kind == "external":
            stroke, sw = t["ink2"], 1
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        g = [f'<rect x="{num(b.x)}" y="{num(b.y)}" width="{num(b.w)}" height="{num(b.h)}" rx="10" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>']
        if kind == "store":  # a data store: a second hairline under the top edge (C4 "cylinder" cue)
            g.append(f'<path d="M{num(b.x + 1)} {num(b.y + 8)} H{num(b.r - 1)}" stroke="{t["hair"]}" stroke-width="1"/>')
        px = b.x + 16
        # eyebrow with the orange signal on the focal element
        ex = px
        if focal:
            g.append(f'<circle cx="{num(px + 4)}" cy="{num(b.y + 24)}" r="4" fill="{ORANGE}"/>')
            ex = px + 14
        g.append(self.mono(eyebrow.upper(), ex, b.y + 28, 11, t["ink2"], track=0.1))
        if logos:
            g.append(self.chips(list(logos), b.r - 12, b.y + 12))
        avail = b.w - 32
        y = b.y + 56
        self.fits("grot_sb", title, 16, avail)
        g.append(self.name(title, px, y))
        if tech:
            y += 18
            self.fits("mono", tech, 12, avail)
            g.append(self.mono(tech, px, y, 12, t["ink2"]))
        last = y
        y += 6
        for line in desc:
            y += 18
            self.fits("grot", line, 13, avail)
            g.append(self.body(line, px, y, 13, t["ink"]))
            last = y
        y = last
        assert y + 12 <= b.b, f"{title}: content overflows the box ({y + 12} > {b.b})"
        self.nodes.append("".join(g))

    def zone(self, b: Box, label: str, sub: str | None = None, at: str = "top"):
        t = self.t
        g = [f'<rect x="{num(b.x)}" y="{num(b.y)}" width="{num(b.w)}" height="{num(b.h)}" rx="14" '
             f'fill="none" stroke="{t["ink2"]}" stroke-width="1" stroke-dasharray="6 5" stroke-opacity="0.7"/>']
        lw = self.fonts["mono_md"].width(label, 12, 0.1) + (self.fonts["mono"].width("  " + sub, 12) if sub else 0)
        ly = b.y if at == "top" else b.b
        g.append(f'<rect x="{num(b.x + 16)}" y="{num(ly - 10)}" width="{num(lw + 16)}" height="20" fill="{t["panel"]}"/>')
        g.append(self.monom(label, b.x + 24, ly + 4, 12, t["ink"], track=0.1))
        if sub:
            g.append(self.mono("  " + sub, b.x + 24 + self.fonts["mono_md"].width(label, 12, 0.1), ly + 4, 12))
        self.zones.append("".join(g))

    def arrow(self, pts: list[tuple[float, float]], dashed: bool = False, color: str | None = None):
        """Orthogonal connector through `pts` with r=8 rounded corners and a drawn arrowhead."""
        color = color or self.t["ink2"]
        d = [f"M{num(pts[0][0])} {num(pts[0][1])}"]
        for i in range(1, len(pts) - 1):
            (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
            r = min(8, abs(x1 - x0 + y1 - y0) / 2, abs(x2 - x1 + y2 - y1) / 2)
            sx = (x1 > x0) - (x1 < x0); sy = (y1 > y0) - (y1 < y0)
            tx = (x2 > x1) - (x2 < x1); ty = (y2 > y1) - (y2 < y1)
            d.append(f"L{num(x1 - sx * r)} {num(y1 - sy * r)} Q{num(x1)} {num(y1)} {num(x1 + tx * r)} {num(y1 + ty * r)}")
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ux = (xb > xa) - (xb < xa); uy = (yb > ya) - (yb < ya)
        d.append(f"L{num(xb - ux * 7)} {num(yb - uy * 7)}")  # stop at the arrowhead
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.lines.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>')
        # arrowhead: 9 long, 7 wide, tip on the box edge
        bx, by = xb - ux * 9, yb - uy * 9
        px, py = -uy * 3.5, ux * 3.5
        self.lines.append(f'<path d="M{num(xb)} {num(yb)} L{num(bx + px)} {num(by + py)} L{num(bx - px)} {num(by - py)} Z" fill="{color}"/>')

    def hop_arrow(self, pts: list[tuple[float, float]], hop: tuple[float, float], dashed: bool = False):
        """Like arrow(), but the vertical segment through `hop` jumps over a crossing line (r=8)."""
        hx, hy = hop
        i = next(k for k in range(len(pts) - 1) if pts[k][0] == pts[k + 1][0] == hx)
        down = pts[i + 1][1] > pts[i][1]
        a, b = (hx, hy - 8 if down else hy + 8), (hx, hy + 8 if down else hy - 8)
        first, second = pts[: i + 1] + [a], [b] + pts[i + 1:]
        color = self.t["ink2"]
        self.arrow(second, dashed)
        # the first part ends in a half-circle bump instead of an arrowhead
        d = [f"M{num(first[0][0])} {num(first[0][1])}"]
        for k in range(1, len(first) - 1):
            (x0, y0), (x1, y1), (x2, y2) = first[k - 1], first[k], first[k + 1]
            r = min(8, abs(x1 - x0 + y1 - y0) / 2, abs(x2 - x1 + y2 - y1) / 2)
            sx = (x1 > x0) - (x1 < x0); sy = (y1 > y0) - (y1 < y0)
            tx = (x2 > x1) - (x2 < x1); ty = (y2 > y1) - (y2 < y1)
            d.append(f"L{num(x1 - sx * r)} {num(y1 - sy * r)} Q{num(x1)} {num(y1)} {num(x1 + tx * r)} {num(y1 + ty * r)}")
        d.append(f"L{num(a[0])} {num(a[1])} A8 8 0 0 {1 if down else 0} {num(b[0])} {num(b[1])}")
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.lines.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>')

    def label(self, lines: list[tuple[str, bool]], x: float, y: float, anchor: str = "middle"):
        """Masked connector label; lines = [(text, is_tech)]. (x, y) = baseline of the first line."""
        t = self.t
        ws = [self.fonts["mono_md" if not tech else "mono"].width(s, 12) for s, tech in lines]
        w = max(ws) + 12
        h = 16 * len(lines) + 4
        x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
        g = [f'<rect x="{num(x0)}" y="{num(y - 14)}" width="{num(w)}" height="{num(h)}" rx="3" fill="{t["panel"]}"/>']
        tx = x0 + w / 2
        for i, (s, tech) in enumerate(lines):
            face = self.mono if tech else self.monom
            g.append(face(s, tx, y + 16 * i, 12, t["ink2"] if tech else t["ink"], "middle"))
        self.labels.append("".join(g))
        return (x0, y - 14, w, h)

    # page -----------------------------------------------------------------
    def frame(self, mark: str, left: str, right: str, title: str, subtitle: str):
        t = self.t
        self.zones.insert(0, "".join([
            f'<rect x="1" y="1" width="{self.W - 2}" height="{self.H - 2}" rx="20" fill="{t["panel"]}" stroke="{t["hair"]}" stroke-width="2"/>',
            f'<svg x="32" y="20" width="32" height="32" viewBox="0 0 64 64">{mark}</svg>',
            self.mono(left, 76, 42, 14, t["ink2"], track=0.03),
            self.mono(right, self.W - 32, 42, 14, t["ink2"], "end", 0.03),
            f'<rect x="32" y="64" width="{self.W - 64}" height="1" fill="{t["hair"]}"/>',
            self.fonts["grot_md"].text(title, 32, 112, 32, t["ink"], track=-0.01),
            self.mono(subtitle, 32, 140, 13, t["ink2"]),
        ]))

    def readout(self, cols: list[tuple[str, str]], right: float, y: float, gap: float = 40):
        """Stat columns, right-aligned: small mono label above a value."""
        t = self.t
        x = right
        out = []
        for label, value in reversed(cols):
            w = max(self.fonts["mono"].width(label, 11, 0.1), self.fonts["mono_md"].width(value, 16))
            x0 = x - w
            out.append(f'<rect x="{num(x0)}" y="{num(y - 6)}" width="{num(w)}" height="1" fill="{t["hair"]}"/>')
            out.append(self.mono(label, x0, y + 12, 11, t["ink2"], track=0.1))
            out.append(self.monom(value, x0, y + 36, 16, t["ink"]))
            x = x0 - gap
        self.top.append("".join(out))

    def legend(self, y: float, items: list[tuple[str, str]], note: str | None = None):
        t = self.t
        g = [f'<rect x="32" y="{num(y)}" width="{self.W - 64}" height="1" fill="{t["hair"]}"/>',
             self.mono("LEGEND", 32, y + 32, 11, t["ink2"], track=0.14)]
        x = 120
        for kind, text in items:
            cy = y + 28
            if kind == "person":
                g.append(f'<rect x="{x}" y="{cy - 8}" width="24" height="16" rx="4" fill="{t["node"]}" stroke="{t["ink2"]}" stroke-width="1.2"/>')
            elif kind == "container":
                g.append(f'<rect x="{x}" y="{cy - 8}" width="24" height="16" rx="4" fill="{t["node"]}" stroke="{t["ink2"]}" stroke-width="1.2"/>')
            elif kind == "focal":
                g.append(f'<rect x="{x}" y="{cy - 8}" width="24" height="16" rx="4" fill="{t["node"]}" stroke="{ORANGE}" stroke-width="2"/>')
            elif kind == "store":
                g.append(f'<rect x="{x}" y="{cy - 8}" width="24" height="16" rx="4" fill="{t["node"]}" stroke="{t["ink2"]}" stroke-width="1.2"/>'
                         f'<path d="M{x + 1} {cy - 4} H{x + 23}" stroke="{t["hair"]}"/>')
            elif kind == "external":
                g.append(f'<rect x="{x}" y="{cy - 8}" width="24" height="16" rx="4" fill="{t["ext"]}" stroke="{t["ink2"]}" stroke-width="1"/>')
            elif kind == "call":
                g.append(f'<path d="M{x} {cy} H{x + 17}" stroke="{t["ink2"]}" stroke-width="1.5"/>'
                         f'<path d="M{x + 24} {cy} L{x + 15} {cy - 3.5} L{x + 15} {cy + 3.5} Z" fill="{t["ink2"]}"/>')
            elif kind == "occasional":
                g.append(f'<path d="M{x} {cy} H{x + 17}" stroke="{t["ink2"]}" stroke-width="1.5" stroke-dasharray="5 4"/>'
                         f'<path d="M{x + 24} {cy} L{x + 15} {cy - 3.5} L{x + 15} {cy + 3.5} Z" fill="{t["ink2"]}"/>')
            g.append(self.mono(text, x + 32, cy + 4, 12, t["ink"]))
            x += 32 + self.fonts["mono"].width(text, 12) + 28
        if note:
            g.append(self.mono(note, self.W - 32, y + 32, 11, t["ink2"], "end"))
        self.top.append("".join(g))

    def svg(self, slug: str, title: str, desc: str, defs: dict[str, str]) -> str:
        glyphs = "".join(f'<path id="{k}" d="{v}"/>' for k, v in defs.items() if v)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" width="{self.W}" '
                f'height="{self.H}" role="img" aria-labelledby="{slug}-title {slug}-desc">'
                f'<title id="{slug}-title">{title}</title><desc id="{slug}-desc">{desc}</desc>'
                f'<defs>{glyphs}</defs>'
                + "".join(self.zones + self.lines + self.labels + self.nodes + self.top) + "</svg>")


# ── C4 level 2 · containers ────────────────────────────────────────────────
def container_view(cv: Canvas):
    # grid (multiples of 4): left column · system boundary with three columns
    learner = Box(32, 320, 168, 144)
    docs = Box(32, 528, 168, 128)
    tunnel = Box(280, 160, 216, 96)
    hub = Box(600, 160, 256, 96)
    web = Box(280, 320, 216, 144)
    api = Box(600, 320, 256, 144)
    cli = Box(280, 528, 216, 128)
    index = Box(600, 528, 256, 128)
    lake = Box(960, 320, 256, 336)

    cv.zone(Box(248, 280, 1000, 464), "EPISTEME", "software system", at="bottom")

    # connectors (drawn under the boxes) and their labels
    cv.arrow([(learner.r, 376), (web.x, 376)])
    cv.label([("asks", False), ("HTTPS", True)], 240, 340)
    cv.arrow([(learner.r, 440), (228, 440), (228, 568), (cli.x, 568)])
    cv.label([("ingests", False), ("mise", True)], 216, 488, "end")
    cv.arrow([(docs.r, 632), (cli.x, 632)])
    cv.label([("reads", False), ("+ OCR", True)], 248, 600)
    cv.arrow([(tunnel.cx, tunnel.b), (tunnel.cx, web.y)], dashed=True)
    cv.label([("forwards", False), ("HTTPS", True)], tunnel.cx + 12, 288, "start")
    cv.arrow([(web.r, 392), (api.x, 392)])
    cv.label([("/ask /ingest", False), ("JSON, HTTP", True)], 548, 356)
    cv.arrow([(api.cx, api.y), (api.cx, hub.b)], dashed=True)
    cv.label([("weights", False), ("first run", True)], api.cx + 12, 288, "start")
    cv.arrow([(api.r, 392), (lake.x, 392)])
    cv.label([("reads, writes", False), ("S3, Iceberg", True)], 908, 356)
    cv.arrow([(api.cx, api.b), (api.cx, index.y)])
    cv.label([("retrieves", False), ("top-k + PageRank", True)], api.cx + 12, 488, "start")
    cv.arrow([(cli.r, 592), (index.x, 592)])
    cv.label([("rebuilds", False), ("from Silver", True)], 548, 556)
    cv.arrow([(cli.cx, cli.b), (cli.cx, 696), (lake.cx, 696), (lake.cx, lake.b)])
    cv.label([("writes Bronze + Silver · S3 API, pyiceberg", False)], 728, 684)

    # elements
    cv.node(learner, "person", "Person", "Learner", None,
            ["Asks in EN, ES, FR, DE;", "adds notes and PDFs"], ["user"])
    cv.node(docs, "external", "External", "Course material", "md · pdf · images",
            ["Obsidian notes, PDFs,", "scanned slides"], ["obsidian", "markdown"])
    cv.node(tunnel, "external", "External · demos", "Cloudflare Tunnel", None,
            ["Public demo URLs"], ["cloudflare"], dashed=True)
    cv.node(hub, "external", "External", "Hugging Face Hub", None,
            ["Qwen2.5-1.5B · multilingual-e5"], ["huggingface"])
    cv.node(web, "container", "Container", "Web app", "Astro 5 · GSAP · Mermaid",
            ["Chat with citations, checks,", "diagrams, live uploads"], ["astro", "greensock", "mermaid"])
    cv.node(api, "container", "Container", "API + RAG engine", "FastAPI · PyTorch · CUDA",
            ["Graph-RAG, chapter mode,", "sentence-level verification"], ["fastapi", "pytorch", "qwen", "nvidia"],
            focal=True)
    cv.node(cli, "container", "Container", "CLI & jobs", "Python · uv · mise",
            ["Ingest with OCR, rebuild,", "QLoRA fine-tuning"], ["python", "uv"])
    cv.node(index, "store", "Data store", "Serving index", "FAISS · 384-d · kNN graph",
            ["Loaded by the API at start;", "PageRank walks the graph"], ["meta"])
    lakehouse(cv, lake)

    cv.readout([("VERIFIED STATEMENTS", "162 / 185 · 88%"), ("TESTS", "96 pytest"),
                ("RUNS ON", "RTX 4060 · 8 GB")], 1248, 96)
    cv.legend(768, [("person", "person"), ("container", "container"), ("focal", "where answers are built"),
                    ("store", "data store"), ("external", "external system"), ("call", "call / read / write"),
                    ("occasional", "occasional")])


def lakehouse(cv: Canvas, b: Box):
    """Data store with its two medallion layers as cells."""
    t = cv.t
    g = [f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="10" fill="{t["node"]}" stroke="{t["ink2"]}" stroke-width="1.2"/>',
         f'<path d="M{b.x + 1} {b.y + 8} H{b.r - 1}" stroke="{t["hair"]}" stroke-width="1"/>',
         cv.mono("DATA STORE", b.x + 16, b.y + 28, 11, t["ink2"], track=0.1),
         cv.name("Lakehouse", b.x + 16, b.y + 60),
         cv.mono("medallion · Bronze, Silver", b.x + 16, b.y + 80, 12)]
    cells = [
        (Box(b.x + 12, b.y + 96, b.w - 24, 112), "BRONZE", "Object storage", "SeaweedFS · S3 API",
         ["Raw files, sha256-deduped"], ["seaweedfs", "docker"]),
        (Box(b.x + 12, b.y + 216, b.w - 24, 108), "SILVER", "Iceberg tables", "SQLite catalog · Parquet",
         ["documents · chunks"], ["iceberg", "sqlite", "apacheparquet"]),
    ]
    for c, eb, name, tech, desc, logos in cells:
        g.append(f'<rect x="{c.x}" y="{c.y}" width="{c.w}" height="{c.h}" rx="8" fill="{t["panel"]}" stroke="{t["hair"]}" stroke-width="1"/>')
        g.append(cv.monom(eb, c.x + 12, c.y + 24, 11, t["ink2"], track=0.12))
        g.append(cv.chips(logos, c.r - 10, c.y + 8, 24))
        cv.fits("grot_sb", name, 15, c.w - 24)
        g.append(cv.fonts["grot_sb"].text(name, c.x + 12, c.y + 56, 15, t["ink"]))
        cv.fits("mono", tech, 12, c.w - 24)
        g.append(cv.mono(tech, c.x + 12, c.y + 74, 12))
        for i, line in enumerate(desc):
            cv.fits("grot", line, 13, c.w - 24)
            g.append(cv.body(line, c.x + 12, c.y + 96 + 18 * i, 13, t["ink"]))
    cv.nodes.append("".join(g))


# ── C4 level 3 · components of the API ─────────────────────────────────────
def component_view(cv: Canvas):
    c1, c2, c3, c4 = 64, 372, 680, 988           # four columns, 228 wide, 80 apart
    W = 228
    web = Box(c1, 160, W, 88)
    index = Box(c2, 160, W, 88)
    router = Box(c1, 312, W, 128)
    retr = Box(c2, 312, W, 128)
    gen = Box(c3, 312, W, 128)
    post = Box(c4, 312, W, 128)
    study = Box(c2, 520, W, 128)
    emb = Box(c3, 520, W, 128)
    verif = Box(c4, 520, W, 128)
    bronze = Box(c2, 712, W, 88)

    cv.zone(Box(32, 280, 1216, 400), "API + RAG ENGINE", "container · FastAPI · one process", at="bottom")

    cv.arrow([(web.cx, web.b), (web.cx, router.y)])
    cv.label([("POST /ask", False), ("answer + checks", True)], web.cx + 12, 268, "start")
    cv.arrow([(retr.cx, retr.y), (retr.cx, index.b)])
    cv.label([("reads", False), ("vectors, graph", True)], retr.cx + 12, 268, "start")
    cv.arrow([(router.r, 360), (retr.x, 360)])
    cv.label([("question", False)], (router.r + retr.x) / 2, 340)
    cv.arrow([(router.cx, router.b), (router.cx, 584), (study.x, 584)])
    cv.label([("chapter", False), ("request", True)], (router.cx + study.x) / 2 + 16, 548)
    cv.arrow([(study.cx, study.b), (study.cx, bronze.y)])
    cv.label([("reads pages", False), ("S3, OCR", True)], study.cx + 12, 676, "start")
    cv.arrow([(retr.r, 360), (gen.x, 360)])
    cv.label([("passages", False), ("cited [n]", True)], (retr.r + gen.x) / 2, 324)
    cv.arrow([(study.r - 36, study.y), (study.r - 36, 480), (gen.x + 40, 480), (gen.x + 40, gen.b)])
    cv.label([("section summaries", False)], 556, 466, "end")
    cv.arrow([(gen.r, 360), (post.x, 360)])
    cv.label([("answer", False), ("draft", True)], (gen.r + post.x) / 2, 324)
    cv.arrow([(post.cx, post.b), (post.cx, verif.y)])
    cv.label([("answer", False), ("+ analogy, figure", True)], post.cx - 12, 472, "end")
    cv.arrow([(verif.x, 584), (emb.r, 584)])
    cv.label([("encodes", False), ("claims", True)], (emb.r + verif.x) / 2, 548)
    cv.hop_arrow([(retr.r, 416), (632, 416), (632, 552), (emb.x, 552)], hop=(632, 480))

    cv.node(web, "container", "Container", "Web app", "Astro · /api proxy", [], ["astro"])
    cv.node(index, "store", "Data store", "Serving index", "FAISS · kNN graph", [], ["meta"])
    cv.node(bronze, "store", "Data store · Bronze", "Object storage", "SeaweedFS · S3 API", [], ["seaweedfs"])
    cv.node(router, "container", "Component", "Router & scope guard", "pipeline.py",
            ["Language, off-topic guard,", "chapter or question?"])
    cv.node(retr, "container", "Component", "Graph-RAG retriever", "graph.py · index.py",
            ["FAISS top-k seeds PageRank", "over the kNN graph"], ["meta"])
    cv.node(study, "container", "Component", "Chapter mode", "study.py · PyMuPDF",
            ["Summarises every section,", "then the overview"])
    cv.node(gen, "container", "Component", "Generator", "llm.py · Qwen2.5-1.5B",
            ["Cited answer, in the", "question's language"], ["qwen", "pytorch"])
    cv.node(post, "container", "Component", "Analogy + diagram", "diagrams.py",
            ["Mermaid built from relations", "found in the sources"], ["mermaid"])
    cv.node(emb, "container", "Component", "Embedder", "embed.py · e5-small",
            ["Retrieval, scope guard", "and verification"], ["huggingface"])
    cv.node(verif, "container", "Component", "Sentence verifier", "verify.py",
            ["Meaning, hard facts and", "absolutes, per sentence"], focal=True)

    cv.readout([("REGULAR ANSWER", "10–27 s"), ("CHAPTER SUMMARY", "1–2 min"), ("VERIFICATION", "< 1 s")],
               1248, 96)
    cv.legend(824, [("container", "component / container"), ("focal", "what makes answers checkable"),
                    ("store", "data store"), ("call", "call / read")])


# ── main ───────────────────────────────────────────────────────────────────
VIEWS = {
    "container": dict(
        fn=container_view, W=1280, H=816,
        right="c4 model · level 2 · containers",
        title="Episteme",
        subtitle="grounded RAG study assistant · one workstation, local model, every sentence checked",
        alt_title="Episteme — C4 container view",
        desc=("C4 container diagram: a learner uses the Astro web app, which calls the FastAPI API and RAG "
              "engine running Qwen2.5 on CUDA; the engine retrieves from a FAISS index and kNN graph and reads "
              "and writes a lakehouse of SeaweedFS object storage (Bronze) and Apache Iceberg tables (Silver); "
              "a Python CLI ingests course material with OCR and rebuilds the index; model weights come from "
              "the Hugging Face Hub and a Cloudflare tunnel exposes demos."),
    ),
    "component": dict(
        fn=component_view, W=1280, H=872,
        right="c4 model · level 3 · API components",
        title="Inside the API",
        subtitle="how a question becomes a checked answer · regular questions on top, chapter requests below",
        alt_title="Episteme — C4 component view of the API",
        desc=("C4 component diagram of the Episteme API: the web app posts a question to the router and scope "
              "guard; regular questions go to the Graph-RAG retriever, which reads the FAISS index and kNN graph, "
              "and chapter requests go to chapter mode, which reads the pages from object storage; the generator "
              "(Qwen2.5-1.5B) writes a cited answer, the analogy and diagram step adds a figure built from "
              "relations found in the sources, and the sentence verifier checks every sentence with the "
              "multilingual-e5 embedder before the answer returns."),
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono", required=True)
    ap.add_argument("--mono-medium", required=True)
    ap.add_argument("--grotesk", required=True)
    ap.add_argument("--mark", required=True)
    ap.add_argument("--only", choices=list(VIEWS))
    a = ap.parse_args()
    mark = re.search(r"<svg[^>]*>(.*)</svg>", Path(a.mark).read_text(), re.S).group(1)
    mark = re.sub(r"<!--.*?-->", "", mark, flags=re.S)
    for view, spec in VIEWS.items():
        if a.only and view != a.only:
            continue
        for theme in THEMES:
            defs: dict[str, str] = {}
            fonts = {
                "mono": Face(a.mono, "m", defs),
                "mono_md": Face(a.mono_medium, "n", defs),
                "grot": Face(a.grotesk, "g", defs, 400),
                "grot_md": Face(a.grotesk, "h", defs, 560),
                "grot_sb": Face(a.grotesk, "s", defs, 640),
            }
            cv = Canvas(theme, fonts, spec["W"], spec["H"])
            cv.frame(mark, "adriansalvadorekomo / episteme", spec["right"], spec["title"], spec["subtitle"])
            spec["fn"](cv)
            slug = f"c4-{view}-{theme}"
            out = HERE / f"{slug}.svg"
            out.write_text(cv.svg(slug, spec["alt_title"], spec["desc"], defs), encoding="utf-8")
            print(f"wrote {out.relative_to(HERE.parent.parent)}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
