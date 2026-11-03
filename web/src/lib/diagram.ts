// Episteme — Mermaid diagrams: lazy-loaded, themed, cached, animated with GSAP.
//
// Mermaid (~1 MB) is imported only when an answer actually contains a diagram.
// Renders are serialized through a queue because mermaid.initialize() is global and
// two diagrams in different accents (LLM orange / RAG teal) must not race.
import { gsap, RM } from './motion';

const ACCENT: Record<string, string> = { llm: '#ff5a1f', rag: '#16c7c7' };
const cache = new Map<string, string>();
let mermaidP: Promise<any> | null = null;
let themedFor = '';
let queue: Promise<unknown> = Promise.resolve();
let seq = 0;

function themeVariables(accent: string) {
  const ink = '#ededed', canvas = '#0a0a0a', surface = '#141516', line = '#3a3c3e';
  return {
    darkMode: true, background: canvas, fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: '13px',
    // flowchart
    primaryColor: surface, primaryTextColor: ink, primaryBorderColor: accent, mainBkg: surface,
    nodeBorder: accent, lineColor: accent, textColor: ink, secondaryColor: '#0e0f10', tertiaryColor: '#0e0f10',
    clusterBkg: '#0e0f10', clusterBorder: line, edgeLabelBackground: canvas, titleColor: ink,
    // sequence
    actorBkg: surface, actorBorder: accent, actorTextColor: ink, actorLineColor: line,
    signalColor: accent, signalTextColor: ink, labelBoxBkgColor: surface, labelBoxBorderColor: accent,
    labelTextColor: ink, loopTextColor: ink, noteBkgColor: '#1a1b1c', noteTextColor: ink, noteBorderColor: line,
    activationBkgColor: '#1f2021', activationBorderColor: accent, sequenceNumberColor: canvas,
  };
}

async function load(accent: string) {
  mermaidP ??= import('mermaid').then((m) => m.default);
  const mermaid = await mermaidP;
  if (themedFor !== accent) {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',          // no click handlers, sanitized labels
      theme: 'base',
      themeVariables: themeVariables(accent),
      flowchart: { curve: 'basis', htmlLabels: false, padding: 14, nodeSpacing: 42, rankSpacing: 50 },
      sequence: { mirrorActors: false, actorMargin: 64, messageMargin: 34 },
    });
    themedFor = accent;
  }
  return mermaid;
}

export type Rendered = { svg: string; cached: boolean };

/** Parse + render to an SVG string. Throws with mermaid's message if the code is invalid. */
export function renderDiagram(code: string, mode = 'llm'): Promise<Rendered> {
  const accent = ACCENT[mode] || ACCENT.llm;
  const key = `${accent}\n${code}`;
  const hit = cache.get(key);
  if (hit) return Promise.resolve({ svg: hit, cached: true });

  const job = queue.then(async () => {
    const mermaid = await load(accent);
    await mermaid.parse(code); // throws on syntax error
    const id = `ep-dia-${++seq}`;
    try {
      const { svg } = await mermaid.render(id, code);
      cache.set(key, svg);
      return { svg, cached: false };
    } finally {
      document.getElementById(id)?.remove();        // mermaid's temp node
      document.getElementById(`d${id}`)?.remove();  // and its error bomb, if any
    }
  });
  queue = job.catch(() => {}); // a failure must not block the next render
  return job;
}

export function kindOf(code: string): 'SEQUENCE' | 'ARCHITECTURE' {
  return /^\s*sequenceDiagram/i.test(code) ? 'SEQUENCE' : 'ARCHITECTURE';
}

/** Choreography: nodes settle in, edges draw themselves on, labels follow. */
export function animateDiagram(svg: SVGSVGElement) {
  if (RM || !svg) return;
  const nodes = svg.querySelectorAll('g.node, g.cluster, rect.actor, g.actor-man, .actor-top');
  const edges = [...svg.querySelectorAll<SVGPathElement | SVGLineElement>(
    'path.flowchart-link, g.edgePaths path, .messageLine0, .messageLine1, line.actor-line')];
  const labels = svg.querySelectorAll('g.edgeLabel, text.messageText, .noteText, .loopText');

  // Dotted/dashed edges keep their pattern and only fade in. (Mermaid ≥ 12 gives even
  // solid edges a dasharray — a gap before the arrowhead — so classify by class, not style.)
  const isPatterned = (e: Element) => /edge-pattern-(dotted|dashed)/.test(e.getAttribute('class') || '');
  const drawable = edges.filter((e) => !isPatterned(e));
  const dashed = edges.filter(isPatterned);
  // remember Mermaid's own inline style so the exact rendering comes back afterwards
  const original = drawable.map((e) => e.getAttribute('style'));
  // Set the dash pattern ourselves: with pathLength=1, "1 1" = one full-length dash and
  // one full-length gap, so offset 1 → hidden, 0 → drawn. (Letting GSAP tween dasharray
  // would merge into Mermaid's own 4-value pattern and the line would never hide.)
  drawable.forEach((e) => {
    e.setAttribute('pathLength', '1');
    e.style.strokeDasharray = '1 1';
    e.style.strokeDashoffset = '1';
  });

  const tl = gsap.timeline({
    defaults: { ease: 'power3.out' },
    onComplete: () => {
      drawable.forEach((e, i) => {
        e.removeAttribute('pathLength');
        if (original[i] === null) e.removeAttribute('style'); else e.setAttribute('style', original[i]!);
      });
    },
  });
  tl.from(nodes, { opacity: 0, scale: 0.9, transformOrigin: '50% 50%', duration: 0.4, stagger: 0.05 }, 0)
    .fromTo(drawable, { strokeDashoffset: 1, opacity: 0.35 },
      { strokeDashoffset: 0, opacity: 1, duration: 0.5, stagger: 0.06, ease: 'power2.inOut' }, 0.18)
    .from(dashed, { opacity: 0, duration: 0.4, stagger: 0.06 }, 0.18)
    .from(labels, { opacity: 0, y: 3, duration: 0.3, stagger: 0.04 }, 0.4);
  return tl;
}
