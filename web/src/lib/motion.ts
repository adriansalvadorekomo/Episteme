// Episteme — motion system (GSAP).
//
// Principle: every exit is the exact reverse of its entrance. `reveal()` and
// `conceal()` build the SAME timeline from the same options; the entrance plays it
// forward, the exit renders its end state and plays it backwards. Same curve, same
// path, same duration — symmetric by construction.
import { gsap } from 'gsap';
import { Flip } from 'gsap/Flip';
import { MotionPathPlugin } from 'gsap/MotionPathPlugin';

gsap.registerPlugin(MotionPathPlugin, Flip);
export { Flip };

export const RM = typeof window !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
export const EASE = 'power3.out';
export const D = RM ? 0 : 0.34;
gsap.defaults({ ease: EASE, duration: D });

export { gsap };

type RevealOpts = {
  y?: number; x?: number; scale?: number; duration?: number; delay?: number;
  width?: boolean; height?: boolean;
};

const SIZE_PROPS = 'width,height,paddingLeft,paddingRight,paddingTop,paddingBottom,marginRight,marginBottom';

function build(el: any, o: RevealOpts) {
  const tl = gsap.timeline({ paused: true, defaults: { duration: o.duration ?? D, ease: EASE } });
  if (o.width || o.height) gsap.set(el, { clearProps: SIZE_PROPS }); // measure the natural box
  if (o.height) {
    const cs = getComputedStyle(el);
    tl.fromTo(el,
      { height: 0, paddingTop: 0, paddingBottom: 0, marginBottom: 0 },
      { height: el.offsetHeight, paddingTop: cs.paddingTop, paddingBottom: cs.paddingBottom, marginBottom: cs.marginBottom },
      0);
  }
  if (o.width) {
    // grow the box from zero so neighbours glide (and glide back on exit)
    const cs = getComputedStyle(el);
    tl.fromTo(el,
      { width: 0, paddingLeft: 0, paddingRight: 0, marginRight: 0 },
      { width: el.offsetWidth, paddingLeft: cs.paddingLeft, paddingRight: cs.paddingRight, marginRight: cs.marginRight },
      0);
  }
  tl.fromTo(el,
    { autoAlpha: 0, y: o.y ?? 8, x: o.x ?? 0, scale: o.scale ?? 0.98 },
    { autoAlpha: 1, y: 0, x: 0, scale: 1 }, 0);
  return tl;
}

/** Entrance. */
export function reveal(el: any, o: RevealOpts = {}) {
  if (!el) return null;
  el.__opts = o;
  const tl = build(el, o);
  if (o.width || o.height) tl.eventCallback('onComplete', () => gsap.set(el, { clearProps: SIZE_PROPS }));
  el.__tl = tl;
  tl.delay(o.delay ?? 0).play();
  return tl;
}

/** Exit = the entrance played backwards (re-measured at the current size). */
export function conceal(el: any): Promise<void> {
  return new Promise((res) => {
    if (!el) { res(); return; }
    if (RM || !el.__opts) { res(); return; }
    el.__tl?.kill();
    const tl = build(el, el.__opts);
    tl.progress(1);
    tl.eventCallback('onReverseComplete', () => res());
    tl.reverse();
  });
}

// ── Modals: scrim fade + card rise; close = the same timeline reversed ───────
export function openModal(modal: any) {
  if (!modal) return;
  modal.__tl?.kill();
  modal.hidden = false;
  const tl = gsap.timeline({ defaults: { duration: RM ? 0 : 0.38, ease: EASE } })
    .fromTo(modal.querySelector('.modal-scrim'), { autoAlpha: 0 }, { autoAlpha: 1 }, 0)
    .fromTo(modal.querySelector('.modal-card'), { autoAlpha: 0, y: 16, scale: 0.97 }, { autoAlpha: 1, y: 0, scale: 1 }, 0.04);
  modal.__tl = tl;
}

export function closeModal(modal: any): Promise<void> {
  return new Promise((res) => {
    if (!modal || modal.hidden) { res(); return; }
    const tl = modal.__tl;
    if (!tl || RM) { modal.hidden = true; res(); return; }
    tl.eventCallback('onReverseComplete', () => { modal.hidden = true; res(); });
    tl.reverse();
  });
}

// ── Logo: draw the mark, pop the nodes, connect them, then a signal travels ──
export function animateLogo(svg: any, { loop = true, delay = 0 } = {}) {
  if (!svg) return;
  const spine = svg.querySelector('.lg-spine');
  const arms = svg.querySelectorAll('.lg-arm');
  const link = svg.querySelector('.lg-link');
  const nodes = svg.querySelectorAll('.lg-node');
  const signal = svg.querySelector('.lg-signal');

  if (RM) { gsap.set(svg, { visibility: 'visible' }); return; }

  gsap.set([spine, ...arms, link], { strokeDasharray: 1, strokeDashoffset: 1 });
  gsap.set(nodes, { scale: 0, transformOrigin: '50% 50%' });
  gsap.set(signal, { autoAlpha: 0 });
  gsap.set(svg, { visibility: 'visible' });

  const intro = gsap.timeline({ delay })
    .to(spine, { strokeDashoffset: 0, duration: 0.5, ease: 'power2.inOut' })
    .to(arms, { strokeDashoffset: 0, duration: 0.42, stagger: 0.1, ease: 'power2.out' }, '-=0.2')
    .to(nodes, { scale: 1, duration: 0.5, stagger: 0.09, ease: 'back.out(3)' }, '-=0.28')
    .to(link, { strokeDashoffset: 0, duration: 0.55, ease: 'power2.inOut' }, '-=0.3');

  if (!loop) return intro;

  // idle: a signal runs Knowledge → Relations → Reasoning; each node answers with a pulse
  const beat = { scale: 1.5, duration: 0.18, yoyo: true, repeat: 1, ease: 'power2.out' };
  const idle = gsap.timeline({ repeat: -1, repeatDelay: 3.4, paused: true })
    .set(signal, { autoAlpha: 1 })
    .to(signal, { duration: 1.6, ease: 'power1.inOut', motionPath: { path: link, align: link, alignOrigin: [0.5, 0.5] } })
    .to(nodes[0], beat, 0)
    .to(nodes[1], beat, 0.78)
    .to(nodes[2], beat, 1.48)
    .to(signal, { autoAlpha: 0, duration: 0.2 }, '-=0.15');

  intro.eventCallback('onComplete', () => idle.play(0));
  svg.__idle = idle;
  return intro;
}

/** Replay the signal immediately (e.g. on hover). */
export function pingLogo(svg: any) {
  if (RM || !svg?.__idle) return;
  svg.__idle.restart();
}
