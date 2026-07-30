"use client";

// Scroll-driven presentation primitives shared by every page.
//
// Everything here is IntersectionObserver + rAF over CSS custom properties, so
// no animation library ships to the visitor and nothing animates off-screen.
// `prefers-reduced-motion` short-circuits all of it to the settled state.

import { useCallback, useEffect, useRef, useState } from "react";

export function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

// True once the element has entered the viewport (latched: it never un-reveals,
// so scrolling back up does not replay everything).
export function useInView({ threshold = 0.18, rootMargin = "0px 0px -8% 0px" } = {}) {
  const ref = useRef(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    if (!("IntersectionObserver" in window)) return setSeen(true);
    const io = new IntersectionObserver(
      ([e]) => e.isIntersecting && (setSeen(true), io.disconnect()),
      { threshold, rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen, threshold, rootMargin]);
  return [ref, seen];
}

/**
 * Fade + travel on entry. `dir` picks the axis the element arrives from, `blur`
 * adds a short defocus, and `delay` staggers siblings.
 */
export function Reveal({
  children,
  dir = "up",
  delay = 0,
  distance = 22,
  scale = 1,
  blur = false,
  className = "",
  as: Tag = "div",
  style,
}) {
  const reduced = useReducedMotion();
  const [ref, seen] = useInView();
  const on = seen || reduced;
  const off = {
    up: `translate3d(0, ${distance}px, 0)`,
    down: `translate3d(0, ${-distance}px, 0)`,
    left: `translate3d(${distance}px, 0, 0)`,
    right: `translate3d(${-distance}px, 0, 0)`,
    none: "none",
  }[dir];
  return (
    <Tag
      ref={ref}
      className={className}
      style={{
        opacity: on ? 1 : 0,
        transform: on ? "none" : `${off} scale(${scale})`,
        filter: on || !blur ? "none" : "blur(7px)",
        transition: reduced
          ? "none"
          : `opacity .7s cubic-bezier(.2,.7,.25,1) ${delay}ms, transform .8s cubic-bezier(.2,.7,.25,1) ${delay}ms, filter .7s ease ${delay}ms`,
        willChange: on ? "auto" : "opacity, transform",
        ...style,
      }}
    >
      {children}
    </Tag>
  );
}

/** Stagger a list of children by a fixed step. */
export function RevealList({ children, step = 70, ...rest }) {
  return (
    <>
      {Array.isArray(children)
        ? children.map((c, i) => (
            <Reveal key={i} delay={i * step} {...rest}>
              {c}
            </Reveal>
          ))
        : children}
    </>
  );
}

/**
 * Progress of an element through the viewport, 0 as its top reaches the bottom
 * edge to 1 as its bottom leaves the top edge. Drives the scroll-scrubbed
 * scenes (rotation, zoom, section-by-section diagrams).
 */
export function useScrollProgress(ref, { clamp = true } = {}) {
  const [p, setP] = useState(0);
  const raf = useRef(0);
  const measure = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight || 1;
    const raw = (vh - r.top) / (vh + r.height);
    setP(clamp ? Math.min(1, Math.max(0, raw)) : raw);
  }, [ref, clamp]);
  useEffect(() => {
    const onScroll = () => {
      cancelAnimationFrame(raf.current);
      raf.current = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(raf.current);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [measure]);
  return p;
}

/** Map t∈[0,1] through [a,b], clamped. */
export const lerp = (a, b, t) => a + (b - a) * Math.min(1, Math.max(0, t));

/** Remap t from [i0,i1] onto [0,1], clamped — for chaining scroll phases. */
export const phase = (t, i0, i1) => Math.min(1, Math.max(0, (t - i0) / (i1 - i0)));

/** Ease for scroll scrubbing: fast in the middle, calm at both ends. */
export const smooth = (t) => t * t * (3 - 2 * t);

/**
 * A figure that can be blown up to fill the window. Everything visual on the
 * site is wrapped in one of these, so any chart or player can be inspected
 * full-bleed and dismissed with Escape.
 */
export function Stage({ children, label, note, className = "", padded = true }) {
  const [full, setFull] = useState(false);
  useEffect(() => {
    if (!full) return;
    const onKey = (e) => e.key === "Escape" && setFull(false);
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [full]);

  const head = (label || note) && (
    <div className="mb-3 flex items-baseline justify-between gap-4">
      <span className="label">{label}</span>
      <div className="flex items-center gap-3">
        {note && <span className="font-mono text-[10px] text-[var(--muted)]">{note}</span>}
        <button
          onClick={() => setFull((f) => !f)}
          aria-label={full ? "exit full screen" : "full screen"}
          className="rounded border border-[var(--hairline)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--muted)] transition hover:border-[var(--signal)] hover:text-[var(--signal)]"
        >
          {full ? "esc ✕" : "⤢"}
        </button>
      </div>
    </div>
  );

  if (full) {
    return (
      <div
        className="fixed inset-0 z-50 overflow-auto bg-[var(--ink)]/97 p-4 backdrop-blur-xl md:p-10"
        onClick={(e) => e.target === e.currentTarget && setFull(false)}
      >
        <div className="mx-auto max-w-[1500px]">
          {head}
          <div className="stage-full">{children}</div>
        </div>
      </div>
    );
  }
  return (
    <div className={`surface ${padded ? "p-5" : "p-3"} ${className}`}>
      {head}
      {children}
    </div>
  );
}

/**
 * Horizontally swipeable rail: drag with a pointer, flick on touch, or use the
 * arrows. Snaps to each child.
 */
export function SwipeRail({ children, className = "" }) {
  const ref = useRef(null);
  const drag = useRef(null);
  const dragged = useRef(false);
  const [edge, setEdge] = useState({ start: true, end: false });

  const sync = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    setEdge({
      start: el.scrollLeft < 8,
      end: el.scrollLeft > el.scrollWidth - el.clientWidth - 8,
    });
  }, []);

  useEffect(() => {
    sync();
  }, [sync, children]);

  const page = (dir) => {
    const el = ref.current;
    if (!el) return;
    el.scrollBy({ left: dir * Math.max(280, el.clientWidth * 0.8), behavior: "smooth" });
  };

  // Drag-to-scroll WITHOUT pointer capture: capturing would retarget pointerup
  // to the rail and swallow clicks on the child buttons. Instead we track the
  // move distance and only suppress the click if the pointer actually dragged.
  const onDown = (e) => {
    if (e.pointerType === "touch") return; // native touch scrolling is smoother
    drag.current = { x: e.clientX, left: ref.current.scrollLeft };
    dragged.current = false;
  };
  const onMove = (e) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    if (Math.abs(dx) > 4) dragged.current = true;
    ref.current.scrollLeft = drag.current.left - dx;
  };
  const onUp = () => {
    drag.current = null;
  };
  const onClickCapture = (e) => {
    if (dragged.current) {
      e.preventDefault();
      e.stopPropagation();
      dragged.current = false;
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div
        ref={ref}
        onScroll={sync}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        onClickCapture={onClickCapture}
        className="swipe-rail flex snap-x snap-mandatory gap-4 overflow-x-auto pb-2"
      >
        {children}
      </div>
      <RailButton side="left" onClick={() => page(-1)} hidden={edge.start} />
      <RailButton side="right" onClick={() => page(1)} hidden={edge.end} />
    </div>
  );
}

function RailButton({ side, onClick, hidden }) {
  return (
    <button
      onClick={onClick}
      aria-label={side === "left" ? "previous" : "next"}
      className="absolute top-1/2 z-10 hidden h-9 w-9 -translate-y-1/2 place-items-center rounded-full border border-[var(--hairline-strong)] bg-[var(--ink)]/85 font-mono text-[13px] text-[var(--text)] backdrop-blur transition hover:border-[var(--signal)] hover:text-[var(--signal)] md:grid"
      style={{
        [side]: -14,
        opacity: hidden ? 0 : 1,
        pointerEvents: hidden ? "none" : "auto",
      }}
    >
      {side === "left" ? "‹" : "›"}
    </button>
  );
}

/**
 * Number that counts up the first time it scrolls into view. It renders the
 * final value on the server and during the first client paint, so the figure is
 * correct with JavaScript disabled and never flashes a zero.
 */
export function Ticker({ value, decimals = 0, suffix = "", prefix = "", duration = 900 }) {
  const reduced = useReducedMotion();
  const [ref, seen] = useInView({ threshold: 0.5 });
  const [v, setV] = useState(value);
  const ran = useRef(false);

  useEffect(() => {
    if (!seen || reduced || ran.current) return;
    ran.current = true;
    let raf;
    let t0;
    const tick = (t) => {
      t0 ??= t;
      const k = Math.min(1, (t - t0) / duration);
      setV(value * (1 - Math.pow(1 - k, 3)));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [seen, value, duration, reduced]);

  return (
    <span ref={ref}>
      {prefix}
      {v.toFixed(decimals)}
      {suffix}
    </span>
  );
}
