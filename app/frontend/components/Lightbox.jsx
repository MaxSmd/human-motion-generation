"use client";

// Fullscreen media preview. Clicking a clip ANYWHERE in the app opens it here
// instead of dumping the raw file into a new browser tab — you stay in the app,
// the clip is big enough to actually judge, and the whole group it came from is
// arrow-key navigable (so GT ⇄ PRED is one keypress apart).
//
// Usage: wrap the app in <LightboxProvider>, then from any tile:
//   const { open } = useLightbox();
//   <button onClick={() => open(items, i)}>…</button>
// `items` are job-output shaped ({media_url, caption, kind, clip_id, npy_url}).

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { mediaUrl } from "@/lib/api";

const LightboxContext = createContext({ open: () => {} });

export function useLightbox() {
  return useContext(LightboxContext);
}

const KIND_META = {
  gt: { label: "GT", color: "var(--muted)" },
  pred: { label: "PRED", color: "var(--signal)" },
  sample: { label: "SAMPLE", color: "var(--accent2)" },
};

const isVideo = (u) => String(u || "").toLowerCase().endsWith(".mp4");

export function LightboxProvider({ children }) {
  const [state, setState] = useState(null); // {items, i}

  const open = useCallback((items, i = 0) => {
    const list = (Array.isArray(items) ? items : [items]).filter((o) => o?.media_url);
    if (list.length) setState({ items: list, i: Math.max(0, Math.min(i, list.length - 1)) });
  }, []);

  const close = useCallback(() => setState(null), []);
  const step = useCallback((d) => {
    setState((s) => (s ? { ...s, i: (s.i + d + s.items.length) % s.items.length } : s));
  }, []);

  // Keyboard: esc closes, ←/→ walk the group. Bound only while open.
  useEffect(() => {
    if (!state) return;
    const onKey = (e) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") step(1);
      else if (e.key === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    // The page behind must not scroll while the overlay owns the viewport.
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [state, close, step]);

  return (
    <LightboxContext.Provider value={{ open }}>
      {children}
      {state && <Overlay state={state} onClose={close} onStep={step} />}
    </LightboxContext.Provider>
  );
}

function Overlay({ state, onClose, onStep }) {
  const o = state.items[state.i];
  const many = state.items.length > 1;
  const k = KIND_META[o.kind];
  const full = mediaUrl(o.media_url);

  return (
    // The backdrop closes on click; the panel stops propagation so clicks on the
    // clip itself (play/scrub) don't dismiss it.
    <div
      className="fixed inset-0 z-50 flex flex-col bg-black/85 backdrop-blur-sm animate-fade-up"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div className="flex shrink-0 items-center gap-3 px-5 py-3 text-[11px]" onClick={(e) => e.stopPropagation()}>
        {k && (
          <span className="rounded px-1.5 py-0.5 font-mono font-medium" style={{ color: k.color, border: `1px solid ${k.color}` }}>
            {k.label}
          </span>
        )}
        {o.clip_id && <span className="font-mono text-slate-300">{o.clip_id}</span>}
        {o.step != null && <span className="font-mono text-[var(--muted)]">step {o.step}</span>}
        {o.from_registry && (
          <span className="rounded border border-[var(--hairline)] px-1.5 py-0.5 font-mono text-[9px] text-[var(--muted)]" title="reused from the GT registry — not re-rendered">
            cached
          </span>
        )}
        {many && <span className="font-mono text-[var(--muted)]">{state.i + 1} / {state.items.length}</span>}
        <div className="ml-auto flex items-center gap-3">
          <a href={full} download className="text-[var(--muted)] hover:text-[var(--signal)]">
            ⤓ {isVideo(o.media_url) ? "mp4" : "gif"}
          </a>
          {o.npy_url && (
            <a href={mediaUrl(o.npy_url)} download className="text-[var(--muted)] hover:text-[var(--signal)]">⤓ npy</a>
          )}
          <button onClick={onClose} className="text-[16px] leading-none text-slate-400 hover:text-white" title="close (esc)">✕</button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 items-center gap-2 px-3 pb-3">
        {many && <NavButton dir="‹" onClick={(e) => { e.stopPropagation(); onStep(-1); }} />}
        <div className="flex min-h-0 flex-1 items-center justify-center" onClick={(e) => e.stopPropagation()}>
          {isVideo(o.media_url) ? (
            <video key={full} src={full} className="max-h-full max-w-full rounded-lg" controls autoPlay loop muted />
          ) : (
            <img key={full} src={full} alt={o.caption || "motion clip"} className="max-h-full max-w-full rounded-lg" />
          )}
        </div>
        {many && <NavButton dir="›" onClick={(e) => { e.stopPropagation(); onStep(1); }} />}
      </div>

      {o.caption && (
        <p className="shrink-0 px-6 pb-5 text-center text-[12px] leading-snug text-slate-400" onClick={(e) => e.stopPropagation()}>
          {o.caption}
        </p>
      )}
    </div>
  );
}

function NavButton({ dir, onClick }) {
  return (
    <button
      onClick={onClick}
      className="shrink-0 rounded-full border border-[var(--hairline-strong)] bg-black/50 px-3 py-4 text-[18px] leading-none text-slate-400 transition hover:border-[var(--signal)] hover:text-[var(--signal)]"
      title={dir === "‹" ? "previous (←)" : "next (→)"}
    >
      {dir}
    </button>
  );
}
