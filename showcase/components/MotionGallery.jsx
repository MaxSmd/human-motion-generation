"use client";

// A single 3D stage driven by a swipeable rail of selectors, with small
// client-side analysis panels below. One canvas is mounted at a time on
// purpose: a wall of simultaneous WebGL views costs the visitor their frame
// rate for no benefit.

import { useState } from "react";
import dynamic from "next/dynamic";
import { Stage, SwipeRail, Reveal } from "./motion";
import ClipStats from "./ClipStats";
import Viewport from "./Viewport";

const SkeletonPlayer = dynamic(() => import("./SkeletonPlayer"), {
  ssr: false,
  loading: () => (
    <div className="viewport grid place-items-center" style={{ height: 420 }}>
      <span className="animate-pulse-soft text-[12px] text-[var(--muted)]">loading motion…</span>
    </div>
  ),
});

/**
 * items: [{
 *   id, label, sub?, caption?, clips: [{url, color, label}],
 *   meta?: [[key, value, tone?]], highlight?, accent?
 * }]
 *
 * `onSelect(index)` fires when the visitor picks a clip, so a summary plot
 * elsewhere on the page can highlight the matching row.
 */
export default function MotionGallery({
  items,
  stageLabel,
  note,
  height = 440,
  initial = 0,
  spread,
  align = false,
  stats = false,
  onSelect,
}) {
  const [i, setI] = useState(initial);
  const it = items[i];

  const choose = (k) => {
    setI(k);
    onSelect?.(k);
  };

  return (
    <div>
      <SwipeRail className="mb-4">
        {items.map((x, k) => (
          <button
            key={x.id}
            onClick={() => choose(k)}
            className="surface min-w-[168px] shrink-0 snap-start p-3 text-left transition"
            style={{
              borderColor: k === i ? "var(--signal)" : undefined,
              background: k === i ? "var(--signal-dim)" : undefined,
            }}
          >
            <div className="text-[12.5px] font-semibold leading-tight text-[var(--text)]">{x.label}</div>
            {x.sub && <div className="mt-1 font-mono text-[10px] text-[var(--muted)]">{x.sub}</div>}
          </button>
        ))}
      </SwipeRail>

      <Stage label={stageLabel} note={note}>
        <Viewport height={height} key={`vp-${it.id}`}>
          <SkeletonPlayer
            key={it.id}
            clips={it.clips}
            height={height}
            highlight={it.highlight}
            accent={it.accent}
            spread={spread}
            align={align}
          />
        </Viewport>
        <div className="mt-4 flex flex-wrap items-start justify-between gap-4">
          {it.caption && (
            <p className="max-w-xl text-[12.5px] leading-relaxed text-[var(--muted)]">
              <span className="text-[var(--muted)]">“</span>
              <span className="text-[var(--text)]">{it.caption}</span>
              <span className="text-[var(--muted)]">”</span>
            </p>
          )}
          {it.meta && (
            <div className="flex flex-wrap gap-2">
              {it.meta.map(([k, v, tone]) => (
                <div key={k} className="rounded-lg border border-[var(--hairline)] px-2.5 py-1.5">
                  <div className="label mb-1">{k}</div>
                  <div
                    className="font-mono text-[12px]"
                    style={{ color: tone === "warn" ? "var(--warn)" : tone === "good" ? "var(--real)" : "var(--text)" }}
                  >
                    {v}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {stats && (
          <div className="mt-4 border-t border-[var(--hairline)] pt-4">
            <ClipStats key={it.id} clips={it.clips} />
          </div>
        )}
      </Stage>
    </div>
  );
}

/** A single always-on stage, used where there is nothing to switch between. */
export function MotionStage({ clips, label, note, height = 420, caption, autoRotate, highlight, accent, align }) {
  return (
    <Reveal>
      <Stage label={label} note={note}>
        <Viewport height={height}>
          <SkeletonPlayer clips={clips} height={height} autoRotate={autoRotate} highlight={highlight} accent={accent} align={align} />
        </Viewport>
        {caption && <p className="mt-3 text-[12.5px] leading-relaxed text-[var(--muted)]">{caption}</p>}
      </Stage>
    </Reveal>
  );
}
