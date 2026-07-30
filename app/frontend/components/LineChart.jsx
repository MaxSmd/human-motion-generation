"use client";

import { useMemo, useRef } from "react";
import { downloadSvg, downloadPng } from "@/lib/exportSvg";
import { lineChartToTikz, downloadText } from "@/lib/exportLatex";
import { niceTicks, fmtNum } from "@/lib/numfmt";

// Colour of the "this is the result you want" marker — the same green the
// analysis tabs use for a passing / desired value (AnalysisKit.GREEN).
const BEST = "#34d399";

// Dependency-free SVG multi-series line chart. `series` = [{label,color,points:[[x,y]]}].
// Points are drawn in given order (so it also renders trajectories/paths).
export default function LineChart({
  series = [],
  width = 540,
  height = 240,
  xLabel = "",
  yLabel = "",
  equal = false,
  yZero = false,
  // "linear" (default) or "log". Log needs strictly-positive y: non-positive
  // points are dropped (with a note left to the caller to surface).
  yScale = "linear",
  // Explicit y-domain [lo, hi]. Points outside it are CLIPPED — drawn as a
  // marker on the frame edge rather than dropped, so an off-scale spike still
  // reads as "it happened here, off the top" instead of vanishing. This is what
  // keeps a loss curve legible when a gradient storm is 50× the working range.
  yDomain = null,
  // When set, a small ⤓ control appears to export the chart as svg/png (paper figures).
  exportName = null,
  // Metric direction — "min" (lower is better, e.g. FID) or "max" (higher is
  // better, e.g. R@1). When given, the chart marks the optimum: a faint ring on
  // each series' own best point, and a labelled halo + crosshair on the single
  // best point across all series. Only meaningful when y is a quality metric;
  // leave null for curves where neither end is "good" (loss over time, cost).
  best = null,
  // How the winning point is described in its label, e.g. "best FID".
  bestLabel = "best",
  // Explicit x tick values (e.g. the exact guidance levels of a sweep). When
  // given, the x-axis ticks at these values only — no auto-spaced ticks — and
  // the domain is padded so edge points/labels don't overflow the frame.
  xTicks = null,
  // Optional overlays (all additive, drawn under the series):
  bands = [],   // horizontal shaded bands: [{ y0, y1, color, label }]
  hlines = [],  // horizontal reference lines: [{ y, color, dash, label }]
  regions = [], // vertical shaded x-regions: [{ x0, x1, color, label }]
  vlines = [],  // vertical event markers: [{ x, color, dash, label, title }]
}) {
  const svgRef = useRef(null);
  const m = { l: 48, r: 12, t: 12, b: 34 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;

  const log = yScale === "log";

  const dom = useMemo(() => {
    const pts = series.flatMap((s) => (Array.isArray(s.points) ? s.points : []));
    if (pts.length === 0) return null;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of pts) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      // A log axis has no room for y ≤ 0 — ignore those when sizing it.
      if (log && !(y > 0)) continue;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    if (!Number.isFinite(ymin)) return null;  // log axis with nothing positive
    // Let target lines / bands stretch the y-domain so they're always in frame.
    for (const b of bands) { ymin = Math.min(ymin, b.y0); ymax = Math.max(ymax, b.y1); }
    for (const h of hlines) { ymin = Math.min(ymin, h.y); ymax = Math.max(ymax, h.y); }
    if (yZero && !log) ymin = Math.min(0, ymin);
    // An explicit domain wins over the data's own extent — that's the whole
    // point of it (clip the storms, keep the detail).
    if (yDomain) {
      ymin = yDomain[0];
      ymax = yDomain[1];
    }
    if (xTicks?.length) {
      // fixed ticks: cover them exactly, then pad both sides so the outermost
      // markers sit inside the plot instead of on (or past) the frame.
      for (const t of xTicks) { xmin = Math.min(xmin, t); xmax = Math.max(xmax, t); }
      const pad = (xmax - xmin || 1) * 0.05;
      xmin -= pad; xmax += pad;
    }
    if (xmin === xmax) xmax = xmin + 1;
    if (ymin === ymax) ymax = ymin + 1;
    if (equal) {
      // square the data window so x/y share a scale (trajectories)
      const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
      const half = Math.max(xmax - xmin, ymax - ymin) / 2 * 1.1;
      xmin = cx - half; xmax = cx + half; ymin = cy - half; ymax = cy + half;
    }
    return { xmin, xmax, ymin, ymax };
  }, [series, equal, yZero, bands, hlines, xTicks, log, yDomain]);

  if (!dom) {
    return (
      <div className="grid place-items-center rounded-lg border border-[var(--hairline)] bg-ink text-[11px] text-[var(--muted)]" style={{ width, height }}>
        no data
      </div>
    );
  }

  const sx = (x) => m.l + ((x - dom.xmin) / (dom.xmax - dom.xmin)) * iw;
  // On a log axis the domain is mapped through log10 — a decade always occupies
  // the same height, which is what lets a 0.01 tail and a 10.0 spike coexist.
  const lo = log ? Math.log10(Math.max(dom.ymin, Number.MIN_VALUE)) : dom.ymin;
  const hi = log ? Math.log10(Math.max(dom.ymax, Number.MIN_VALUE)) : dom.ymax;
  const sy = (y) => {
    const v = log ? Math.log10(Math.max(y, Number.MIN_VALUE)) : y;
    return m.t + ih - ((v - lo) / (hi - lo || 1)) * ih;
  };
  // Off-domain points are pinned to the frame edge (and flagged) rather than
  // dropped — a clipped spike must still be visible AS a spike.
  const clampY = (y) => Math.min(dom.ymax, Math.max(dom.ymin, y));
  const clipDir = (y) => (y > dom.ymax ? 1 : y < dom.ymin ? -1 : 0);
  // Log ticks land on the decades inside the domain (plus the endpoints), so the
  // labels stay round numbers instead of arbitrary log-spaced values.
  const yNice = log ? null : niceTicks(dom.ymin, dom.ymax, 5);
  const yTickVals = log
    ? (() => {
        const out = [];
        for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) out.push(10 ** e);
        if (!out.length || out[0] > dom.ymin) out.unshift(dom.ymin);
        if (out[out.length - 1] < dom.ymax) out.push(dom.ymax);
        return out;
      })()
    : yNice.ticks;
  // Auto x ticks are round numbers too (frame counts, steps, guidance levels).
  const xNice = niceTicks(dom.xmin, dom.xmax, 5);
  // Decimals are chosen from the tick spacing, so an axis never shows more
  // digits than it actually resolves — and never falls back to 1.0e+3.
  const fmtY = (v) => fmtNum(v, log ? null : yNice.step);
  const fmtX = (v) => fmtNum(v, xTicks?.length ? null : xNice.step);

  // Optimum markers. `perSeries` = each curve's own best point (faint ring, so a
  // multi-run sweep still shows where each config peaks); `overall` = the single
  // winner across all series, which gets the halo, the crosshair and the label.
  // Ties keep the first point encountered, matching the table's winner rule.
  const better = (a, b) => (best === "min" ? a < b : a > b);
  const optima = useMemo(() => {
    if (best !== "min" && best !== "max") return { perSeries: [], overall: null };
    const perSeries = [];
    let overall = null;
    series.forEach((s, i) => {
      let win = null;
      for (const [x, y] of Array.isArray(s.points) ? s.points : []) {
        if (!Number.isFinite(y) || (log && !(y > 0))) continue;
        if (!win || better(y, win.y)) win = { x, y, color: s.color, label: s.label, i };
      }
      if (!win) return;
      perSeries.push(win);
      if (!overall || better(win.y, overall.y)) overall = win;
    });
    return { perSeries, overall };
  }, [series, best, log]);

  return (
    <div className="relative">
      {exportName && (
        <div className="absolute right-0 top-0 z-10 flex gap-1 opacity-40 transition hover:opacity-100">
          <button type="button" onClick={() => downloadSvg(svgRef.current, `${exportName}.svg`)} title="download SVG"
            className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">⤓ svg</button>
          <button type="button" onClick={() => downloadPng(svgRef.current, `${exportName}.png`)} title="download PNG"
            className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">⤓ png</button>
          <button type="button" title="download pgfplots/TikZ LaTeX"
            onClick={() => downloadText(lineChartToTikz(series, { xLabel, yLabel, yScale, width, height }), `${exportName}.tex`)}
            className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">⤓ tex</button>
        </div>
      )}
      <svg ref={svgRef} width={width} height={height} className="overflow-visible">
        {/* gridlines + y ticks */}
        {yTickVals.map((t, i) => (
          <g key={`y${i}`}>
            <line x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} stroke="var(--hairline)" strokeWidth="1" />
            <text x={m.l - 6} y={sy(t) + 3} textAnchor="end" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>
              {fmtY(t)}
            </text>
          </g>
        ))}
        {/* x ticks — explicit values (with tick marks) when given, else auto */}
        {(xTicks?.length ? xTicks : xNice.ticks).map((t, i) => (
          <g key={`x${i}`}>
            {xTicks?.length ? (
              <line x1={sx(t)} x2={sx(t)} y1={m.t + ih} y2={m.t + ih + 4} stroke="var(--hairline-strong)" strokeWidth="1" />
            ) : null}
            <text x={sx(t)} y={height - m.b + 16} textAnchor="middle" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>
              {fmtX(t)}
            </text>
          </g>
        ))}
        {/* vertical regions (e.g. a constraint's active frame window) */}
        {regions.map((r, i) => {
          const x0 = Math.max(m.l, sx(r.x0));
          const x1 = Math.min(width - m.r, sx(r.x1));
          return <rect key={`rg${i}`} x={x0} y={m.t} width={Math.max(0, x1 - x0)} height={ih}
            fill={r.color || "rgba(148,163,184,0.10)"} />;
        })}
        {/* horizontal bands (e.g. an allowed bend range) */}
        {bands.map((b, i) => {
          const y0 = sy(Math.max(b.y0, b.y1));
          const y1 = sy(Math.min(b.y0, b.y1));
          return <rect key={`bd${i}`} x={m.l} y={y0} width={iw} height={Math.max(0, y1 - y0)}
            fill={b.color || "rgba(52,211,153,0.12)"} />;
        })}
        {/* horizontal reference lines (e.g. a pin target) */}
        {hlines.map((h, i) => (
          <line key={`hl${i}`} x1={m.l} x2={width - m.r} y1={sy(h.y)} y2={sy(h.y)}
            stroke={h.color || "#fbbf24"} strokeWidth="1.4" strokeDasharray={h.dash || "5 4"} />
        ))}
        {/* vertical event markers (e.g. a resume-from-checkpoint seam), drawn
            under the series with a small label hung off the top of the frame */}
        {vlines.map((v, i) => {
          const x = sx(v.x);
          if (x < m.l || x > width - m.r) return null;
          return (
            <g key={`vl${i}`}>
              <line x1={x} x2={x} y1={m.t} y2={m.t + ih}
                stroke={v.color || "var(--hairline-strong)"} strokeWidth="1.2"
                strokeDasharray={v.dash || "3 3"}>
                {v.title && <title>{v.title}</title>}
              </line>
              {v.label && (
                <text x={x + 3} y={m.t + 8} className="fill-[var(--muted)]"
                  style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>
                  {v.label}
                </text>
              )}
            </g>
          );
        })}
        {/* series — polyline + point markers (so single-point series are visible).
            `marker` (circle|square|triangle|diamond|cross) is a secondary identity
            channel on top of color — e.g. one shape per ODE step count. */}
        {series.map((s, i) => {
          // A log axis can't place y ≤ 0 at all; everything else is clamped into
          // the domain so the line stays continuous across a clipped excursion.
          const pts = (Array.isArray(s.points) ? s.points : []).filter(
            ([, y]) => !log || y > 0
          );
          const clipped = pts.filter(([, y]) => clipDir(y) !== 0);
          return (
            <g key={i}>
              {/* `scatter` drops the connecting line: for a paired plot the x
                  order carries no meaning, so joining the dots would invent a
                  trend that isn't there. */}
              {pts.length > 1 && !s.scatter && (
                <polyline
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.width || 1.6}
                  // `dash` is a second identity channel independent of colour, so
                  // one hue can carry a pair (e.g. same text, projection on/off).
                  strokeDasharray={s.dash || undefined}
                  strokeOpacity={s.opacity ?? 1}
                  points={pts.map(([x, y]) => `${sx(x)},${sy(clampY(y))}`).join(" ")}
                />
              )}
              {/* markers are the whole plot for a scatter series, so they're
                  never thinned out there */}
              {(pts.length <= 60 || s.scatter) && pts.map(([x, y], j) => (
                <Mark key={j} shape={s.marker} x={sx(x)} y={sy(clampY(y))} color={s.color} />
              ))}
              {/* clipped excursions: an arrow on the frame edge pointing the way
                  the value actually went, titled with its true magnitude */}
              {clipped.map(([x, y], j) => (
                <ClipMark key={`c${j}`} x={sx(x)} y={sy(clampY(y))} up={clipDir(y) > 0}
                  color={s.color} value={y} />
              ))}
            </g>
          );
        })}
        {/* desired-optimum markers (drawn over the series) */}
        {optima.perSeries.map((p, i) => (
          <circle key={`ob${i}`} cx={sx(p.x)} cy={sy(clampY(p.y))} r="5.2"
            fill="none" stroke={p.color} strokeWidth="1.2" strokeOpacity="0.55">
            <title>{`${p.label ? `${p.label} — ` : ""}${bestLabel}: ${fmtNum(p.y)} at ${fmtX(p.x)}`}</title>
          </circle>
        ))}
        {optima.overall && (() => {
          const x = sx(optima.overall.x), y = sy(clampY(optima.overall.y));
          // Flip the label inboard near the frame edges so it never clips out.
          const flipX = x > m.l + iw * 0.72;
          const flipY = y < m.t + 24;
          return (
            <g pointerEvents="none">
              <line x1={m.l} x2={x} y1={y} y2={y} stroke={BEST} strokeWidth="1" strokeDasharray="2 3" strokeOpacity="0.55" />
              <line x1={x} x2={x} y1={y} y2={m.t + ih} stroke={BEST} strokeWidth="1" strokeDasharray="2 3" strokeOpacity="0.55" />
              <circle cx={x} cy={y} r="8.5" fill={BEST} fillOpacity="0.14" />
              <circle cx={x} cy={y} r="8.5" fill="none" stroke={BEST} strokeWidth="1.5" />
              <text x={flipX ? x - 12 : x + 12} y={flipY ? y + 16 : y - 12}
                textAnchor={flipX ? "end" : "start"}
                style={{ fontSize: 9.5, fontFamily: "var(--font-mono)", fill: BEST }}>
                ★ {bestLabel} {fmtNum(optima.overall.y)}
              </text>
            </g>
          );
        })()}

        {/* axis labels */}
        {xLabel && (
          <text x={m.l + iw / 2} y={height - 2} textAnchor="middle" className="fill-[var(--muted)]" style={{ fontSize: 10 }}>
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text transform={`translate(11,${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" className="fill-[var(--muted)]" style={{ fontSize: 10 }}>
            {yLabel}
          </text>
        )}
      </svg>
      {series.length > 1 && (
        <div className="mt-1 flex flex-wrap gap-3 pl-12">
          {series.map((s, i) => (
            <span key={i} className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <svg width="18" height="10" className="shrink-0">
                <line x1="0" y1="5" x2="18" y2="5" stroke={s.color} strokeWidth="1.6" />
                <Mark shape={s.marker} x={9} y={5} color={s.color} />
              </svg>
              {s.label}
              {/* which curve holds the overall optimum — readable without
                  tracing the halo back to a colour in a crowded sweep */}
              {optima.overall?.i === i && <span style={{ color: BEST }}>★</span>}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// An out-of-domain point: a solid triangle on the frame edge pointing where the
// value went. Hover gives the true number, so clipping hides magnitude, never
// the event itself.
function ClipMark({ x, y, up, color, value }) {
  const r = 3.4;
  const tip = up ? y - r : y + r;
  const base = up ? y + r : y - r;
  return (
    <polygon points={`${x},${tip} ${x - r},${base} ${x + r},${base}`} fill={color} opacity="0.9">
      <title>{`${fmtNum(value)} (off scale)`}</title>
    </polygon>
  );
}

function Mark({ shape = "circle", x, y, color, r = 2.7 }) {
  switch (shape) {
    case "square":
      return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} fill={color} />;
    case "triangle":
      return <polygon points={`${x},${y - r * 1.2} ${x - r * 1.2},${y + r} ${x + r * 1.2},${y + r}`} fill={color} />;
    case "diamond":
      return <polygon points={`${x},${y - r * 1.35} ${x + r * 1.35},${y} ${x},${y + r * 1.35} ${x - r * 1.35},${y}`} fill={color} />;
    case "cross":
      return <path d={`M${x - r},${y - r}L${x + r},${y + r}M${x - r},${y + r}L${x + r},${y - r}`} stroke={color} strokeWidth="1.7" fill="none" />;
    default:
      return <circle cx={x} cy={y} r={r} fill={color} />;
  }
}
