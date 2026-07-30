"use client";

// Dependency-free SVG charts. All of them are responsive (viewBox + width 100 %)
// so the same component reads correctly inline and blown up full screen.
//
// Conventions kept across every chart here:
//   · log scale whenever the data spans more than ~1.5 decades (FID does)
//   · series labelled directly at the end of the line, no separate legend key
//   · one hairline gridline per tick, nothing heavier
//   · values in the same tabular mono face as the rest of the site

import { useEffect, useMemo, useRef, useState } from "react";

const MONO = { fontFamily: "var(--font-mono)" };

/**
 * Render an SVG chart at its container's true pixel width, never wider than the
 * width it was designed at. On a phone the viewBox shrinks to the real width, so
 * SVG units equal screen pixels and axis text keeps its designed size instead of
 * being scaled down into an unreadable smear; on a wide screen it caps at the
 * design width and the browser upscales as before. Returns [ref, width].
 */
function useFluidWidth(designW) {
  const ref = useRef(null);
  const [w, setW] = useState(designW);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => {
      const cw = e.contentRect.width;
      if (cw > 0) setW(Math.min(Math.round(cw), designW));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [designW]);
  return [ref, w];
}

function fmt(v, d) {
  if (d != null) return v.toFixed(d);
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 0.001 || a >= 10000) return v.toExponential(1);
  return Number(v.toFixed(a < 0.01 ? 4 : a < 1 ? 3 : a < 100 ? 2 : 0)).toString();
}

// Decade ticks at 1/2/5. Over a range narrower than a decade that can leave one
// tick or none, so progressively finer multiplier sets are tried before giving
// up and labelling the endpoints.
function logTicks(lo, hi) {
  for (const mults of [[1, 2, 5], [1, 1.5, 2, 3, 5, 7], [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]]) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of mults) {
        const v = m * 10 ** e;
        if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
      }
    }
    out.sort((a, b) => a - b);
    if (out.length >= 3) return out;
  }
  return [lo, (lo * hi) ** 0.5, hi];
}

/**
 * Drop ticks that would render closer together than `minPx`, keeping the first
 * of each cluster. A log axis over several decades otherwise collides its labels
 * into an unreadable smear at the compressed end.
 */
function thin(ticks, scale, minPx = 40) {
  const out = [];
  let last = -Infinity;
  for (const t of ticks) {
    const x = scale(t);
    if (Math.abs(x - last) >= minPx) {
      out.push(t);
      last = x;
    }
  }
  return out;
}

// Round tick values at a 1 / 2 / 5 × 10ⁿ step, covering the domain from the
// inside so a tick never sits outside the plotted range.
function linTicks(lo, hi, n = 5) {
  const raw = (hi - lo) / n;
  if (!(raw > 0)) return [lo];
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Number(v.toPrecision(12)));
  }
  return out.length ? out : [lo, hi];
}

/**
 * Multi-series line/scatter plot.
 *
 * series: [{ label, color, points: [[x, y]], err?: [[x, lo, hi]], dashed?, area? }]
 */
export function LinePlot({
  series = [],
  width: designW = 640,
  height = 320,
  xLabel,
  yLabel,
  yLog = false,
  xLog = false,
  xTicks,
  yTicksN = 5,
  hlines = [],
  vlines = [],
  bands = [],
  annotations = [],
  markers = [], // [{x, y, label, color}] — highlighted "best" points
  labelSeries = true,
  yPad = 0.08,
  valueDigits,
  right = 96,
}) {
  const [wrapRef, width] = useFluidWidth(designW);
  const narrow = width < 480;
  const m = {
    l: narrow ? 44 : 54,
    r: labelSeries ? (narrow ? Math.min(right, 38) : right) : narrow ? 12 : 18,
    t: 16,
    b: 42,
  };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const [hover, setHover] = useState(null);

  const dom = useMemo(() => {
    const pts = series.flatMap((s) => s.points);
    if (!pts.length) return null;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of pts) {
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    for (const s of series) for (const e of s.err ?? []) { ymin = Math.min(ymin, e[1]); ymax = Math.max(ymax, e[2]); }
    for (const h of hlines) { ymin = Math.min(ymin, h.y); ymax = Math.max(ymax, h.y); }
    for (const b of bands) { ymin = Math.min(ymin, b.y0); ymax = Math.max(ymax, b.y1); }
    if (yLog) {
      ymin = Math.max(ymin, 1e-6);
      const lo = Math.log10(ymin), hi = Math.log10(ymax), pad = (hi - lo || 1) * yPad;
      ymin = 10 ** (lo - pad); ymax = 10 ** (hi + pad);
    } else {
      const pad = (ymax - ymin || 1) * yPad;
      ymin -= pad; ymax += pad;
    }
    if (xTicks?.length) for (const t of xTicks) { xmin = Math.min(xmin, t); xmax = Math.max(xmax, t); }
    if (xLog) {
      const lo = Math.log10(xmin), hi = Math.log10(xmax), pad = (hi - lo || 1) * 0.05;
      xmin = 10 ** (lo - pad); xmax = 10 ** (hi + pad);
    } else {
      const pad = (xmax - xmin || 1) * 0.045;
      xmin -= pad; xmax += pad;
    }
    return { xmin, xmax, ymin, ymax };
  }, [series, hlines, bands, yLog, xLog, xTicks, yPad]);

  if (!dom) return <Empty width={width} height={height} />;

  const sx = (x) =>
    m.l + (xLog
      ? (Math.log10(x) - Math.log10(dom.xmin)) / (Math.log10(dom.xmax) - Math.log10(dom.xmin))
      : (x - dom.xmin) / (dom.xmax - dom.xmin)) * iw;
  const sy = (y) =>
    m.t + ih - (yLog
      ? (Math.log10(Math.max(y, 1e-9)) - Math.log10(dom.ymin)) / (Math.log10(dom.ymax) - Math.log10(dom.ymin))
      : (y - dom.ymin) / (dom.ymax - dom.ymin)) * ih;

  const yt = thin(yLog ? logTicks(dom.ymin, dom.ymax) : linTicks(dom.ymin, dom.ymax, yTicksN), sy, 22);
  const xt = xTicks?.length
    ? xTicks
    : thin(xLog ? logTicks(dom.xmin, dom.xmax) : linTicks(dom.xmin, dom.xmax, 5), sx, 44);

  return (
    <svg ref={wrapRef} viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ overflow: "visible" }} role="img">
      {bands.map((b, i) => (
        <rect key={`b${i}`} x={m.l} y={sy(b.y1)} width={iw} height={Math.abs(sy(b.y0) - sy(b.y1))}
          fill={b.color || "rgba(52,211,153,0.10)"} />
      ))}
      {yt.map((t, i) => (
        <g key={`y${i}`}>
          <line x1={m.l} x2={m.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--hairline)" />
          <text x={m.l - 8} y={sy(t) + 3.5} textAnchor="end" fill="var(--muted)" style={{ ...MONO, fontSize: 9.5 }}>
            {fmt(t)}
          </text>
        </g>
      ))}
      {xt.map((t, i) => (
        <g key={`x${i}`}>
          <line x1={sx(t)} x2={sx(t)} y1={m.t + ih} y2={m.t + ih + 4} stroke="var(--hairline-strong)" />
          <text x={sx(t)} y={m.t + ih + 17} textAnchor="middle" fill="var(--muted)" style={{ ...MONO, fontSize: 9.5 }}>
            {t >= 1000 ? `${t / 1000}k` : fmt(t)}
          </text>
        </g>
      ))}
      {vlines.map((v, i) => (
        <g key={`v${i}`}>
          <line x1={sx(v.x)} x2={sx(v.x)} y1={m.t} y2={m.t + ih} stroke={v.color || "var(--warn)"}
            strokeWidth="1.2" strokeDasharray="4 4" opacity="0.75" />
          {v.label && (
            <text x={sx(v.x) + 5} y={m.t + 11} fill={v.color || "var(--warn)"} style={{ ...MONO, fontSize: 9.5 }}>
              {v.label}
            </text>
          )}
        </g>
      ))}
      {hlines.map((h, i) => {
        const left = h.anchor === "left";
        return (
          <g key={`h${i}`}>
            <line x1={m.l} x2={m.l + iw} y1={sy(h.y)} y2={sy(h.y)} stroke={h.color || "var(--muted)"}
              strokeWidth="1.3" strokeDasharray={h.dash || "5 4"} />
            {h.label && (
              <text x={left ? m.l + 4 : m.l + iw - 3} y={sy(h.y) - 5} textAnchor={left ? "start" : "end"}
                fill={h.color || "var(--muted)"} style={{ ...MONO, fontSize: 9.5 }}>
                {h.label}
              </text>
            )}
          </g>
        );
      })}

      {series.map((s, i) => {
        const dim = hover != null && hover !== i;
        const last = s.points[s.points.length - 1];
        return (
          <g key={i} opacity={dim ? 0.22 : 1} style={{ transition: "opacity .18s" }}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            {s.area && s.points.length > 1 && (
              <polygon fill={s.color} opacity="0.09"
                points={`${sx(s.points[0][0])},${m.t + ih} ${s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} ${sx(last[0])},${m.t + ih}`} />
            )}
            {(s.err ?? []).map(([x, lo, hi], j) => (
              <g key={`e${j}`} stroke={s.color} strokeWidth="1.2">
                <line x1={sx(x)} x2={sx(x)} y1={sy(lo)} y2={sy(hi)} />
                <line x1={sx(x) - 4} x2={sx(x) + 4} y1={sy(lo)} y2={sy(lo)} />
                <line x1={sx(x) - 4} x2={sx(x) + 4} y1={sy(hi)} y2={sy(hi)} />
              </g>
            ))}
            {s.points.length > 1 && (
              <polyline fill="none" stroke={s.color} strokeWidth={s.width ?? 2}
                strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" strokeLinecap="round"
                points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
            )}
            {s.points.length <= 40 &&
              s.points.map(([x, y], j) => (
                <circle key={j} cx={sx(x)} cy={sy(y)} r={s.dot ?? 3} fill="var(--ink)" stroke={s.color} strokeWidth="1.8" />
              ))}
            {labelSeries && s.label && (
              <text x={sx(last[0]) + 8} y={sy(last[1]) + 3.5} fill={s.color} style={{ ...MONO, fontSize: 10, fontWeight: 600 }}>
                {s.label}
              </text>
            )}
            {s.values &&
              s.points.map(([x, y], j) => {
                // Sit above the error bar's upper cap when there is one, so the
                // value never lands on top of the whisker it belongs to.
                const e = s.err?.[j];
                const top = e ? Math.min(sy(y), sy(e[2])) : sy(y);
                return (
                  <text key={`v${j}`} x={sx(x)} y={top - 9} textAnchor="middle" fill="var(--text)"
                    style={{ ...MONO, fontSize: 9.5 }}>
                    {fmt(y, valueDigits)}
                  </text>
                );
              })}
          </g>
        );
      })}

      {annotations.map((a, i) => (
        <g key={`a${i}`}>
          <line x1={sx(a.x)} y1={sy(a.y)} x2={sx(a.x) + (a.dx ?? 0)} y2={sy(a.y) + (a.dy ?? -22)}
            stroke="var(--hairline-strong)" strokeWidth="1" />
          <text x={sx(a.x) + (a.dx ?? 0)} y={sy(a.y) + (a.dy ?? -22) - 5} textAnchor={a.anchor || "middle"}
            fill={a.color || "var(--text)"} style={{ ...MONO, fontSize: 9.5 }}>
            {a.text}
          </text>
        </g>
      ))}

      {markers.map((mk, i) => {
        const col = mk.color || "var(--signal)";
        return (
          <g key={`mk${i}`}>
            <circle cx={sx(mk.x)} cy={sy(mk.y)} r="9" fill="none" stroke={col} strokeWidth="1.2" opacity="0.35">
              <animate attributeName="r" values="7;12;7" dur="2.4s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.5;0;0.5" dur="2.4s" repeatCount="indefinite" />
            </circle>
            <circle cx={sx(mk.x)} cy={sy(mk.y)} r="4.5" fill={col} stroke="var(--ink)" strokeWidth="1.5" />
            {mk.label && (
              <text x={sx(mk.x)} y={sy(mk.y) - 20} textAnchor="middle" fill={col} style={{ ...MONO, fontSize: 10, fontWeight: 700 }}>
                {mk.label}
              </text>
            )}
          </g>
        );
      })}

      {xLabel && (
        <text x={m.l + iw / 2} y={height - 4} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10.5 }}>
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text transform={`translate(13,${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10.5 }}>
          {yLabel.replace(/[↑↓→]/g, "").trim()}
        </text>
      )}
    </svg>
  );
}

/**
 * Horizontal DOT plot on a log axis — the right shape for comparing FIDs.
 *
 * Deliberately not a bar chart. A bar encodes its value as a length measured
 * from zero, and a log axis has no zero: the left edge is whatever the smallest
 * value happens to be, so bar lengths would encode a ratio to an arbitrary
 * baseline and would change if a row were added. The dot's position carries the
 * value; nothing else does. `rule` draws a faint leader to the label for
 * scanning only, off by default.
 */
export function LogBars({ rows, width: designW = 640, rowH = 46, unit = "FID", digits = 3, rule = false }) {
  const [wrapRef, width] = useFluidWidth(designW);
  // Clamp the label gutter and value margin to a fraction of the real width, so
  // the plot never collapses on a phone; at the design width these stay at 224/70.
  const m = { l: Math.min(224, width * 0.48), r: Math.min(70, width * 0.16), t: 10, b: 40 };
  const height = m.t + m.b + rows.length * rowH;
  const iw = width - m.l - m.r;
  const lo = Math.min(...rows.map((r) => r.v)) * 0.55;
  const hi = Math.max(...rows.map((r) => r.v)) * 1.7;
  const sx = (v) => m.l + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * iw;

  return (
    <svg ref={wrapRef} viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ overflow: "visible" }} role="img">
      {thin(logTicks(lo, hi), sx, 46).map((t, i) => (
        <g key={i}>
          <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={m.t + rows.length * rowH} stroke="var(--hairline)" />
          <text x={sx(t)} y={height - 24} textAnchor="middle" fill="var(--muted)" style={{ ...MONO, fontSize: 9.5 }}>
            {fmt(t)}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = m.t + i * rowH + rowH / 2;
        return (
          <g key={i}>
            <text x={m.l - 12} y={y - 3} textAnchor="end" fill={r.hi ? "var(--text)" : "var(--muted)"}
              style={{ fontSize: 11.5, fontWeight: r.hi ? 600 : 400 }}>
              {r.label}
            </text>
            {r.sub && (
              <text x={m.l - 12} y={y + 9} textAnchor="end" fill="var(--muted)" style={{ ...MONO, fontSize: 9 }}>
                {r.sub}
              </text>
            )}
            {rule && (
              <line x1={m.l} x2={sx(r.v)} y1={y} y2={y} stroke={r.color} strokeWidth="1" strokeDasharray="2 3" opacity="0.3" />
            )}
            <circle cx={sx(r.v)} cy={y} r={r.hi ? 6.5 : 5} fill={r.color} stroke="var(--ink)" strokeWidth={r.hi ? 1.6 : 0} />
            <text x={sx(r.v) + 12} y={y + 4} fill={r.color} style={{ ...MONO, fontSize: 11.5, fontWeight: 700 }}>
              {fmt(r.v, digits)}
            </text>
          </g>
        );
      })}
      <text x={m.l + iw / 2} y={height - 7} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10 }}>
        {unit}
      </text>
    </svg>
  );
}

/**
 * One row per group, every individual value plotted on a shared log axis, with
 * a reference line. For small n this is the honest alternative to a summary
 * mark: the reader sees the spread rather than a mean that hides it.
 */
export function PairSpread({
  rows,
  width: designW = 720,
  rowH = 34,
  refLine,
  refLabel,
  color = "#22d3ee",
  warnAbove,
  xLabel = "constrained ÷ free (jerk)",
}) {
  const [wrapRef, width] = useFluidWidth(designW);
  const m = { l: Math.min(210, width * 0.42), r: Math.min(52, width * 0.12), t: 12, b: 40 };
  const height = m.t + m.b + rows.length * rowH;
  const iw = width - m.l - m.r;
  const all = rows.flatMap((r) => r.values);
  const lo = Math.min(...all, refLine ?? Infinity) * 0.7;
  const hi = Math.max(...all, refLine ?? -Infinity) * 1.4;
  const sx = (v) => m.l + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * iw;

  return (
    <svg ref={wrapRef} viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ overflow: "visible" }} role="img">
      {thin(logTicks(lo, hi), sx, 40).map((t, i) => (
        <g key={i}>
          <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={m.t + rows.length * rowH} stroke="var(--hairline)" />
          <text x={sx(t)} y={height - 22} textAnchor="middle" fill="var(--muted)" style={{ ...MONO, fontSize: 9.5 }}>
            {fmt(t)}×
          </text>
        </g>
      ))}
      {refLine != null && (
        <g>
          <line x1={sx(refLine)} x2={sx(refLine)} y1={m.t - 4} y2={m.t + rows.length * rowH} stroke="var(--real)" strokeWidth="1.4" strokeDasharray="5 4" />
          {refLabel && (
            <text x={sx(refLine)} y={m.t - 8} textAnchor="middle" fill="var(--real)" style={{ ...MONO, fontSize: 9.5 }}>
              {refLabel}
            </text>
          )}
        </g>
      )}
      {rows.map((r, i) => {
        const y = m.t + i * rowH + rowH / 2;
        const sorted = [...r.values].sort((a, b) => a - b);
        return (
          <g key={`${r.label}-${r.sub}-${i}`}>
            <text x={m.l - 12} y={y - 2} textAnchor="end" fill="var(--text)" style={{ fontSize: 11 }}>
              {r.label}
            </text>
            {r.sub && (
              <text x={m.l - 12} y={y + 9} textAnchor="end" fill="var(--muted)" style={{ ...MONO, fontSize: 8.5 }}>
                {r.sub.length > 34 ? r.sub.slice(0, 33) + "…" : r.sub}
              </text>
            )}
            <line x1={sx(sorted[0])} x2={sx(sorted[sorted.length - 1])} y1={y} y2={y} stroke="var(--hairline-strong)" strokeWidth="1.5" />
            {r.values.map((v, k) => (
              <circle key={k} cx={sx(v)} cy={y} r="4" fill={warnAbove && v > warnAbove ? "var(--warn)" : color} opacity="0.9" />
            ))}
          </g>
        );
      })}
      <text x={m.l + iw / 2} y={height - 5} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10 }}>
        {xLabel}
      </text>
    </svg>
  );
}

/** Matrix of values with a sequential ramp — the ω × steps grid. */
export function HeatMap({ rows, cols, values, rowLabel, colLabel, digits = 3, best = "min" }) {
  const flat = values.flat().filter((v) => v != null);
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const bestV = best === "min" ? lo : hi;
  // Wide-range cyan ramp so adjacent cells are easy to tell apart: bright/light
  // = good (low FID), deep navy = poor. `t` runs 0 (best) → 1 (worst) in log.
  const tt = (v) => (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1);
  const shade = (v) => {
    const t = tt(v);
    return `hsl(${190 - t * 14} ${82 - t * 30}% ${72 - t * 52}%)`;
  };
  const ink = (v) => (tt(v) > 0.42 ? "#dbe4f0" : "#04121a"); // readable on light or dark cells
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse" style={{ ...MONO, fontSize: 12 }}>
        <thead>
          <tr>
            <th className="px-2 py-1.5 text-left text-[9.5px] uppercase tracking-[0.16em] text-[var(--muted)]">
              {rowLabel} \ {colLabel}
            </th>
            {cols.map((c) => (
              <th key={c} className="px-3 py-1.5 text-[10.5px] font-normal text-[var(--muted)]">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r}>
              <td className="whitespace-nowrap px-2 py-1.5 text-right text-[10.5px] text-[var(--muted)]">{r}</td>
              {cols.map((c, j) => {
                const v = values[i][j];
                if (v == null) return <td key={c} className="px-1 py-1" />;
                const isBest = v === bestV;
                return (
                  <td key={c} className="px-1 py-1">
                    <div
                      className="grid h-12 w-[80px] place-items-center rounded-md transition-transform duration-200 hover:scale-[1.08]"
                      style={{
                        background: shade(v),
                        color: ink(v),
                        fontWeight: isBest ? 800 : 600,
                        boxShadow: isBest ? "0 0 0 2px var(--signal), 0 0 26px -3px var(--signal)" : "none",
                      }}
                    >
                      {v.toFixed(digits)}
                      {isBest && <span style={{ fontSize: 8, letterSpacing: "0.1em", opacity: 0.8 }}>BEST</span>}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Paired comparison: two points per row joined by a rule, log x. */
export function Dumbbell({ rows, width: designW = 640, rowH = 34, aLabel, bLabel, aColor, bColor, digits = 4, highlight = -1 }) {
  const [wrapRef, width] = useFluidWidth(designW);
  const m = { l: Math.min(168, width * 0.4), r: Math.min(84, width * 0.2), t: 26, b: 26 };
  const height = m.t + m.b + rows.length * rowH;
  const iw = width - m.l - m.r;
  const all = rows.flatMap((r) => [r.a, r.b]);
  const lo = Math.min(...all) * 0.6, hi = Math.max(...all) * 1.5;
  const sx = (v) => m.l + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * iw;
  return (
    <svg ref={wrapRef} viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ overflow: "visible" }} role="img">
      {thin(logTicks(lo, hi), sx, 46).map((t, i) => (
        <g key={i}>
          <line x1={sx(t)} x2={sx(t)} y1={m.t - 6} y2={m.t + rows.length * rowH} stroke="var(--hairline)" />
          <text x={sx(t)} y={height - 8} textAnchor="middle" fill="var(--muted)" style={{ ...MONO, fontSize: 9.5 }}>{fmt(t)}</text>
        </g>
      ))}
      <text x={m.l} y={12} fill={aColor} style={{ ...MONO, fontSize: 10, fontWeight: 700 }}>{aLabel}</text>
      <text x={m.l + Math.min(110, iw * 0.5)} y={12} fill={bColor} style={{ ...MONO, fontSize: 10, fontWeight: 700 }}>{bLabel}</text>
      {rows.map((r, i) => {
        const y = m.t + i * rowH + rowH / 2;
        const on = i === highlight;
        return (
          <g key={r.label} opacity={highlight < 0 || on ? 1 : 0.4} style={{ transition: "opacity .2s" }}>
            {on && <rect x={m.l - 2} y={y - rowH / 2 + 3} width={iw + 4} height={rowH - 6} rx="5" fill="var(--signal-dim)" />}
            <text x={m.l - 12} y={y + 4} textAnchor="end" fill={on ? "var(--text)" : "var(--muted)"}
              style={{ fontSize: 11.5, fontWeight: on ? 600 : 400 }}>{r.label}</text>
            <line x1={sx(r.a)} x2={sx(r.b)} y1={y} y2={y} stroke="var(--hairline-strong)" strokeWidth="2" />
            <circle cx={sx(r.a)} cy={y} r={on ? 5.5 : 4.5} fill={aColor} />
            <circle cx={sx(r.b)} cy={y} r={on ? 5.5 : 4.5} fill={bColor} />
            <text x={sx(r.b) + 12} y={y + 4} fill={on ? "var(--text)" : "var(--muted)"} style={{ ...MONO, fontSize: 10 }}>
              {(r.b / r.a).toFixed(1)}×
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Vertical bars, one value each, direct-labelled. */
export function BarPlot({ bars, width: designW = 480, height = 240, ymax, hline, digits = 3, yLabel }) {
  const [wrapRef, width] = useFluidWidth(designW);
  const m = { l: 42, r: 12, t: 26, b: 46 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const max = ymax ?? Math.max(...bars.map((b) => b.v)) * 1.25;
  const sy = (v) => m.t + ih - (v / max) * ih;
  const bw = (iw / bars.length) * 0.6;
  return (
    <svg ref={wrapRef} viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ overflow: "visible" }} role="img">
      {linTicks(0, max, 4).map((t, i) => (
        <g key={i}>
          <line x1={m.l} x2={m.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--hairline)" />
          <text x={m.l - 7} y={sy(t) + 3.5} textAnchor="end" fill="var(--muted)" style={{ ...MONO, fontSize: 9 }}>{fmt(t)}</text>
        </g>
      ))}
      {hline != null && (
        <>
          <line x1={m.l} x2={m.l + iw} y1={sy(hline.v)} y2={sy(hline.v)} stroke="var(--real)" strokeWidth="1.4" strokeDasharray="5 4" />
          <text x={m.l + iw} y={sy(hline.v) - 5} textAnchor="end" fill="var(--real)" style={{ ...MONO, fontSize: 9.5 }}>{hline.label}</text>
        </>
      )}
      {bars.map((b, i) => {
        const x = m.l + (i + 0.5) * (iw / bars.length) - bw / 2;
        return (
          <g key={i}>
            <rect x={x} y={sy(b.v)} width={bw} height={m.t + ih - sy(b.v)} fill={b.color} rx="3" opacity="0.92" />
            <text x={x + bw / 2} y={sy(b.v) - 7} textAnchor="middle" fill="var(--text)" style={{ ...MONO, fontSize: 10.5, fontWeight: 700 }}>
              {fmt(b.v, digits)}
            </text>
            <text x={x + bw / 2} y={m.t + ih + 15} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10 }}>{b.label}</text>
            {b.sub && (
              <text x={x + bw / 2} y={m.t + ih + 28} textAnchor="middle" fill="var(--muted)" style={{ ...MONO, fontSize: 8.5 }}>{b.sub}</text>
            )}
          </g>
        );
      })}
      {yLabel && (
        <text transform={`translate(11,${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10 }}>{yLabel}</text>
      )}
    </svg>
  );
}

/**
 * Vertical violin plot: one kernel-density silhouette per group, with the
 * quartiles marked and the individual samples jittered over it. Density is
 * estimated in the axis's own space, so on a log axis the kernel is symmetric
 * in log-jerk, which is where these distributions are roughly symmetric.
 */
export function Violin({ groups, width: designW = 560, height = 330, yLog = true, yLabel, refBand, digits = 4 }) {
  const [wrapRef, width] = useFluidWidth(designW);
  const m = { l: 56, r: 16, t: 18, b: 46 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const tx = yLog ? (v) => Math.log10(Math.max(v, 1e-9)) : (v) => v;

  const all = groups.flatMap((g) => g.values);
  let a = Math.min(...all), b = Math.max(...all);
  if (refBand) {
    a = Math.min(a, refBand.lo);
    b = Math.max(b, refBand.hi);
  }
  a = tx(a);
  b = tx(b);
  const pad = (b - a || 1) * 0.12;
  a -= pad;
  b += pad;
  const sy = (v) => m.t + ih - ((tx(v) - a) / (b - a)) * ih;
  const yt = yLog ? logTicks(10 ** a, 10 ** b) : linTicks(10 ** (a * 0 + a), b, 5);
  const ticks = thin(yLog ? yt : linTicks(a, b, 5), sy, 24);

  const slot = iw / groups.length;

  const kde = (vals) => {
    const xs = vals.map(tx).sort((p, q) => p - q);
    const n = xs.length;
    const mean = xs.reduce((s, v) => s + v, 0) / n;
    const sd = Math.sqrt(xs.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, n - 1)) || 1;
    const h = Math.max(1.06 * sd * n ** -0.2, (b - a) * 0.02);
    const grid = [];
    for (let i = 0; i <= 40; i++) {
      const y = a + ((b - a) * i) / 40;
      let d = 0;
      for (const x of xs) {
        const u = (y - x) / h;
        d += Math.exp(-0.5 * u * u);
      }
      grid.push([y, d / (n * h * Math.sqrt(2 * Math.PI))]);
    }
    const q = (p) => {
      const idx = (n - 1) * p;
      const lo = Math.floor(idx), hi = Math.ceil(idx);
      return 10 ** (xs[lo] + (xs[hi] - xs[lo]) * (idx - lo));
    };
    return { grid, q1: q(0.25), med: q(0.5), q3: q(0.75) };
  };

  return (
    <svg ref={wrapRef} viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ overflow: "visible" }} role="img">
      {refBand && (
        <rect x={m.l} y={sy(refBand.hi)} width={iw} height={Math.abs(sy(refBand.lo) - sy(refBand.hi))}
          fill={refBand.color || "var(--real)"} opacity="0.09" />
      )}
      {refBand?.line != null && (
        <g>
          <line x1={m.l} x2={m.l + iw} y1={sy(refBand.line)} y2={sy(refBand.line)}
            stroke={refBand.color || "var(--real)"} strokeWidth="1.2" strokeDasharray="5 4" />
          <text x={m.l + 4} y={sy(refBand.line) - 5} fill={refBand.color || "var(--real)"} style={{ ...MONO, fontSize: 9.5 }}>
            {refBand.label}
          </text>
        </g>
      )}
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={m.l} x2={m.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--hairline)" />
          <text x={m.l - 8} y={sy(t) + 3.5} textAnchor="end" fill="var(--muted)" style={{ ...MONO, fontSize: 9.5 }}>
            {fmt(t)}
          </text>
        </g>
      ))}

      {groups.map((g, gi) => {
        const cx = m.l + slot * (gi + 0.5);
        const { grid, q1, med, q3 } = kde(g.values);
        const maxD = Math.max(...grid.map((p) => p[1])) || 1;
        const halfW = slot * 0.34;
        const left = grid.map(([y, d]) => `${cx - (d / maxD) * halfW},${sy(10 ** y)}`);
        const right = grid.map(([y, d]) => `${cx + (d / maxD) * halfW},${sy(10 ** y)}`).reverse();
        return (
          <g key={gi}>
            <polygon points={[...left, ...right].join(" ")} fill={g.color} opacity="0.18"
              stroke={g.color} strokeWidth="1.3" strokeOpacity="0.55" />
            {/* quartile box + median */}
            <line x1={cx} x2={cx} y1={sy(q1)} y2={sy(q3)} stroke={g.color} strokeWidth="6" opacity="0.5" strokeLinecap="round" />
            <line x1={cx - halfW * 0.5} x2={cx + halfW * 0.5} y1={sy(med)} y2={sy(med)} stroke="#fff" strokeWidth="1.6" />
            {/* individual samples, jittered */}
            {g.values.map((v, i) => {
              const jx = cx + ((i * 2654435761) % 1000 / 1000 - 0.5) * halfW * 0.9;
              return <circle key={i} cx={jx} cy={sy(v)} r="1.7" fill={g.color} opacity="0.8" />;
            })}
            <text x={cx} y={m.t + ih + 15} textAnchor="middle" fill="var(--text)" style={{ fontSize: 11 }}>{g.label}</text>
            <text x={cx} y={m.t + ih + 28} textAnchor="middle" fill="var(--muted)" style={{ ...MONO, fontSize: 9 }}>
              med {med.toFixed(digits)} · n {g.values.length}
            </text>
          </g>
        );
      })}
      {yLabel && (
        <text transform={`translate(13,${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" fill="var(--muted)" style={{ fontSize: 10.5 }}>
          {yLabel.replace(/[↑↓→]/g, "").trim()}
        </text>
      )}
    </svg>
  );
}


function Empty({ width, height }) {
  return (
    <div className="grid place-items-center rounded-lg border border-[var(--hairline)] text-[11px] text-[var(--muted)]"
      style={{ width: "100%", aspectRatio: `${width} / ${height}` }}>
      no data
    </div>
  );
}
