"use client";

import { useMemo } from "react";

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
  // Optional overlays (all additive, drawn under the series):
  bands = [],   // horizontal shaded bands: [{ y0, y1, color, label }]
  hlines = [],  // horizontal reference lines: [{ y, color, dash, label }]
  regions = [], // vertical shaded x-regions: [{ x0, x1, color, label }]
}) {
  const m = { l: 48, r: 12, t: 12, b: 34 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;

  const dom = useMemo(() => {
    const pts = series.flatMap((s) => s.points);
    if (pts.length === 0) return null;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of pts) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    // Let target lines / bands stretch the y-domain so they're always in frame.
    for (const b of bands) { ymin = Math.min(ymin, b.y0); ymax = Math.max(ymax, b.y1); }
    for (const h of hlines) { ymin = Math.min(ymin, h.y); ymax = Math.max(ymax, h.y); }
    if (yZero) ymin = Math.min(0, ymin);
    if (xmin === xmax) xmax = xmin + 1;
    if (ymin === ymax) ymax = ymin + 1;
    if (equal) {
      // square the data window so x/y share a scale (trajectories)
      const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
      const half = Math.max(xmax - xmin, ymax - ymin) / 2 * 1.1;
      xmin = cx - half; xmax = cx + half; ymin = cy - half; ymax = cy + half;
    }
    return { xmin, xmax, ymin, ymax };
  }, [series, equal, yZero, bands, hlines]);

  if (!dom) {
    return (
      <div className="grid place-items-center rounded-lg border border-[var(--hairline)] bg-ink text-[11px] text-[var(--muted)]" style={{ width, height }}>
        no data
      </div>
    );
  }

  const sx = (x) => m.l + ((x - dom.xmin) / (dom.xmax - dom.xmin)) * iw;
  const sy = (y) => m.t + ih - ((y - dom.ymin) / (dom.ymax - dom.ymin)) * ih;
  const ticks = (lo, hi, n = 5) =>
    Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
  const fmtTick = (v) => {
    const a = Math.abs(v);
    if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(1);
    return Number(v.toFixed(a < 1 ? 3 : a < 100 ? 1 : 0)).toString();
  };

  return (
    <div>
      <svg width={width} height={height} className="overflow-visible">
        {/* gridlines + y ticks */}
        {ticks(dom.ymin, dom.ymax).map((t, i) => (
          <g key={`y${i}`}>
            <line x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} stroke="var(--hairline)" strokeWidth="1" />
            <text x={m.l - 6} y={sy(t) + 3} textAnchor="end" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>
              {fmtTick(t)}
            </text>
          </g>
        ))}
        {/* x ticks */}
        {ticks(dom.xmin, dom.xmax).map((t, i) => (
          <text key={`x${i}`} x={sx(t)} y={height - m.b + 16} textAnchor="middle" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>
            {fmtTick(t)}
          </text>
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
        {/* series — polyline + point markers (so single-point series are visible) */}
        {series.map((s, i) => (
          <g key={i}>
            {s.points.length > 1 && (
              <polyline
                fill="none"
                stroke={s.color}
                strokeWidth="1.6"
                points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")}
              />
            )}
            {s.points.map(([x, y], j) => (
              <circle key={j} cx={sx(x)} cy={sy(y)} r={s.points.length > 60 ? 0 : 2.4} fill={s.color} />
            ))}
          </g>
        ))}
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
              <span className="inline-block h-2 w-3 rounded-sm" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
