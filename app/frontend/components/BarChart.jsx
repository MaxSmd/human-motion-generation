"use client";

import { useRef } from "react";
import { downloadSvg, downloadPng } from "@/lib/exportSvg";
import { barChartToTikz, downloadText } from "@/lib/exportLatex";
import { niceTicks, fmtNum } from "@/lib/numfmt";

// Dependency-free grouped bar chart. `bars` = [{ label, values: [{v, color}] }].
// One group per label; bars within a group are drawn side by side (e.g. GT vs
// gen for the same joint). Horizontal by default (good for 22 joint names).
//
// `hlines` = [{ v, color, dash, label }] draws reference marks on the VALUE axis
// (vertical rules when horizontal, horizontal when not) — for a baseline the bars
// should be read against, e.g. ground-truth jerk.
export default function BarChart({
  bars = [],
  width = 420,
  height = 300,
  unit = "",
  horizontal = true,
  exportName = null,
  legend = null, // [{ label, color }]
  hlines = [],
  labelWidth = 78, // value-axis gutter; widen for labels longer than a joint name
}) {
  const svgRef = useRef(null);
  // Tolerate malformed bars (missing/undefined `values`) so a partial payload
  // renders an empty chart rather than crashing the whole tab.
  bars = bars.filter((b) => Array.isArray(b?.values));
  hlines = hlines.filter((h) => Number.isFinite(h?.v));
  // reference lines join the max so one that outruns every bar still lands on-axis
  const max = Math.max(1e-9, ...bars.flatMap((b) => b.values.map((x) => x.v || 0)), ...hlines.map((h) => h.v));
  const m = horizontal ? { l: labelWidth, r: 40, t: 6, b: 22 } : { l: 44, r: 8, t: 8, b: 46 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;

  if (bars.length === 0) {
    return <div className="grid place-items-center rounded-lg border border-[var(--hairline)] bg-ink text-[11px] text-[var(--muted)]" style={{ width, height }}>no data</div>;
  }

  const band = (horizontal ? ih : iw) / bars.length;
  const inner = Math.min(band * 0.7, 18);
  // 4 ticks along the long (value) axis, 3 when it's the short vertical one
  const valTicks = niceTicks(0, max, horizontal ? 4 : 3);

  return (
    <div className="relative">
      {exportName && (
        <div className="absolute right-0 top-0 z-10 flex gap-1 opacity-40 transition hover:opacity-100">
          <button type="button" onClick={() => downloadSvg(svgRef.current, `${exportName}.svg`)}
            className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">⤓ svg</button>
          <button type="button" onClick={() => downloadPng(svgRef.current, `${exportName}.png`)}
            className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">⤓ png</button>
          <button type="button" title="download pgfplots/TikZ LaTeX"
            onClick={() => downloadText(barChartToTikz(bars, { unit, horizontal, legend: legend || [], width, height }), `${exportName}.tex`)}
            className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">⤓ tex</button>
        </div>
      )}
      <svg ref={svgRef} width={width} height={height}>
        {/* value axis on round numbers (0, 0.25, 0.5 … ) rather than fractions
            of the data max, which read as 1.0e+3 / 0.033 */}
        {valTicks.ticks.map((v) => {
          const f = v / max;
          const x = horizontal ? m.l + f * iw : null;
          const y = horizontal ? null : m.t + ih - f * ih;
          return horizontal ? (
            <g key={v}>
              <line x1={x} x2={x} y1={m.t} y2={m.t + ih} stroke="var(--hairline)" />
              <text x={x} y={height - 8} textAnchor="middle" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>{fmtNum(v, valTicks.step)}</text>
            </g>
          ) : (
            <g key={v}>
              <line x1={m.l} x2={m.l + iw} y1={y} y2={y} stroke="var(--hairline)" />
              <text x={m.l - 6} y={y + 3} textAnchor="end" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>{fmtNum(v, valTicks.step)}</text>
            </g>
          );
        })}
        {bars.map((b, i) => {
          const nv = b.values.length;
          const sub = inner / nv;
          return b.values.map((val, k) => {
            const len = (Math.abs(val.v || 0) / max) * (horizontal ? iw : ih);
            if (horizontal) {
              const y = m.t + i * band + (band - inner) / 2 + k * sub;
              return <rect key={`${i}-${k}`} x={m.l} y={y} width={Math.max(0, len)} height={Math.max(1, sub - 1)} fill={val.color} rx="1" />;
            }
            const x = m.l + i * band + (band - inner) / 2 + k * sub;
            return <rect key={`${i}-${k}`} x={x} y={m.t + ih - len} width={Math.max(1, sub - 1)} height={Math.max(0, len)} fill={val.color} rx="1" />;
          });
        })}
        {hlines.map((h, i) => {
          const f = (h.v || 0) / max;
          const stroke = h.color || "var(--muted)";
          if (horizontal) {
            const x = m.l + f * iw;
            return (
              <g key={`h${i}`}>
                <line x1={x} x2={x} y1={m.t} y2={m.t + ih} stroke={stroke} strokeDasharray={h.dash || "4 4"} />
                {h.label && (
                  <text x={x + 3} y={m.t + 8} className="fill-current" style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: stroke }}>{h.label}</text>
                )}
              </g>
            );
          }
          const y = m.t + ih - f * ih;
          return (
            <g key={`h${i}`}>
              <line x1={m.l} x2={m.l + iw} y1={y} y2={y} stroke={stroke} strokeDasharray={h.dash || "4 4"} />
              {h.label && (
                <text x={m.l + iw - 3} y={y - 3} textAnchor="end" style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: stroke }}>{h.label}</text>
              )}
            </g>
          );
        })}
        {bars.map((b, i) => {
          const pos = m.t + i * band + band / 2;
          return horizontal ? (
            <text key={i} x={m.l - 6} y={pos + 3} textAnchor="end" className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>{b.label}</text>
          ) : (
            <text key={i} x={m.l + i * band + band / 2} y={height - 6} textAnchor="end" transform={`rotate(-45 ${m.l + i * band + band / 2} ${height - 6})`} className="fill-[var(--muted)]" style={{ fontSize: 8, fontFamily: "var(--font-mono)" }}>{b.label}</text>
          );
        })}
      </svg>
      {legend && legend.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-3 pl-2">
          {legend.map((l) => (
            <span key={l.label} className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: l.color }} /> {l.label}{unit ? ` (${unit})` : ""}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// Bin a flat value series into `nbins` for an overlaid histogram. Returns
// { bars, legend } ready for BarChart, given one or more named+coloured series.
export function histogramBars(seriesList, nbins = 20) {
  const all = seriesList.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  if (all.length === 0) return { bars: [], legend: [] };
  const lo = Math.min(...all), hi = Math.max(...all);
  const w = (hi - lo) / nbins || 1;
  const counts = seriesList.map(() => new Array(nbins).fill(0));
  seriesList.forEach((s, si) => {
    for (const v of s.values) {
      if (!Number.isFinite(v)) continue;
      const b = Math.min(nbins - 1, Math.max(0, Math.floor((v - lo) / w)));
      counts[si][b]++;
    }
  });
  // normalize each series to a fraction so different-length clips compare fairly
  const totals = counts.map((c) => c.reduce((a, b) => a + b, 0) || 1);
  const bars = Array.from({ length: nbins }, (_, b) => ({
    label: fmtNum(lo + (b + 0.5) * w, w),
    values: seriesList.map((s, si) => ({ v: counts[si][b] / totals[si], color: s.color })),
  }));
  return { bars, legend: seriesList.map((s) => ({ label: s.label, color: s.color })) };
}
