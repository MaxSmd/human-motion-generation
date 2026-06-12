"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import Katex from "./Katex";
import LineChart from "./LineChart";

const RUN_COLORS = ["#22d3ee", "#fbbf24", "#a78bfa", "#34d399", "#fb7185", "#60a5fa"];

// Mirrors server/analysis/eval_tables.COLUMNS (key + KaTeX header).
const COLS = [
  { key: "fid", tex: "\\text{FID}\\downarrow" },
  { key: "r1", tex: "\\text{R@1}\\uparrow" },
  { key: "r2", tex: "\\text{R@2}\\uparrow" },
  { key: "r3", tex: "\\text{R@3}\\uparrow" },
  { key: "mm_dist", tex: "\\text{MM-Dist}\\downarrow" },
  { key: "diversity", tex: "\\text{Diversity}" },
  { key: "multimodality", tex: "\\text{MModality}" },
];

const fmt = (v) =>
  v == null ? "—" : Math.abs(v) < 100 ? v.toFixed(3) : v.toFixed(1);

export default function EvalTab() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [cmp, setCmp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api.evalRuns()
      .then((r) => {
        setRuns(r);
        setSelected(new Set(r.slice(0, 4).map((x) => x.run)));
      })
      .catch((e) => setError(e.message));
  }, []);

  const toggle = (run) =>
    setSelected((s) => {
      const n = new Set(s);
      n.has(run) ? n.delete(run) : n.add(run);
      return n;
    });

  async function compare() {
    setLoading(true);
    setError(null);
    try {
      // ONE request — the comparison response carries the full per-ω sweep too
      // (the head node serializes SSH, so we avoid a parallel fan-out).
      setCmp(await api.analysisTable([...selected]));
    } catch (e) {
      setError(e.message);
      setCmp(null);
    } finally {
      setLoading(false);
    }
  }

  // Build [{label,color,points:[[omega,value]]}] for a metric across runs from
  // the comparison response's `sweeps` map ({run: {omega: metrics}}).
  function sweepSeries(metric) {
    const runs = Object.keys(cmp?.sweeps || {});
    return runs.map((run, i) => ({
      label: run.length > 18 ? run.slice(0, 17) + "…" : run,
      color: RUN_COLORS[i % RUN_COLORS.length],
      points: Object.entries(cmp.sweeps[run])
        .map(([w, m]) => [parseFloat(w), m[metric]])
        .filter(([, y]) => y != null)
        .sort((a, b) => a[0] - b[0]),
    }));
  }

  async function copyLatex() {
    if (!cmp?.latex) return;
    await navigator.clipboard.writeText(cmp.latex);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[320px_1fr]">
      {/* run picker */}
      <div className="surface p-5">
        <div className="label mb-3">eval runs · {runs.length}</div>
        {error && <p className="mb-2 text-[12px] text-rose-300">⚠ {error}</p>}
        <ul className="max-h-[60vh] space-y-1 overflow-y-auto pr-1">
          {runs.map((r) => (
            <li key={r.run}>
              <label className="flex cursor-pointer items-center gap-2 rounded-md border border-[var(--hairline)] px-3 py-2 text-[12px] hover:border-[var(--hairline-strong)]">
                <input
                  type="checkbox"
                  checked={selected.has(r.run)}
                  onChange={() => toggle(r.run)}
                  className="accent-[var(--signal)]"
                />
                <span className="truncate font-mono text-slate-300">{r.run}</span>
              </label>
            </li>
          ))}
          {runs.length === 0 && !error && (
            <li className="text-[12px] text-[var(--muted)]">
              No eval runs yet — launch one from the Cluster tab.
            </li>
          )}
        </ul>
        <button
          onClick={compare}
          disabled={loading || selected.size === 0}
          className="btn-signal mt-4 w-full"
        >
          {loading ? "COMPARING…" : `COMPARE ${selected.size} RUN${selected.size === 1 ? "" : "S"}`}
        </button>
      </div>

      {/* comparison table */}
      <div className="surface p-5">
        <div className="mb-3 flex items-center justify-between">
          <span className="label">run comparison · best-FID ω, winner bolded</span>
          {cmp?.latex && (
            <button onClick={copyLatex} className="btn-ghost text-[11px]">
              {copied ? "✓ COPIED" : "COPY LATEX"}
            </button>
          )}
        </div>

        {!cmp ? (
          <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
            Select runs and compare to render a paper-style results table.
          </p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                    <th className="py-2 pr-4 font-medium">Run</th>
                    <th className="py-2 pr-4 font-medium"><Katex tex="\omega" /></th>
                    {COLS.map((c) => (
                      <th key={c.key} className="py-2 pr-4 font-medium">
                        <Katex tex={c.tex} />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {cmp.rows.map((r, i) => (
                    <tr key={r.run} className="border-b border-[var(--hairline)]">
                      <td className="py-2 pr-4 text-slate-300">{r.run}</td>
                      <td className="py-2 pr-4 text-[var(--muted)]">{r.omega?.toFixed(1)}</td>
                      {COLS.map((c) => {
                        const win = cmp.best?.[c.key] === i;
                        return (
                          <td
                            key={c.key}
                            className={`py-2 pr-4 ${win ? "font-bold text-[var(--signal)]" : "text-slate-400"}`}
                          >
                            {fmt(r[c.key])}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {cmp.errors && Object.keys(cmp.errors).length > 0 && (
              <p className="mt-3 text-[11px] text-[var(--amber)]">
                skipped: {Object.keys(cmp.errors).join(", ")}
              </p>
            )}

            <details className="mt-4">
              <summary className="label cursor-pointer">LaTeX source</summary>
              <pre className="mt-2 overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink p-3 text-[11px] text-slate-400">
                {cmp.latex}
              </pre>
            </details>

            {cmp.sweeps && Object.keys(cmp.sweeps).length > 0 && (
              <div className="mt-6">
                <div className="label mb-3">guidance sweep · metric vs ω</div>
                <div className="grid gap-5 lg:grid-cols-2">
                  <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                    <figcaption className="label mb-2">FID ↓ vs ω</figcaption>
                    <LineChart series={sweepSeries("fid")} width={360} height={240} xLabel="ω" yLabel="FID" yZero />
                  </figure>
                  <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                    <figcaption className="label mb-2">R@3 ↑ vs ω</figcaption>
                    <LineChart series={sweepSeries("r3")} width={360} height={240} xLabel="ω" yLabel="R@3" yZero />
                  </figure>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
