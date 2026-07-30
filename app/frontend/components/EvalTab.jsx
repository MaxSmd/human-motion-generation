"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { // shared with §04 — see lib/evalGrid
  FAMILY_RAMPS, MARKERS, cellKey, ckptStepOf, gridCells, loadEvalData, modelOf, stepsOf, stepsVal,
} from "@/lib/evalGrid";
import Katex from "./Katex";
import LineChart from "./LineChart";
import { TableTools } from "./ExportButtons";

// Mirrors server/analysis/eval_tables.COLUMNS (key + KaTeX header). `dir`
// drives winner-bolding: "min"/"max" metrics have a best, others don't.
const COLS = [
  { key: "fid", tex: "\\text{FID}\\downarrow", dir: "min" },
  { key: "r1", tex: "\\text{R@1}\\uparrow", dir: "max" },
  { key: "r2", tex: "\\text{R@2}\\uparrow", dir: "max" },
  { key: "r3", tex: "\\text{R@3}\\uparrow", dir: "max" },
  { key: "mm_dist", tex: "\\text{MM-Dist}\\downarrow", dir: "min" },
  { key: "diversity", tex: "\\text{Diversity}", dir: null },
  { key: "multimodality", tex: "\\text{MModality}", dir: null },
];

const fmt = (v) =>
  v == null ? "—" : Math.abs(v) < 100 ? v.toFixed(3) : v.toFixed(1);

// Run → config / ODE steps / checkpoint resolution, the shared (config × steps ×
// ω) grid and the batched fetch all live in lib/evalGrid — §04 (Model
// head-to-head) reads the same payload and must resolve runs identically.

// §02 Eval comparison: parses ALL available eval runs (no picker — one batched
// fetch), reduces the headline table to the best cell per model config (ω and
// ODE steps both free), and plots every run's guidance sweep.
export default function EvalTab() {
  const [cmp, setCmp] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [nRuns, setNRuns] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [cfgFilter, setCfgFilter] = useState("all"); // chart filter — base's FID ~8 flattens mid's sub-1 curves
  const [xAxis, setXAxis] = useState("omega"); // "omega" (guidance sweep) | "steps" (ODE convergence)
  const headlineRef = useRef(null);
  const perRunRef = useRef(null);

  async function load(force = false) {
    setLoading(true);
    setError(null);
    try {
      // ONE request for everything — the head node serializes SSH, so we avoid
      // a parallel fan-out; previous render stays up while this refreshes. The
      // payload is shared with §04 (loadEvalData caches it), so ↻ must force.
      const { runs, jobs: js, cmp: c } = await loadEvalData({ force });
      setJobs(js);
      setNRuns(runs.length);
      setCmp(c);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Best cell per model config across every run × ω (each run = one step count).
  const bestRows = useMemo(() => {
    const best = {};
    for (const [run, byW] of Object.entries(cmp?.sweeps || {})) {
      const config = modelOf(jobs, run, cmp?.meta);
      for (const [w, m] of Object.entries(byW)) {
        if (m?.fid == null) continue;
        if (!best[config] || m.fid < best[config].fid) {
          best[config] = { config, run, omega: parseFloat(w), steps: stepsOf(jobs, run, cmp?.meta), ...m };
        }
      }
    }
    return Object.values(best).sort((a, b) => a.fid - b.fid);
  }, [cmp, jobs]);

  // Winner per metric column across the config rows (only directed metrics).
  const colWinner = useMemo(() => {
    const out = {};
    for (const { key, dir } of COLS) {
      if (!dir) continue;
      let bi = -1;
      bestRows.forEach((r, i) => {
        if (r[key] == null) return;
        if (bi < 0 || (dir === "min" ? r[key] < bestRows[bi][key] : r[key] > bestRows[bi][key])) bi = i;
      });
      out[key] = bi;
    }
    return out;
  }, [bestRows]);

  // Shared (config × steps × ω) grid — the matrix and the charts read the
  // same cells, so they can never disagree.
  const gridData = useMemo(() => gridCells(cmp?.sweeps, jobs, cmp?.meta), [cmp, jobs]);
  const configs = useMemo(() => [...new Set(gridData.rows.map((r) => r.config))].sort(), [gridData]);
  const chartRows = useMemo(
    () => gridData.rows.filter((r) => cfgFilter === "all" || r.config === cfgFilter),
    [gridData, cfgFilter]
  );

  // Ramp step i of n from a family's validated ramp (dim → bright).
  const rampAt = (ramp, i, n) =>
    n === 1 ? ramp[ramp.length - 2] : ramp[Math.min(ramp.length - 1, Math.round((i * (ramp.length - 1)) / (n - 1)))];

  // x = ω: one series per (config, steps). Hue = config family; lightness =
  // ODE steps rank (dim = few → bright = many); marker shape = step count,
  // shared across configs. Assignment runs over the FULL row set so the
  // config filter never repaints or reshapes surviving series.
  function seriesVsOmega(metric) {
    const stepsOrder = [...new Set(gridData.rows.map((r) => r.steps ?? "?"))]
      .sort((a, b) => stepsVal(a === "?" ? null : a) - stepsVal(b === "?" ? null : b));
    const byFam = {};
    for (const r of gridData.rows) {
      const fam = FAMILY_RAMPS[r.config] ? r.config : "other";
      (byFam[fam] = byFam[fam] || []).push(r);
    }
    const colorOf = new Map();
    for (const rs of Object.values(byFam)) {
      const ramp = FAMILY_RAMPS[FAMILY_RAMPS[rs[0].config] ? rs[0].config : "other"];
      rs.forEach((r, i) => colorOf.set(r, rampAt(ramp, i, rs.length)));
    }
    return chartRows.map((r) => ({
      label: `${r.config.replace(/^dit_/, "")} · ${r.steps ?? "?"} steps`,
      color: colorOf.get(r),
      marker: MARKERS[stepsOrder.indexOf(r.steps ?? "?") % MARKERS.length],
      points: gridData.omegas
        .map((w) => {
          const m = gridData.cells.get(cellKey(r, w));
          return m?.[metric] != null ? [w, m[metric]] : null;
        })
        .filter(Boolean),
    })).filter((s) => s.points.length > 0);
  }

  // ω levels a config has any known-steps result for. Computed over the full
  // grid (filter-stable color/shape assignment).
  const configOmegas = (config) => {
    const rs = gridData.rows.filter((r) => r.config === config && r.steps != null);
    return gridData.omegas.filter((w) => rs.some((r) => gridData.cells.has(cellKey(r, w))));
  };

  // x = ODE steps (transposed): one series per (config, ω) — ALL results, not
  // just multi-step ones; an ω evaluated at a single step count shows as a
  // lone marker at that x. Lightness = ω rank (dim = low ω); marker shape = ω.
  // Only rows with an unknown step count can't be placed on this axis.
  function seriesVsSteps(metric) {
    const omegaOrder = [...new Set(
      [...new Set(gridData.rows.map((r) => r.config))].flatMap((c) => configOmegas(c))
    )].sort((a, b) => a - b);
    const out = [];
    for (const config of [...new Set(chartRows.map((r) => r.config))]) {
      const ws = configOmegas(config);
      const ramp = FAMILY_RAMPS[config] || FAMILY_RAMPS.other;
      const rows = gridData.rows.filter((r) => r.config === config && r.steps != null);
      ws.forEach((w, i) => {
        const points = rows
          .map((r) => {
            const m = gridData.cells.get(cellKey(r, w));
            return m?.[metric] != null ? [r.steps, m[metric]] : null;
          })
          .filter(Boolean)
          .sort((a, b) => a[0] - b[0]);
        if (points.length > 0) out.push({
          label: `${config.replace(/^dit_/, "")} · ω=${w}`,
          color: rampAt(ramp, i, ws.length),
          marker: MARKERS[omegaOrder.indexOf(w) % MARKERS.length],
          points,
        });
      });
    }
    return out;
  }

  // x = training checkpoint step: one series per config, y = best metric (at the
  // best-FID cell) over every ω / ODE-step for that (config, checkpoint). Shows
  // whether more training budget is still buying quality. Runs whose checkpoint
  // step is unknown are dropped from this axis only.
  function seriesVsCheckpoint(metric) {
    const byCfgCkpt = new Map(); // `${config}|${ckpt}` → best-fid cell
    for (const [run, byW] of Object.entries(cmp?.sweeps || {})) {
      const config = modelOf(jobs, run, cmp?.meta);
      const ckpt = ckptStepOf(jobs, run);
      if (ckpt == null) continue;
      for (const m of Object.values(byW)) {
        if (m?.fid == null) continue;
        const key = `${config}|${ckpt}`;
        const cur = byCfgCkpt.get(key);
        if (!cur || m.fid < cur.fid) byCfgCkpt.set(key, { config, ckpt, ...m });
      }
    }
    const byCfg = {};
    for (const cell of byCfgCkpt.values()) (byCfg[cell.config] = byCfg[cell.config] || []).push(cell);
    return Object.entries(byCfg)
      .filter(([config]) => cfgFilter === "all" || config === cfgFilter)
      .map(([config, cells]) => {
        const ramp = FAMILY_RAMPS[config] || FAMILY_RAMPS.other;
        return {
          label: config.replace(/^dit_/, ""),
          color: ramp[ramp.length - 2],
          marker: "circle",
          points: cells.filter((c) => c[metric] != null).sort((a, b) => a.ckpt - b.ckpt).map((c) => [c.ckpt, c[metric]]),
        };
      })
      .filter((s) => s.points.length > 0);
  }

  const chartSeries = (metric) =>
    xAxis === "steps" ? seriesVsSteps(metric)
    : xAxis === "ckpt" ? seriesVsCheckpoint(metric)
    : seriesVsOmega(metric);

  const ckptTicks = useMemo(() => {
    const s = new Set();
    for (const run of Object.keys(cmp?.sweeps || {})) {
      const c = ckptStepOf(jobs, run);
      if (c != null && (cfgFilter === "all" || modelOf(jobs, run, cmp?.meta) === cfgFilter)) s.add(c);
    }
    return [...s].sort((a, b) => a - b);
  }, [cmp, jobs, cfgFilter]);

  const omegas = useMemo(
    () => gridData.omegas.filter((w) => chartRows.some((r) => gridData.cells.has(cellKey(r, w)))),
    [gridData, chartRows]
  );
  const stepTicks = useMemo(
    () => [...new Set(chartRows.filter((r) => r.steps != null).map((r) => r.steps))].sort((a, b) => a - b),
    [chartRows]
  );

  async function copyLatex() {
    if (!cmp?.latex) return;
    await navigator.clipboard.writeText(cmp.latex);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="label">
          best result · per model config (ω and ODE steps free) · {nRuns} runs parsed
          {loading ? " · loading…" : ""}
        </span>
        <span className="flex gap-2">
          {cmp?.latex && (
            <button onClick={copyLatex} className="btn-ghost text-[11px]">
              {copied ? "✓ COPIED" : "COPY LATEX"}
            </button>
          )}
          <button onClick={() => load(true)} disabled={loading} className="btn-ghost text-[11px]">↻ REFRESH</button>
        </span>
      </div>

      {error && <p className="mb-2 text-[12px] text-rose-300">⚠ {error}</p>}

      {!cmp ? (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
          {loading ? "Comparing all eval runs…" : "No eval runs yet — launch one from Lab ▸ Train / Eval."}
        </p>
      ) : (
        <div style={{ opacity: loading ? 0.6 : 1 }}>
          {/* headline: one row per config, its best cell over every run/ω/step */}
          <div className="mb-2 flex justify-end">
            <TableTools getTable={() => headlineRef.current} name="eval-best-per-config"
              caption="Best result per model config (ω and ODE steps free)." label="tab:eval-best" />
          </div>
          <div className="overflow-x-auto">
            <table ref={headlineRef} className="w-full border-collapse text-left text-[12px]">
              <thead>
                <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                  <th className="py-2 pr-4 font-medium">Config</th>
                  <th className="py-2 pr-4 font-medium"><Katex tex="\omega" /></th>
                  <th className="py-2 pr-4 font-medium">Steps</th>
                  {COLS.map((c) => (
                    <th key={c.key} className="py-2 pr-4 font-medium"><Katex tex={c.tex} /></th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono">
                {bestRows.map((r, i) => (
                  <tr key={r.config} className="border-b border-[var(--hairline)]" title={r.run}>
                    <td className="py-2 pr-4 text-slate-300">{r.config}</td>
                    <td className="py-2 pr-4 text-[var(--muted)]">{r.omega?.toFixed(1)}</td>
                    <td className="py-2 pr-4 text-[var(--muted)]">{r.steps ?? "?"}</td>
                    {COLS.map((c) => (
                      <td key={c.key}
                        className={`py-2 pr-4 ${colWinner[c.key] === i ? "font-bold text-[var(--signal)]" : "text-slate-400"}`}>
                        {fmt(r[c.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="label mt-2 normal-case tracking-normal">
            hover a row for its source run · per-run table under “all runs” below
          </p>

          {cmp.errors && Object.keys(cmp.errors).length > 0 && (
            <p className="mt-3 text-[11px] text-[var(--amber)]">
              skipped (no readable results): {Object.keys(cmp.errors).join(", ")}
            </p>
          )}

          {gridData.rows.length > 0 && <FidMatrix grid={gridData} />}

          {/* every (filtered) run, plotted — colors: hue = config, lightness = run */}
          {Object.keys(cmp.sweeps || {}).length > 0 && (
            <div className="mt-6">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <span className="flex items-center gap-3">
                  <span className="label">sweeps · {xAxis === "steps" ? "ODE convergence" : "guidance"}</span>
                  <span className="flex gap-1.5">
                    {["all", ...configs].map((c) => (
                      <button key={c} onClick={() => setCfgFilter(c)}
                        className={`rounded-md px-2.5 py-1 text-[11px] transition ${cfgFilter === c ? "bg-[var(--signal-dim)] text-[var(--signal)] border border-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                        {c}
                      </button>
                    ))}
                  </span>
                  <span className="flex gap-1.5">
                    {[["omega", "x: ω"], ["steps", "x: ODE steps"], ["ckpt", "x: checkpoint"]].map(([k, l]) => (
                      <button key={k} onClick={() => setXAxis(k)}
                        className={`rounded-md px-2.5 py-1 text-[11px] transition ${xAxis === k ? "bg-[var(--signal-dim)] text-[var(--signal)] border border-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                        {l}
                      </button>
                    ))}
                  </span>
                </span>
                <span className="font-mono text-[10px] text-[var(--muted)]">
                  {xAxis === "steps"
                    ? <>steps ∈ {"{"}{stepTicks.join(", ")}{"}"} · cyan = mid · amber = base</>
                    : xAxis === "ckpt"
                    ? <>ckpt ∈ {"{"}{ckptTicks.map((c) => `${(c / 1000).toFixed(0)}k`).join(", ")}{"}"} · best FID per config</>
                    : <>ω ∈ {"{"}{omegas.join(", ")}{"}"} · cyan = mid · amber = base</>}
                </span>
              </div>
              {xAxis === "steps" && (
                <p className="mb-2 text-[11px] text-[var(--muted)]">
                  one series per ω — an ω evaluated at a single step count shows as a lone marker
                </p>
              )}
              {xAxis === "ckpt" && (
                <p className="mb-2 text-[11px] text-[var(--muted)]">
                  best metric over ω / ODE-steps at each training checkpoint — needs ≥2 evals of one config at different checkpoint steps to form a curve
                </p>
              )}
              <p className="mb-2 text-[11px] text-[var(--muted)]">
                <span className="text-[#34d399]">★ green halo</span> = the best value on the plot in that metric&rsquo;s own
                direction (↓ lowest / ↑ highest); the faint ring on each curve is that series&rsquo; own best.
              </p>
              <div className="grid gap-5 lg:grid-cols-2">
                {[
                  { metric: "fid", label: "FID ↓", y: "FID", yZero: true },
                  { metric: "r1", label: "R@1 ↑", y: "R@1", yZero: false },
                  { metric: "r3", label: "R@3 ↑", y: "R@3", yZero: false },
                  { metric: "mm_dist", label: "MM-Dist ↓", y: "MM-Dist", yZero: false },
                ].map((c) => (
                  <figure key={c.metric} className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                    <figcaption className="label mb-2">{c.label}</figcaption>
                    <LineChart series={chartSeries(c.metric)} width={420} height={240}
                      xLabel={xAxis === "steps" ? "ODE steps" : xAxis === "ckpt" ? "training step" : "ω"} yLabel={c.y} yZero={c.yZero}
                      xTicks={xAxis === "steps" ? stepTicks : xAxis === "ckpt" ? ckptTicks : omegas}
                      // direction comes from the same table COLS drive winner-bolding with,
                      // so the highlighted point and the bolded cell can't disagree
                      best={COLS.find((x) => x.key === c.metric)?.dir}
                      bestLabel={`best ${c.y}`}
                      exportName={`eval-${c.metric}-vs-${xAxis}`} />
                  </figure>
                ))}
              </div>
            </div>
          )}

          <details className="mt-5">
            <summary className="label cursor-pointer">all runs · per-run best-FID ω table + LaTeX</summary>
            <div className="mt-3 flex justify-end">
              <TableTools getTable={() => perRunRef.current} name="eval-per-run"
                caption="Per-run best-FID ω results." label="tab:eval-per-run" />
            </div>
            <div className="mt-2 overflow-x-auto">
              <table ref={perRunRef} className="w-full border-collapse text-left text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                    <th className="py-2 pr-4 font-medium">Run</th>
                    <th className="py-2 pr-4 font-medium"><Katex tex="\omega" /></th>
                    {COLS.map((c) => (
                      <th key={c.key} className="py-2 pr-4 font-medium"><Katex tex={c.tex} /></th>
                    ))}
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {(cmp.rows || []).map((r, i) => (
                    <tr key={r.run} className="border-b border-[var(--hairline)]">
                      <td className="py-2 pr-4 text-slate-300">{r.run}</td>
                      <td className="py-2 pr-4 text-[var(--muted)]">{r.omega?.toFixed(1)}</td>
                      {COLS.map((c) => (
                        <td key={c.key}
                          className={`py-2 pr-4 ${cmp.best?.[c.key] === i ? "font-bold text-[var(--signal)]" : "text-slate-400"}`}>
                          {fmt(r[c.key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {cmp.latex && (
              <pre className="mt-3 overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink p-3 text-[11px] text-slate-400">
                {cmp.latex}
              </pre>
            )}
          </details>
        </div>
      )}
    </div>
  );
}

// FID matrix, rows = config × ODE steps, columns = ω. Every row is
// protocol-consistent (one step count throughout), so numbers within a row are
// honestly comparable — mixing step counts per cell made columns uneven AND
// quietly favored ω levels that happened to have high-step runs. Replicates at
// the same (config, steps, ω) collapse to their best; the source run is in the
// cell tooltip. Runs whose step count is unknowable show as "? steps".
function FidMatrix({ grid }) {
  const { cells, rows, omegas } = grid;
  const matrixRef = useRef(null);
  if (cells.size === 0) return null;
  const rowBest = (r) => {
    const vals = omegas.map((w) => cells.get(cellKey(r, w))?.fid).filter((v) => v != null);
    return vals.length ? Math.min(...vals) : null;
  };

  return (
    <div className="mt-6">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="label">FID · config × ODE steps vs ω</span>
        <span className="flex items-center gap-2 font-mono text-[10px] text-[var(--muted)]">
          <span>row best bolded · hover a cell for its run</span>
          <TableTools getTable={() => matrixRef.current} name="eval-fid-matrix"
            caption="FID by config × ODE steps vs guidance ω." label="tab:eval-fid-matrix" />
        </span>
      </div>
      <div className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink px-3 py-2">
        <table ref={matrixRef} className="w-full border-collapse text-left text-[12px]">
          <thead>
            <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
              <th className="py-1.5 pr-4 font-medium">config</th>
              <th className="py-1.5 pr-4 font-medium">steps</th>
              {omegas.map((w) => <th key={w} className="py-1.5 pr-4 font-mono font-medium">ω={w}</th>)}
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => {
              const rb = rowBest(r);
              return (
                <tr key={`${r.config}|${r.steps}`} className="border-b border-[var(--hairline)]">
                  <td className="py-1.5 pr-4 text-slate-300">{r.config}</td>
                  <td className="py-1.5 pr-4 text-[var(--muted)]">{r.steps ?? "?"}</td>
                  {omegas.map((w) => {
                    const b = cells.get(cellKey(r, w));
                    return (
                      <td key={w} className="py-1.5 pr-4" title={b?.run || ""}>
                        {!b ? <span className="text-[var(--muted)]">—</span> : (
                          <span className={b.fid === rb ? "font-bold text-[var(--signal)]" : "text-slate-300"}>{fmt(b.fid)}</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
