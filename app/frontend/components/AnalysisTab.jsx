"use client";

import { useEffect, useMemo, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import LineChart from "./LineChart";
import EvalComparison from "./EvalTab";
import ValidationPanel from "./ValidationPanel";
import { loadNpy } from "@/lib/npy";
import { clipDistance, clipDistanceDetailed } from "@/lib/motionMetrics";
import { Chart, EmptyHint, Section, SIGNAL, SLATE } from "./AnalysisKit";
import BarChart from "./BarChart";
import { cellKey, familyColor, gridCells, loadEvalData, stepsVal } from "@/lib/evalGrid";
import { FALLBACK_JOINTS } from "./ConstraintsTab";

const JOINT_NAMES = FALLBACK_JOINTS.map((j) => j.name);

// Clip comparison and constraint fidelity used to live here as §03/§04; they're
// now tabs of their own (ClipComparisonTab · ConstraintAnalysisTab) beside this
// one in the Lab, so this tab is the run-level view: how training went, how the
// eval scores it, and whether the harness can be trusted.
export default function AnalysisTab() {
  return (
    <div className="space-y-8">
      <Section n="01" title="Training" sub="curves · config · duration"><TrainingCurves /></Section>
      <Section n="02" title="Eval comparison" sub="metrics · guidance + ODE sweeps · LaTeX"><EvalComparison /></Section>
      <Section n="03" title="Sampling behavior" sub="diversity across seeds · ODE-step convergence — from your own renders"><SamplingBehavior /></Section>
      <Section n="04" title="Model head-to-head" sub="mid vs base at matched settings · sample efficiency · guidance robustness"><HeadToHead /></Section>
      <Section n="05" title="Harness calibration" sub="does the eval measure what it claims? — GT self-checks vs published"><ValidationPanel /></Section>
    </div>
  );
}

// ───────────────────────────────────────────────────── training curves + config

const fmtStep = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n % 1000 ? 1 : 0)}k` : String(n));

// A resume that rewinds less than this is a plain requeue picking up where it
// left off — stitched silently. Anything longer is a real rollback to an older
// checkpoint and gets a ↺ seam on the plot.
const MINOR_RESUME_STEPS = 1000;

function fmtDuration(s) {
  if (s == null) return "—";
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

// Robust y-domain for a noisy curve.
//
// A training loss is a low, slowly-decaying band punctuated by gradient storms
// orders of magnitude above it (dit_mid spikes ~50× its working range). Scaling
// to the true max spends the whole axis on those few spikes and flattens the
// part you actually read. So the default domain covers the bulk of the data —
// the boxplot IQR fence — and LineChart clips the storms onto the frame edge
// instead of dropping them.
//
// The fence must come from the quartiles ALONE, never from a tail quantile like
// p99: the backend downsampler (curves._downsample) force-includes storm rows on
// top of the strided backbone precisely so they survive as clip markers, which
// enriches the outliers to a few percent of the delivered rows. A p99 computed on
// that sample sits inside the storm population and blows the axis right back out.
// Quartiles are unaffected by the enrichment, so the fence is stable.
function robustDomain(values, { yZero = false } = {}) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length < 4) return null;
  const q = (p) => v[Math.min(v.length - 1, Math.max(0, Math.round(p * (v.length - 1))))];
  const q1 = q(0.25), q3 = q(0.75), iqr = q3 - q1;
  if (!(iqr > 0)) return null;  // flat column — nothing to clip
  // 3×IQR (not the boxplot's 1.5×) so a clean, steeply-decaying curve keeps its
  // real early-transient top and only genuine storms clip.
  const fence = 3 * iqr;
  // Cap at the fence, but never below the data's own extent — a curve with no
  // outliers is framed exactly as "full range" would frame it.
  const hi = Math.min(v[v.length - 1], q3 + fence);
  const lo = yZero ? 0 : Math.max(v[0], q1 - fence);
  if (!(hi > lo)) return null;
  const pad = (hi - lo) * 0.05;
  return [yZero ? 0 : lo - pad, hi + pad];
}

function TrainingCurves() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [data, setData] = useState(null);
  const [info, setInfo] = useState(null);
  const [col, setCol] = useState("loss");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  // Y-axis framing. "robust" clips outliers to the frame edge, "full" scales to
  // everything, "log" fits a wide range in decades. null = follow the per-column
  // default (`autoMode` below); any click pins an explicit choice.
  const [yMode, setYMode] = useState(null);

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_metrics))).catch(() => {});
  }, []);

  useEffect(() => {
    if (!run) return;
    setLoading(true);
    setError(null);
    setInfo(null);
    api.runMetrics(run)
      .then((d) => { setData(d); if (!d.columns.includes(col)) setCol(d.columns[0]); })
      .catch((e) => { setError(e.message); setData(null); })
      .finally(() => setLoading(false));
    api.runInfo(run).then(setInfo).catch(() => {});
  }, [run]); // eslint-disable-line react-hooks/exhaustive-deps

  // Framing is a property of the column, not a sticky global — loss wants log,
  // grad_norm wants clipping. Drop back to the auto default whenever either the
  // run or the plotted column changes.
  useEffect(() => { setYMode(null); }, [run, col]);

  // `loss` is logged as `accum_loss × grad_accum` (train.py), i.e. grad_accum ×
  // the real per-sample loss. Relaunch a run with a different grad_accum and the
  // curve takes a step that is pure bookkeeping: rmg_mid went 2 → 4 at its tf32
  // relaunch, which doubled the logged loss at 160.1k while training carried on
  // improving (1.70 → 1.55 per sample). Dividing each era by its own grad_accum
  // is what makes the two halves the same quantity. Only `loss` carries the
  // multiplier — grad_norm and the rest are logged raw.
  const eras = data?.eras || null;
  const canNormalize = !!eras && col === "loss" && new Set(eras.map((e) => e.grad_accum)).size > 1;
  const [normalize, setNormalize] = useState(true);
  const divisorAt = (step) => {
    if (!eras) return 1;
    let ga = 1;
    for (const e of eras) if (step >= e.from_step) ga = e.grad_accum;
    return ga || 1;
  };

  const series = useMemo(() => {
    if (!data) return [];
    const xi = data.columns.indexOf("step");
    const yi = data.columns.indexOf(col);
    if (yi < 0) return [];
    const on = canNormalize && normalize;
    const points = data.rows
      .map((r) => {
        const x = xi >= 0 ? r[xi] : 0;
        const y = r[yi];
        return [x, on && y != null ? y / divisorAt(x) : y];
      })
      .filter(([x, y]) => x != null && y != null);
    return [{ label: col, color: SIGNAL, points }];
  }, [data, col, canNormalize, normalize]); // eslint-disable-line react-hooks/exhaustive-deps

  // The clamp only applies in "robust" mode; the count of clipped points is what
  // the caption reports, so the storms stay accounted for even when off-frame.
  const yValues = useMemo(() => series[0]?.points.map(([, y]) => y) || [], [series]);

  // Per-column default framing, decided from the data's shape.
  //
  // Robust clipping was the wrong default for `loss`: on rmg_mid every value
  // above the fence sits in the first 9.6k steps — that's the warmup decay
  // (66 → 5), not a storm, and clipping it amputates the most informative part
  // of the curve. A column with a wide positive dynamic range belongs on a log
  // axis, where a decay AND a spiky band (grad_norm: median 7, max 278) both
  // stay readable with nothing hidden. Clipping is for the rest: bounded columns
  // (lr, t_mean, steps_per_s) where one excursion would flatten the detail.
  const autoMode = useMemo(() => {
    if (yValues.length < 8) return "full";
    const v = yValues.filter(Number.isFinite).sort((a, b) => a - b);
    if (!v.length) return "full";
    const q = (p) => v[Math.round(p * (v.length - 1))];
    // `med / v[0]` keeps a bounded column with one near-zero point off the log
    // axis: a big max/min ratio alone can come from a single small value.
    if (v[0] > 0 && v[v.length - 1] / v[0] >= 8 && q(0.5) / v[0] >= 1.5) return "log";
    return "robust";
  }, [yValues]);
  const mode = yMode ?? autoMode;

  const yDomain = useMemo(
    () => (mode === "robust" ? robustDomain(yValues, { yZero: col === "loss" }) : null),
    [yValues, mode, col],
  );
  const nClipped = yDomain ? yValues.filter((y) => y > yDomain[1] || y < yDomain[0]).length : 0;

  // Resume seams. The backend has already pruned the abandoned branch, so the
  // curve is monotonic in step; these just mark where the run was picked back up
  // from an earlier checkpoint, and how much progress that rollback cost.
  //
  // Only the seams that actually cost progress get a marker: a requeue resuming
  // from the checkpoint it had just written rewinds by one log interval, which
  // is bookkeeping, not history — annotating all of those is the same clutter
  // the abandoned branches were.
  const restarts = data?.restarts || [];
  const vlines = useMemo(
    () => restarts
      .filter((r) => r.abandoned_to - r.step >= MINOR_RESUME_STEPS)
      .map((r) => ({
        x: r.step,
        color: "#fbbf24",
        label: "↺",
        title: `resumed from step ${fmtStep(r.step)} — rolled back ${fmtStep(r.abandoned_to - r.step)} steps `
          + `(${r.dropped} logged point${r.dropped === 1 ? "" : "s"} discarded)`,
      })),
    [restarts],
  );
  // A log axis needs positive values; loss/grad-norm qualify, signed columns don't.
  const canLog = yValues.length > 0 && yValues.every((y) => y > 0);

  // Total training time = Σ Δstep / steps_per_s across logged rows. This is the
  // actual compute time and survives resumes (file mtimes don't — they only
  // reflect the last resume job, which is why mtime under-reports badly).
  const metricDuration = useMemo(() => {
    if (!data) return null;
    const si = data.columns.indexOf("step");
    const ri = data.columns.indexOf("steps_per_s");
    if (si < 0 || ri < 0) return null;
    const rows = data.rows.filter((r) => r[si] != null).sort((a, b) => a[si] - b[si]);
    let sec = 0;
    for (let i = 1; i < rows.length; i++) {
      const ds = rows[i][si] - rows[i - 1][si];
      const rate = rows[i][ri];
      if (ds > 0 && rate > 0) sec += ds / rate;
    }
    return sec > 0 ? Math.round(sec) : null;
  }, [data]);

  // Curve-derived summary: where the run got to and how fast it trained —
  // readable at a glance without hunting through the plotted series.
  const summary = useMemo(() => {
    if (!data) return null;
    const si = data.columns.indexOf("step");
    const li = data.columns.indexOf("loss");
    const ri = data.columns.indexOf("steps_per_s");
    const rows = si >= 0 ? data.rows.filter((r) => r[si] != null).sort((a, b) => a[si] - b[si]) : data.rows;
    if (rows.length === 0) return null;
    const out = [];
    if (si >= 0) out.push(["last step", rows[rows.length - 1][si].toLocaleString()]);
    if (li >= 0) {
      const tail = rows.slice(-20).map((r) => r[li]).filter((v) => v != null);
      if (tail.length) out.push(["final loss (tail avg)", (tail.reduce((a, b) => a + b, 0) / tail.length).toFixed(4)]);
      let best = null, bestStep = null;
      for (const r of rows) if (r[li] != null && (best == null || r[li] < best)) { best = r[li]; bestStep = si >= 0 ? r[si] : null; }
      if (best != null) out.push(["best loss", `${best.toFixed(4)}${bestStep != null ? ` @ ${bestStep.toLocaleString()}` : ""}`]);
    }
    if (ri >= 0) {
      const rates = rows.map((r) => r[ri]).filter((v) => v > 0);
      if (rates.length) out.push(["throughput", `${(rates.reduce((a, b) => a + b, 0) / rates.length).toFixed(2)} it/s`]);
    }
    return out;
  }, [data]);

  const cfg = info?.config;
  const facts = [
    ...(cfg
      ? [
          ["model", cfg.model?.name],
          ["repr", cfg.representation?.name],
          ["max_steps", cfg.train?.max_steps],
          ["subset_n", cfg.data?.subset_n],
          ["subset_frac", cfg.data?.subset_fraction],
          ["lr", cfg.train?.optimizer?.lr],
          ["precision", cfg.train?.precision],
          ["train time", fmtDuration(metricDuration ?? info?.duration_seconds)],
        ]
      : []),
    ...(summary || []),
  ].filter(([, v]) => v !== undefined && v !== null);

  return (
    <div className="surface p-5">
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="block min-w-[220px]">
          <span className="label mb-1.5 block">training run (metrics.csv)</span>
          <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">select run…</option>
            {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
          </select>
        </label>
        {data && (
          <div className="flex flex-wrap gap-1.5">
            {data.columns.filter((c) => c !== "step").map((c) => (
              <button key={c} onClick={() => setCol(c)}
                className={`rounded-md px-2.5 py-1 text-[11px] transition ${col === c ? "bg-[var(--signal-dim)] text-[var(--signal)] border border-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                {c}
              </button>
            ))}
          </div>
        )}
      </div>

      {facts.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-[var(--hairline)] bg-ink px-4 py-3">
          {facts.map(([k, v]) => (
            <div key={k}>
              <div className="label">{k}</div>
              <div className="font-mono text-[13px] text-slate-200">{String(v)}</div>
            </div>
          ))}
        </div>
      )}

      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {loading && <p className="text-[12px] text-[var(--muted)]">loading metrics…</p>}
      {data && (
        <>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="label">y-axis</span>
            {[
              ["robust", "robust", "clip outlier spikes to the frame edge — keeps the working range readable"],
              ["full", "full range", "scale to every point, spikes included"],
              ["log", "log", canLog ? "log scale — spikes and tail both in frame" : `${col} has non-positive values; log needs y > 0`],
            ].map(([id, label, title]) => (
              <button key={id} type="button" onClick={() => setYMode(id)} title={title}
                disabled={id === "log" && !canLog}
                className={`rounded-md px-2.5 py-1 text-[11px] transition disabled:opacity-30 ${mode === id ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                {label}
              </button>
            ))}
            {mode === "robust" && nClipped > 0 && (
              <span className="text-[11px] text-[var(--muted)]">
                {nClipped} of {yValues.length} point{nClipped === 1 ? "" : "s"} off scale (▲ on the edge · hover for the value)
              </span>
            )}
            {restarts.length > 0 && (
              <span className="text-[11px] text-amber-300/70" title={`${restarts.length} resume${restarts.length === 1 ? "" : "s"} total (requeues under ${fmtStep(MINOR_RESUME_STEPS)} steps stitched silently)`}>
                {vlines.length > 0
                  ? `↺ ${vlines.length} rollback${vlines.length === 1 ? "" : "s"} to an older checkpoint — superseded steps pruned (hover a seam)`
                  : "↺ resumes stitched — superseded steps pruned"}
              </span>
            )}
          </div>
          {canNormalize && (
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setNormalize((v) => !v)}
                title="loss is logged as accum_loss × grad_accum, so a relaunch with a different grad_accum steps the curve for free. Dividing each era by its own grad_accum recovers the per-sample loss."
                className={`rounded-md px-2.5 py-1 text-[11px] transition ${normalize ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                ÷ grad_accum
              </button>
              <span className="text-[11px] text-[var(--muted)]">
                {normalize
                  ? `per-sample loss — ${eras.map((e) => `÷${e.grad_accum} from ${fmtStep(e.from_step)}`).join(" · ")}`
                  : `raw logged loss — grad_accum changes ${eras.map((e) => e.grad_accum).join(" → ")} at ${eras.slice(1).map((e) => fmtStep(e.from_step)).join(", ")}, stepping the curve`}
              </span>
            </div>
          )}
          <LineChart series={series} width={760} height={280} xLabel="step"
            yLabel={canNormalize && normalize ? "loss / grad_accum" : col}
            yZero={col === "loss" && mode !== "log"}
            yScale={mode === "log" ? "log" : "linear"}
            yDomain={yDomain} vlines={vlines} />
        </>
      )}
      {!run && !error && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
          Pick a run to plot its training curves + config.
        </p>
      )}
    </div>
  );
}

// ───────────────────────────────── sampling behavior (client-side .npy)
//
// Two questions the model's own renders can answer, both from the pulled joint
// .npy via a root-aligned clip-distance:
//   • DIVERSITY — same prompt, different seeds: how far apart are the samples?
//     (mode-collapse at high ω shows up as near-zero spread.) Re-roll (⟳) or
//     batch the same prompt at different seeds in Create to populate this.
//   • ODE CONVERGENCE — same prompt + seed, different ODE step counts: distance
//     to the highest-step render (a reference "exact" solve). A curve that flattens
//     means the sampler has converged; still-dropping means more steps still help.
function SamplingBehavior() {
  const [clips, setClips] = useState([]);
  const cache = useMemo(() => new Map(), []); // `${job}::${name}` → loadNpy promise

  useEffect(() => {
    api.jobs().then((js) => {
      const out = [];
      for (const j of js) {
        if (j.state !== "done") continue;
        for (const o of j.outputs || []) {
          if (!o.npy_url || o.kind === "gt") continue; // model renders only
          out.push({
            job: j.id, name: o.npy_url.split("/").pop(), url: o.npy_url,
            caption: (o.caption || j.params?.prompts || "").trim(),
            seed: j.params?.seed ?? 0,
            steps: j.params?.num_steps ?? null,
            guidance: j.params?.guidance ?? null,
          });
        }
      }
      setClips(out);
    }).catch(() => {});
  }, []);

  const load = (c) => {
    const k = `${c.job}::${c.name}`;
    if (!cache.has(k)) cache.set(k, loadNpy(mediaUrl(c.url)));
    return cache.get(k);
  };

  return (
    <div className="space-y-6">
      <Diversity clips={clips} load={load} />
      <Convergence clips={clips} load={load} />
    </div>
  );
}

// Seed cloud → distance matrix, decomposed. Same-prompt renders at different
// seeds are the samples of the conditional the model learned; how they spread is
// the whole story. We compute the full pairwise pose-distance matrix once and
// read five things off it: the spread stats + a collapse flag (nearest pair vs
// median), a heatmap (which seeds cluster / which is the outlier), a per-JOINT
// decomposition (where on the body the variation lives), and a per-FRAME envelope
// (when in the clip the samples fan out).
function Diversity({ clips, load }) {
  const [sel, setSel] = useState("");
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);

  // Group by prompt (+ sampling knobs) with ≥2 distinct seeds.
  const groups = useMemo(() => {
    const by = {};
    for (const c of clips) {
      if (!c.caption) continue;
      const k = `${c.caption}|s${c.steps}|w${c.guidance}`;
      (by[k] = by[k] || { key: k, caption: c.caption, steps: c.steps, guidance: c.guidance, members: [] }).members.push(c);
    }
    return Object.values(by)
      .map((g) => ({ ...g, seeds: [...new Set(g.members.map((m) => m.seed))] }))
      .filter((g) => g.seeds.length >= 2);
  }, [clips]);

  const g = groups.find((x) => x.key === sel);
  useEffect(() => {
    if (!g) { setRes(null); return; }
    setBusy(true);
    (async () => {
      try {
        // one clip per distinct seed, ordered by seed for a stable matrix
        const uniq = [];
        const seen = new Set();
        for (const m of g.members) if (!seen.has(m.seed)) { seen.add(m.seed); uniq.push(m); }
        uniq.sort((a, b) => a.seed - b.seed);
        const npys = await Promise.all(uniq.map(load));
        const N = npys.length;

        const mat = Array.from({ length: N }, () => new Array(N).fill(0));
        const rowSum = new Array(N).fill(0);
        const jointAcc = [];         // Σ per-joint distance over pairs
        const frameSum = [];         // Σ per-frame distance over pairs (variable T)
        const frameCnt = [];
        const values = [];           // off-diagonal distances (upper triangle)
        for (let i = 0; i < N; i++)
          for (let k = i + 1; k < N; k++) {
            const dd = clipDistanceDetailed(npys[i], npys[k]);
            if (!dd) continue;
            mat[i][k] = mat[k][i] = dd.mean;
            rowSum[i] += dd.mean; rowSum[k] += dd.mean;
            values.push({ i, k, d: dd.mean });
            dd.perJoint.forEach((v, j) => { jointAcc[j] = (jointAcc[j] || 0) + v; });
            dd.perFrame.forEach((v, f) => { frameSum[f] = (frameSum[f] || 0) + v; frameCnt[f] = (frameCnt[f] || 0) + 1; });
          }
        if (values.length === 0) { setRes(null); return; }

        const ds = values.map((v) => v.d).sort((a, b) => a - b);
        const mean = ds.reduce((a, b) => a + b, 0) / ds.length;
        const median = ds[Math.floor((ds.length - 1) / 2)];
        const min = ds[0], max = ds[ds.length - 1];
        const std = Math.sqrt(ds.reduce((a, b) => a + (b - mean) ** 2, 0) / ds.length);
        const cv = mean ? std / mean : 0;
        const nearest = values.reduce((a, b) => (b.d < a.d ? b : a));
        const farthest = values.reduce((a, b) => (b.d > a.d ? b : a));
        // Centrality: mean distance to the other seeds. Medoid = smallest (most
        // representative render), outlier = largest (the odd one out).
        const central = rowSum.map((s, i) => ({ i, avg: s / (N - 1) }));
        const medoid = central.reduce((a, b) => (b.avg < a.avg ? b : a));
        const outlier = central.reduce((a, b) => (b.avg > a.avg ? b : a));
        // Collapse signal: two seeds far closer than the rest cluster → the model
        // put near-identical mass on them (a fold in the conditional).
        const collapse = median > 0 && nearest.d < 0.45 * median;

        const nPairs = values.length;
        const perJoint = jointAcc.map((v) => v / nPairs);
        const perFrame = frameSum.map((v, f) => [f, v / frameCnt[f]]);

        setRes({
          seeds: uniq.map((m) => m.seed), N, mat, mean, median, min, max, cv,
          nearest, farthest, medoid, outlier, collapse, central,
          perJoint, perFrame, fps: 20,
        });
      } catch { setRes(null); }
      finally { setBusy(false); }
    })();
  }, [sel]); // eslint-disable-line react-hooks/exhaustive-deps

  // Per-joint bars: the joints carrying the most spread, most-varied first.
  const jointBars = useMemo(() => {
    if (!res) return [];
    return res.perJoint
      .map((v, j) => ({ label: JOINT_NAMES[j] || `j${j}`, v }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 12)
      .map((b) => ({ label: b.label, values: [{ v: b.v, color: SIGNAL }] }));
  }, [res]);

  return (
    <div className="surface p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="label">diversity across seeds · pairwise pose-distance decomposed</span>
        <span className="font-mono text-[10px] text-[var(--muted)]">{groups.length} eligible prompt{groups.length === 1 ? "" : "s"}</span>
      </div>
      {groups.length === 0 ? (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-8 text-center text-[12px] text-[var(--muted)]">
          Need the same prompt sampled at ≥2 seeds. In <span className="text-slate-300">Create ▸ Prompt</span>, generate one, hit <span className="text-[var(--signal)]">⟳</span> (new seed), generate again.
        </p>
      ) : (
        <>
          <select className="field-input max-w-2xl" value={sel} onChange={(e) => setSel(e.target.value)}>
            <option value="">select a prompt…</option>
            {groups.map((x) => (
              <option key={x.key} value={x.key}>{`“${x.caption.slice(0, 44)}” · ${x.seeds.length} seeds${x.steps ? ` · ${x.steps} steps` : ""}`}</option>
            ))}
          </select>
          {busy && <p className="mt-3 text-[12px] text-[var(--muted)]">measuring spread…</p>}
          {res && (
            <div className="mt-4 space-y-5">
              {/* spread stat row */}
              <div className="flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-[var(--hairline)] bg-ink px-4 py-3">
                <Stat k="mean spread" v={`${res.mean.toFixed(3)} m`} hi />
                <Stat k="range" v={`${res.min.toFixed(3)} – ${res.max.toFixed(3)}`} />
                <Stat k="uniformity (cv)" v={res.cv.toFixed(2)} />
                <Stat k="medoid seed" v={res.seeds[res.medoid.i]} sub="most typical" />
                <Stat k="outlier seed" v={res.seeds[res.outlier.i]} sub={`${res.outlier.avg.toFixed(3)} m out`} />
                <Stat k="nearest pair"
                  v={`${res.seeds[res.nearest.i]}↔${res.seeds[res.nearest.k]}`}
                  sub={`${res.nearest.d.toFixed(3)} m`} warn={res.collapse} />
              </div>
              {res.collapse && (
                <p className="-mt-2 text-[11px] text-[var(--amber,#fbbf24)]">
                  ⚠ seeds {res.seeds[res.nearest.i]} &amp; {res.seeds[res.nearest.k]} sit far closer than the rest — partial mode-collapse.
                </p>
              )}

              <div className="grid gap-5 lg:grid-cols-2">
                {/* distance-matrix heatmap */}
                <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                  <figcaption className="label mb-2">seed × seed distance (m)</figcaption>
                  <Heatmap seeds={res.seeds} mat={res.mat} max={res.max} />
                </figure>

                {/* per-joint spread bars */}
                <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                  <figcaption className="label mb-2">spread by joint · top movers (m)</figcaption>
                  <BarChart bars={jointBars} width={380} height={260} horizontal labelWidth={64}
                    exportName="seed-diversity-by-joint" />
                </figure>
              </div>

              {/* per-frame divergence envelope */}
              <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                <figcaption className="label mb-2">divergence over the clip · mean pairwise distance per frame</figcaption>
                <LineChart series={[{ label: "spread", color: SIGNAL, points: res.perFrame }]}
                  width={760} height={220} xLabel="frame" yLabel="spread (m)" yZero
                  exportName="seed-divergence-envelope" />
              </figure>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Compact labelled stat for the spread row.
function Stat({ k, v, sub, hi, warn }) {
  return (
    <div>
      <div className="label">{k}</div>
      <div className={`font-mono text-[13px] ${warn ? "text-[var(--amber,#fbbf24)]" : hi ? "text-[var(--signal)]" : "text-slate-200"}`}>{String(v)}</div>
      {sub && <div className="font-mono text-[9px] text-[var(--muted)]">{sub}</div>}
    </div>
  );
}

// N×N distance matrix as a cyan-intensity grid. Diagonal is blank; every cell
// shows its distance and deepens with it, so clusters (dark blocks) and the
// outlier row (a bright band) read at a glance. Cheap inline SVG — N is small.
function Heatmap({ seeds, mat, max }) {
  const N = seeds.length;
  const cell = N <= 5 ? 46 : N <= 8 ? 34 : 26;
  const gut = 26; // label gutter
  const size = gut + N * cell;
  const norm = (d) => (max > 0 ? Math.min(1, d / max) : 0);
  return (
    <svg width={size} height={size} className="overflow-visible">
      {seeds.map((s, i) => (
        <text key={`r${i}`} x={gut - 5} y={gut + i * cell + cell / 2 + 3} textAnchor="end"
          className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>{s}</text>
      ))}
      {seeds.map((s, j) => (
        <text key={`c${j}`} x={gut + j * cell + cell / 2} y={gut - 6} textAnchor="middle"
          className="fill-[var(--muted)]" style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}>{s}</text>
      ))}
      {mat.map((row, i) => row.map((d, j) => {
        const x = gut + j * cell, y = gut + i * cell;
        if (i === j) return <rect key={`${i}-${j}`} x={x + 1} y={y + 1} width={cell - 2} height={cell - 2} rx="2" fill="var(--hairline)" opacity="0.35" />;
        const a = 0.1 + 0.9 * norm(d);
        return (
          <g key={`${i}-${j}`}>
            <rect x={x + 1} y={y + 1} width={cell - 2} height={cell - 2} rx="2" fill={SIGNAL} opacity={a}>
              <title>{`seed ${seeds[i]} ↔ ${seeds[j]}: ${d.toFixed(4)} m`}</title>
            </rect>
            {cell >= 34 && (
              <text x={x + cell / 2} y={y + cell / 2 + 3} textAnchor="middle" pointerEvents="none"
                style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: a > 0.55 ? "#04121a" : "var(--muted)" }}>
                {d.toFixed(2)}
              </text>
            )}
          </g>
        );
      }))}
    </svg>
  );
}

function Convergence({ clips, load }) {
  const [sel, setSel] = useState("");
  const [res, setRes] = useState(null); // { points:[[steps,dist]], ref }
  const [busy, setBusy] = useState(false);

  // Group by (prompt, seed) with ≥2 distinct ODE-step counts.
  const groups = useMemo(() => {
    const by = {};
    for (const c of clips) {
      if (!c.caption || c.steps == null) continue;
      const k = `${c.caption}|seed${c.seed}|w${c.guidance}`;
      (by[k] = by[k] || { key: k, caption: c.caption, seed: c.seed, guidance: c.guidance, members: [] }).members.push(c);
    }
    return Object.values(by)
      .map((grp) => ({ ...grp, steps: [...new Set(grp.members.map((m) => m.steps))].sort((a, b) => a - b) }))
      .filter((grp) => grp.steps.length >= 2);
  }, [clips]);

  const g = groups.find((x) => x.key === sel);
  useEffect(() => {
    if (!g) { setRes(null); return; }
    setBusy(true);
    (async () => {
      try {
        const uniq = [];
        const seen = new Set();
        for (const m of g.members.slice().sort((a, b) => a.steps - b.steps)) if (!seen.has(m.steps)) { seen.add(m.steps); uniq.push(m); }
        const npys = await Promise.all(uniq.map(load));
        const ref = npys[npys.length - 1]; // highest step count = reference solve
        const refSteps = uniq[uniq.length - 1].steps;
        const points = uniq.slice(0, -1).map((m, i) => [m.steps, clipDistance(npys[i], ref)]).filter(([, d]) => d != null);
        setRes({ points, refSteps });
      } catch { setRes(null); }
      finally { setBusy(false); }
    })();
  }, [sel]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="surface p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="label">ODE convergence · distance to the highest-step render</span>
        <span className="font-mono text-[10px] text-[var(--muted)]">{groups.length} eligible</span>
      </div>
      {groups.length === 0 ? (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-8 text-center text-[12px] text-[var(--muted)]">
          Need the same prompt + <span className="text-slate-300">seed</span> at ≥2 ODE-step counts. Generate one at e.g. 20 steps, then bump <span className="text-slate-300">ODE steps</span> (keep the seed) and generate again.
        </p>
      ) : (
        <>
          <select className="field-input max-w-2xl" value={sel} onChange={(e) => setSel(e.target.value)}>
            <option value="">select a prompt + seed…</option>
            {groups.map((x) => (
              <option key={x.key} value={x.key}>{`“${x.caption.slice(0, 40)}” · seed ${x.seed} · steps {${x.steps.join(",")}}`}</option>
            ))}
          </select>
          {busy && <p className="mt-3 text-[12px] text-[var(--muted)]">measuring convergence…</p>}
          {res && res.points.length > 0 && (
            <div className="mt-3">
              <LineChart series={[{ label: `distance to ${res.refSteps}-step`, color: SIGNAL, points: res.points }]}
                width={520} height={240} xLabel="ODE steps" yLabel="Δ to reference (m)" yZero
                xTicks={res.points.map((p) => p[0])} exportName="ode-convergence" />
              <p className="label mt-1 normal-case tracking-normal">flattening → converged; still dropping → more steps still change the sample.</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────── model head-to-head (mid vs base)
//
// §02 plots every eval curve; this section answers the question those curves are
// evidence FOR: is the bigger config actually better, and where does the win
// come from? Everything here is derived from the same batched eval payload
// (lib/evalGrid) so the two sections can't disagree.
//
//   • MATCHED-PAIR DELTA — the headline. Compare the two configs only on cells
//     where BOTH were evaluated at the same (ODE steps, ω), so the gap can't be
//     an artifact of one config having been tuned harder. Per metric: median
//     relative gain, how many pairs the challenger wins, and an exact two-sided
//     sign test on the win count — with a handful of ω levels, "wins 5 of 6" is
//     the kind of claim that needs a p-value attached.
//   • SAMPLE EFFICIENCY — best FID per ODE-step count (ω free), normalized to
//     each config's own optimum. Not "which is better" but "how many sampling
//     steps does each need to reach its own ceiling" — the quality-per-step read
//     that pairs with §03's convergence measurement.
//   • GUIDANCE ROBUSTNESS — the spread of FID across ω for each config. A config
//     whose score collapses away from its best ω is one whose headline number is
//     largely tuning; a flat one is genuinely better everywhere.

// Metrics compared pairwise. `dir` = which way is better; `div_gap` is derived
// (|diversity − diversity_real|), since diversity alone has no direction —
// matching the real data's diversity is the goal, exceeding it is not.
const H2H_METRICS = [
  { key: "fid", label: "FID", dir: "min" },
  { key: "r1", label: "R@1", dir: "max" },
  { key: "r2", label: "R@2", dir: "max" },
  { key: "r3", label: "R@3", dir: "max" },
  { key: "mm_dist", label: "MM-Dist", dir: "min" },
  { key: "div_gap", label: "|Div − Div_real|", dir: "min" },
];

const metricVal = (m, key) => {
  if (key !== "div_gap") return m?.[key] ?? null;
  if (m?.diversity == null || m?.diversity_real == null) return null;
  return Math.abs(m.diversity - m.diversity_real);
};

const median = (xs) => {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return null;
  const i = v.length >> 1;
  return v.length % 2 ? v[i] : (v[i - 1] + v[i]) / 2;
};

// Exact two-sided sign test: P(at least as lopsided as `w` wins in `n` ties-free
// trials | p = 0.5). Kept exact rather than normal-approximated because n here
// is the number of matched (steps, ω) cells — typically under 20.
function signTestP(w, n) {
  if (!n) return null;
  const k = Math.min(w, n - w);
  let logC = 0, sum = 0; // log C(n, i) built up incrementally to avoid overflow
  for (let i = 0; i <= k; i++) {
    if (i > 0) logC += Math.log((n - i + 1) / i);
    sum += Math.exp(logC - n * Math.LN2);
  }
  return Math.min(1, 2 * sum);
}

const pct = (v) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`);
const num = (v) => (v == null ? "—" : Math.abs(v) < 100 ? v.toFixed(3) : v.toFixed(1));
const short = (c) => c.replace(/^dit_/, "");

function HeadToHead() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pick, setPick] = useState(null); // [A, B] — null = auto (baseline vs best)

  useEffect(() => {
    loadEvalData()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const grid = useMemo(
    () => gridCells(data?.cmp?.sweeps, data?.jobs || [], data?.cmp?.meta),
    [data],
  );

  // Per-config summary over ALL its cells: best FID (and where), plus the
  // step counts and ω levels it was evaluated at.
  const byConfig = useMemo(() => {
    const out = new Map();
    for (const r of grid.rows) {
      for (const w of grid.omegas) {
        const m = grid.cells.get(cellKey(r, w));
        if (!m) continue;
        const cur = out.get(r.config) || { config: r.config, cells: [], best: null };
        const cell = { steps: r.steps, omega: w, ...m };
        cur.cells.push(cell);
        if (!cur.best || cell.fid < cur.best.fid) cur.best = cell;
        out.set(r.config, cur);
      }
    }
    return [...out.values()].sort((a, b) => a.best.fid - b.best.fid);
  }, [grid]);

  const configs = useMemo(() => byConfig.map((c) => c.config), [byConfig]);

  // Auto pick: challenger = best-FID config, baseline = the next one. Explicit
  // A/B selection wins, and survives a refetch only if both are still present.
  const [A, B] = useMemo(() => {
    if (pick && configs.includes(pick[0]) && configs.includes(pick[1])) return pick;
    if (configs.length < 2) return [configs[0] || null, null];
    return [configs[1], configs[0]]; // A = baseline (worse best FID), B = challenger
  }, [pick, configs]);

  // Matched cells: same ODE-step count AND same ω, evaluated for both configs.
  const paired = useMemo(() => {
    if (!A || !B) return [];
    const stepsSet = new Set(grid.rows.filter((r) => r.config === A || r.config === B).map((r) => r.steps ?? "?"));
    const out = [];
    for (const s of stepsSet) {
      const steps = s === "?" ? null : s;
      for (const w of grid.omegas) {
        const a = grid.cells.get(cellKey({ config: A, steps }, w));
        const b = grid.cells.get(cellKey({ config: B, steps }, w));
        if (a && b) out.push({ steps, omega: w, a, b });
      }
    }
    return out.sort((p, q) => (stepsVal(p.steps) - stepsVal(q.steps)) || (p.omega - q.omega));
  }, [grid, A, B]);

  // Per-metric matched-pair stats. Relative gain is signed so positive always
  // means "B better", whichever way the metric points.
  const stats = useMemo(() => {
    return H2H_METRICS.map(({ key, label, dir }) => {
      const rels = [], as = [], bs = [];
      let wins = 0, ties = 0;
      for (const p of paired) {
        const av = metricVal(p.a, key), bv = metricVal(p.b, key);
        if (av == null || bv == null) continue;
        as.push(av); bs.push(bv);
        const gain = dir === "min" ? (av - bv) / Math.abs(av) : (bv - av) / Math.abs(av);
        rels.push(gain);
        if (gain > 0) wins++;
        else if (gain === 0) ties++;
      }
      const n = rels.length, decided = n - ties;
      return {
        key, label, dir, n, wins,
        medA: median(as), medB: median(bs),
        rel: median(rels),
        p: decided ? signTestP(wins, decided) : null,
      };
    }).filter((s) => s.n > 0);
  }, [paired]);

  const fidStat = stats.find((s) => s.key === "fid");

  // FID scatter: one point per matched cell, (baseline, challenger). Everything
  // under the y = x diagonal is a cell the challenger wins — the whole claim in
  // one picture, with no averaging to hide behind.
  const scatter = useMemo(() => {
    const pts = paired.filter((p) => p.a.fid != null && p.b.fid != null).map((p) => [p.a.fid, p.b.fid]);
    if (!pts.length) return null;
    const lo = Math.min(...pts.flat()), hi = Math.max(...pts.flat());
    return {
      series: [
        { label: "parity (y = x)", color: SLATE, dash: "5 4", points: [[lo, lo], [hi, hi]] },
        { label: `${short(A)} → ${short(B)} · matched cells`, color: SIGNAL, marker: "circle", scatter: true, points: pts },
      ],
    };
  }, [paired, A, B]);

  // Sample efficiency: best FID at each ODE-step count (ω free), as a ratio to
  // that config's own best. 1.0 = at its ceiling; 1.2 = 20% worse than the best
  // that config ever reaches, purely from having fewer sampling steps.
  const efficiency = useMemo(() => {
    const series = [], summary = [];
    for (const { config, cells } of byConfig) {
      const byStep = new Map();
      for (const c of cells) {
        if (c.steps == null) continue;
        const cur = byStep.get(c.steps);
        if (!cur || c.fid < cur) byStep.set(c.steps, c.fid);
      }
      if (byStep.size < 2) continue;
      const best = Math.min(...byStep.values());
      const pts = [...byStep.entries()].sort((x, y) => x[0] - y[0]).map(([s, f]) => [s, f / best]);
      series.push({ label: short(config), color: familyColor(config), marker: "circle", points: pts });
      // Cheapest step count still within 5% of this config's own ceiling.
      const sat = pts.find(([, r]) => r <= 1.05);
      summary.push({ config, best, saturateAt: sat ? sat[0] : null, steps: pts.map((p) => p[0]) });
    }
    return { series, summary };
  }, [byConfig]);

  // Guidance robustness at each config's best step count: how far the score
  // travels across ω. spread = (worst − best) / best.
  const robustness = useMemo(() =>
    byConfig.map(({ config, cells, best }) => {
      const atBestSteps = cells.filter((c) => c.steps === best.steps && c.fid != null);
      const fids = atBestSteps.map((c) => c.fid);
      if (fids.length < 2) return { config, n: fids.length, bestOmega: best.omega, best: best.fid };
      const lo = Math.min(...fids), hi = Math.max(...fids);
      return {
        config, n: fids.length, steps: best.steps,
        bestOmega: atBestSteps.find((c) => c.fid === lo)?.omega ?? best.omega,
        best: lo, worst: hi, spread: (hi - lo) / lo,
      };
    }), [byConfig]);

  const spreadBars = useMemo(
    () => robustness.filter((r) => r.spread != null).map((r) => ({
      label: `${short(r.config)} (${r.n} ω)`,
      values: [{ v: r.spread * 100, color: familyColor(r.config) }],
    })),
    [robustness],
  );

  if (loading && !data) {
    return <div className="surface p-5"><p className="text-[12px] text-[var(--muted)]">loading eval results…</p></div>;
  }
  if (error) {
    return <div className="surface p-5"><p className="text-[12px] text-rose-300">⚠ {error}</p></div>;
  }
  if (configs.length < 2) {
    return (
      <div className="surface p-5">
        <EmptyHint>
          Needs eval results for at least <span className="text-slate-300">two model configs</span> — currently{" "}
          {configs.length ? `only ${configs.join(", ")}` : "none"}. Run an eval sweep for the other config from Lab ▸ Train / Eval.
        </EmptyHint>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ── matched-pair delta ── */}
      <div className="surface p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <span className="label">head-to-head · matched (ODE steps, ω)</span>
            <select value={A} onChange={(e) => setPick([e.target.value, B])}
              className="rounded-md border border-[var(--hairline)] bg-ink px-2 py-1 font-mono text-[11px] text-slate-300">
              {configs.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <span className="text-[11px] text-[var(--muted)]">→</span>
            <select value={B} onChange={(e) => setPick([A, e.target.value])}
              className="rounded-md border border-[var(--hairline)] bg-ink px-2 py-1 font-mono text-[11px] text-[var(--signal)]">
              {configs.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </span>
          <span className="font-mono text-[10px] text-[var(--muted)]">
            {paired.length} matched cell{paired.length === 1 ? "" : "s"}
          </span>
        </div>

        {A === B ? (
          <EmptyHint>Pick two different configs.</EmptyHint>
        ) : paired.length === 0 ? (
          <EmptyHint>
            No (ODE steps, ω) setting was evaluated for <span className="text-slate-300">both</span> {short(A)} and{" "}
            {short(B)}. Re-run one config's eval at a step count and guidance the other already has — an unmatched
            comparison can't separate the model from its tuning.
          </EmptyHint>
        ) : (
          <>
            <div className="mb-4 flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-[var(--hairline)] bg-ink px-4 py-3">
              <Stat k="matched cells" v={paired.length} sub="same steps + ω" />
              {fidStat && <Stat k="median FID gain" v={pct(fidStat.rel)} hi={fidStat.rel > 0} warn={fidStat.rel <= 0}
                sub={`${short(B)} vs ${short(A)}`} />}
              {fidStat && <Stat k="FID wins" v={`${fidStat.wins} / ${fidStat.n}`} />}
              {fidStat?.p != null && <Stat k="sign test" v={fidStat.p < 0.001 ? "p < 0.001" : `p = ${fidStat.p.toFixed(3)}`}
                sub="two-sided, exact" />}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                    <th className="py-2 pr-4 font-medium">Metric</th>
                    <th className="py-2 pr-4 font-medium">median {short(A)}</th>
                    <th className="py-2 pr-4 font-medium">median {short(B)}</th>
                    <th className="py-2 pr-4 font-medium">median gain</th>
                    <th className="py-2 pr-4 font-medium">wins</th>
                    <th className="py-2 pr-4 font-medium">sign test</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {stats.map((s) => (
                    <tr key={s.key} className="border-b border-[var(--hairline)]">
                      <td className="py-2 pr-4 text-slate-300">
                        {s.label} <span className="text-[10px] text-[var(--muted)]">{s.dir === "min" ? "↓" : "↑"}</span>
                      </td>
                      <td className="py-2 pr-4 text-slate-400">{num(s.medA)}</td>
                      <td className="py-2 pr-4 text-slate-400">{num(s.medB)}</td>
                      <td className={`py-2 pr-4 font-bold ${s.rel > 0 ? "text-[var(--signal)]" : s.rel < 0 ? "text-rose-300" : "text-slate-400"}`}>
                        {pct(s.rel)}
                      </td>
                      <td className="py-2 pr-4 text-slate-400">{s.wins} / {s.n}</td>
                      <td className="py-2 pr-4 text-[var(--muted)]">
                        {s.p == null ? "—" : s.p < 0.001 ? "<0.001" : s.p.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {scatter && (
              <div className="mt-4">
                <Chart title={`FID per matched cell · below the diagonal = ${short(B)} wins`}>
                  <LineChart series={scatter.series} width={520} height={280} equal
                    xLabel={`${short(A)} FID`} yLabel={`${short(B)} FID`} exportName="head-to-head-fid" />
                </Chart>
              </div>
            )}

            <p className="label mt-2 normal-case tracking-normal">
              Only settings evaluated for both configs are compared, so the gap can&apos;t come from one config having
              been swept harder than the other. Gain is signed toward {short(B)} for every metric (a lower MM-Dist and a
              smaller diversity gap both count as positive). The sign test asks how likely this many wins would be from
              a coin flip — with a handful of ω levels, a clean sweep can still land at p ≈ 0.06.
            </p>
          </>
        )}
      </div>

      {/* ── sample efficiency ── */}
      <div className="surface p-5">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="label">sample efficiency · FID vs ODE steps, relative to each config&apos;s own best</span>
          <span className="font-mono text-[10px] text-[var(--muted)]">{efficiency.series.length} config{efficiency.series.length === 1 ? "" : "s"}</span>
        </div>
        {efficiency.series.length === 0 ? (
          <EmptyHint>
            Needs a config evaluated at ≥2 distinct <span className="text-slate-300">ODE step</span> counts. Re-run one
            eval with a different <code>num_sample_steps</code> and this curve appears.
          </EmptyHint>
        ) : (
          <>
            <div className="mb-4 flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-[var(--hairline)] bg-ink px-4 py-3">
              {efficiency.summary.map((s) => (
                <Stat key={s.config} k={short(s.config)}
                  v={s.saturateAt != null ? `${s.saturateAt} steps` : "not reached"}
                  sub={`within 5% of its best (FID ${num(s.best)})`} />
              ))}
            </div>
            <Chart title="best FID at each step count ÷ that config's best FID (1.0 = at its ceiling)">
              <LineChart series={efficiency.series} width={520} height={260}
                xLabel="ODE steps" yLabel="FID ÷ config's best"
                xTicks={[...new Set(efficiency.summary.flatMap((s) => s.steps))].sort((a, b) => a - b)}
                hlines={[{ y: 1.05, color: SLATE, dash: "4 4", label: "+5%" }]}
                exportName="sample-efficiency" />
            </Chart>
            <p className="label mt-2 normal-case tracking-normal">
              Each curve is normalized to its OWN best, so this is not a quality comparison — it&apos;s how many sampling
              steps each config needs to reach its ceiling. A curve that is already flat at 20 steps buys nothing from
              more; one still falling at the right edge is being under-sampled and its headline number understates it.
              §03 measures the same convergence on your own renders, without an eval.
            </p>
          </>
        )}
      </div>

      {/* ── guidance robustness ── */}
      <div className="surface p-5">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="label">guidance robustness · FID spread across ω</span>
          <span className="font-mono text-[10px] text-[var(--muted)]">at each config&apos;s best step count</span>
        </div>
        {spreadBars.length === 0 ? (
          <EmptyHint>Needs ≥2 guidance levels evaluated for a config at one step count.</EmptyHint>
        ) : (
          <>
            <Chart title="(worst ω − best ω) ÷ best ω, in % — lower = less of the score is tuning">
              <BarChart bars={spreadBars} width={520} height={Math.max(110, 40 + spreadBars.length * 34)}
                horizontal labelWidth={110} unit="%" exportName="guidance-robustness" />
            </Chart>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full border-collapse text-left text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                    <th className="py-2 pr-4 font-medium">Config</th>
                    <th className="py-2 pr-4 font-medium">ω*</th>
                    <th className="py-2 pr-4 font-medium">FID at ω*</th>
                    <th className="py-2 pr-4 font-medium">worst ω</th>
                    <th className="py-2 pr-4 font-medium">spread</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {robustness.map((r) => (
                    <tr key={r.config} className="border-b border-[var(--hairline)]">
                      <td className="py-2 pr-4 text-slate-300">{r.config}</td>
                      <td className="py-2 pr-4 text-slate-400">{r.bestOmega?.toFixed(1) ?? "—"}</td>
                      <td className="py-2 pr-4 text-[var(--signal)]">{num(r.best)}</td>
                      <td className="py-2 pr-4 text-slate-400">{num(r.worst)}</td>
                      <td className="py-2 pr-4 text-slate-400">{r.spread == null ? "—" : `${(r.spread * 100).toFixed(0)}%`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="label mt-2 normal-case tracking-normal">
              How much a config&apos;s headline FID depends on landing the right guidance scale. A large spread means the
              published number is a tuned corner of the ω sweep and will not survive a different ω; a small spread means
              the config is better everywhere. §02 has the underlying curves.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

