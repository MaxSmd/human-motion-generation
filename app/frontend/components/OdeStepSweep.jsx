"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import LineChart from "./LineChart";
import { ACTIVE } from "@/lib/useVizJob";

const SIGNAL = "#22d3ee";

// Every metric the eval writes — the sweep plots all of them vs ODE step count.
const METRICS = [
  { key: "fid", label: "FID ↓ vs ODE steps", y: "FID", yZero: true },
  { key: "r1", label: "R@1 ↑ vs ODE steps", y: "R@1" },
  { key: "r2", label: "R@2 ↑ vs ODE steps", y: "R@2" },
  { key: "r3", label: "R@3 ↑ vs ODE steps", y: "R@3" },
  { key: "mm_dist", label: "MM-Dist ↓ vs ODE steps", y: "MM-Dist" },
  { key: "diversity", label: "Diversity vs ODE steps", y: "Diversity" },
  { key: "multimodality", label: "MModality vs ODE steps", y: "MModality" },
];

const stepRange = (start, stop, inc) => {
  const out = [];
  if (!(inc > 0)) return out;
  for (let s = start; s <= stop; s += inc) out.push(s);
  return out;
};

// Parse "[6.5]" (or "6.5") → 6.5.
const parseGuidance = (s) => {
  const v = parseFloat(String(s ?? "").replace(/[[\]]/g, ""));
  return Number.isFinite(v) ? v : null;
};

// An ODE-step sweep launches one eval run per num_sample_steps value (results.json
// is keyed by ω, and num_sample_steps is a per-run scalar), all at the same fixed
// guidance. Jobs share a `sweep_id` so they can be regrouped after a reload; the
// backend's local queue runs them serially. Once each run's results.json lands,
// analysisTable() batches the fetch (one SSH round-trip) and we plot every metric
// against the step count.
export default function OdeStepSweep() {
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);
  const [form, setForm] = useState({ guidance: 6.5, start: 50, stop: 250, inc: 50, max_clips: 256 });
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState(null);

  const [jobs, setJobs] = useState([]);
  const [activeSweepId, setActiveSweepId] = useState("");

  const [results, setResults] = useState(null);
  const [resultsError, setResultsError] = useState(null);
  const [resultsLoading, setResultsLoading] = useState(false);

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const steps = useMemo(() => stepRange(form.start, form.stop, form.inc), [form.start, form.stop, form.inc]);

  // Poll all jobs so a sweep's progress updates live (and past sweeps stay listed).
  useEffect(() => {
    let alive = true;
    const tick = () => api.jobs().then((j) => alive && setJobs(j)).catch(() => {});
    tick();
    const t = setInterval(tick, 4000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // Regroup eval jobs into sweeps: explicitly by shared sweep_id (launched from
  // this panel), and implicitly by (checkpoint, guidance) for manually launched
  // eval runs — so e.g. a 200-step headline run plus a 400-step confirm at the
  // same ω form a plottable step sweep without having been tagged. (newest first)
  const sweeps = useMemo(() => {
    const by = new Map();
    for (const j of jobs) {
      if (j.kind !== "eval") continue;
      let id = j.params?.sweep_id;
      if (!id) {
        const ckpt = j.params?.checkpoint;
        if (!ckpt || j.params?.num_sample_steps == null) continue;
        id = `auto:${ckpt}|ω${parseGuidance(j.params?.guidance_scales) ?? "?"}`;
      }
      if (!by.has(id)) by.set(id, []);
      by.get(id).push(j);
    }
    return [...by.entries()]
      .map(([id, js]) => {
        const sorted = [...js].sort((a, b) => (a.params?.num_sample_steps ?? 0) - (b.params?.num_sample_steps ?? 0));
        const createdAt = Math.min(...js.map((j) => j.submitted_at || 0));
        return {
          id, jobs: sorted, createdAt,
          implicit: id.startsWith("auto:"),
          checkpoint: sorted[0]?.params?.checkpoint,
          guidance: parseGuidance(sorted[0]?.params?.guidance_scales),
          stepList: [...new Set(sorted.map((j) => j.params?.num_sample_steps).filter((s) => s != null))],
        };
      })
      .sort((a, b) => b.createdAt - a.createdAt);
  }, [jobs]);

  const selected = sweeps.find((s) => s.id === activeSweepId) || sweeps[0] || null;

  // Auto-select the newest sweep once one exists (keeps view pinned to a launch).
  useEffect(() => {
    if (!activeSweepId && sweeps.length) setActiveSweepId(sweeps[0].id);
  }, [sweeps, activeSweepId]);

  // The runs of the selected sweep that already wrote results — refetched (batched)
  // as more complete, so the charts fill in incrementally.
  const doneRuns = useMemo(
    () => (selected ? selected.jobs.filter((j) => j.state === "done" && j.run_name).map((j) => j.run_name) : []),
    [selected]
  );
  const doneKey = doneRuns.join(",");

  useEffect(() => {
    if (!doneKey) { setResults(null); setResultsError(null); return; }
    let alive = true;
    setResultsLoading(true);
    setResultsError(null);
    api.analysisTable(doneKey.split(","))
      .then((r) => alive && setResults(r))
      .catch((e) => alive && setResultsError(e.message))
      .finally(() => alive && setResultsLoading(false));
    return () => { alive = false; };
  }, [doneKey]);

  async function launch() {
    if (!checkpoint || steps.length === 0) return;
    const id = `ode-${Date.now()}`;
    setLaunching(true);
    setLaunchError(null);
    try {
      // Serial POSTs — each enqueues on the backend's local queue (one cluster
      // job at a time), so we never stack the SLURM queue directly.
      for (const s of steps) {
        await api.submitEval({
          checkpoint,
          model_preset: presets?.model_preset,
          train_preset: presets?.train_preset,
          guidance_scales: `[${form.guidance}]`,
          num_sample_steps: s,
          max_clips: form.max_clips,
          sweep_id: id,
        });
      }
      setActiveSweepId(id);
    } catch (e) {
      setLaunchError(e.message);
    } finally {
      setLaunching(false);
    }
  }

  // [num_sample_steps, metric] points for the selected sweep, from the batched
  // comparison response (sweeps[run] is keyed by ω; a step sweep has one ω/run).
  function metricSeries(metric) {
    if (!results || !selected) return [];
    const gKey = selected.guidance != null ? String(selected.guidance) : null;
    const points = selected.jobs
      .map((j) => {
        const byW = results.sweeps?.[j.run_name];
        if (!byW) return null;
        const wk = gKey && byW[gKey] ? gKey : Object.keys(byW)[0];
        const v = byW[wk]?.[metric];
        const x = j.params?.num_sample_steps;
        return x != null && v != null ? [x, v] : null;
      })
      .filter(Boolean)
      .sort((a, b) => a[0] - b[0]);
    return [{ label: selected.guidance != null ? `ω=${selected.guidance}` : "sweep", color: SIGNAL, points }];
  }

  const xTicks = selected?.stepList?.length ? selected.stepList : null;
  const nActive = selected ? selected.jobs.filter((j) => ACTIVE.has(j.state)).length : 0;
  const nDone = doneRuns.length;
  const ckptLabel = (p) => (p ? p.split("/").slice(-2).join("/") : "—");

  return (
    <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
      {/* launcher */}
      <div className="surface p-5">
        <div className="label mb-3">launch ODE-step sweep</div>
        <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
          <Field label="max_clips"><input type="number" className="field-input" value={form.max_clips} onChange={set("max_clips")} /></Field>
          <Field label="steps start"><input type="number" className="field-input" value={form.start} onChange={set("start")} /></Field>
          <Field label="steps stop"><input type="number" className="field-input" value={form.stop} onChange={set("stop")} /></Field>
          <Field label="increment"><input type="number" className="field-input" value={form.inc} onChange={set("inc")} /></Field>
        </div>
        <p className="label mt-3 normal-case tracking-normal">
          {steps.length
            ? <>{steps.length} eval run{steps.length === 1 ? "" : "s"} · steps {steps.join(", ")} · fixed ω={form.guidance}</>
            : "set a positive increment with stop ≥ start"}
        </p>
        {steps.length > 8 && (
          <p className="mt-1 text-[11px] text-[var(--amber)]">
            {steps.length} runs ≈ {steps.length}× a single eval (~40 min each on the full split) — they queue serially.
          </p>
        )}
        <button onClick={launch} disabled={launching || !checkpoint || steps.length === 0} className="btn-signal mt-4 w-full">
          {launching ? "LAUNCHING…" : `▶  LAUNCH ${steps.length} EVAL${steps.length === 1 ? "" : "S"}`}
        </button>
        {launchError && <p className="mt-2 text-[12px] text-rose-300">⚠ {launchError}</p>}
        <p className="label mt-3 text-center">one eval run per step count · all at the same guidance</p>
      </div>

      {/* results */}
      <div className="surface p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <span className="label">metric vs ODE steps · fixed guidance</span>
          {sweeps.length > 0 && (
            <select className="field-input max-w-[280px]" value={selected?.id || ""} onChange={(e) => setActiveSweepId(e.target.value)}>
              {sweeps.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.implicit ? "manual · " : ""}{ckptLabel(s.checkpoint)} · ω={s.guidance ?? "?"} · {s.jobs.length} run{s.jobs.length === 1 ? "" : "s"}
                </option>
              ))}
            </select>
          )}
        </div>

        {!selected ? (
          <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
            Launch a sweep to plot every metric against the number of ODE sampling steps.
          </p>
        ) : (
          <>
            {/* per-step job status */}
            <div className="mb-4 overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink px-3 py-2">
              <div className="mb-1 flex items-center justify-between">
                <span className="label normal-case tracking-normal">
                  {ckptLabel(selected.checkpoint)} · ω={selected.guidance ?? "?"} · {nDone}/{selected.jobs.length} done
                  {nActive > 0 ? ` · ${nActive} running/queued` : ""}
                </span>
                {resultsLoading && <span className="text-[11px] text-[var(--muted)]">loading…</span>}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {selected.jobs.map((j) => (
                  <span key={j.id} className="rounded-md border border-[var(--hairline)] px-2 py-0.5 font-mono text-[10px]"
                    style={{ color: j.state === "done" ? SIGNAL : j.state === "failed" ? "#fb7185" : "var(--muted)" }}>
                    {j.params?.num_sample_steps}s · {j.state}
                  </span>
                ))}
              </div>
            </div>

            {resultsError && <p className="mb-3 text-[12px] text-rose-300">⚠ {resultsError}</p>}
            {results?.errors && Object.keys(results.errors).length > 0 && (
              <p className="mb-3 text-[11px] text-[var(--amber)]">skipped: {Object.keys(results.errors).length} run(s) had no readable results yet</p>
            )}

            {nDone === 0 ? (
              <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
                No results yet — charts fill in as each step's eval run completes.
              </p>
            ) : (
              <div className="grid gap-5 lg:grid-cols-2">
                {METRICS.map((c) => (
                  <figure key={c.key} className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
                    <figcaption className="label mb-2">{c.label}</figcaption>
                    <LineChart series={metricSeries(c.key)} width={420} height={240}
                      xLabel="ODE steps" yLabel={c.y} yZero={c.yZero} xTicks={xTicks} />
                  </figure>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}
