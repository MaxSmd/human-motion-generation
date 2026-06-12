"use client";

import { useEffect, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import VizJobResult from "./VizJobResult";
import { useVizJob } from "@/lib/useVizJob";

export default function TrainingViewerTab({ clusterMode }) {
  return clusterMode ? <TrainingCluster /> : <TrainingLocal />;
}

// ───────────────────────────────────────────── cluster: render a run's sample dumps

function TrainingCluster() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const { job, error, submitting, run: launch } = useVizJob();

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_samples))).catch(() => {});
  }, []);

  return (
    <div className="space-y-5">
      <form
        className="surface flex flex-wrap items-end gap-4 p-5"
        onSubmit={(e) => { e.preventDefault(); if (run) launch({ mode: "samples", run }); }}
      >
        <label className="block min-w-[240px]">
          <span className="label mb-1.5 block">run (remote, has samples)</span>
          <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">select run…</option>
            {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
          </select>
        </label>
        <button type="submit" className="btn-signal" disabled={submitting || !run}>
          {submitting ? "SUBMITTING…" : "▶  RENDER SAMPLES"}
        </button>
        <p className="label">renders the trainer's fixed-prompt dumps across saved steps — scrub the strip to watch it learn</p>
      </form>

      <div className="surface p-5">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Pick a run and render its training-time samples." />
      </div>
    </div>
  );
}

// ───────────────────────────────────────────── local mode (RMG_CLUSTER_MODE=0)

function TrainingLocal() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [steps, setSteps] = useState([]);
  const [stepIdx, setStepIdx] = useState(0);
  const [clips, setClips] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => { api.runs().then(setRuns).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    if (!run) return;
    setError(null);
    api.runSteps(run).then((s) => { setSteps(s); setStepIdx(0); }).catch((e) => setError(e.message));
  }, [run]);
  const step = steps[stepIdx];
  useEffect(() => {
    if (!run || step == null) return;
    setLoading(true); setError(null);
    api.runSample(run, step).then((res) => setClips(res.clips)).catch((e) => { setError(e.message); setClips([]); }).finally(() => setLoading(false));
  }, [run, step]);

  return (
    <div className="space-y-5">
      <div className="surface flex flex-wrap items-center gap-5 p-5">
        <label className="block min-w-[200px]">
          <span className="label mb-1.5 block">run</span>
          <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">select a run…</option>
            {runs.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </label>
        {steps.length > 0 && (
          <div className="flex-1">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="label">training step</span>
              <span className="font-mono text-sm text-[var(--signal)]">{step}<span className="ml-2 text-[11px] text-[var(--muted)]">{stepIdx + 1}/{steps.length}</span></span>
            </div>
            <input type="range" min={0} max={steps.length - 1} value={stepIdx} onChange={(e) => setStepIdx(Number(e.target.value))} className="w-full accent-[var(--signal)]" />
          </div>
        )}
      </div>
      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {clips.map((c, i) => (
          <figure key={i} className="surface overflow-hidden p-2">
            {c.media_url.toLowerCase().endsWith(".mp4")
              ? <video src={mediaUrl(c.media_url)} className="w-full rounded" controls autoPlay loop muted />
              : <img src={mediaUrl(c.media_url)} alt={c.text} className="w-full rounded" />}
            <figcaption className="px-1.5 pb-1 pt-2 text-[12px] text-slate-400">{c.text}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
