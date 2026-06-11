"use client";

import { useEffect, useState } from "react";
import { api, mediaUrl } from "@/lib/api";

export default function TrainingViewerTab() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [steps, setSteps] = useState([]);
  const [stepIdx, setStepIdx] = useState(0);
  const [clips, setClips] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!run) return;
    setError(null);
    api
      .runSteps(run)
      .then((s) => {
        setSteps(s);
        setStepIdx(0);
      })
      .catch((e) => setError(e.message));
  }, [run]);

  const step = steps[stepIdx];

  useEffect(() => {
    if (!run || step == null) return;
    setLoading(true);
    setError(null);
    api
      .runSample(run, step)
      .then((res) => setClips(res.clips))
      .catch((e) => {
        setError(e.message);
        setClips([]);
      })
      .finally(() => setLoading(false));
  }, [run, step]);

  return (
    <div className="space-y-5">
      {/* transport bar */}
      <div className="surface flex flex-wrap items-center gap-5 p-5">
        <label className="block min-w-[200px]">
          <span className="label mb-1.5 block">run</span>
          <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">select a run…</option>
            {runs.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>

        {steps.length > 0 && (
          <div className="flex-1">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="label">training step</span>
              <span className="font-mono text-sm text-[var(--signal)]">
                {step}
                <span className="ml-2 text-[11px] text-[var(--muted)]">
                  {stepIdx + 1}/{steps.length}
                </span>
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={steps.length - 1}
              value={stepIdx}
              onChange={(e) => setStepIdx(Number(e.target.value))}
              className="w-full accent-[var(--signal)]"
            />
            <div className="mt-1 flex justify-between text-[10px] text-[var(--muted)]">
              <span>{steps[0]}</span>
              <span>watch the model learn →</span>
              <span>{steps[steps.length - 1]}</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <p className="rounded-lg border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-[12px] text-rose-300">
          ⚠ {error}
        </p>
      )}

      {!run && !error && (
        <div className="surface px-6 py-16 text-center">
          <p className="label">select a run to scrub its fixed-prompt samples</p>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {clips.map((c, i) => (
          <figure key={i} className="surface overflow-hidden p-2">
            <div className="viewport aspect-square">
              <div className="flex h-full w-full items-center justify-center p-2">
                {loading ? (
                  <span className="h-7 w-7 animate-spin rounded-full border-2 border-[var(--hairline-strong)] border-t-[var(--signal)]" />
                ) : c.media_url.toLowerCase().endsWith(".mp4") ? (
                  <video src={mediaUrl(c.media_url)} className="max-h-full max-w-full rounded" controls autoPlay loop muted />
                ) : (
                  <img src={mediaUrl(c.media_url)} alt={c.text} className="max-h-full max-w-full rounded" />
                )}
              </div>
            </div>
            <figcaption className="px-1.5 pb-1 pt-2.5 text-[12px] text-slate-400">{c.text}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
