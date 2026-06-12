"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import { useVizJob } from "@/lib/useVizJob";

const MODES = [
  { id: "clip", label: "GT clips", hint: "render stored ground-truth motions" },
  { id: "compare", label: "GT vs prediction", hint: "GT + the model's prediction on each clip's caption" },
  { id: "samples", label: "Training samples", hint: "the trainer's fixed-prompt dumps across saved steps" },
];

// Parse / serialise the comma-separated clip-id field as a set so the dropdown,
// the chips and the text box all stay in sync.
const parseClips = (s) => (s || "").split(",").map((c) => c.trim()).filter(Boolean);
const joinClips = (arr) => arr.join(",");

export default function VisualizeTab() {
  const [mode, setMode] = useState("clip");
  const [clips, setClips] = useState("000019,000021,000022");
  const [checkpoint, setCheckpoint] = useState("");
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  // GT-clip browse (id + assigned caption) — hints which clips exist.
  const [gtClips, setGtClips] = useState([]);
  const [gtErr, setGtErr] = useState(null);
  const [subsetFraction, setSubsetFraction] = useState(0.01);
  const [subsetSeed, setSubsetSeed] = useState(0);
  // Training-sample step selection (avoid rendering an entire dir of dumps).
  const [steps, setSteps] = useState([]);
  const [stepsErr, setStepsErr] = useState(null);
  const [selSteps, setSelSteps] = useState([]);
  const { job, error, submitting, run: launch } = useVizJob();

  const selected = parseClips(clips);
  const isSel = (cid) => selected.includes(cid);
  const toggleClip = (cid) =>
    setClips(joinClips(isSel(cid) ? selected.filter((c) => c !== cid) : [...selected, cid]));

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_samples))).catch(() => {});
  }, []);

  // Load the GT clip list for the dropdown (train subset by fraction/seed).
  useEffect(() => {
    if (mode !== "clip" && mode !== "compare") return;
    setGtErr(null);
    api
      .clusterGtClips({ subset_fraction: subsetFraction, subset_seed: subsetSeed, limit: 120 })
      .then(setGtClips)
      .catch((e) => { setGtClips([]); setGtErr(e.message); });
  }, [mode, subsetFraction, subsetSeed]);

  // Load saved sample steps when a run is picked.
  useEffect(() => {
    if (mode !== "samples" || !run) { setSteps([]); setSelSteps([]); return; }
    setStepsErr(null);
    api
      .runSampleSteps(run)
      .then((s) => { setSteps(s); setSelSteps(s.map((x) => x.step)); })
      .catch((e) => { setSteps([]); setStepsErr(e.message); });
  }, [mode, run]);

  function go(e) {
    e.preventDefault();
    if (mode === "clip") launch({ mode: "clip", clips });
    else if (mode === "compare") launch({ mode: "compare", clips, checkpoint, subset_fraction: subsetFraction, subset_seed: subsetSeed });
    else launch({ mode: "samples", run, steps: selSteps });
  }

  const cur = MODES.find((m) => m.id === mode);

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_1.1fr]">
      <form className="surface space-y-4 p-6" onSubmit={go}>
        <div className="grid grid-cols-3 gap-1.5">
          {MODES.map((m) => (
            <button key={m.id} type="button" onClick={() => setMode(m.id)}
              className={`rounded-md px-2 py-2 text-[12px] font-medium transition ${mode === m.id ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
              {m.label}
            </button>
          ))}
        </div>
        <p className="label">{cur.hint}</p>

        {(mode === "clip" || mode === "compare") && (
          <>
            <label className="block">
              <span className="label mb-1.5 block">clip ids (comma-sep{mode === "compare" ? ", or 'auto'" : ""})</span>
              <input className="field-input" value={clips} onChange={(e) => setClips(e.target.value)} />
            </label>

            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="label mb-1.5 block">subset fraction</span>
                <input type="number" step="0.01" min="0" max="1" className="field-input"
                  value={subsetFraction} onChange={(e) => setSubsetFraction(Number(e.target.value))} />
              </label>
              <label className="block">
                <span className="label mb-1.5 block">subset seed</span>
                <input type="number" className="field-input"
                  value={subsetSeed} onChange={(e) => setSubsetSeed(Number(e.target.value))} />
              </label>
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <span className="label">available GT clips · click to select</span>
                <span className="font-mono text-[11px] text-[var(--muted)]">{selected.length} selected</span>
              </div>
              {gtErr ? (
                <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-3 text-[11px] text-[var(--muted)]">
                  GT clip list unavailable ({gtErr}). Enter ids manually above{mode === "compare" ? " or use 'auto'" : ""}.
                </p>
              ) : gtClips.length === 0 ? (
                <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-3 text-[11px] text-[var(--muted)]">loading clips…</p>
              ) : (
                <div className="max-h-52 space-y-1 overflow-y-auto rounded-lg border border-[var(--hairline)] bg-ink p-1.5">
                  {gtClips.map((c) => (
                    <button key={c.cid} type="button" onClick={() => toggleClip(c.cid)}
                      className={`flex w-full items-start gap-2 rounded px-2 py-1 text-left text-[11px] transition ${isSel(c.cid) ? "bg-[var(--signal-dim)] text-[var(--signal)]" : "text-slate-400 hover:bg-white/5"}`}>
                      <span className="mt-px font-mono">{isSel(c.cid) ? "▣" : "▢"}</span>
                      <span className="shrink-0 font-mono text-slate-300">{c.cid}</span>
                      <span className="truncate">{c.caption || "—"}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </>
        )}

        {mode === "compare" && <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} />}

        {mode === "samples" && (
          <>
            <label className="block">
              <span className="label mb-1.5 block">run (remote, has samples)</span>
              <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
                <option value="">select run…</option>
                {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
              </select>
            </label>

            {run && (
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="label">steps to render</span>
                  <span className="flex gap-2 text-[11px]">
                    <button type="button" className="text-slate-400 hover:text-[var(--signal)]" onClick={() => setSelSteps(steps.map((s) => s.step))}>all</button>
                    <button type="button" className="text-slate-400 hover:text-[var(--signal)]" onClick={() => setSelSteps([])}>none</button>
                  </span>
                </div>
                {stepsErr ? (
                  <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-3 text-[11px] text-[var(--muted)]">no steps found ({stepsErr})</p>
                ) : steps.length === 0 ? (
                  <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-3 text-[11px] text-[var(--muted)]">loading steps…</p>
                ) : (
                  <>
                    <div className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto rounded-lg border border-[var(--hairline)] bg-ink p-2">
                      {steps.map((s) => {
                        const on = selSteps.includes(s.step);
                        return (
                          <button key={s.step} type="button"
                            onClick={() => setSelSteps(on ? selSteps.filter((x) => x !== s.step) : [...selSteps, s.step])}
                            className={`rounded-full border px-2.5 py-1 font-mono text-[11px] transition ${on ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                            {s.step.toLocaleString()}
                          </button>
                        );
                      })}
                    </div>
                    <p className="mt-1.5 label">
                      {selSteps.length} step{selSteps.length === 1 ? "" : "s"} → ~{selSteps.length * 3} GIFs (3 prompts/step)
                    </p>
                  </>
                )}
              </div>
            )}
          </>
        )}

        <button type="submit" className="btn-signal w-full"
          disabled={submitting
            || ((mode === "clip" || mode === "compare") && selected.length === 0)
            || (mode === "compare" && !checkpoint)
            || (mode === "samples" && (!run || selSteps.length === 0))}>
          {submitting ? "SUBMITTING…" : `▶  RENDER ${cur.label.toUpperCase()}`}
        </button>
      </form>

      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Rendered clips will appear here." />
      </div>
    </div>
  );
}
