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
const EXAMPLES = ["000019", "000021", "000022", "000026"];

export default function VisualizeTab() {
  const [mode, setMode] = useState("clip");
  const [clips, setClips] = useState("000019,000021,000022");
  const [checkpoint, setCheckpoint] = useState("");
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const { job, error, submitting, run: launch } = useVizJob();

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_samples))).catch(() => {});
  }, []);

  function go(e) {
    e.preventDefault();
    if (mode === "clip") launch({ mode: "clip", clips });
    else if (mode === "compare") launch({ mode: "compare", clips, checkpoint });
    else launch({ mode: "samples", run });
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
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map((c) => (
                <button key={c} type="button" onClick={() => setClips(c)}
                  className="rounded-full border border-[var(--hairline)] px-2.5 py-1 font-mono text-[11px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">
                  {c}
                </button>
              ))}
            </div>
          </>
        )}

        {mode === "compare" && <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} />}

        {mode === "samples" && (
          <label className="block">
            <span className="label mb-1.5 block">run (remote, has samples)</span>
            <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
              <option value="">select run…</option>
              {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
            </select>
          </label>
        )}

        <button type="submit" className="btn-signal w-full"
          disabled={submitting || (mode === "compare" && !checkpoint) || (mode === "samples" && !run)}>
          {submitting ? "SUBMITTING…" : `▶  RENDER ${cur.label.toUpperCase()}`}
        </button>
      </form>

      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Rendered clips will appear here." />
      </div>
    </div>
  );
}
