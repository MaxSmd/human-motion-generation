"use client";

// LAB workspace — the train → evaluate → read-results loop in one place. Folds
// the old Model (train + eval submission) and Analysis (curves · eval LaTeX ·
// clip comparison) tabs together so the "now go to the Analysis tab" hop is just
// a mode switch. A run-history rail lists train/eval jobs beside it.

import { useState } from "react";
import ModelTab from "./ModelTab";
import AnalysisTab from "./AnalysisTab";
import HistoryRail from "./HistoryRail";
import TrainingProgress from "./TrainingProgress";
import { useJobHistory } from "@/lib/useJobHistory";
import { WORKSPACE_CATEGORIES } from "@/lib/classifyJob";

const MODES = [
  { id: "run", label: "Train / Eval", sub: "submit GPU jobs" },
  { id: "analysis", label: "Analysis", sub: "curves · eval · clip metrics" },
];

export default function LabWorkspace({ model }) {
  const [mode, setMode] = useState("run");
  const [showRail, setShowRail] = useState(true);
  const { jobs, refresh } = useJobHistory(WORKSPACE_CATEGORIES.lab);

  const main = (
    <div className="min-w-0 space-y-5">
      {/* live progress for whatever training currently holds the GPU slot —
          surfaces runs launched in earlier sessions, not just this one */}
      <TrainingProgress jobs={jobs} onChange={refresh} />
      {mode === "run" ? <ModelTab model={model} /> : <AnalysisTab />}
    </div>
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap gap-1.5">
          {MODES.map((m) => (
            <button key={m.id} onClick={() => setMode(m.id)}
              className={`rounded-md border px-3.5 py-2 text-left transition ${mode === m.id ? "border-[var(--signal)] bg-[var(--signal-dim)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"}`}>
              <div className={`text-[13px] font-bold ${mode === m.id ? "text-[var(--signal)]" : "text-slate-300"}`}>{m.label}</div>
              <div className="label mt-0.5 normal-case">{m.sub}</div>
            </button>
          ))}
        </div>
        <button onClick={() => setShowRail((s) => !s)}
          className="ml-auto rounded-md border border-[var(--hairline)] px-3 py-1.5 text-[12px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]">
          {showRail ? "hide" : "show"} history · {jobs.length}
        </button>
      </div>

      {showRail ? (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
          {main}
          <HistoryRail jobs={jobs} categories={WORKSPACE_CATEGORIES.lab}
            emptyHint="Train & eval runs collect here." />
        </div>
      ) : (
        main
      )}
    </div>
  );
}
