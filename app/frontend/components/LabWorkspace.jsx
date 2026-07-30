"use client";

// LAB workspace — the train → evaluate → read-results loop in one place. Folds
// the old Model (train + eval submission) and Analysis tabs together so the "now
// go to the Analysis tab" hop is just a mode switch. A run-history rail lists
// train/eval jobs beside it.
//
// Analysis is split by the question being asked, not by the code that answers it:
// run-level (curves · eval · calibration), one clip's dynamics, and constraints
// (did the pin hold, what did it cost, and does saying it in the prompt help).

import { useState } from "react";
import ModelTab from "./ModelTab";
import AnalysisTab from "./AnalysisTab";
import ClipComparisonTab from "./ClipComparisonTab";
import ConstraintAnalysisTab from "./ConstraintAnalysisTab";
import HistoryRail from "./HistoryRail";
import HistoryPreview from "./HistoryPreview";
import { useJobHistory } from "@/lib/useJobHistory";
import { WORKSPACE_CATEGORIES } from "@/lib/classifyJob";

const MODES = [
  { id: "run", label: "Train / Eval", sub: "submit GPU jobs" },
  { id: "analysis", label: "Analysis", sub: "curves · eval · sampling" },
  { id: "clips", label: "Clip comparison", sub: "real vs gen — one clip" },
  { id: "constraints", label: "Constraint analysis", sub: "fidelity · cost · text ablation" },
];

const PANES = {
  run: (model) => <ModelTab model={model} />,
  analysis: () => <AnalysisTab />,
  clips: () => <ClipComparisonTab />,
  constraints: () => <ConstraintAnalysisTab />,
};

export default function LabWorkspace({ model }) {
  const [mode, setMode] = useState("run");
  const [showRail, setShowRail] = useState(true);
  const [preview, setPreview] = useState(null); // history job loaded into the centre
  const { jobs, all } = useJobHistory(WORKSPACE_CATEGORIES.lab);
  // Track the previewed job in the live poll so a running train/eval opened
  // from history updates in place; progress bars live in the System drawer.
  const previewJob = preview ? all.find((j) => j.id === preview.id) || preview : null;

  const main = (
    <div className="min-w-0 space-y-5">
      {previewJob && <HistoryPreview job={previewJob} onClose={() => setPreview(null)} />}
      {(PANES[mode] || PANES.run)(model)}
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
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px] 2xl:grid-cols-[minmax(0,1fr)_380px]">
          {main}
          <div className="xl:sticky xl:top-6 xl:self-start">
            <HistoryRail jobs={jobs} categories={WORKSPACE_CATEGORIES.lab}
              onOpen={setPreview} emptyHint="Train & eval runs collect here." />
          </div>
        </div>
      ) : (
        main
      )}
    </div>
  );
}
