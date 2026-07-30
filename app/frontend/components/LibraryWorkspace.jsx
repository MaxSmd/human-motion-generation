"use client";

// LIBRARY workspace — render & browse motion that already exists (no new
// sampling). Folds the old Visualize, Ground-Truth and Training tabs together.
//
//   cluster: VisualizeTab already covers clip renders (always paired GT vs
//            prediction) and training samples in one form, so it IS the library;
//            a render-history rail sits beside.
//   local  : a small switch between the dataset browser and the training-sample
//            scrubber (these read in-process, not via the job queue).

import { useState } from "react";
import VisualizeTab from "./VisualizeTab";
import GTBrowserTab from "./GTBrowserTab";
import TrainingViewerTab from "./TrainingViewerTab";
import HistoryRail from "./HistoryRail";
import HistoryPreview from "./HistoryPreview";
import { useJobHistory } from "@/lib/useJobHistory";
import { WORKSPACE_CATEGORIES } from "@/lib/classifyJob";

export default function LibraryWorkspace({ clusterMode }) {
  if (clusterMode) return <LibraryCluster />;
  return <LibraryLocal />;
}

function LibraryCluster() {
  const [showRail, setShowRail] = useState(true);
  const [preview, setPreview] = useState(null); // history job loaded into the centre
  const { jobs, all } = useJobHistory(WORKSPACE_CATEGORIES.library);
  // Track the previewed job in the live poll so an in-flight render's preview
  // updates as it completes; progress bars live in the System drawer.
  const previewJob = preview ? all.find((j) => j.id === preview.id) || preview : null;

  const main = (
    <div className="min-w-0 space-y-5">
      {previewJob && <HistoryPreview job={previewJob} onClose={() => setPreview(null)} />}
      <VisualizeTab />
    </div>
  );

  return (
    <div className="space-y-5">
      <div className="flex items-center">
        <div>
          <div className="display text-base font-bold text-white">Library</div>
          <div className="label mt-0.5">render clips (always GT + prediction) · training samples</div>
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
            <HistoryRail jobs={jobs} categories={WORKSPACE_CATEGORIES.library}
              onOpen={setPreview} emptyHint="Rendered clips collect here by kind." />
          </div>
        </div>
      ) : (
        main
      )}
    </div>
  );
}

const LOCAL_MODES = [
  { id: "gt", label: "Ground truth", sub: "dataset browser" },
  { id: "training", label: "Training", sub: "sample scrubber" },
];

function LibraryLocal() {
  const [mode, setMode] = useState("gt");
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-1.5">
        {LOCAL_MODES.map((m) => (
          <button key={m.id} onClick={() => setMode(m.id)}
            className={`rounded-md border px-3.5 py-2 text-left transition ${mode === m.id ? "border-[var(--signal)] bg-[var(--signal-dim)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"}`}>
            <div className={`text-[13px] font-bold ${mode === m.id ? "text-[var(--signal)]" : "text-slate-300"}`}>{m.label}</div>
            <div className="label mt-0.5 normal-case">{m.sub}</div>
          </button>
        ))}
      </div>
      {mode === "gt" ? <GTBrowserTab clusterMode={false} /> : <TrainingViewerTab clusterMode={false} />}
    </div>
  );
}
