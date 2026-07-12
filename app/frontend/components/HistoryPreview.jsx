"use client";

// A history job loaded into the workspace's main preview area. Clicking a row
// in the HistoryRail lands here: the job renders through the same VizJobResult
// used for freshly completed jobs, plus a header identifying what was opened.

import VizJobResult from "./VizJobResult";
import { classifyJob, categoryMeta } from "@/lib/classifyJob";

function ago(sec) {
  if (!sec) return "";
  const d = Math.max(0, Date.now() / 1000 - sec);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

export default function HistoryPreview({ job, onClose }) {
  if (!job) return null;
  const cat = classifyJob(job);
  const meta = categoryMeta(cat);
  const caption = job.outputs?.[0]?.caption || job.params?.prompts || job.params?.text || job.run_name || "";

  return (
    <div className="surface animate-fade-up p-5" style={{ borderColor: "var(--hairline-strong)" }}>
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="label flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
            history preview · {meta.label}
            {job.submitted_at ? <span className="normal-case tracking-normal">· {ago(job.submitted_at)}</span> : null}
          </div>
          {caption && <div className="mt-1 truncate text-[13px] text-slate-300">{caption}</div>}
        </div>
        <button onClick={onClose}
          className="shrink-0 rounded-md border border-[var(--hairline)] px-2.5 py-1 text-[11px] text-[var(--muted)] hover:border-[var(--signal)] hover:text-[var(--signal)]">
          ✕ close
        </button>
      </div>
      <VizJobResult job={job} emptyHint="This job has no pulled outputs." />
    </div>
  );
}
