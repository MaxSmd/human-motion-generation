"use client";

import { useMemo, useState } from "react";
import { mediaUrl } from "@/lib/api";
import { classifyJob, categoryMeta } from "@/lib/classifyJob";
import { useLightbox } from "./Lightbox";

const STATE_COLOR = {
  queued: "var(--amber)", submitting: "var(--muted)", pending: "var(--amber)",
  running: "var(--signal)", pulling: "var(--accent2)", paused: "var(--accent2)",
  done: "#34d399", failed: "#fb7185", cancelled: "#fb7185",
};
const ACTIVE = new Set(["queued", "submitting", "pending", "running", "pulling"]);

function ago(sec) {
  if (!sec) return "";
  const d = Math.max(0, Date.now() / 1000 - sec);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

// A per-workspace history list. `jobs` arrives already filtered to the
// workspace's categories (newest first). Chips narrow further by category;
// clicking a row loads the job into the workspace's centre preview (onOpen);
// the chevron expands the row to show the pulled clips + the exact params, and
// (when supported) push those params back into the editor via onRestore.
export default function HistoryRail({ jobs = [], categories, onRestore, onOpen, emptyHint }) {
  const [chip, setChip] = useState("all");

  // Only offer chips for categories actually present, so the bar stays tidy.
  const present = useMemo(() => {
    const seen = new Set(jobs.map(classifyJob));
    return (categories || [...seen]).filter((c) => seen.has(c));
  }, [jobs, categories]);

  const shown = chip === "all" ? jobs : jobs.filter((j) => classifyJob(j) === chip);

  return (
    <div className="surface flex max-h-[78vh] flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="label">history · {jobs.length}</span>
      </div>

      {present.length > 1 && (
        <div className="mb-3 flex flex-wrap gap-1.5">
          <Chip active={chip === "all"} onClick={() => setChip("all")} label="all" color="var(--signal)" />
          {present.map((c) => (
            <Chip key={c} active={chip === c} onClick={() => setChip(c)}
              label={categoryMeta(c).label} color={categoryMeta(c).color} />
          ))}
        </div>
      )}

      <div className="-mr-1.5 space-y-1.5 overflow-y-auto pr-1.5">
        {shown.length === 0 ? (
          <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-8 text-center text-[11px] text-[var(--muted)]">
            {emptyHint || "Generations you launch will collect here."}
          </p>
        ) : (
          shown.map((j) => <Row key={j.id} job={j} onRestore={onRestore} onOpen={onOpen} />)
        )}
      </div>
    </div>
  );
}

function Chip({ active, onClick, label, color }) {
  return (
    <button onClick={onClick}
      className={`rounded-full border px-2.5 py-0.5 text-[10px] font-medium tracking-wide transition ${active ? "" : "border-[var(--hairline)] text-[var(--muted)] hover:text-slate-200"}`}
      style={active ? { borderColor: color, color, background: "var(--signal-dim)" } : undefined}>
      {label}
    </button>
  );
}

function Row({ job, onRestore, onOpen }) {
  const [open, setOpen] = useState(false);
  const { open: openLightbox } = useLightbox();
  const cat = classifyJob(job);
  const meta = categoryMeta(cat);
  const sColor = STATE_COLOR[job.state] || "var(--muted)";
  const busy = ACTIVE.has(job.state);
  // A representative caption: the first output's caption, else the prompt param.
  const caption = job.outputs?.[0]?.caption || job.params?.prompts || job.params?.text || job.run_name || "";
  const outputs = job.outputs || [];

  return (
    <div className="rounded-lg border border-[var(--hairline)] bg-ink transition hover:border-[var(--hairline-strong)]">
      <div className="flex w-full items-start gap-2 px-2.5 py-2">
        {/* clicking the row body loads the job into the centre preview; the
            chevron alone toggles the inline detail expansion */}
        <button onClick={() => (onOpen ? onOpen(job) : setOpen((o) => !o))}
          className="flex min-w-0 flex-1 items-start gap-2 text-left"
          title={onOpen ? "load into preview" : undefined}>
          <span className="mt-0.5 h-2 w-2 shrink-0 rounded-full" style={{ background: meta.color }} title={meta.label} />
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-1.5">
              <span className="truncate text-[11px] text-slate-300">{caption || meta.label}</span>
            </span>
            <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-[var(--muted)]">
              <span style={{ color: sColor }} className={busy ? "animate-pulse-soft" : ""}>{job.state}</span>
              <span>· {meta.label}</span>
              {job.submitted_at ? <span>· {ago(job.submitted_at)}</span> : null}
            </span>
          </span>
        </button>
        <div className="flex shrink-0 items-center gap-1">
          <button onClick={() => setOpen((o) => !o)} title="details"
            className="px-1 text-[10px] text-[var(--muted)] hover:text-slate-200">
            {open ? "▾" : "▸"}
          </button>
        </div>
      </div>

      {open && (
        <div className="space-y-2 border-t border-[var(--hairline)] px-2.5 py-2">
          {job.error && <p className="text-[11px] text-rose-300">⚠ {job.error}</p>}
          {outputs.length > 0 && (
            <div className="flex gap-1.5 overflow-x-auto">
              {outputs.map((o, i) => (
                <button key={i} type="button" onClick={() => openLightbox(outputs, i)}
                  className="shrink-0" title={o.caption}>
                  {o.media_url.toLowerCase().endsWith(".mp4")
                    ? <video src={mediaUrl(o.media_url)} className="h-16 w-16 rounded border border-[var(--hairline)] object-cover transition hover:border-[var(--signal)]" muted />
                    : <img src={mediaUrl(o.media_url)} alt={o.caption} className="h-16 w-16 rounded border border-[var(--hairline)] object-cover transition hover:border-[var(--signal)]" />}
                </button>
              ))}
            </div>
          )}
          <Params params={job.params} />
          {onRestore && job.params && (
            <button onClick={() => onRestore(job.params, cat, job)}
              className="w-full rounded border border-[var(--hairline)] py-1 text-[10px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]">
              {cat === "scene" ? "↻ load room + clip into editor" : "↻ use these settings"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Compact key/value view of the params that produced the job (skips empties and
// the bulky scene/constraint arrays, which get a count badge instead).
function Params({ params }) {
  if (!params) return null;
  const rows = [];
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === "" || (Array.isArray(v) && v.length === 0)) continue;
    if (k === "scene") { rows.push([k, `${v.objects?.length ?? 0} obj`]); continue; }
    if (k === "constraints" || k === "ranges") { rows.push([k, `${v.length}`]); continue; }
    if (typeof v === "object") continue;
    rows.push([k, String(v)]);
  }
  if (rows.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[10px] text-slate-400">
      {rows.map(([k, v]) => (
        <div key={k} className="truncate" title={`${k}=${v}`}>
          <span className="text-[var(--muted)]">{k}</span> {v}
        </div>
      ))}
    </div>
  );
}
