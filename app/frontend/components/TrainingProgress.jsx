"use client";

// Live progress for the current training run — the cluster runs ONE job at a
// time, so a long training monopolises the slot. Shows step/percent, loss,
// it/s, ETA + walltime-resubmit count, and lets you Pause (scancel + free the
// slot, keep the run so it resumes from its latest checkpoint), Resume, or
// Cancel. Shared by the Lab workspace (top of the train/eval view) and the
// System drawer. Renders nothing when no train job is live.

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";

const POLL_MS = 5000;
const ACTIVE = new Set(["queued", "submitting", "pending", "running", "pulling"]);
const LIVE = new Set([...ACTIVE, "paused"]);
// Train states from which a long run can be picked up where it left off.
const TRAIN_LIVE = new Set(["queued", "submitting", "pending", "running", "paused"]);

const STATE_COLOR = {
  queued: "var(--amber)", submitting: "var(--muted)", pending: "var(--amber)",
  running: "var(--signal)", pulling: "var(--accent2)", paused: "var(--accent2)",
  done: "#34d399", failed: "#fb7185", cancelled: "#fb7185",
};

function fmtDuration(s) {
  if (s == null) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m`;
  return `${s}s`;
}

export default function TrainingProgress({ jobs, onChange }) {
  // Newest live train job (jobs arrive newest-first).
  const job = jobs.find((j) => j.kind === "train" && TRAIN_LIVE.has(j.state)) || null;
  const [prog, setProg] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const jobId = job?.id;
  const fetchProg = useCallback(async () => {
    if (!jobId) return;
    try { setProg(await api.jobProgress(jobId)); }
    catch { /* transient — keep last value */ }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) { setProg(null); return; }
    fetchProg();
    const t = setInterval(fetchProg, POLL_MS);
    return () => clearInterval(t);
  }, [jobId, job?.state, fetchProg]);

  if (!job) return null;

  const act = (fn, confirmMsg) => async () => {
    if (confirmMsg && !confirm(confirmMsg)) return;
    setBusy(true); setErr(null);
    try { await fn(job.id); await fetchProg(); onChange?.(); }
    catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  };

  const color = STATE_COLOR[job.state] || "var(--muted)";
  const step = prog?.step, maxSteps = prog?.max_steps;
  const pct = prog?.pct ?? (step != null && maxSteps ? (100 * step) / maxSteps : null);
  const paused = job.state === "paused";
  const canPause = job.state === "pending" || job.state === "running";

  return (
    <div className="surface p-5" style={{ borderColor: paused ? "var(--accent2)" : "var(--hairline)" }}>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="label">current training</span>
        <span className={`rounded px-2 py-0.5 font-mono text-[11px] ${job.state === "running" ? "animate-pulse-soft" : ""}`} style={{ color, border: `1px solid ${color}` }}>
          {job.state}
        </span>
        <span className="truncate font-mono text-[12px] text-slate-300">{job.run_name}</span>
        {prog?.resubmit_count > 0 && (
          <span className="rounded border border-[var(--hairline-strong)] px-2 py-0.5 text-[10px] text-slate-400" title="walltime cycles auto-resumed">
            ↻ {prog.resubmit_count} resubmit{prog.resubmit_count === 1 ? "" : "s"}
          </span>
        )}
        {prog?.complete && <span className="text-[11px] text-emerald-400">✓ complete</span>}
        <span className="ml-auto flex items-center gap-2">
          {canPause && (
            <button onClick={act(api.pauseJob)} disabled={busy}
              className="rounded border border-[var(--accent2)]/50 px-3 py-1 text-[11px] text-[var(--accent2)] hover:bg-[var(--accent2)]/10 disabled:opacity-40">
              ⏸ pause
            </button>
          )}
          {paused && (
            <button onClick={act(api.resumeJob)} disabled={busy}
              className="btn-signal px-3 py-1 text-[11px] disabled:opacity-40">
              ▶ resume
            </button>
          )}
          {LIVE.has(job.state) && (
            <button onClick={act(api.cancelJob, "Cancel this training run for good?")} disabled={busy}
              className="rounded border border-rose-500/40 px-3 py-1 text-[11px] text-rose-300 hover:bg-rose-500/10 disabled:opacity-40">
              cancel
            </button>
          )}
        </span>
      </div>

      {/* progress bar */}
      <div className="mb-2">
        <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
          <span className="text-slate-200">
            step {step != null ? step.toLocaleString() : "—"}
            <span className="text-[var(--muted)]"> / {maxSteps ? maxSteps.toLocaleString() : "—"}</span>
          </span>
          <span className="text-[var(--signal)]">{pct != null ? `${pct.toFixed(1)}%` : "—"}</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-[var(--hairline)]">
          <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(100, pct ?? 0)}%`, background: paused ? "var(--accent2)" : "var(--signal)" }} />
        </div>
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-slate-400">
        <span>loss <span className="text-slate-200">{prog?.loss != null ? prog.loss.toFixed(4) : "—"}</span></span>
        <span>speed <span className="text-slate-200">{prog?.steps_per_s != null ? `${prog.steps_per_s.toFixed(2)} it/s` : "—"}</span></span>
        <span>eta <span className="text-slate-200">{prog?.complete ? "done" : fmtDuration(prog?.eta_seconds)}</span></span>
        {job.slurm_id && <span>slurm <span className="text-[var(--muted)]">#{job.slurm_id}</span></span>}
      </div>

      {paused && (
        <p className="mt-3 rounded border border-[var(--accent2)]/30 bg-[var(--accent2)]/5 px-3 py-2 text-[11px] text-[var(--accent2)]">
          Slot is free — launch eval / viz from the other tabs, then hit <b>resume</b> to continue this run from its latest checkpoint.
        </p>
      )}
      {err && <p className="mt-2 text-[11px] text-rose-300">⚠ {err}</p>}
    </div>
  );
}
