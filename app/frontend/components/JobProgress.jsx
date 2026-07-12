"use client";

// Live progress for the CURRENT cluster job — the cluster runs ONE job at a
// time, so whatever is live monopolises the slot. This panel auto-adapts to the
// job's kind (from /jobs/{id}/progress):
//   • train — step / max_steps, loss, it/s, ETA + resubmit count; pause / resume
//             / cancel (pause frees the slot but keeps the run for a resume).
//   • eval  — TWO bars: the ω guidance sweep (level i/N) + the batch within the
//             current level; cancel only (eval isn't resumable).
//   • viz   — one coarse bar over rendered clips / prompts; cancel only.
// Renders nothing when no job is live. Shared by every workspace + the System
// drawer; only one workspace mounts at a time, so it polls a single job.

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

const KIND_LABEL = { train: "current training", eval: "current evaluation", viz: "current render" };

function fmtDuration(s) {
  if (s == null) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m`;
  return `${s}s`;
}

// Pick the newest job worth showing progress for. One job runs on the cluster at
// a time; a paused train can coexist with a later run, so prefer an actively
// on-cluster job, else fall back to a paused train (so resume stays reachable).
function pickLiveJob(jobs) {
  const active = jobs.find((j) => ACTIVE.has(j.state));
  if (active) return active;
  return jobs.find((j) => j.kind === "train" && j.state === "paused") || null;
}

function Bar({ pct, color }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-[var(--hairline)]">
      <div className="h-full rounded-full transition-all"
        style={{ width: `${Math.min(100, Math.max(0, pct ?? 0))}%`, background: color }} />
    </div>
  );
}

// ---- per-kind bodies -------------------------------------------------------

function TrainBody({ prog, color, paused }) {
  const step = prog?.step, maxSteps = prog?.max_steps;
  const pct = prog?.pct ?? (step != null && maxSteps ? (100 * step) / maxSteps : null);
  return (
    <>
      <div className="mb-2">
        <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
          <span className="text-slate-200">
            step {step != null ? step.toLocaleString() : "—"}
            <span className="text-[var(--muted)]"> / {maxSteps ? maxSteps.toLocaleString() : "—"}</span>
          </span>
          <span className="text-[var(--signal)]">{pct != null ? `${pct.toFixed(1)}%` : "—"}</span>
        </div>
        <Bar pct={pct} color={paused ? "var(--accent2)" : "var(--signal)"} />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-slate-400">
        <span>loss <span className="text-slate-200">{prog?.loss != null ? prog.loss.toFixed(4) : "—"}</span></span>
        <span>speed <span className="text-slate-200">{prog?.steps_per_s != null ? `${prog.steps_per_s.toFixed(2)} it/s` : "—"}</span></span>
        <span>eta <span className="text-slate-200">{prog?.complete ? "done" : fmtDuration(prog?.eta_seconds)}</span></span>
      </div>
    </>
  );
}

function EvalBody({ prog, color }) {
  const outer = prog?.outer, inner = prog?.inner;
  const outerPct = outer?.n ? (100 * outer.i) / outer.n : null;
  const innerPct = inner?.n ? (100 * inner.i) / inner.n : null;
  const stageLabel = prog?.stage === "multimodality" ? "multimodality"
    : prog?.stage === "done" ? "done" : "sampling batches";
  return (
    <>
      {/* outer: guidance sweep */}
      <div className="mb-3">
        <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
          <span className="text-slate-200">
            guidance level {outer ? `${outer.i} / ${outer.n}` : "—"}
            {outer?.label && <span className="text-[var(--muted)]"> · {outer.label}</span>}
          </span>
          <span className="text-[var(--signal)]">{outerPct != null ? `${outerPct.toFixed(0)}%` : "—"}</span>
        </div>
        <Bar pct={outerPct} color="var(--signal)" />
      </div>
      {/* inner: within the current ω */}
      <div className="mb-1">
        <div className="mb-1 flex items-baseline justify-between font-mono text-[11px] text-slate-400">
          <span>{stageLabel} {inner ? `${inner.i} / ${inner.n}` : ""}</span>
          <span className="text-[var(--accent2)]">{innerPct != null ? `${innerPct.toFixed(0)}%` : ""}</span>
        </div>
        <Bar pct={innerPct} color="var(--accent2)" />
      </div>
    </>
  );
}

function VizBody({ prog }) {
  const inner = prog?.inner;
  const pct = inner?.n ? (100 * inner.i) / inner.n : (prog?.complete ? 100 : null);
  const stageLabel = prog?.stage === "sampling" ? "sampling motion"
    : prog?.stage === "done" ? "done" : "rendering";
  return (
    <div className="mb-1">
      <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
        <span className="text-slate-200">
          {stageLabel} {inner ? <span className="text-[var(--muted)]">{inner.i} / {inner.n}</span> : ""}
        </span>
        <span className="text-[var(--signal)]">{pct != null ? `${pct.toFixed(0)}%` : "—"}</span>
      </div>
      <Bar pct={pct} color="var(--signal)" />
    </div>
  );
}

export default function JobProgress({ jobs, onChange }) {
  const job = pickLiveJob(jobs);
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
  const paused = job.state === "paused";
  const isTrain = job.kind === "train";
  const canPause = isTrain && (job.state === "pending" || job.state === "running");
  const cancelMsg = isTrain
    ? "Cancel this training run for good?"
    : `Cancel this ${job.kind} job?`;

  return (
    <div className="surface p-5" style={{ borderColor: paused ? "var(--accent2)" : "var(--hairline)" }}>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="label">{KIND_LABEL[job.kind] || `current ${job.kind}`}</span>
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
            <button onClick={act(api.cancelJob, cancelMsg)} disabled={busy}
              className="rounded border border-rose-500/40 px-3 py-1 text-[11px] text-rose-300 hover:bg-rose-500/10 disabled:opacity-40">
              cancel
            </button>
          )}
        </span>
      </div>

      {job.kind === "train" && <TrainBody prog={prog} color={color} paused={paused} />}
      {job.kind === "eval" && <EvalBody prog={prog} color={color} />}
      {job.kind === "viz" && <VizBody prog={prog} />}

      {job.slurm_id && (
        <div className="mt-2 font-mono text-[11px] text-[var(--muted)]">slurm #{job.slurm_id}</div>
      )}

      {paused && (
        <p className="mt-3 rounded border border-[var(--accent2)]/30 bg-[var(--accent2)]/5 px-3 py-2 text-[11px] text-[var(--accent2)]">
          Slot is free — launch eval / viz from the other tabs, then hit <b>resume</b> to continue this run from its latest checkpoint.
        </p>
      )}
      {err && <p className="mt-2 text-[11px] text-rose-300">⚠ {err}</p>}
    </div>
  );
}
