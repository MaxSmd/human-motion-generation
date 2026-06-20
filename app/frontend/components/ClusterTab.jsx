"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, mediaUrl } from "@/lib/api";

const POLL_MS = 5000;
const ACTIVE = new Set(["queued", "submitting", "pending", "running", "pulling"]);
// "Live" = still ours to manage (includes paused, which holds no cluster slot
// but is resumable). Used to keep these rows visible and to scope action buttons.
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

// Monitor only: link status, the remote squeue, and our local job queue. All
// job *launching* lives in the Generate / Visualize / Model tabs.
export default function ClusterTab({ status }) {
  const [squeue, setSqueue] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [err, setErr] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [q, j] = await Promise.all([api.clusterSqueue(), api.jobs()]);
      setSqueue(q);
      setJobs(j);
      setErr(null);
    } catch (e) {
      setErr(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  return (
    <div className="space-y-5">
      <div className="surface flex flex-wrap items-center gap-x-8 gap-y-2 p-4">
        <Readout k="head" v={status?.hostname || status?.host} />
        <Readout k="latency" v={status?.latency_ms != null ? `${status.latency_ms} ms` : "—"} />
        <Readout k="partitions" v={(status?.partitions || []).join(" · ") || "—"} />
        <span className="ml-auto flex items-center gap-2 text-[11px] text-[var(--muted)]">
          <span className="dot animate-pulse-soft" style={{ color: "var(--signal)" }} />
          auto-refresh {POLL_MS / 1000}s
        </span>
      </div>

      {err && <p className="rounded-lg border border-rose-500/30 bg-rose-500/5 px-4 py-2 text-[12px] text-rose-300">⚠ {err}</p>}

      <TrainingPanel jobs={jobs} onChange={refresh} />
      <Jobs jobs={jobs} onChange={refresh} />
      <Queue squeue={squeue} onChange={refresh} />
    </div>
  );
}

function Readout({ k, v }) {
  return (
    <div>
      <div className="label">{k}</div>
      <div className="font-mono text-sm text-slate-200">{v}</div>
    </div>
  );
}

// ───────────────────────────────────────────────────── current training run
//
// The cluster runs ONE job at a time, so a long training monopolises the slot.
// This panel lets you Pause it (scancel + free the slot, but keep the run so it
// resumes from its latest checkpoint), do ad-hoc eval/viz, then Resume — plus a
// live read of how far along the run is.

function TrainingPanel({ jobs, onChange }) {
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

// ─────────────────────────────────────────────────────────── remote squeue

function Queue({ squeue, onChange }) {
  async function cancel(id) {
    if (!confirm(`scancel job ${id}?`)) return;
    await api.clusterCancel(id);
    onChange();
  }
  return (
    <div className="surface p-5">
      <div className="label mb-3">squeue · --me ({squeue.length})</div>
      {squeue.length === 0 ? (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-6 text-center text-[12px] text-[var(--muted)]">no jobs in the queue</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-[12px]">
            <thead className="label">
              <tr><th className="pb-2 pr-3">jobid</th><th className="pb-2 pr-3">name</th><th className="pb-2 pr-3">st</th><th className="pb-2 pr-3">time</th><th className="pb-2 pr-3">node/reason</th><th /></tr>
            </thead>
            <tbody>
              {squeue.map((r) => (
                <tr key={r.jobid} className="border-t border-[var(--hairline)]">
                  <td className="py-1.5 pr-3 text-[var(--signal)]">{r.jobid}</td>
                  <td className="py-1.5 pr-3 text-slate-300">{r.name}</td>
                  <td className="py-1.5 pr-3">{r.state}</td>
                  <td className="py-1.5 pr-3 text-slate-400">{r.time}</td>
                  <td className="py-1.5 pr-3 text-slate-500">{r.nodes || r.reason}</td>
                  <td className="py-1.5"><button onClick={() => cancel(r.jobid)} className="rounded border border-rose-500/40 px-2 py-0.5 text-[10px] text-rose-300 hover:bg-rose-500/10">cancel</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────── our jobs + local queue

const RECENT_N = 3;

function Jobs({ jobs, onChange }) {
  // Keep only the most-recent jobs visible by default (the list grew unwieldy);
  // toggle off to see the full history. `jobs` arrives newest-first.
  const [recentOnly, setRecentOnly] = useState(true);

  if (jobs.length === 0) {
    return (
      <div className="surface px-6 py-10 text-center">
        <p className="label">no jobs yet — launch from Generate / Visualize / Model</p>
      </div>
    );
  }
  const queued = jobs.filter((j) => j.state === "queued").length;
  // Always keep live jobs visible (incl. paused); only collapse the older
  // finished ones.
  const active = jobs.filter((j) => LIVE.has(j.state));
  const rest = jobs.filter((j) => !LIVE.has(j.state));
  const shown = recentOnly ? [...active, ...rest.slice(0, RECENT_N)] : jobs;
  const hidden = jobs.length - shown.length;

  return (
    <div className="surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="label">launched jobs · {jobs.length}{queued ? ` · ${queued} queued` : ""}</span>
        <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <input type="checkbox" checked={recentOnly} onChange={(e) => setRecentOnly(e.target.checked)} />
          recent only (≤{RECENT_N})
        </label>
      </div>
      <div className="space-y-2">
        {shown.map((j) => <JobRow key={j.id} job={j} onChange={onChange} />)}
      </div>
      {hidden > 0 && (
        <button onClick={() => setRecentOnly(false)} className="mt-3 w-full rounded border border-dashed border-[var(--hairline)] py-1.5 text-[11px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">
          show {hidden} older job{hidden === 1 ? "" : "s"}
        </button>
      )}
    </div>
  );
}

function JobRow({ job, onChange }) {
  const [open, setOpen] = useState(false);
  const [meta, setMeta] = useState(false);
  const [log, setLog] = useState(null);
  const [loading, setLoading] = useState(false);
  const preRef = useRef(null);
  const color = STATE_COLOR[job.state] || "var(--muted)";

  const fetchLog = useCallback(async () => {
    setLoading(true);
    try { setLog((await api.jobLog(job.id)).log || "(empty)"); }
    catch (e) { setLog(`error: ${e.message}`); }
    finally { setLoading(false); }
  }, [job.id]);

  useEffect(() => {
    if (!open) return;
    fetchLog();
    if (!ACTIVE.has(job.state)) return;
    const t = setInterval(fetchLog, 3000);
    return () => clearInterval(t);
  }, [open, job.state, fetchLog]);

  useEffect(() => { if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight; }, [log]);

  return (
    <div className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span className={`rounded px-2 py-0.5 font-mono text-[11px] ${ACTIVE.has(job.state) ? "animate-pulse-soft" : ""}`} style={{ color, border: `1px solid ${color}` }}>
          {job.state}{job.state === "queued" && job.queue_pos ? ` #${job.queue_pos}` : ""}
        </span>
        <span className="font-mono text-slate-300">{job.kind}{job.mode ? `/${job.mode}` : ""}</span>
        {job.slurm_id && <span className="font-mono text-[var(--muted)]">#{job.slurm_id}</span>}
        <span className="truncate font-mono text-[var(--muted)]">{job.run_name}</span>
        {job.error && <span className="text-rose-300">{job.error}</span>}
        <span className="ml-auto flex items-center gap-2">
          <button onClick={() => setMeta((m) => !m)} className={`rounded border px-2 py-0.5 text-[10px] transition ${meta ? "border-[var(--signal)] text-[var(--signal)]" : "border-[var(--hairline-strong)] text-slate-300 hover:border-[var(--signal)]"}`}>
            {meta ? "hide info" : "info"}
          </button>
          {job.slurm_id && (
            <button onClick={() => setOpen((o) => !o)} className={`rounded border px-2 py-0.5 text-[10px] transition ${open ? "border-[var(--signal)] text-[var(--signal)]" : "border-[var(--hairline-strong)] text-slate-300 hover:border-[var(--signal)]"}`}>
              {open ? "hide log" : "log"}
            </button>
          )}
          {job.state === "paused" && (
            <button onClick={async () => { await api.resumeJob(job.id); onChange?.(); }} className="rounded border border-[var(--signal)]/50 px-2 py-0.5 text-[10px] text-[var(--signal)] hover:bg-[var(--signal)]/10">resume</button>
          )}
          {LIVE.has(job.state) && (
            <button onClick={async () => { await api.cancelJob(job.id); onChange?.(); }} className="rounded border border-rose-500/40 px-2 py-0.5 text-[10px] text-rose-300 hover:bg-rose-500/10">cancel</button>
          )}
        </span>
      </div>
      {meta && <JobMeta job={job} />}
      {job.outputs?.length > 0 && (
        <div className="mt-2 flex gap-2 overflow-x-auto">
          {job.outputs.map((o, i) => (
            <a key={i} href={mediaUrl(o.media_url)} target="_blank" rel="noreferrer" className="shrink-0">
              {o.media_url.toLowerCase().endsWith(".mp4")
                ? <video src={mediaUrl(o.media_url)} className="h-16 w-16 rounded border border-[var(--hairline)] object-cover" muted />
                : <img src={mediaUrl(o.media_url)} alt={o.caption} className="h-16 w-16 rounded border border-[var(--hairline)] object-cover" />}
            </a>
          ))}
        </div>
      )}
      {open && (
        <div className="mt-2">
          <div className="mb-1 flex items-center gap-2 label">
            slurm/logs · {job.slurm_id}
            {ACTIVE.has(job.state) && <span className="text-[var(--signal)]">live</span>}
            <button onClick={fetchLog} className="ml-auto text-slate-400 hover:text-[var(--signal)]">{loading ? "…" : "↻"}</button>
          </div>
          <pre ref={preRef} className="max-h-72 overflow-auto rounded border border-[var(--hairline)] bg-black p-2 text-[10px] leading-relaxed text-slate-400">{log ?? "loading…"}</pre>
        </div>
      )}
    </div>
  );
}

// What the job was actually submitted to do: the resolved params + the exact
// remote command. Lets you audit a job from the Cluster tab without reading the
// log (e.g. which checkpoint / clips / guidance it ran with).
function JobMeta({ job }) {
  const params = Object.entries(job.params || {}).filter(
    ([, v]) => v !== null && v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0)
  );
  return (
    <div className="mt-2 space-y-2 rounded border border-[var(--hairline)] bg-ink p-2.5 text-[11px]">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[var(--muted)]">
        <span>kind <span className="font-mono text-slate-300">{job.kind}{job.mode ? `/${job.mode}` : ""}</span></span>
        {job.run_name && <span>run <span className="font-mono text-slate-300">{job.run_name}</span></span>}
        {job.job_name && <span>name <span className="font-mono text-slate-300">{job.job_name}</span></span>}
        {job.submitted_at ? <span>at <span className="font-mono text-slate-300">{new Date(job.submitted_at * 1000).toLocaleString()}</span></span> : null}
      </div>
      {params.length > 0 && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 font-mono text-slate-400 sm:grid-cols-3">
          {params.map(([k, v]) => (
            <div key={k} className="truncate" title={`${k}=${fmt(v)}`}>
              <span className="text-[var(--muted)]">{k}</span> {fmt(v)}
            </div>
          ))}
        </div>
      )}
      {job.command && (
        <div>
          <div className="label mb-1">submit command</div>
          <pre className="overflow-x-auto rounded border border-[var(--hairline)] bg-black p-2 text-[10px] leading-relaxed text-[var(--signal)]">{job.command}</pre>
        </div>
      )}
    </div>
  );
}

function fmt(v) {
  if (Array.isArray(v)) return `[${v.join(",")}]`;
  return String(v);
}
