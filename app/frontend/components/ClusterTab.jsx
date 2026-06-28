"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import TrainingProgress from "./TrainingProgress";

const POLL_MS = 5000;
const ACTIVE = new Set(["queued", "submitting", "pending", "running", "pulling"]);
// "Live" = still ours to manage (includes paused, which holds no cluster slot
// but is resumable). Used to keep these rows visible and to scope action buttons.
const LIVE = new Set([...ACTIVE, "paused"]);

const STATE_COLOR = {
  queued: "var(--amber)", submitting: "var(--muted)", pending: "var(--amber)",
  running: "var(--signal)", pulling: "var(--accent2)", paused: "var(--accent2)",
  done: "#34d399", failed: "#fb7185", cancelled: "#fb7185",
};

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

      <TrainingProgress jobs={jobs} onChange={refresh} />
      <Jobs jobs={jobs} onChange={refresh} />
      <Queue squeue={squeue} jobs={jobs} onChange={refresh} />
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

// ─────────────────────────────────────────────────────────── remote squeue

function Queue({ squeue, jobs, onChange }) {
  // slurm ids the app already tracks (any live state) — those rows can't be
  // adopted again, so we hide the adopt affordance for them.
  const tracked = new Set((jobs || []).filter((j) => j.slurm_id && LIVE.has(j.state)).map((j) => j.slurm_id));
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
                <QueueRow key={r.jobid} row={r} tracked={tracked.has(r.jobid)} onChange={onChange} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function QueueRow({ row: r, tracked, onChange }) {
  const [adopting, setAdopting] = useState(false);
  async function cancel() {
    if (!confirm(`scancel job ${r.jobid}?`)) return;
    await api.clusterCancel(r.jobid);
    onChange();
  }
  return (
    <>
      <tr className="border-t border-[var(--hairline)]">
        <td className="py-1.5 pr-3 text-[var(--signal)]">{r.jobid}</td>
        <td className="py-1.5 pr-3 text-slate-300">{r.name}</td>
        <td className="py-1.5 pr-3">{r.state}</td>
        <td className="py-1.5 pr-3 text-slate-400">{r.time}</td>
        <td className="py-1.5 pr-3 text-slate-500">{r.nodes || r.reason}</td>
        <td className="py-1.5">
          <span className="flex items-center justify-end gap-2">
            {tracked ? (
              <span className="text-[10px] text-[var(--muted)]" title="already tracked by the app">tracked</span>
            ) : (
              <button onClick={() => setAdopting((a) => !a)}
                className={`rounded border px-2 py-0.5 text-[10px] transition ${adopting ? "border-[var(--signal)] text-[var(--signal)]" : "border-[var(--hairline-strong)] text-slate-300 hover:border-[var(--signal)]"}`}
                title="track this run for live progress + walltime auto-resubmit">
                {adopting ? "close" : "adopt"}
              </button>
            )}
            <button onClick={cancel} className="rounded border border-rose-500/40 px-2 py-0.5 text-[10px] text-rose-300 hover:bg-rose-500/10">cancel</button>
          </span>
        </td>
      </tr>
      {adopting && (
        <tr className="border-t border-[var(--hairline)]/40">
          <td colSpan={6} className="py-2">
            <AdoptForm jobid={r.jobid} guessName={r.name} onDone={() => { setAdopting(false); onChange(); }} />
          </td>
        </tr>
      )}
    </>
  );
}

// Adopt a train job that's already on the cluster (e.g. a raw `sbatch`) into the
// app registry so it gets the live progress panel + walltime auto-resubmit. The
// run_name is REQUIRED (it locates the run dir + is what a resubmit resumes); the
// presets/max_steps let a resubmit rebuild the same launch command. Assumes the
// run used the standard slurm/rmg/train.sbatch layout.
function AdoptForm({ jobid, guessName, onDone }) {
  const [runName, setRunName] = useState("");
  const [modelPreset, setModelPreset] = useState("dit_base");
  const [trainPreset, setTrainPreset] = useState("rmg_base");
  const [maxSteps, setMaxSteps] = useState("");
  const [autoResubmit, setAutoResubmit] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function submit() {
    if (!runName.trim()) { setErr("run_name is required"); return; }
    setBusy(true); setErr(null);
    try {
      await api.adoptJob({
        slurm_id: jobid,
        run_name: runName.trim(),
        model_preset: modelPreset,
        train_preset: trainPreset,
        max_steps: maxSteps ? Number(maxSteps) : null,
        auto_resubmit: autoResubmit,
      });
      onDone();
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  const field = "rounded border border-[var(--hairline)] bg-ink px-2 py-1 text-[11px] text-slate-200";
  return (
    <div className="space-y-2 rounded-lg border border-[var(--hairline)] bg-ink p-3">
      <p className="text-[11px] text-[var(--muted)]">
        Adopt slurm <span className="font-mono text-[var(--signal)]">#{jobid}</span> as a tracked train run.
        Enter its <span className="font-mono">RUN_NAME</span> (the run dir under <span className="font-mono">runs/&lt;model&gt;/train</span>) and the presets it was launched with —
        a resubmit will resume the same run from latest.pt.
      </p>
      <div className="flex flex-wrap items-end gap-2 font-mono">
        <label className="flex flex-col gap-0.5">
          <span className="label">run_name *</span>
          <input value={runName} onChange={(e) => setRunName(e.target.value)} placeholder={guessName || "rmg-mid-…"} className={`${field} w-56`} />
        </label>
        <label className="flex flex-col gap-0.5">
          <span className="label">model_preset</span>
          <input value={modelPreset} onChange={(e) => setModelPreset(e.target.value)} className={`${field} w-32`} />
        </label>
        <label className="flex flex-col gap-0.5">
          <span className="label">train_preset</span>
          <input value={trainPreset} onChange={(e) => setTrainPreset(e.target.value)} className={`${field} w-32`} />
        </label>
        <label className="flex flex-col gap-0.5">
          <span className="label">max_steps</span>
          <input value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} placeholder="from config" className={`${field} w-28`} />
        </label>
        <label className="flex items-center gap-1.5 pb-1.5 text-[11px] text-slate-300">
          <input type="checkbox" checked={autoResubmit} onChange={(e) => setAutoResubmit(e.target.checked)} />
          auto-resubmit
        </label>
        <button onClick={submit} disabled={busy} className="btn-signal px-3 py-1 text-[11px] disabled:opacity-40">
          {busy ? "adopting…" : "adopt"}
        </button>
      </div>
      {err && <p className="text-[11px] text-rose-300">⚠ {err}</p>}
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
          {!LIVE.has(job.state) && (
            <button title="remove from list" onClick={async () => { await api.deleteJob(job.id); onChange?.(); }} className="rounded border border-[var(--hairline-strong)] px-2 py-0.5 text-[10px] text-slate-400 hover:border-rose-500/40 hover:text-rose-300">✕</button>
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
