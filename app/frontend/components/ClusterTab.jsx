"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, mediaUrl } from "@/lib/api";

const POLL_MS = 5000;
const ACTIVE = new Set(["queued", "submitting", "pending", "running", "pulling"]);

const STATE_COLOR = {
  queued: "var(--amber)", submitting: "var(--muted)", pending: "var(--amber)",
  running: "var(--signal)", pulling: "var(--accent2)", done: "#34d399",
  failed: "#fb7185", cancelled: "#fb7185",
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

function Jobs({ jobs, onChange }) {
  if (jobs.length === 0) {
    return (
      <div className="surface px-6 py-10 text-center">
        <p className="label">no jobs yet — launch from Generate / Visualize / Model</p>
      </div>
    );
  }
  const queued = jobs.filter((j) => j.state === "queued").length;
  return (
    <div className="surface p-5">
      <div className="label mb-3">launched jobs · {jobs.length}{queued ? ` · ${queued} queued` : ""}</div>
      <div className="space-y-2">
        {jobs.map((j) => <JobRow key={j.id} job={j} onChange={onChange} />)}
      </div>
    </div>
  );
}

function JobRow({ job, onChange }) {
  const [open, setOpen] = useState(false);
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
          {job.slurm_id && (
            <button onClick={() => setOpen((o) => !o)} className={`rounded border px-2 py-0.5 text-[10px] transition ${open ? "border-[var(--signal)] text-[var(--signal)]" : "border-[var(--hairline-strong)] text-slate-300 hover:border-[var(--signal)]"}`}>
              {open ? "hide log" : "log"}
            </button>
          )}
          {ACTIVE.has(job.state) && (
            <button onClick={async () => { await api.cancelJob(job.id); onChange?.(); }} className="rounded border border-rose-500/40 px-2 py-0.5 text-[10px] text-rose-300 hover:bg-rose-500/10">cancel</button>
          )}
        </span>
      </div>
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
