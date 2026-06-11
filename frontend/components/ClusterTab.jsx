"use client";

import { useCallback, useEffect, useState } from "react";
import { api, mediaUrl } from "@/lib/api";

const POLL_MS = 5000;

const STATE_COLOR = {
  submitting: "var(--muted)",
  pending: "var(--amber)",
  running: "var(--signal)",
  pulling: "var(--accent2)",
  done: "#34d399",
  failed: "#fb7185",
  cancelled: "#fb7185",
};

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
      {/* node summary */}
      <div className="surface flex flex-wrap items-center gap-x-8 gap-y-2 p-4">
        <Readout k="head" v={status?.hostname || status?.host} />
        <Readout k="latency" v={status?.latency_ms != null ? `${status.latency_ms} ms` : "—"} />
        <Readout k="partitions" v={(status?.partitions || []).join(" · ") || "—"} />
        <span className="ml-auto flex items-center gap-2 text-[11px] text-[var(--muted)]">
          <span className="dot animate-pulse-soft" style={{ color: "var(--signal)" }} />
          auto-refresh {POLL_MS / 1000}s
        </span>
      </div>

      {err && (
        <p className="rounded-lg border border-rose-500/30 bg-rose-500/5 px-4 py-2 text-[12px] text-rose-300">
          ⚠ {err}
        </p>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Launcher onLaunched={refresh} />
        <Queue squeue={squeue} onChange={refresh} />
      </div>

      <Jobs jobs={jobs} />
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

// ─────────────────────────────────────────────────────────────── live squeue

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
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-6 text-center text-[12px] text-[var(--muted)]">
          no jobs in the queue
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-[12px]">
            <thead className="label">
              <tr>
                <th className="pb-2 pr-3">jobid</th>
                <th className="pb-2 pr-3">name</th>
                <th className="pb-2 pr-3">st</th>
                <th className="pb-2 pr-3">time</th>
                <th className="pb-2 pr-3">node/reason</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {squeue.map((r) => (
                <tr key={r.jobid} className="border-t border-[var(--hairline)]">
                  <td className="py-1.5 pr-3 text-[var(--signal)]">{r.jobid}</td>
                  <td className="py-1.5 pr-3 text-slate-300">{r.name}</td>
                  <td className="py-1.5 pr-3">{r.state}</td>
                  <td className="py-1.5 pr-3 text-slate-400">{r.time}</td>
                  <td className="py-1.5 pr-3 text-slate-500">{r.nodes || r.reason}</td>
                  <td className="py-1.5">
                    <button
                      onClick={() => cancel(r.jobid)}
                      className="rounded border border-rose-500/40 px-2 py-0.5 text-[10px] text-rose-300 hover:bg-rose-500/10"
                    >
                      cancel
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────── launcher

const KINDS = [
  { id: "viz", label: "Visualize" },
  { id: "train", label: "Train" },
  { id: "eval", label: "Eval" },
];

function Launcher({ onLaunched }) {
  const [kind, setKind] = useState("viz");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [preview, setPreview] = useState(null);

  // shared run/checkpoint pickers
  const [runs, setRuns] = useState([]);
  useEffect(() => {
    api.clusterRuns().then(setRuns).catch(() => {});
  }, []);

  const [form, setForm] = useState({
    // viz
    mode: "clip", clips: "000021,000019,000022", prompts: "a person walks forward",
    checkpoint: "", run: "",
    // train
    train_preset: "rmg_base", model_preset: "dit_base", subset_n: 16, max_steps: 15000,
    lr: "", overrides: "",
    // eval
    max_clips: 256, guidance_scales: "[6.5]",
  });
  const set = (k) => (e) => {
    const val = e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setForm((f) => ({ ...f, [k]: val }));
  };

  async function launch() {
    setBusy(true);
    setMsg(null);
    try {
      let res;
      if (kind === "viz") res = await api.submitViz(form);
      else if (kind === "train") res = await api.submitTrain(form);
      else res = await api.submitEval(form);
      setMsg({ ok: true, text: `launched ${res.kind} → slurm ${res.slurm_id || "?"}` });
      setPreview(null);
      onLaunched();
    } catch (e) {
      setMsg({ ok: false, text: e.message });
    } finally {
      setBusy(false);
    }
  }

  async function doPreview() {
    setMsg(null);
    try {
      const res = await api.previewTrain(form);
      setPreview(res.command);
    } catch (e) {
      setMsg({ ok: false, text: e.message });
    }
  }

  return (
    <div className="surface p-5">
      <div className="mb-4 flex gap-1.5">
        {KINDS.map((k) => (
          <button
            key={k.id}
            onClick={() => { setKind(k.id); setPreview(null); setMsg(null); }}
            className={`rounded-md px-3 py-1.5 text-[12px] font-medium transition ${
              kind === k.id
                ? "bg-[var(--signal-dim)] text-[var(--signal)] border border-[var(--signal)]"
                : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"
            }`}
          >
            {k.label}
          </button>
        ))}
      </div>

      {kind === "viz" && (
        <div className="space-y-3">
          <Field label="mode">
            <select className="field-input" value={form.mode} onChange={set("mode")}>
              {["clip", "prompt", "compare", "samples"].map((m) => <option key={m}>{m}</option>)}
            </select>
          </Field>
          {(form.mode === "clip" || form.mode === "compare") && (
            <Field label="clips (comma-sep, or 'auto')">
              <input className="field-input" value={form.clips} onChange={set("clips")} />
            </Field>
          )}
          {form.mode === "prompt" && (
            <Field label="prompts (| separated)">
              <input className="field-input" value={form.prompts} onChange={set("prompts")} />
            </Field>
          )}
          {form.mode === "samples" && (
            <RunPicker runs={runs.filter((r) => r.has_samples)} value={form.run} onChange={set("run")} />
          )}
          {(form.mode === "prompt" || form.mode === "compare") && (
            <CheckpointPicker runs={runs.filter((r) => r.has_checkpoints)} value={form.checkpoint} onChange={(v) => setForm((f) => ({ ...f, checkpoint: v }))} />
          )}
        </div>
      )}

      {kind === "train" && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Field label="model"><select className="field-input" value={form.model_preset} onChange={set("model_preset")}><option>dit_base</option><option>dit_large</option></select></Field>
            <Field label="train preset"><select className="field-input" value={form.train_preset} onChange={set("train_preset")}><option>rmg_base</option><option>rmg_large</option></select></Field>
            <Field label="subset_n"><input type="number" className="field-input" value={form.subset_n} onChange={set("subset_n")} /></Field>
            <Field label="max_steps"><input type="number" className="field-input" value={form.max_steps} onChange={set("max_steps")} /></Field>
          </div>
          <Field label="extra overrides (hydra, space-sep)">
            <input className="field-input" placeholder="train.optimizer.lr=1e-4 train.precision=bf16" value={form.overrides} onChange={set("overrides")} />
          </Field>
          {preview && (
            <pre className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink p-3 text-[11px] text-[var(--signal)]">{preview}</pre>
          )}
          <button onClick={doPreview} className="btn-ghost w-full text-[12px]">PREVIEW COMMAND</button>
        </div>
      )}

      {kind === "eval" && (
        <div className="space-y-3">
          <CheckpointPicker runs={runs.filter((r) => r.has_checkpoints)} value={form.checkpoint} onChange={(v) => setForm((f) => ({ ...f, checkpoint: v }))} />
          <div className="grid grid-cols-2 gap-3">
            <Field label="max_clips"><input type="number" className="field-input" value={form.max_clips} onChange={set("max_clips")} /></Field>
            <Field label="guidance_scales"><input className="field-input" value={form.guidance_scales} onChange={set("guidance_scales")} /></Field>
          </div>
        </div>
      )}

      {msg && (
        <p className={`mt-3 text-[12px] ${msg.ok ? "text-emerald-300" : "text-rose-300"}`}>
          {msg.ok ? "✓ " : "⚠ "}{msg.text}
        </p>
      )}

      <button onClick={launch} disabled={busy} className="btn-signal mt-4 w-full">
        {busy ? "LAUNCHING…" : `▶  LAUNCH ${kind.toUpperCase()} JOB`}
      </button>
      {kind === "train" && (
        <p className="label mt-2 text-center">launches a real GPU training run — confirm the preview first</p>
      )}
    </div>
  );
}

function RunPicker({ runs, value, onChange }) {
  return (
    <Field label="run">
      <select className="field-input" value={value} onChange={onChange}>
        <option value="">select run…</option>
        {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
      </select>
    </Field>
  );
}

function CheckpointPicker({ runs, value, onChange }) {
  const [run, setRun] = useState("");
  const [ckpts, setCkpts] = useState([]);
  useEffect(() => {
    if (!run) { setCkpts([]); return; }
    api.clusterCheckpoints(run).then(setCkpts).catch(() => setCkpts([]));
  }, [run]);
  return (
    <div className="grid grid-cols-2 gap-3">
      <Field label="run">
        <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
          <option value="">select run…</option>
          {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
        </select>
      </Field>
      <Field label="checkpoint">
        <select className="field-input" value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="">select…</option>
          {ckpts.map((c) => <option key={c} value={c}>{c.split("/").pop()}</option>)}
        </select>
      </Field>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────── our jobs

function Jobs({ jobs }) {
  if (jobs.length === 0) return null;
  return (
    <div className="surface p-5">
      <div className="label mb-3">launched jobs · {jobs.length}</div>
      <div className="space-y-2">
        {jobs.map((j) => <JobRow key={j.id} job={j} />)}
      </div>
    </div>
  );
}

function JobRow({ job }) {
  const [log, setLog] = useState(null);
  const color = STATE_COLOR[job.state] || "var(--muted)";
  async function showLog() {
    try { setLog((await api.jobLog(job.id)).log || "(empty)"); }
    catch (e) { setLog(`error: ${e.message}`); }
  }
  return (
    <div className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span className="rounded px-2 py-0.5 font-mono text-[11px]" style={{ color, border: `1px solid ${color}` }}>
          {job.state}
        </span>
        <span className="font-mono text-slate-300">{job.kind}{job.mode ? `/${job.mode}` : ""}</span>
        {job.slurm_id && <span className="font-mono text-[var(--muted)]">#{job.slurm_id}</span>}
        <span className="truncate font-mono text-[var(--muted)]">{job.run_name}</span>
        {job.error && <span className="text-rose-300">{job.error}</span>}
        {job.slurm_id && (
          <button onClick={showLog} className="ml-auto rounded border border-[var(--hairline-strong)] px-2 py-0.5 text-[10px] text-slate-300 hover:border-[var(--signal)]">
            log
          </button>
        )}
      </div>
      {job.outputs?.length > 0 && (
        <div className="mt-2 flex gap-2 overflow-x-auto">
          {job.outputs.map((o, i) => (
            <a key={i} href={mediaUrl(o.media_url)} target="_blank" rel="noreferrer" className="shrink-0">
              {o.media_url.toLowerCase().endsWith(".mp4") ? (
                <video src={mediaUrl(o.media_url)} className="h-16 w-16 rounded border border-[var(--hairline)] object-cover" muted />
              ) : (
                <img src={mediaUrl(o.media_url)} alt={o.caption} className="h-16 w-16 rounded border border-[var(--hairline)] object-cover" />
              )}
            </a>
          ))}
        </div>
      )}
      {log != null && (
        <pre className="mt-2 max-h-48 overflow-auto rounded border border-[var(--hairline)] bg-black p-2 text-[10px] text-slate-400">{log}</pre>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="label mb-1.5 block">{label}</span>
      {children}
    </label>
  );
}
