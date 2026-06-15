"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import LineChart from "./LineChart";
import EvalComparison from "./EvalTab";

const SIGNAL = "#22d3ee";
const AMBER = "#fbbf24";
const ROLE_COLOR = { real: SIGNAL, gen: AMBER, clip: SIGNAL };

export default function AnalysisTab() {
  return (
    <div className="space-y-8">
      <Section n="01" title="Training" sub="curves · config · duration"><TrainingCurves /></Section>
      <Section n="02" title="Eval comparison" sub="metrics · guidance sweep · LaTeX"><EvalComparison /></Section>
      <Section n="03" title="Clip comparison" sub="real vs gen — trajectory · jitter · contact"><JointAnalysis /></Section>
    </div>
  );
}

function Section({ n, title, sub, children }) {
  return (
    <section>
      <div className="mb-3 flex items-baseline gap-3">
        <span className="font-mono text-[11px] tracking-widest text-[var(--signal)]">{n}</span>
        <h2 className="display text-lg font-bold text-white">{title}</h2>
        <span className="label">{sub}</span>
      </div>
      {children}
    </section>
  );
}

// ───────────────────────────────────────────────────── training curves + config

function fmtDuration(s) {
  if (s == null) return "—";
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function TrainingCurves() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [data, setData] = useState(null);
  const [info, setInfo] = useState(null);
  const [col, setCol] = useState("loss");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_metrics))).catch(() => {});
  }, []);

  useEffect(() => {
    if (!run) return;
    setLoading(true);
    setError(null);
    setInfo(null);
    api.runMetrics(run)
      .then((d) => { setData(d); if (!d.columns.includes(col)) setCol(d.columns[0]); })
      .catch((e) => { setError(e.message); setData(null); })
      .finally(() => setLoading(false));
    api.runInfo(run).then(setInfo).catch(() => {});
  }, [run]); // eslint-disable-line react-hooks/exhaustive-deps

  const series = useMemo(() => {
    if (!data) return [];
    const xi = data.columns.indexOf("step");
    const yi = data.columns.indexOf(col);
    if (yi < 0) return [];
    const points = data.rows.map((r) => [xi >= 0 ? r[xi] : 0, r[yi]]).filter(([x, y]) => x != null && y != null);
    return [{ label: col, color: SIGNAL, points }];
  }, [data, col]);

  // Total training time = Σ Δstep / steps_per_s across logged rows. This is the
  // actual compute time and survives resumes (file mtimes don't — they only
  // reflect the last resume job, which is why mtime under-reports badly).
  const metricDuration = useMemo(() => {
    if (!data) return null;
    const si = data.columns.indexOf("step");
    const ri = data.columns.indexOf("steps_per_s");
    if (si < 0 || ri < 0) return null;
    const rows = data.rows.filter((r) => r[si] != null).sort((a, b) => a[si] - b[si]);
    let sec = 0;
    for (let i = 1; i < rows.length; i++) {
      const ds = rows[i][si] - rows[i - 1][si];
      const rate = rows[i][ri];
      if (ds > 0 && rate > 0) sec += ds / rate;
    }
    return sec > 0 ? Math.round(sec) : null;
  }, [data]);

  const cfg = info?.config;
  const facts = cfg && [
    ["model", cfg.model?.name],
    ["repr", cfg.representation?.name],
    ["max_steps", cfg.train?.max_steps],
    ["subset_n", cfg.data?.subset_n],
    ["subset_frac", cfg.data?.subset_fraction],
    ["lr", cfg.train?.optimizer?.lr],
    ["precision", cfg.train?.precision],
    ["train time", fmtDuration(metricDuration ?? info?.duration_seconds)],
  ].filter(([, v]) => v !== undefined && v !== null);

  return (
    <div className="surface p-5">
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="block min-w-[220px]">
          <span className="label mb-1.5 block">training run (metrics.csv)</span>
          <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">select run…</option>
            {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
          </select>
        </label>
        {data && (
          <div className="flex flex-wrap gap-1.5">
            {data.columns.filter((c) => c !== "step").map((c) => (
              <button key={c} onClick={() => setCol(c)}
                className={`rounded-md px-2.5 py-1 text-[11px] transition ${col === c ? "bg-[var(--signal-dim)] text-[var(--signal)] border border-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                {c}
              </button>
            ))}
          </div>
        )}
      </div>

      {facts && (
        <div className="mb-4 flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-[var(--hairline)] bg-ink px-4 py-3">
          {facts.map(([k, v]) => (
            <div key={k}>
              <div className="label">{k}</div>
              <div className="font-mono text-[13px] text-slate-200">{String(v)}</div>
            </div>
          ))}
        </div>
      )}

      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {loading && <p className="text-[12px] text-[var(--muted)]">loading metrics…</p>}
      {data && <LineChart series={series} width={640} height={260} xLabel="step" yLabel={col} yZero={col === "loss"} />}
      {!run && !error && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
          Pick a run to plot its training curves + config.
        </p>
      )}
    </div>
  );
}

// ───────────────────────────────────── joint .npy analysis (GT vs gen overlay)

function JointAnalysis() {
  const [jobs, setJobs] = useState([]);
  const [sel, setSel] = useState("");
  const [series, setSeries] = useState(null); // {trajectory,jitter,speed,foot_height: LineChart series[]}
  const [stats, setStats] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.jobs()
      .then((j) => setJobs(j.filter((x) => x.state === "done" && (x.outputs || []).some((o) => o.npy_url))))
      .catch(() => {});
  }, []);

  // Group each done job's .npy outputs by clip id, pairing real-<cid>/gen-<cid>.
  const groups = useMemo(() =>
    jobs.flatMap((j) => {
      const ckpt = ckptLabel(j.params?.checkpoint);
      const byId = {};
      (j.outputs || []).filter((o) => o.npy_url).forEach((o) => {
        const name = o.npy_url.split("/").pop();
        const m = name.match(/^(real|gen)-(.+)\.npy$/);
        const role = m ? m[1] : "clip";
        const cid = m ? m[2] : name.replace(/\.npy$/, "");
        (byId[cid] = byId[cid] || { job: j.id, cid, items: [], mode: j.mode || j.kind, ckpt }).items.push({ role, name });
      });
      return Object.values(byId).map((g) => ({
        ...g, key: `${g.job}::${g.cid}`,
        label: `${g.cid} · ${g.items.map((i) => i.role).join("+")}${g.ckpt ? ` · ${g.ckpt}` : ` · ${g.mode}`}`,
      }));
    }), [jobs]);

  const selGroup = groups.find((x) => x.key === sel);

  useEffect(() => {
    const g = groups.find((x) => x.key === sel);
    if (!g) return;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        // serialized backend → fetch sequentially
        const results = [];
        for (const it of g.items) results.push({ role: it.role, data: await api.analysisNpy(g.job, it.name) });
        const mk = (field) => results.map((r) => ({ label: r.role, color: ROLE_COLOR[r.role] || SIGNAL, points: r.data[field] }));
        setSeries({ trajectory: mk("trajectory"), jitter: mk("jitter"), speed: mk("speed"), foot_height: mk("foot_height") });
        setStats(results.map((r) => ({ role: r.role, jerk: r.data.jerk_mean, frames: r.data.frames })));
      } catch (e) { setError(e.message); setSeries(null); }
      finally { setLoading(false); }
    })();
  }, [sel, groups]);

  return (
    <div className="surface p-5">
      <div className="mb-4 flex flex-wrap items-end gap-4">
        <label className="block min-w-[300px]">
          <span className="label mb-1.5 block">clip (.npy from a viz job — compare jobs overlay GT vs gen)</span>
          <select className="field-input" value={sel} onChange={(e) => setSel(e.target.value)}>
            <option value="">select clip…</option>
            {groups.map((g) => <option key={g.key} value={g.key}>{g.label}</option>)}
          </select>
        </label>
        {selGroup?.ckpt && (
          <div>
            <div className="label">checkpoint</div>
            <div className="font-mono text-sm text-slate-200">{selGroup.ckpt}</div>
          </div>
        )}
        {stats.map((s) => (
          <div key={s.role}>
            <div className="label" style={{ color: ROLE_COLOR[s.role] }}>{s.role} · jerk</div>
            <div className="font-mono text-sm text-slate-200">{s.jerk ?? "—"}</div>
          </div>
        ))}
        {stats.length === 2 && (
          <div>
            <div className="label">gen/GT jerk</div>
            <div className="font-mono text-sm text-[var(--amber)]">
              {(() => { const r = stats.find((x) => x.role === "real")?.jerk, g = stats.find((x) => x.role === "gen")?.jerk; return r && g ? `${(g / r).toFixed(1)}×` : "—"; })()}
            </div>
          </div>
        )}
      </div>

      {groups.length === 0 && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
          Run a viz job (Cluster / GT tab) — its rendered clips appear here. Use a <span className="text-slate-300">compare</span> job to overlay GT vs prediction.
        </p>
      )}
      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {loading && <p className="text-[12px] text-[var(--muted)]">analysing joints…</p>}

      {series && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Chart title="root trajectory (top-down)"><LineChart series={series.trajectory} equal width={360} height={300} xLabel="x" yLabel="z" /></Chart>
          <Chart title="jitter — ‖Δ³x‖ per frame (lower = smoother)"><LineChart series={series.jitter} width={360} height={300} xLabel="frame" yLabel="jerk" yZero /></Chart>
          <Chart title="root speed (m/s)"><LineChart series={series.speed} width={360} height={260} xLabel="frame" yLabel="m/s" yZero /></Chart>
          <Chart title="min foot height (contact near 0)"><LineChart series={series.foot_height} width={360} height={260} xLabel="frame" yLabel="height" yZero /></Chart>
        </div>
      )}
    </div>
  );
}

// /…/runs/<kind>/<run>/checkpoints/<file>.pt → "<run>/<file>.pt"
function ckptLabel(path) {
  if (!path) return null;
  const p = path.split("/");
  return p.length >= 3 ? `${p[p.length - 3]}/${p[p.length - 1]}` : p[p.length - 1];
}

function Chart({ title, children }) {
  return (
    <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
      <figcaption className="label mb-2">{title}</figcaption>
      {children}
    </figure>
  );
}
