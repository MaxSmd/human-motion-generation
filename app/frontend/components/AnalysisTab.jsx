"use client";

import { useEffect, useMemo, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import LineChart from "./LineChart";
import EvalComparison from "./EvalTab";
import OdeStepSweep from "./OdeStepSweep";
import ValidationPanel from "./ValidationPanel";

const SIGNAL = "#22d3ee";
const AMBER = "#fbbf24";
const ROLE_COLOR = { real: SIGNAL, gen: AMBER, clip: SIGNAL };

export default function AnalysisTab() {
  return (
    <div className="space-y-8">
      <Section n="01" title="Training" sub="curves · config · duration"><TrainingCurves /></Section>
      <Section n="02" title="Eval comparison" sub="metrics · guidance sweep · LaTeX"><EvalComparison /></Section>
      <Section n="03" title="ODE step sweep" sub="fixed guidance · metrics vs num_sample_steps"><OdeStepSweep /></Section>
      <Section n="04" title="Clip comparison" sub="real vs gen — trajectory · jitter · contact"><JointAnalysis /></Section>
      <Section n="05" title="Validation" sub="is the gap a bug? — end-to-end audit"><ValidationPanel /></Section>
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

  // Curve-derived summary: where the run got to and how fast it trained —
  // readable at a glance without hunting through the plotted series.
  const summary = useMemo(() => {
    if (!data) return null;
    const si = data.columns.indexOf("step");
    const li = data.columns.indexOf("loss");
    const ri = data.columns.indexOf("steps_per_s");
    const rows = si >= 0 ? data.rows.filter((r) => r[si] != null).sort((a, b) => a[si] - b[si]) : data.rows;
    if (rows.length === 0) return null;
    const out = [];
    if (si >= 0) out.push(["last step", rows[rows.length - 1][si].toLocaleString()]);
    if (li >= 0) {
      const tail = rows.slice(-20).map((r) => r[li]).filter((v) => v != null);
      if (tail.length) out.push(["final loss (tail avg)", (tail.reduce((a, b) => a + b, 0) / tail.length).toFixed(4)]);
      let best = null, bestStep = null;
      for (const r of rows) if (r[li] != null && (best == null || r[li] < best)) { best = r[li]; bestStep = si >= 0 ? r[si] : null; }
      if (best != null) out.push(["best loss", `${best.toFixed(4)}${bestStep != null ? ` @ ${bestStep.toLocaleString()}` : ""}`]);
    }
    if (ri >= 0) {
      const rates = rows.map((r) => r[ri]).filter((v) => v > 0);
      if (rates.length) out.push(["throughput", `${(rates.reduce((a, b) => a + b, 0) / rates.length).toFixed(2)} it/s`]);
    }
    return out;
  }, [data]);

  const cfg = info?.config;
  const facts = [
    ...(cfg
      ? [
          ["model", cfg.model?.name],
          ["repr", cfg.representation?.name],
          ["max_steps", cfg.train?.max_steps],
          ["subset_n", cfg.data?.subset_n],
          ["subset_frac", cfg.data?.subset_fraction],
          ["lr", cfg.train?.optimizer?.lr],
          ["precision", cfg.train?.precision],
          ["train time", fmtDuration(metricDuration ?? info?.duration_seconds)],
        ]
      : []),
    ...(summary || []),
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

      {facts.length > 0 && (
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
      {data && <LineChart series={series} width={760} height={280} xLabel="step" yLabel={col} yZero={col === "loss"} />}
      {!run && !error && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
          Pick a run to plot its training curves + config.
        </p>
      )}
    </div>
  );
}

// ───────────────────────────────────── joint .npy analysis (GT vs gen overlay)

const ROLE_LABEL = { real: "GT", gen: "GEN", clip: "CLIP" };

// Derived per-clip stats from the backend's analysis response — computed here
// (the series are already in the payload) so richer stats need no backend change.
function clipStats(d) {
  const fps = d.fps || 20;
  const speeds = (d.speed || []).map(([, v]) => v);
  const meanSpeed = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null;
  const maxSpeed = speeds.length ? Math.max(...speeds) : null;
  let path = 0;
  for (let i = 1; i < (d.trajectory || []).length; i++) {
    const [x0, z0] = d.trajectory[i - 1], [x1, z1] = d.trajectory[i];
    path += Math.hypot(x1 - x0, z1 - z0);
  }
  const thr = d.contact_threshold ?? 0.05;
  const contactPct = d.foot_height?.length
    ? (100 * d.foot_height.filter(([, h]) => h < thr).length) / d.foot_height.length
    : null;
  return {
    frames: d.frames,
    duration: d.frames / fps,
    jerk: d.jerk_mean,
    accel: d.accel_mean,
    meanSpeed, maxSpeed,
    path: d.trajectory?.length > 1 ? path : null,
    contactPct,
  };
}

// [key, label, unit, formatter, lowerBetter] rows of the stats table.
const STAT_ROWS = [
  ["frames", "frames", "", (v) => v, null],
  ["duration", "duration", "s", (v) => v.toFixed(1), null],
  ["jerk", "jerk ‖Δ³x‖", "", (v) => v.toFixed(4), true],
  ["accel", "accel ‖Δ²x‖", "", (v) => v.toFixed(4), true],
  ["meanSpeed", "mean root speed", "m/s", (v) => v.toFixed(2), null],
  ["maxSpeed", "max root speed", "m/s", (v) => v.toFixed(2), null],
  ["path", "path length", "m", (v) => v.toFixed(2), null],
  ["contactPct", "foot contact", "%", (v) => v.toFixed(0), null],
];

function JointAnalysis() {
  const [jobs, setJobs] = useState([]);
  const [sel, setSel] = useState("");
  const [series, setSeries] = useState(null); // {trajectory,jitter,speed,foot_height: LineChart series[]}
  const [stats, setStats] = useState([]);    // [{role, ...clipStats}]
  const [thr, setThr] = useState(0.05);      // foot-contact threshold from backend
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.jobs()
      .then((j) => setJobs(j.filter((x) => x.state === "done" && (x.outputs || []).some((o) => o.npy_url))))
      .catch(() => {});
  }, []);

  // Group each done job's .npy outputs by clip id, pairing real-<cid>/gen-<cid>.
  // Each item keeps its full output (caption, media, kind) so the selected clip
  // can be shown, not just named.
  const groups = useMemo(() =>
    jobs.flatMap((j) => {
      const ckpt = ckptLabel(j.params?.checkpoint);
      const byId = {};
      (j.outputs || []).filter((o) => o.npy_url).forEach((o) => {
        const name = o.npy_url.split("/").pop();
        const m = name.match(/^(real|gen)-(.+)\.npy$/);
        const role = m ? m[1] : "clip";
        const cid = m ? m[2] : name.replace(/\.npy$/, "");
        (byId[cid] = byId[cid] || { job: j.id, cid, items: [], mode: j.mode || j.kind, ckpt }).items.push({ role, name, out: o });
      });
      return Object.values(byId).map((g) => {
        const caption = g.items.find((i) => i.out.caption)?.out.caption || "";
        return {
          ...g, caption, key: `${g.job}::${g.cid}`,
          label: `${g.cid} · ${caption ? `“${caption.slice(0, 60)}${caption.length > 60 ? "…" : ""}” · ` : ""}${g.items.map((i) => ROLE_LABEL[i.role] || i.role).join("+")}${g.ckpt ? ` · ${g.ckpt}` : ` · ${g.mode}`}`,
        };
      });
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
        const mk = (field) => results.map((r) => ({
          label: ROLE_LABEL[r.role] || r.role, color: ROLE_COLOR[r.role] || SIGNAL, points: r.data[field],
        }));
        setSeries({ trajectory: mk("trajectory"), jitter: mk("jitter"), speed: mk("speed"), foot_height: mk("foot_height") });
        setStats(results.map((r) => ({ role: r.role, ...clipStats(r.data) })));
        setThr(results[0]?.data?.contact_threshold ?? 0.05);
      } catch (e) { setError(e.message); setSeries(null); setStats([]); }
      finally { setLoading(false); }
    })();
  }, [sel, groups]);

  const real = stats.find((s) => s.role === "real");
  const gen = stats.find((s) => s.role === "gen");

  return (
    <div className="surface p-5">
      <label className="mb-4 block max-w-2xl">
        <span className="label mb-1.5 block">clip (.npy from a viz job — compare jobs overlay GT vs gen)</span>
        <select className="field-input" value={sel} onChange={(e) => setSel(e.target.value)}>
          <option value="">select clip…</option>
          {groups.map((g) => <option key={g.key} value={g.key}>{g.label}</option>)}
        </select>
      </label>

      {groups.length === 0 && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
          Run a viz job (Library) — its rendered clips appear here. Use a <span className="text-slate-300">compare</span> job to overlay GT vs prediction.
        </p>
      )}
      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {loading && <p className="text-[12px] text-[var(--muted)]">analysing joints…</p>}

      {selGroup && (
        <div className="mb-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          {/* which clip is which: the actual renders, badged by role */}
          <div>
            <div className="label mb-2">clip · {selGroup.cid}{selGroup.ckpt ? ` · ${selGroup.ckpt}` : ""}</div>
            {selGroup.caption && (
              <p className="mb-2 text-[12px] leading-snug text-slate-300">“{selGroup.caption}”</p>
            )}
            <div className="grid grid-cols-2 gap-3">
              {selGroup.items.map((it) => (
                <figure key={it.name} className="overflow-hidden rounded-lg border bg-black"
                  style={{ borderColor: ROLE_COLOR[it.role] || "var(--hairline)" }}>
                  {it.out.media_url?.toLowerCase().endsWith(".mp4") ? (
                    <video src={mediaUrl(it.out.media_url)} className="w-full" autoPlay loop muted />
                  ) : it.out.media_url ? (
                    <img src={mediaUrl(it.out.media_url)} alt={it.role} className="w-full" />
                  ) : null}
                  <figcaption className="px-2 py-1 font-mono text-[10px]" style={{ color: ROLE_COLOR[it.role] || "var(--muted)" }}>
                    {ROLE_LABEL[it.role] || it.role}{it.role === "gen" && selGroup.ckpt ? ` · ${selGroup.ckpt}` : it.role === "real" ? " · dataset" : ""}
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>

          {/* side-by-side stats, with gen/GT ratios when both roles exist */}
          {stats.length > 0 && (
            <div>
              <div className="label mb-2">clip metrics</div>
              <table className="w-full border-collapse text-left text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                    <th className="py-1.5 pr-3 font-medium">metric</th>
                    {stats.map((s) => (
                      <th key={s.role} className="py-1.5 pr-3 font-medium" style={{ color: ROLE_COLOR[s.role] }}>
                        {ROLE_LABEL[s.role] || s.role}
                      </th>
                    ))}
                    {real && gen && <th className="py-1.5 font-medium text-[var(--amber)]">gen / GT</th>}
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {STAT_ROWS.map(([key, label, unit, f, lowerBetter]) => {
                    const ratio = real && gen && real[key] > 1e-9 && gen[key] != null ? gen[key] / real[key] : null;
                    const off = ratio != null && lowerBetter != null && Math.abs(ratio - 1) > 0.25;
                    return (
                      <tr key={key} className="border-b border-[var(--hairline)]">
                        <td className="py-1.5 pr-3 text-[var(--muted)]">{label}{unit ? ` (${unit})` : ""}</td>
                        {stats.map((s) => (
                          <td key={s.role} className="py-1.5 pr-3 text-slate-300">{s[key] == null ? "—" : f(s[key])}</td>
                        ))}
                        {real && gen && (
                          <td className="py-1.5" style={{ color: off ? "#fb7185" : "var(--amber)" }}>
                            {ratio == null ? "—" : `${ratio.toFixed(2)}×`}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {real && gen && (
                <p className="label mt-2 normal-case tracking-normal">
                  ratios ≈ 1 mean the generation matches the GT clip's dynamics; jerk/accel ≫ 1 = rougher motion.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {series && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Chart title="root trajectory (top-down)"><LineChart series={series.trajectory} equal width={420} height={300} xLabel="x" yLabel="z" /></Chart>
          <Chart title="jitter — ‖Δ³x‖ per frame (lower = smoother)"><LineChart series={series.jitter} width={420} height={300} xLabel="frame" yLabel="jerk" yZero /></Chart>
          <Chart title="root speed (m/s)"><LineChart series={series.speed} width={420} height={260} xLabel="frame" yLabel="m/s" yZero /></Chart>
          <Chart title="min foot height — dashed line = contact threshold">
            <LineChart series={series.foot_height} width={420} height={260} xLabel="frame" yLabel="height" yZero
              hlines={[{ y: thr, color: "#34d399", dash: "4 4" }]} />
          </Chart>
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
