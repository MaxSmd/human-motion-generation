"use client";

// ── Clip comparison ────────────────────────────────────────────────────────
// Was §03 of the Analysis tab; promoted to a tab of its own so "how does this
// one clip actually move?" isn't buried under the training curves.
//
// Pick a rendered clip and read its dynamics against the ground truth: root
// trajectory, per-frame jitter, speed, foot height / foot-skate, and where the
// roughness lives per joint. A `compare` viz job pairs real-<cid>/gen-<cid> so
// GT and prediction overlay on every chart; a plain prompt render shows alone.

import { useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import LineChart from "./LineChart";
import BarChart, { histogramBars } from "./BarChart";
import { FALLBACK_JOINTS } from "./ConstraintsTab";
import { SIGNAL, AMBER, ROLE_COLOR, ROLE_LABEL, Chart, EmptyHint, ckptLabel } from "./AnalysisKit";
import { TableTools } from "./ExportButtons";

const JOINT_NAMES = FALLBACK_JOINTS.map((j) => j.name);

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
    footSkate: d.foot_skate_mean,
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
  ["footSkate", "foot-skate", "m/s", (v) => v.toFixed(3), true],
  ["meanSpeed", "mean root speed", "m/s", (v) => v.toFixed(2), null],
  ["maxSpeed", "max root speed", "m/s", (v) => v.toFixed(2), null],
  ["path", "path length", "m", (v) => v.toFixed(2), null],
  ["contactPct", "foot contact", "%", (v) => v.toFixed(0), null],
];

// Viz-job params → clip-group metadata for filtering. GT-only clip renders
// have no model in the loop → config "gt"; generated ones resolve their config
// from model_preset (checkpoint-path heuristic as fallback).
function vizMeta(j) {
  const p = j.params || {};
  const mode = j.mode || j.kind;
  const ckpt = p.checkpoint || "";
  const cfg = mode === "clip"
    ? "gt"
    : (p.model_preset || (/base/i.test(ckpt) ? "dit_base" : /mid/i.test(ckpt) ? "dit_mid" : "?")).replace(/^dit_/, "");
  return {
    cfg, mode,
    w: p.guidance != null ? parseFloat(p.guidance) : null,
    steps: p.num_steps != null ? parseInt(p.num_steps, 10) : null,
  };
}

const numSort = (a, b) => (a === "—" ? 1 : b === "—" ? -1 : a - b);

export default function ClipComparisonTab() {
  const clipMetricsRef = useRef(null);
  const [jobs, setJobs] = useState([]);
  const [sel, setSel] = useState("");
  const [filters, setFilters] = useState({ cfg: "all", w: "all", steps: "all", mode: "all" });
  const [series, setSeries] = useState(null);
  const [stats, setStats] = useState([]);
  const [perJoint, setPerJoint] = useState(null);
  const [dist, setDist] = useState(null);
  const [thr, setThr] = useState(0.05);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.jobs()
      .then((j) => setJobs(j.filter((x) => x.state === "done" && (x.outputs || []).some((o) => o.npy_url))))
      .catch(() => {});
  }, []);

  // Group each done job's .npy outputs by clip id, pairing real-<cid>/gen-<cid>.
  // Each item keeps its full output (caption, media, kind) so the selected clip
  // can be shown, not just named; group meta (config/ω/steps/mode) drives filters.
  const groups = useMemo(() =>
    jobs.flatMap((j) => {
      const ckpt = ckptLabel(j.params?.checkpoint);
      const meta = vizMeta(j);
      const byId = {};
      (j.outputs || []).filter((o) => o.npy_url).forEach((o) => {
        const name = o.npy_url.split("/").pop();
        const m = name.match(/^(real|gen)-(.+)\.npy$/);
        const role = m ? m[1] : "clip";
        const cid = m ? m[2] : name.replace(/\.npy$/, "");
        (byId[cid] = byId[cid] || { job: j.id, cid, items: [], ...meta, ckpt }).items.push({ role, name, out: o });
      });
      return Object.values(byId).map((g) => {
        const caption = g.items.find((i) => i.out.caption)?.out.caption || "";
        return {
          ...g, caption, key: `${g.job}::${g.cid}`,
          label: `${g.cid} · ${caption ? `“${caption.slice(0, 48)}${caption.length > 48 ? "…" : ""}” · ` : ""}${g.items.map((i) => ROLE_LABEL[i.role] || i.role).join("+")}`,
        };
      });
    }), [jobs]);

  const dims = useMemo(() => ({
    cfg: [...new Set(groups.map((g) => g.cfg))].sort(),
    w: [...new Set(groups.map((g) => g.w ?? "—"))].sort(numSort),
    steps: [...new Set(groups.map((g) => g.steps ?? "—"))].sort(numSort),
    mode: [...new Set(groups.map((g) => g.mode))].sort(),
  }), [groups]);

  const visible = useMemo(() => groups.filter((g) =>
    (filters.cfg === "all" || g.cfg === filters.cfg) &&
    (filters.w === "all" || String(g.w ?? "—") === String(filters.w)) &&
    (filters.steps === "all" || String(g.steps ?? "—") === String(filters.steps)) &&
    (filters.mode === "all" || g.mode === filters.mode)
  ), [groups, filters]);

  useEffect(() => {
    if (sel && !visible.some((g) => g.key === sel)) setSel("");
  }, [visible, sel]);

  const selGroup = visible.find((x) => x.key === sel);

  useEffect(() => {
    const g = groups.find((x) => x.key === sel);
    if (!g) { setSeries(null); setPerJoint(null); setDist(null); setStats([]); return; }
    setLoading(true);
    setError(null);
    (async () => {
      try {
        // serialized backend → fetch sequentially
        const results = [];
        for (const it of g.items) results.push({ role: it.role, data: await api.analysisNpy(g.job, it.name) });
        // Default missing series to [] so an older backend (no foot_skate /
        // per_joint_jerk) degrades to an empty chart instead of white-screening
        // the whole tab (undefined points → LineChart destructure crash).
        const mk = (field) => results.map((r) => ({
          label: ROLE_LABEL[r.role] || r.role, color: ROLE_COLOR[r.role] || SIGNAL,
          points: Array.isArray(r.data?.[field]) ? r.data[field] : [],
        }));
        setSeries({
          trajectory: mk("trajectory"), jitter: mk("jitter"), speed: mk("speed"),
          foot_height: mk("foot_height"), foot_skate: mk("foot_skate"),
        });
        setStats(results.map((r) => ({ role: r.role, ...clipStats(r.data) })));
        setThr(results[0]?.data?.contact_threshold ?? 0.05);

        const pjJerk = JOINT_NAMES.map((name, j) => ({
          label: name,
          values: results.map((r) => ({ v: r.data.per_joint_jerk?.[j] ?? 0, color: ROLE_COLOR[r.role] || SIGNAL })),
        }));

        const realIt = g.items.find((i) => i.role === "real");
        const genIt = g.items.find((i) => i.role === "gen");
        let err = null;
        if (realIt && genIt) {
          try {
            const c = await api.compareNpy(g.job, realIt.name, genIt.name);
            err = {
              mpjpe: c.mpjpe,
              bars: JOINT_NAMES.map((name, j) => ({ label: name, values: [{ v: c.per_joint_error?.[j] ?? 0, color: AMBER }] })),
            };
          } catch { /* mismatched lengths etc — skip */ }
        }
        setPerJoint({ jerk: pjJerk, err });

        const vals = (r, f) => (r.data[f] || []).map(([, v]) => v);
        const distSeries = (f) => results.map((r) => ({ label: ROLE_LABEL[r.role] || r.role, color: ROLE_COLOR[r.role] || SIGNAL, values: vals(r, f) }));
        setDist({ jerk: histogramBars(distSeries("jitter")), speed: histogramBars(distSeries("speed")) });
      } catch (e) { setError(e.message); setSeries(null); setStats([]); setPerJoint(null); setDist(null); }
      finally { setLoading(false); }
    })();
  }, [sel, groups]);

  const real = stats.find((s) => s.role === "real");
  const gen = stats.find((s) => s.role === "gen");

  const chip = (dim, v) => (
    <button key={String(v)} onClick={() => setFilters((f) => ({ ...f, [dim]: String(f[dim]) === String(v) ? "all" : v }))}
      className={`rounded-md px-2.5 py-1 text-[11px] transition ${String(filters[dim]) === String(v) ? "bg-[var(--signal-dim)] text-[var(--signal)] border border-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
      {String(v)}
    </button>
  );

  return (
    <div className="surface p-5">
      <div className="mb-4">
        <h2 className="display text-lg font-bold text-white">Clip comparison</h2>
        <p className="label mt-0.5 normal-case tracking-normal">
          real vs gen — trajectory · jitter · foot-skate · per-joint
        </p>
      </div>

      {groups.length > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2">
          {[["cfg", "config"], ["w", "ω"], ["steps", "ODE steps"], ["mode", "job"]].map(([dim, label]) =>
            dims[dim].length > 1 ? (
              <span key={dim} className="flex items-center gap-1.5">
                <span className="label">{label}</span>
                {dims[dim].map((v) => chip(dim, v))}
              </span>
            ) : null
          )}
          <span className="ml-auto font-mono text-[10px] text-[var(--muted)]">{visible.length}/{groups.length} clips</span>
        </div>
      )}

      <label className="mb-4 block max-w-2xl">
        <span className="label mb-1.5 block">clip (.npy from a viz job — clip renders overlay GT vs gen)</span>
        <select className="field-input" value={sel} onChange={(e) => setSel(e.target.value)}>
          <option value="">select clip…</option>
          {visible.map((g) => <option key={g.key} value={g.key}>{g.label}</option>)}
        </select>
      </label>

      {groups.length === 0 && (
        <EmptyHint>
          Render clips in <span className="text-slate-300">Library</span> — they appear here. Every clip render carries its GT, so the GT-vs-prediction overlay is always available.
        </EmptyHint>
      )}
      {groups.length > 0 && visible.length === 0 && <EmptyHint>No clips match the active filters.</EmptyHint>}
      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {loading && <p className="text-[12px] text-[var(--muted)]">analysing joints…</p>}

      {selGroup && (
        <div className="mb-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <div>
            <div className="label mb-2">
              clip · {selGroup.cid} · {selGroup.cfg}
              {selGroup.w != null ? ` · ω=${selGroup.w}` : ""}
              {selGroup.steps != null ? ` · ${selGroup.steps} ODE steps` : ""}
              {selGroup.ckpt ? ` · ${selGroup.ckpt}` : ""}
            </div>
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

          {stats.length > 0 && (
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="label">clip metrics</span>
                <TableTools getTable={() => clipMetricsRef.current} name={`clip-metrics-${selGroup?.cid || ""}`}
                  caption="Per-clip motion metrics." label="tab:clip-metrics" />
              </div>
              <table ref={clipMetricsRef} className="w-full border-collapse text-left text-[12px]">
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
          <Chart title="root trajectory (top-down)"><LineChart series={series.trajectory} equal width={420} height={300} xLabel="x" yLabel="z" exportName={`traj-${selGroup?.cid}`} /></Chart>
          <Chart title="jitter — ‖Δ³x‖ per frame (lower = smoother)"><LineChart series={series.jitter} width={420} height={300} xLabel="frame" yLabel="jerk" yZero exportName={`jitter-${selGroup?.cid}`} /></Chart>
          <Chart title="root speed (m/s)"><LineChart series={series.speed} width={420} height={260} xLabel="frame" yLabel="m/s" yZero exportName={`speed-${selGroup?.cid}`} /></Chart>
          <Chart title="min foot height — dashed line = contact threshold">
            <LineChart series={series.foot_height} width={420} height={260} xLabel="frame" yLabel="height" yZero
              hlines={[{ y: thr, color: "#34d399", dash: "4 4" }]} exportName={`footheight-${selGroup?.cid}`} />
          </Chart>
          <Chart title="foot-skate — planted-foot drift (m/s, lower = less sliding)">
            <LineChart series={series.foot_skate} width={420} height={260} xLabel="frame" yLabel="m/s" yZero exportName={`footskate-${selGroup?.cid}`} />
          </Chart>
          {dist && dist.jerk.bars.length > 0 && (
            <Chart title="jerk distribution (normalized) — rougher motion = mass to the right">
              <BarChart bars={dist.jerk.bars} legend={dist.jerk.legend} width={420} height={260} horizontal={false} exportName={`jerkdist-${selGroup?.cid}`} />
            </Chart>
          )}
        </div>
      )}

      {perJoint && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Chart title="per-joint mean jerk — which joints are roughest">
            <BarChart bars={perJoint.jerk} width={420} height={360}
              legend={stats.map((s) => ({ label: ROLE_LABEL[s.role] || s.role, color: ROLE_COLOR[s.role] }))}
              exportName={`perjoint-jerk-${selGroup?.cid}`} />
          </Chart>
          {perJoint.err && (
            <Chart title={`per-joint GT↔gen position error · MPJPE ${perJoint.err.mpjpe} m (root-aligned)`}>
              <BarChart bars={perJoint.err.bars} width={420} height={360} unit="m"
                legend={[{ label: "error", color: AMBER }]} exportName={`perjoint-err-${selGroup?.cid}`} />
            </Chart>
          )}
        </div>
      )}
    </div>
  );
}
