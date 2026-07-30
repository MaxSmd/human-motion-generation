"use client";

// ── Obstacle-course lab ────────────────────────────────────────────────────
// Author-free by design: the three courses (slab / stairs / corner) live in
// `backend.analysis.courses` and arrive over `/cluster/analysis/courses`. This
// component never declares geometry of its own, so the staircase drawn here is
// provably the staircase that was sampled and the one the metrics scored.
//
// What it shows, per course:
//   • a PLAN view (x–z) and an ELEVATION view (z–y) of the room, the obstacles,
//     the spawn and the authored pelvis path, with every finished clip's actual
//     root path drawn over them, coloured by arm;
//   • the (arm × metric) table with mean ± sd over clips;
//   • the task-success readout — mounted / treads climbed / turned left — which
//     is the only thing that answers "did it do the course".
//
// The elevation view is the one that matters for slab and stairs: a clip that
// walks THROUGH the flight and one that climbs it have nearly identical plan
// views and completely different profiles.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";

const ARM_COLOR = {
  free: "#737f96",
  room: "#fbbf24",
  "room+traj": "#22d3ee",
};
const ARM_ORDER = ["free", "room", "room+traj"];

// Headline metrics for the comparison table. `lower` marks which direction is
// an improvement; `null` means the metric is descriptive, not a score — root
// speed and pelvis rise say what the body DID, not whether it did well.
const TABLE_ROWS = [
  ["penetration_max", "penetration max", "m", true, "deepest any joint sits inside a solid"],
  ["penetration_frame_frac", "frames penetrating", "", true, "share of frames >2 cm inside geometry"],
  ["path_dev_mean", "path deviation", "m", true, "mean distance from the authored path"],
  ["contact_frame_frac", "frames in contact", "", false, "share of frames with a foot on its support \u2014 the DENOMINATOR of the skate number"],
  ["n_planted_transitions", "planted transitions", "", false, "skate samples; below 10 the skate reads n/a rather than a flattering small number"],
  ["foot_skate_mean", "foot skate", "m/s", true, "planted-foot slide, measured against its OWN support"],
  ["slide_per_root_step", "slide ÷ root step", "", true, "1.0 = the body slid as far as it travelled"],
  ["support_clearance_mean", "swing clearance", "m", false, "how high the feet lift above their support"],
  ["jerk_mean", "jerk", "", true, "smoothness"],
  ["root_speed_mean", "root speed", "m/s", null, "gait speed"],
  ["pelvis_rise", "pelvis rise", "m", null, "net height gained over the clip"],
];

function fmt(v, digits = 3) {
  if (v == null) return "—";
  return Number(v).toFixed(digits);
}

// A metric that is null because there was nothing to measure is NOT a zero.
// `foot_skate_mean` returns null when no foot ever touched its support, and
// printing 0.000 there would award the best possible score to a clip that
// levitated the whole way.
function statCell(stat) {
  if (!stat) return <span className="text-[var(--muted)]">n/a</span>;
  return (
    <span className="font-mono">
      {fmt(stat.mean)}
      {stat.n > 1 && <span className="ml-1 text-[10px] text-[var(--muted)]">±{fmt(stat.sd, 3)}</span>}
    </span>
  );
}

// ───────────────────────────────────────── geometry views

function useProjection(course, pad = 0.35) {
  // Fit the course (obstacles + path + spawn), not the whole 6 m room — a room
  // drawn to scale leaves the interesting 2 m as a smudge in the middle.
  return useMemo(() => {
    if (!course) return null;
    const xs = [], zs = [], ys = [0];
    for (const o of course.objects) {
      xs.push(o.x - o.w / 2, o.x + o.w / 2);
      zs.push(o.z - o.d / 2, o.z + o.d / 2);
      ys.push(o.y + o.h / 2);
    }
    for (const p of course.path) { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); }
    xs.push(course.spawn.x); zs.push(course.spawn.z);
    return {
      x0: Math.min(...xs) - pad, x1: Math.max(...xs) + pad,
      z0: Math.min(...zs) - pad, z1: Math.max(...zs) + pad,
      y0: 0, y1: Math.max(...ys) + pad,
    };
  }, [course, pad]);
}

function PlanView({ course, clips, width = 340, height = 300 }) {
  const b = useProjection(course);
  if (!b) return null;
  const m = { l: 8, r: 8, t: 8, b: 8 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const s = Math.min(iw / (b.x1 - b.x0), ih / (b.z1 - b.z0));
  // +Z is forward and drawn UP the page; +X right. Matches the spawn arrow.
  const X = (x) => m.l + iw / 2 + (x - (b.x0 + b.x1) / 2) * s;
  const Z = (z) => m.t + ih / 2 - (z - (b.z0 + b.z1) / 2) * s;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-lg bg-ink">
      <rect x={0} y={0} width={width} height={height} fill="none" />
      {course.objects.map((o) => (
        <g key={o.id}>
          <rect x={X(o.x - o.w / 2)} y={Z(o.z + o.d / 2)}
            width={o.w * s} height={o.d * s}
            fill="rgba(255,255,255,0.06)" stroke="var(--hairline-strong)" strokeWidth="1" />
          <text x={X(o.x)} y={Z(o.z) + 3} textAnchor="middle" fontSize="8"
            fill="var(--muted)" className="font-mono">{(o.y + o.h / 2).toFixed(2)}</text>
        </g>
      ))}
      {/* authored path */}
      <polyline points={course.path.map((p) => `${X(p[0])},${Z(p[2])}`).join(" ")}
        fill="none" stroke="var(--signal)" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.75" />
      {/* achieved root paths */}
      {clips.map((c, i) => (
        <polyline key={i} points={c.root_xz.map(([x, z]) => `${X(x)},${Z(z)}`).join(" ")}
          fill="none" stroke={ARM_COLOR[c.arm]} strokeWidth="1" opacity="0.55" />
      ))}
      {/* spawn */}
      <circle cx={X(course.spawn.x)} cy={Z(course.spawn.z)} r="3.5" fill="var(--signal)" />
      <line x1={X(course.spawn.x)} y1={Z(course.spawn.z)}
        x2={X(course.spawn.x)} y2={Z(course.spawn.z) - 12}
        stroke="var(--signal)" strokeWidth="1.5" markerEnd="url(#ar)" />
      <defs>
        <marker id="ar" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto">
          <path d="M0,0 L5,2.5 L0,5 z" fill="var(--signal)" />
        </marker>
      </defs>
    </svg>
  );
}

function ElevationView({ course, clips, width = 340, height = 220 }) {
  const b = useProjection(course);
  if (!b) return null;
  const m = { l: 26, r: 8, t: 8, b: 18 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  // Height is exaggerated relative to depth — a 0.15 m riser against a 3 m walk
  // is invisible to scale, and the riser is the whole question.
  const sz = iw / (b.z1 - b.z0);
  const sy = ih / (b.y1 - b.y0);
  const Zp = (z) => m.l + (z - b.z0) * sz;
  const Yp = (y) => m.t + ih - (y - b.y0) * sy;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-lg bg-ink">
      <line x1={m.l} y1={Yp(0)} x2={width - m.r} y2={Yp(0)} stroke="var(--hairline-strong)" strokeWidth="1" />
      <text x={4} y={Yp(0) + 3} fontSize="8" fill="var(--muted)" className="font-mono">0</text>
      <text x={4} y={Yp(b.y1) + 8} fontSize="8" fill="var(--muted)" className="font-mono">{b.y1.toFixed(1)}m</text>
      {course.objects.map((o) => (
        <rect key={o.id} x={Zp(o.z - o.d / 2)} y={Yp(o.y + o.h / 2)}
          width={o.d * sz} height={o.h * sy}
          fill="rgba(255,255,255,0.07)" stroke="var(--hairline-strong)" strokeWidth="1" />
      ))}
      <polyline points={course.path.map((p) => `${Zp(p[2])},${Yp(p[1])}`).join(" ")}
        fill="none" stroke="var(--signal)" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.75" />
      {clips.map((c, i) => (
        <polyline key={i}
          points={c.root_xz.map(([, z], k) => `${Zp(z)},${Yp(c.root_y[k])}`).join(" ")}
          fill="none" stroke={ARM_COLOR[c.arm]} strokeWidth="1" opacity="0.55" />
      ))}
    </svg>
  );
}

// ───────────────────────────────────────── success readout

function SuccessCard({ row }) {
  const s = row.success;
  const color = ARM_COLOR[row.arm];
  let headline, detail;
  if (s.kind === "mount") {
    headline = `${Math.round(s.mounted_rate * 100)}% mounted`;
    detail = `${fmt(s.frames_supported_on?.mean, 0)} frames on the slab`;
  } else if (s.kind === "climb") {
    headline = `${fmt(s.treads_climbed?.mean, 1)} / ${4} treads`;
    detail = `top reached in ${Math.round(s.top_reached_rate * 100)}% of clips`;
  } else {
    headline = `${fmt(s.turn_left_deg?.mean, 0)}° left`;
    detail = `${Math.round(s.exit_reached_rate * 100)}% reached the exit · ` +
      `${Math.round((s.inside_corridor_frac?.mean ?? 0) * 100)}% in-corridor`;
  }
  return (
    <div className="rounded-md border border-[var(--hairline)] p-3">
      <div className="mb-1 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ background: color }} />
        <span className="font-mono text-[11px] text-slate-200">{row.arm}</span>
        <span className="text-[10px] text-[var(--muted)]">n={row.n}</span>
      </div>
      <div className="font-mono text-[15px]" style={{ color }}>{headline}</div>
      <div className="text-[10px] text-[var(--muted)]">{detail}</div>
    </div>
  );
}

// ───────────────────────────────────────── main

export default function CourseLab({ clusterMode }) {
  const [cat, setCat] = useState(null);
  const [active, setActive] = useState("slab");
  const [study, setStudy] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);
  const [cfg, setCfg] = useState({ num_steps: 800, guidance: 6.5, room_guidance: 0.75, seeds: "0,1" });

  useEffect(() => { api.courses().then(setCat).catch((e) => setError(e.message)); }, []);

  const refresh = useCallback(() => {
    api.courseStudy().then(setStudy).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 20000);   // cheap: local .npy only, no SSH
    return () => clearInterval(t);
  }, [refresh]);

  const course = useMemo(
    () => cat?.courses.find((c) => c.key === active) || null, [cat, active]);
  const clips = useMemo(
    () => (study?.clips || []).filter((c) => c.course === active), [study, active]);
  const rows = useMemo(
    () => (study?.table || []).filter((r) => r.course === active), [study, active]);

  async function launch() {
    if (!presets?.model_preset) {
      setError("pick a checkpoint first — without its run config the wrong network gets built");
      return;
    }
    setBusy(true); setError(null);
    try {
      const res = await api.submitCourseStudy({
        checkpoint,
        model_preset: presets.model_preset,
        train_preset: presets.train_preset,
        courses: [active],
        seeds: cfg.seeds.split(",").map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n)),
        num_steps: Number(cfg.num_steps),
        guidance: Number(cfg.guidance),
        room_guidance: Number(cfg.room_guidance),
      });
      setError(`queued ${res.n} cells (${res.n * 3} clips)`);
      refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (!cat) {
    return <section className="surface p-5"><p className="text-[13px] text-[var(--muted)]">
      {error || "loading course catalogue…"}</p></section>;
  }

  const speed = course?.speed;

  return (
    <div className="space-y-4">
      {/* course picker */}
      <section className="surface p-5">
        <div className="mb-3 flex items-center justify-between">
          <span className="label">05 · euclidean obstacle courses</span>
          <span className="text-[10px] text-[var(--muted)]">
            {study?.n_clips ?? 0} clips scored
            {study?.num_steps ? ` · ${study.num_steps} ODE steps` : ""}
            {study?.other_configs?.length
              ? ` · ${study.other_configs.map((c) => `${c.n_clips}@${c.num_steps}`).join(", ")} excluded`
              : ""}
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {cat.courses.map((c) => (
            <button key={c.key} onClick={() => setActive(c.key)}
              className={`rounded-md border px-3 py-2 text-left transition ${
                active === c.key
                  ? "border-[var(--signal)] bg-[var(--signal-dim)]"
                  : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"}`}>
              <div className="font-mono text-[12px] text-slate-200">{c.label}</div>
              <div className="text-[10px] text-[var(--muted)]">{c.title}</div>
            </button>
          ))}
        </div>
        {course && (
          <p className="mt-3 text-[12px] text-[var(--muted)]">
            {course.blurb} — {course.num_frames} frames,
            path {speed.length_m} m in {speed.duration_s} s ={" "}
            <span style={{ color: speed.ok ? "var(--signal)" : "#f87171" }}>
              {speed.speed_mps} m/s
            </span>{" "}
            (natural gait ≈ 0.5 m/s; a faster request manufactures foot-skate).
          </p>
        )}
      </section>

      {/* geometry + arms */}
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="surface p-4">
          <div className="label mb-2">plan · x–z</div>
          <PlanView course={course} clips={clips} />
        </section>
        <section className="surface p-4">
          <div className="label mb-2">elevation · z–y <span className="text-[10px] normal-case text-[var(--muted)]">(height exaggerated)</span></div>
          <ElevationView course={course} clips={clips} />
          <div className="mt-2 flex flex-wrap gap-3">
            {ARM_ORDER.map((a) => (
              <span key={a} className="flex items-center gap-1.5 text-[10px] text-[var(--muted)]">
                <span className="h-2 w-2 rounded-full" style={{ background: ARM_COLOR[a] }} />{a}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-[10px] text-[var(--muted)]">
              <span className="h-[2px] w-4" style={{ background: "var(--signal)" }} />authored path
            </span>
          </div>
        </section>
      </div>

      {/* run */}
      {clusterMode && (
        <section className="surface space-y-3 p-5">
          <div className="label">run the study</div>
          <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
          <div className="grid grid-cols-4 gap-3">
            {[["num_steps", "ODE steps"], ["guidance", "ω (text)"],
              ["room_guidance", "ω (room)"], ["seeds", "seeds"]].map(([k, label]) => (
              <label key={k} className="block">
                <span className="label mb-1 block">{label}</span>
                <input className="field-input" value={cfg[k]}
                  onChange={(e) => setCfg({ ...cfg, [k]: e.target.value })} />
              </label>
            ))}
          </div>
          <p className="text-[11px] text-[var(--muted)]">
            Queues one cell per (arm × seed) for <span className="font-mono">{course?.label}</span> —
            all three arms in one sbatch, same seed and batch size, so position i of
            each arm draws the same noise and the arms are a matched pair.
          </p>
          <button onClick={launch} disabled={busy || !checkpoint}
            className="rounded-md border border-[var(--signal)] bg-[var(--signal-dim)] px-4 py-2 text-[12px] disabled:opacity-40">
            {busy ? "queueing…" : `queue ${course?.label} study`}
          </button>
          {error && <p className="text-[11px] text-[var(--amber)]">{error}</p>}
        </section>
      )}

      {/* results */}
      {rows.length > 0 && (
        <>
          <section className="surface p-5">
            <div className="label mb-3">task success · {course?.title}</div>
            <div className="grid gap-3 sm:grid-cols-3">
              {rows.map((r) => <SuccessCard key={r.arm} row={r} />)}
            </div>
          </section>

          <section className="surface p-5">
            <div className="label mb-3">metrics · mean ± sd over clips</div>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="text-left text-[10px] uppercase tracking-wider text-[var(--muted)]">
                    <th className="py-1.5 pr-3">metric</th>
                    {rows.map((r) => (
                      <th key={r.arm} className="py-1.5 pr-3" style={{ color: ARM_COLOR[r.arm] }}>
                        {r.arm}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {TABLE_ROWS.map(([key, label, unit, lower, hint]) => (
                    <tr key={key} className="border-t border-[var(--hairline)]">
                      <td className="py-1.5 pr-3">
                        <span className="text-slate-300">{label}</span>
                        {unit && <span className="ml-1 text-[10px] text-[var(--muted)]">{unit}</span>}
                        <div className="text-[10px] text-[var(--muted)]">{hint}</div>
                      </td>
                      {rows.map((r) => (
                        <td key={r.arm} className="py-1.5 pr-3">{statCell(r.metrics[key])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="surface p-5">
            <div className="label mb-3">clips</div>
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {clips.map((c) => (
                <a key={`${c.job}-${c.clip}`} href={mediaUrl(c.media_url)} target="_blank"
                  rel="noreferrer" className="block rounded-md border border-[var(--hairline)] p-2 hover:border-[var(--hairline-strong)]">
                  <div className="mb-1 flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full" style={{ background: ARM_COLOR[c.arm] }} />
                    <span className="font-mono text-[10px] text-slate-300">{c.arm}</span>
                    <span className="text-[9px] text-[var(--muted)]">seed {c.seed}</span>
                  </div>
                  <img src={mediaUrl(c.media_url)} alt={c.caption} className="w-full rounded" />
                  <div className="mt-1 font-mono text-[9px] text-[var(--muted)]">
                    pen {fmt(c.penetration_max, 2)}m · dev {fmt(c.path_dev_mean, 2)}m
                  </div>
                </a>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
