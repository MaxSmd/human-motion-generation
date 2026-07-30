"use client";

// TRAJECTORY CONTROL — spatial (mask-control) constraints on RMG.
//
// The task from the mask-control line of work (OmniControl / GMD / MaskControl):
// give the model world-space positions for a joint at chosen frames and make it
// hit them while staying on the prompt. Draw a path on the floor plan, pick a
// joint and a keyframe density, sample, and read back how far the delivered
// motion actually landed from each target.
//
// The keyframe list and target points are computed HERE and sent as explicit
// frames+points, so the numbers the backend enforces and the numbers this panel
// scores against are the same arrays — no resampling logic duplicated across the
// wire. (`flow.trajectory` also accepts a raw path, for scripted use.)
//
// Floor plan is a top-down map of the FK world frame: +x → right, +z → up the
// page, y is height and is left free unless "xyz" is picked. Note it plots WORLD
// axes — the body's own facing is not fixed to +z (see TrajectoryScene3D), so the
// path you draw is a world-space route, not a "walk forward N metres" instruction.

import { useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import { loadNpy } from "@/lib/npy";
import { useJobHistory } from "@/lib/useJobHistory";
import { WORKSPACE_CATEGORIES } from "@/lib/classifyJob";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import MediaViewer from "./MediaViewer";
import LineChart from "./LineChart";
import HistoryRail from "./HistoryRail";
import TrajectoryScene3D from "./TrajectoryScene3D";
import TrajectoryEvalPanel from "./TrajectoryEvalPanel";

const ACTIVE = new Set(["queued", "submitted", "pending", "running"]);

// Joints the literature controls. Indices are the 22-joint HumanML3D skeleton.
const CONTROL_JOINTS = [
  { name: "pelvis", idx: 0, hint: "the trajectory task — exact, it IS the root" },
  { name: "L_Wrist", idx: 20, hint: "reach target" },
  { name: "R_Wrist", idx: 21, hint: "reach target" },
  { name: "Head", idx: 15, hint: "duck / height" },
  { name: "L_Foot", idx: 10, hint: "foot placement" },
  { name: "R_Foot", idx: 11, hint: "foot placement" },
];

const MODES = [
  { id: "project", label: "project", hint: "exact — the root absorbs the correction each ODE step" },
  { id: "guide", label: "guide", hint: "gradient through FK only — the diffusion baselines' mechanism" },
  { id: "hybrid", label: "hybrid", hint: "project, then guide whatever residual is left" },
];

const BLENDS = [
  { id: "interp", label: "interp", hint: "warp the whole root path through the targets — smooth" },
  { id: "contact", label: "contact", hint: "as interp, but move the root while the feet are AIRBORNE — halves foot-skate" },
  { id: "local", label: "local", hint: "cosine falloff near each keyframe — local edit" },
  { id: "none", label: "none", hint: "write only at keyframes — exact but kinks the path" },
];

const PRESETS = {
  line: [[0, 0], [0, 3]],
  L: [[0, 0], [0, 2], [2, 2]],
  S: [[0, 0], [1.2, 1], [-1.2, 2], [0, 3]],
  circle: Array.from({ length: 13 }, (_, i) => {
    const a = (i / 12) * Math.PI * 2;
    return [1.5 * Math.sin(a), 1.5 - 1.5 * Math.cos(a)];
  }),
};

// ── path maths (mirrors flow.trajectory.resample_path) ──────────────────────

function resamplePath(pts, n) {
  if (pts.length === 0) return [];
  if (pts.length === 1 || n <= 1) return Array.from({ length: n }, () => pts[0]);
  const seg = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    total += d;
    seg.push(total);
  }
  if (total < 1e-9) return Array.from({ length: n }, () => pts[0]);
  const out = [];
  for (let k = 0; k < n; k++) {
    const want = (total * k) / (n - 1);
    let i = seg.findIndex((c) => c >= want);
    if (i < 0) i = seg.length - 1;
    const lo = i === 0 ? 0 : seg[i - 1];
    const u = seg[i] - lo < 1e-9 ? 0 : (want - lo) / (seg[i] - lo);
    out.push([
      pts[i][0] * (1 - u) + pts[i + 1][0] * u,
      pts[i][1] * (1 - u) + pts[i + 1][1] * u,
    ]);
  }
  return out;
}

/** Keyframe indices + world targets for the drawn path. */
function buildTargets(pathPts, numFrames, every) {
  if (pathPts.length < 1) return { frames: [], points: [] };
  const dense = resamplePath(pathPts, numFrames);
  const frames = [];
  for (let f = 0; f < numFrames; f += Math.max(1, every)) frames.push(f);
  if (frames[frames.length - 1] !== numFrames - 1) frames.push(numFrames - 1);
  return {
    frames,
    points: frames.map((f) => [dense[f][0], 0, dense[f][1]]),
    dense,
  };
}

function errorsAgainst(joints, jointIdx, frames, points, axes) {
  if (!joints) return null;
  const [T, J] = joints.shape;
  const useY = axes === "xyz";
  const errs = [];
  for (let k = 0; k < frames.length; k++) {
    const f = frames[k];
    if (f >= T) continue;
    const o = (f * J + jointIdx) * 3;
    const dx = joints.data[o] - points[k][0];
    const dy = useY ? joints.data[o + 1] - points[k][1] : 0;
    const dz = joints.data[o + 2] - points[k][2];
    errs.push({ frame: f, err: Math.hypot(dx, dy, dz) });
  }
  if (!errs.length) return null;
  const vals = errs.map((e) => e.err);
  const over = (t) => vals.filter((v) => v > t).length / vals.length;
  return {
    perFrame: errs,
    avg: vals.reduce((a, b) => a + b, 0) / vals.length,
    max: Math.max(...vals),
    loc50: over(0.5),
    loc20: over(0.2),
    n: vals.length,
  };
}

/** Pelvis (or any joint) path as [[x, z], …] for the overlay. */
function jointPath(joints, jointIdx) {
  if (!joints) return [];
  const [T, J] = joints.shape;
  const out = [];
  for (let f = 0; f < T; f++) {
    const o = (f * J + jointIdx) * 3;
    out.push([joints.data[o], joints.data[o + 2]]);
  }
  return out;
}

// ── floor-plan editor ───────────────────────────────────────────────────────

function PathCanvas({ pts, setPts, dense, achieved, targetsAt, extent, size = 460 }) {
  const ref = useRef(null);
  const [drag, setDrag] = useState(null);
  // A drag ends with mouseup, and the browser fires `click` on the SVG straight
  // after — which would drop a spurious waypoint every time one is moved. Latch
  // that a drag actually moved something and swallow the click that follows.
  const moved = useRef(false);

  const toPx = (p) => [
    ((p[0] + extent) / (2 * extent)) * size,
    size - ((p[1] + extent) / (2 * extent)) * size,   // +z up the page
  ];
  const toWorld = (px, py) => [
    (px / size) * 2 * extent - extent,
    ((size - py) / size) * 2 * extent - extent,
  ];

  function evPos(e) {
    const r = ref.current.getBoundingClientRect();
    return toWorld(
      ((e.clientX - r.left) / r.width) * size,
      ((e.clientY - r.top) / r.height) * size
    );
  }

  const grid = [];
  for (let m = -Math.floor(extent); m <= Math.floor(extent); m++) {
    const [gx] = toPx([m, 0]);
    const [, gy] = toPx([0, m]);
    const major = m === 0;
    grid.push(
      <g key={`g${m}`} stroke={major ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.055)"}>
        <line x1={gx} y1={0} x2={gx} y2={size} />
        <line x1={0} y1={gy} x2={size} y2={gy} />
      </g>
    );
  }

  const poly = (list) => list.map(toPx).map((p) => p.join(",")).join(" ");

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${size} ${size}`}
      className="w-full touch-none rounded-xl border border-[var(--hairline)] bg-black/40"
      onClick={(e) => {
        if (drag !== null || moved.current) {
          moved.current = false;
          return;
        }
        setPts([...pts, evPos(e)]);
      }}
      onMouseMove={(e) => {
        if (drag === null) return;
        moved.current = true;
        const next = pts.slice();
        next[drag] = evPos(e);
        setPts(next);
      }}
      onMouseUp={() => setDrag(null)}
      onMouseLeave={() => { setDrag(null); moved.current = false; }}
    >
      {grid}
      {/* achieved path first, so targets draw on top */}
      {achieved.length > 1 && (
        <polyline points={poly(achieved)} fill="none" stroke="var(--amber)" strokeWidth="2" opacity="0.9" />
      )}
      {dense.length > 1 && (
        <polyline points={poly(dense)} fill="none" stroke="var(--signal)" strokeWidth="2"
                  strokeDasharray="6 4" opacity="0.8" />
      )}
      {targetsAt.map((p, i) => {
        const [x, y] = toPx([p[0], p[2]]);
        return <circle key={`t${i}`} cx={x} cy={y} r="3" fill="var(--signal)" opacity="0.85" />;
      })}
      {pts.map((p, i) => {
        const [x, y] = toPx(p);
        return (
          <g key={`w${i}`}>
            <circle
              cx={x} cy={y} r="8" fill="rgba(34,211,238,0.22)" stroke="var(--signal)"
              className="cursor-grab"
              onMouseDown={(e) => { e.stopPropagation(); setDrag(i); }}
              onClick={(e) => {
                e.stopPropagation();
                if (e.shiftKey) setPts(pts.filter((_, k) => k !== i));
              }}
            />
            <text x={x} y={y + 3.5} textAnchor="middle" fontSize="9"
                  fill="#04212a" style={{ pointerEvents: "none", fontWeight: 700 }}>
              {i + 1}
            </text>
          </g>
        );
      })}
      <text x="8" y={size - 8} fontSize="10" fill="var(--muted)" fontFamily="var(--font-mono)">
        {`±${extent} m · click to add · drag to move · shift-click to delete`}
      </text>
    </svg>
  );
}

// ── the tab ─────────────────────────────────────────────────────────────────

export default function TrajectoryTab({ clusterMode, checkpoints = [] }) {
  const [text, setText] = useState("a person walks forward");
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);

  const [pts, setPts] = useState(PRESETS.L);
  const [jointIdx, setJointIdx] = useState(0);
  const [axes, setAxes] = useState("xz");
  const [every, setEvery] = useState(10);
  const [mode, setMode] = useState("project");
  const [blend, setBlend] = useState("interp");
  const [blendFrames, setBlendFrames] = useState(10);
  const [guidanceWeight, setGuidanceWeight] = useState(1.0);
  const [facePath, setFacePath] = useState(true);
  const [retime, setRetime] = useState(true);
  const [extent, setExtent] = useState(4);

  const [numFrames, setNumFrames] = useState(120);
  const [numSteps, setNumSteps] = useState(50);
  const [guidance, setGuidance] = useState(6.5);
  const [seed, setSeed] = useState(0);

  const [busy, setBusy] = useState(false);
  const [busyState, setBusyState] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);   // { joints, mediaUrl, frames, points, jointIdx, axes }
  const [showRail, setShowRail] = useState(true);
  const [loadingJob, setLoadingJob] = useState(null);

  const { jobs } = useJobHistory(WORKSPACE_CATEGORIES.trajectory);

  const targets = useMemo(
    () => buildTargets(pts, numFrames, every),
    [pts, numFrames, every]
  );

  // Score against the SIGNAL THAT WAS SENT, not the current editor state — the
  // panel must not silently re-grade an old clip after the path is edited.
  const scored = useMemo(() => {
    if (!result?.joints) return null;
    return errorsAgainst(result.joints, result.jointIdx, result.frames, result.points, result.axes);
  }, [result]);

  const achieved = useMemo(
    () => (result?.joints ? jointPath(result.joints, result.jointIdx) : []),
    [result]
  );

  const trajectoryPayload = () => ({
    mode,
    blend,
    blend_frames: Number(blendFrames),
    guidance_weight: Number(guidanceWeight),
    retime: retime && jointIdx === 0,
    face_path: facePath && jointIdx === 0,
    face_strength: 1.0,
    constraints: [
      {
        joint: CONTROL_JOINTS.find((j) => j.idx === jointIdx).name,
        axes,
        frames: targets.frames,
        points: targets.points,
      },
    ],
  });

  async function runCluster(trajectory) {
    // The checkpoint's dims come from its run config. Submitting without them
    // silently falls back to dit_base, which mismatches a dit_mid checkpoint and
    // dies on the GPU with a size-mismatch after ~30 s of queue + startup. Fail
    // here instead, where it costs nothing and says why.
    if (!presets?.model_preset) {
      throw new Error(
        "checkpoint config not loaded yet — its model/train preset is unknown, " +
        "and guessing would build the wrong network for these weights. " +
        "Re-pick the run, or wait for the preset line under the picker."
      );
    }
    let job = await api.submitViz({
      mode: "prompt",
      checkpoint,
      prompts: text,
      guidance: Number(guidance),
      num_steps: Number(numSteps),
      num_frames: Number(numFrames),
      seed: Number(seed),
      model_preset: presets?.model_preset,
      train_preset: presets?.train_preset,
      trajectory,
    });
    setBusyState(job.state);
    while (ACTIVE.has(job.state)) {
      await new Promise((r) => setTimeout(r, 3000));
      job = await api.job(job.id);
      setBusyState(job.state);
    }
    if (job.state !== "done") throw new Error(job.error || `job ${job.state}`);
    const out = (job.outputs || []).find((o) => o.npy_url);
    if (!out) throw new Error("job finished but no joints .npy was pulled");
    return { joints: await loadNpy(mediaUrl(out.npy_url)), media: mediaUrl(out.media_url) };
  }

  async function runLocal(trajectory) {
    const body = {
      text, guidance: Number(guidance), num_steps: Number(numSteps),
      num_frames: Number(numFrames), seed: Number(seed), trajectory,
    };
    if (checkpoint) body.checkpoint = checkpoint;
    const res = await api.generate(body);
    return {
      joints: await loadNpy(mediaUrl(res.joints_npy_url)),
      media: mediaUrl(res.media_url),
    };
  }

  async function run() {
    if (!targets.frames.length) {
      setError("draw a path first — click on the floor plan to lay down waypoints");
      return;
    }
    setBusy(true);
    setError(null);
    const trajectory = trajectoryPayload();
    try {
      const { joints, media } = clusterMode ? await runCluster(trajectory) : await runLocal(trajectory);
      setResult({
        joints, mediaUrl: media,
        frames: targets.frames, points: targets.points,
        jointIdx, axes, dense: targets.dense,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      setBusyState(null);
    }
  }

  // ── load a past run back into the tab ──────────────────────────────────────
  // Restores BOTH halves: the control signal that was sent (so the 3D view can
  // score the clip against its own targets, and so the run can be tweaked and
  // re-sent) and the motion that came back. The floor-plan waypoints are rebuilt
  // from the control points themselves, which reproduces the target path exactly
  // rather than guessing at the polyline that was originally drawn.
  async function openJob(job) {
    const p = job?.params || {};
    const t = p.trajectory;
    setError(null);
    setLoadingJob(job.id);
    try {
      if (p.prompts) setText(String(p.prompts).split("|")[0].trim());
      if (p.checkpoint) setCheckpoint(p.checkpoint);
      if (p.num_frames) setNumFrames(p.num_frames);
      if (p.num_steps) setNumSteps(p.num_steps);
      if (p.guidance != null) setGuidance(p.guidance);
      if (p.seed != null) setSeed(p.seed);

      let frames = [], points = [], jIdx = 0, ax = "xyz";
      const c = t?.constraints?.[0];
      if (c) {
        setMode(t.mode || "project");
        setBlend(t.blend || "interp");
        if (t.blend_frames != null) setBlendFrames(t.blend_frames);
        if (t.guidance_weight != null) setGuidanceWeight(t.guidance_weight);
        setFacePath(!!t.face_path);
        setRetime(!!t.retime);
        ax = c.axes || "xyz";
        setAxes(ax);
        const named = CONTROL_JOINTS.find((j) => j.name === c.joint);
        jIdx = named ? named.idx : (Number.isInteger(c.joint) ? c.joint : 0);
        setJointIdx(jIdx);
        frames = c.frames || [];
        points = c.points || [];
        if (points.length) setPts(points.map((q) => [q[0], q[2]]));
        // Recover the keyframe spacing so the editor, if re-run, reproduces the
        // same control signal instead of silently switching density.
        if (frames.length > 1) setEvery(Math.max(1, frames[1] - frames[0]));
      }

      const out = (job.outputs || []).find((o) => o.npy_url);
      if (!out) throw new Error(`job ${job.state} — no joints .npy to preview`);
      const joints = await loadNpy(mediaUrl(out.npy_url));
      setResult({
        joints, mediaUrl: mediaUrl(out.media_url),
        frames, points, jointIdx: jIdx, axes: ax, jobId: job.id,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingJob(null);
    }
  }

  const jointName = CONTROL_JOINTS.find((j) => j.idx === jointIdx)?.name;

  const main = (
      <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="surface space-y-4 p-5">
          {/* 3D first: the control points and both paths only make sense together
              in the volume they live in — the floor plan below is the editor. */}
          <div className="flex items-center justify-between gap-3">
            {/* Says which of the two it is: the measured pair (target + the clip
                that was actually produced against it) or a preview of a signal
                that has not been sampled yet. They must not be confused. */}
            <div className="label">
              {result
                ? `3D · measured run · ${result.frames.length} control points`
                : `3D · target preview · ${targets.frames.length} control points`}
            </div>
            {result?.jobId && (
              <span className="font-mono text-[10px] text-[var(--muted)]">
                loaded {result.jobId.slice(0, 8)}
              </span>
            )}
          </div>
          <TrajectoryScene3D
            joints={result?.joints || null}
            jointIdx={result ? result.jointIdx : jointIdx}
            frames={result ? result.frames : targets.frames}
            points={result ? result.points : targets.points}
            axes={result ? result.axes : axes}
            fps={20}
            height={420}
          />

          <div className="flex items-center justify-between gap-3 border-t border-[var(--hairline)] pt-4">
            {/* World axes, not body axes. "+Z forward" describes HumanML3D's
                first-frame canonicalisation, not the clips shown here — measured
                headings scatter across the circle — so the map is labelled by
                the world axes it actually plots. */}
            <div className="label">floor plan · top-down · +x right, +z up</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.keys(PRESETS).map((k) => (
                <button key={k} onClick={() => setPts(PRESETS[k])}
                        className="btn-ghost !px-2.5 !py-1 text-[11px]">{k}</button>
              ))}
              <button onClick={() => setPts([])} className="btn-ghost !px-2.5 !py-1 text-[11px]">
                clear
              </button>
            </div>
          </div>

          <PathCanvas
            pts={pts} setPts={setPts}
            dense={targets.dense || []}
            achieved={achieved}
            targetsAt={targets.points}
            extent={extent}
          />

          <div className="flex flex-wrap items-center gap-4 text-[12px] text-[var(--muted)]">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-6" style={{ background: "var(--signal)" }} />
              target path ({targets.frames.length} keyframes)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-6" style={{ background: "var(--amber)" }} />
              achieved {jointName}
            </span>
            <label className="ml-auto flex items-center gap-2">
              <span className="label">extent</span>
              <input type="range" min="2" max="10" step="1" value={extent}
                     onChange={(e) => setExtent(Number(e.target.value))} />
              <span className="font-mono text-[11px]">{extent} m</span>
            </label>
          </div>

          {scored && (
            <div className="rounded-xl border border-[var(--hairline)] bg-black/25 p-4">
              <div className="label mb-3">control fidelity · {scored.n} keyframes</div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="avg err" value={`${scored.avg.toFixed(3)} m`} good={scored.avg < 0.05} />
                <Stat label="max err" value={`${scored.max.toFixed(3)} m`} good={scored.max < 0.1} />
                <Stat label="loc err @50cm" value={scored.loc50.toFixed(3)} good={scored.loc50 === 0} />
                <Stat label="loc err @20cm" value={scored.loc20.toFixed(3)} good={scored.loc20 === 0} />
              </div>
              <div className="mt-4">
                <LineChart
                  series={[{
                    label: "error (m)",
                    color: "#fbbf24",
                    points: scored.perFrame.map((p) => [p.frame, p.err]),
                  }]}
                  height={130}
                  xLabel="frame"
                  yLabel="m"
                />
              </div>
            </div>
          )}

          {result?.mediaUrl && (
            <MediaViewer url={result.mediaUrl} caption={text} />
          )}
        </div>

        <div className="surface space-y-4 p-5">
          <div>
            <div className="label mb-1.5">prompt</div>
            <input className="field-input" value={text} onChange={(e) => setText(e.target.value)} />
          </div>

          <div>
            <div className="label mb-1.5">checkpoint</div>
            {clusterMode ? (
              <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
            ) : (
              <select className="field-input" value={checkpoint}
                      onChange={(e) => setCheckpoint(e.target.value)}>
                <option value="">server default</option>
                {checkpoints.map((c) => (
                  <option key={c.path} value={c.path}>{c.run} · {c.name}</option>
                ))}
              </select>
            )}
          </div>

          <div>
            <div className="label mb-1.5">controlled joint</div>
            <select className="field-input" value={jointIdx}
                    onChange={(e) => setJointIdx(Number(e.target.value))}>
              {CONTROL_JOINTS.map((j) => (
                <option key={j.idx} value={j.idx}>{j.name} — {j.hint}</option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="axes">
              <select className="field-input" value={axes} onChange={(e) => setAxes(e.target.value)}>
                <option value="xz">xz — floor path, height free</option>
                <option value="xyz">xyz — pin height too</option>
              </select>
            </Field>
            <Field label={`keyframe every ${every} fr`}>
              <input type="range" min="1" max="40" step="1" value={every}
                     onChange={(e) => setEvery(Number(e.target.value))} className="w-full" />
            </Field>
          </div>

          <Field label="enforcement">
            <div className="grid grid-cols-3 gap-1.5">
              {MODES.map((m) => (
                <Chip key={m.id} on={mode === m.id} onClick={() => setMode(m.id)}
                      label={m.label} title={m.hint} />
              ))}
            </div>
          </Field>

          {mode !== "guide" && (
            <Field label="root correction spread">
              <div className="grid grid-cols-2 gap-1.5">
                {BLENDS.map((b) => (
                  <Chip key={b.id} on={blend === b.id} onClick={() => setBlend(b.id)}
                        label={b.label} title={b.hint} />
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-[var(--muted)]">
                {BLENDS.find((b) => b.id === blend).hint}
              </p>
            </Field>
          )}

          {jointIdx === 0 && (
            <label className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-[var(--hairline)] p-3">
              <input type="checkbox" checked={retime} className="mt-0.5"
                     onChange={(e) => setRetime(e.target.checked)} />
              <span>
                <span className="text-[13px] font-semibold text-slate-200">follow path on the model's own timing</span>
                <span className="mt-1 block text-[11px] leading-relaxed text-[var(--muted)]">
                  You choose WHERE, the model chooses WHEN. A constant-speed path demands
                  full walking speed from frame 0, but a generated clip stands still for
                  ~1.9 s first — forcing the schedule drags a standing body. Off = each
                  point is pinned to its frame (what the benchmark scores).
                </span>
              </span>
            </label>
          )}

          {jointIdx === 0 && (
            <label className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-[var(--hairline)] p-3">
              <input type="checkbox" checked={facePath} className="mt-0.5"
                     onChange={(e) => setFacePath(e.target.checked)} />
              <span>
                <span className="text-[13px] font-semibold text-slate-200">turn to face the path</span>
                <span className="mt-1 block text-[11px] leading-relaxed text-[var(--muted)]">
                  Moving the root's position without its orientation drags the body
                  sideways or backwards along the route — measured 125° between facing
                  and travel, against 13° for a free walk. Heading is exactly projectable
                  too, so this yaws the root to follow the path. Position stays exact.
                </span>
              </span>
            </label>
          )}

          {mode !== "guide" && blend === "local" && (
            <Num label="falloff (frames)" value={blendFrames} onChange={setBlendFrames} min={1} max={60} />
          )}
          {mode !== "project" && (
            <Num label="guidance weight" value={guidanceWeight} onChange={setGuidanceWeight}
                 min={0} max={5} step={0.1} />
          )}

          <div className="grid grid-cols-2 gap-3">
            <Num label="frames" value={numFrames} onChange={setNumFrames} min={20} max={196} />
            <Num label="ODE steps" value={numSteps} onChange={setNumSteps} min={10} max={400} />
            <Num label="ω (CFG)" value={guidance} onChange={setGuidance} min={1} max={12} step={0.5} />
            <Num label="seed" value={seed} onChange={setSeed} min={0} max={9999} />
          </div>

          <button onClick={run}
                  disabled={busy || (clusterMode && (!checkpoint || !presets?.model_preset))}
                  className="btn-signal w-full">
            {busy ? busyState || "sampling…" : "Sample with trajectory"}
          </button>
          {clusterMode && !checkpoint && (
            <p className="text-[11px] text-[var(--muted)]">
              pick a run + checkpoint — sampling runs as a cluster viz job
            </p>
          )}
          {clusterMode && checkpoint && !presets?.model_preset && (
            <p className="text-[11px] text-[var(--amber)]">
              waiting for the run config — the checkpoint's model preset is needed
              so the network is built at the right size
            </p>
          )}
          {clusterMode && presets?.model_preset && (
            <p className="text-[11px] text-[var(--muted)]">
              building <span className="font-mono">{presets.model_preset}</span> /{" "}
              <span className="font-mono">{presets.train_preset}</span> to match this checkpoint
            </p>
          )}
          {error && <p className="text-[12px] text-[var(--amber)]">{error}</p>}
        </div>
      </div>
  );

  return (
    <div className="space-y-5">
      <header className="surface p-5">
        <div className="label">04 · trajectory control</div>
        <h2 className="display mt-1.5 text-lg font-bold text-white">
          Spatial constraints — drive a joint through world-space targets
        </h2>
        <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-[var(--muted)]">
          Draw a path, pick how many keyframes to pin it at, and sample. RMG stores the
          pelvis as the Euclidean factor of its manifold, so a root target is met by an
          exact projection rather than by gradient guidance — the question this panel is
          built to answer is what that exactness costs the rest of the motion.
        </p>
        <div className="mt-3 flex items-center gap-3">
          <button onClick={() => setShowRail((s) => !s)}
                  className="rounded-md border border-[var(--hairline)] px-3 py-1.5 text-[12px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]">
            {showRail ? "hide" : "show"} history · {jobs.length}
          </button>
          {loadingJob && (
            <span className="font-mono text-[11px] text-[var(--muted)]">loading run…</span>
          )}
        </div>
      </header>

      {showRail ? (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px] 2xl:grid-cols-[minmax(0,1fr)_380px]">
          {main}
          <div className="xl:sticky xl:top-6 xl:self-start">
            <HistoryRail
              jobs={jobs}
              categories={WORKSPACE_CATEGORIES.trajectory}
              onOpen={openJob}
              onRestore={openJob}
              emptyHint="Trajectory-controlled generations collect here."
            />
          </div>
        </div>
      ) : (
        main
      )}

      {/* Benchmark results last: the panel above is for authoring one clip, this
          is how the same control scores across the whole test split. */}
      <TrajectoryEvalPanel />
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <div className="label mb-1.5">{label}</div>
      {children}
    </div>
  );
}

function Num({ label, value, onChange, min, max, step = 1 }) {
  return (
    <Field label={label}>
      <input type="number" className="field-input" value={value} min={min} max={max} step={step}
             onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

function Chip({ on, onClick, label, title }) {
  return (
    <button onClick={onClick} title={title}
            className={`rounded-md border px-2 py-1.5 font-mono text-[11px] transition ${
              on ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                 : "border-[var(--hairline)] text-slate-300 hover:border-[var(--hairline-strong)]"
            }`}>
      {label}
    </button>
  );
}

function Stat({ label, value, good }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`mt-1 font-mono text-[15px] font-bold ${good ? "text-[var(--signal)]" : "text-white"}`}>
        {value}
      </div>
    </div>
  );
}
