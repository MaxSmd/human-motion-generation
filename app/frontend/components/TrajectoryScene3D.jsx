"use client";

// 3D viewport for spatial (mask-control) trajectory constraints.
//
// Shows, in one orbitable box, the three things you need to judge a controlled
// generation against each other:
//
//   • the DISCRETE control points — the (frame, joint) slots that were actually
//     pinned, drawn as spheres with a drop-line to the floor so their height
//     reads correctly and you can see which are mid-air;
//   • the TARGET path they interpolate;
//   • the ACHIEVED path the controlled joint actually took, plus the skeleton
//     animating along it.
//
// An error is only legible next to the thing it is an error *from*, so a miss is
// drawn as an explicit segment from control point to achieved position — the
// picture and the avg-err number describe the same quantity.
//
// World frame is HumanML3D's: Y up, floor at y=0 — the same frame the joints
// .npy is already in, so nothing is transformed here.
//
// A note on "forward": the docs describe +Z as forward because HumanML3D
// canonicalises a clip's FIRST FRAME to face +Z. That is a statement about the
// dataset's preprocessing, not about the clips this viewer is handed — measured
// on real output, ground-truth renders come back with first-frame headings all
// over the circle (−137°, +76°, +90°, …) and this model's samples sit near −83°.
// So the camera must not assume a facing: it is derived per clip from the body's
// own hips/shoulders, which is what `_canonicalize_first_frame` uses upstream.
// X and Z are labelled as the world axes they are, and the character's actual
// facing is drawn in the scene rather than asserted in a caption.

import { useMemo, useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, GizmoHelper, GizmoViewport, Line, OrbitControls } from "@react-three/drei";

const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];

const SIGNAL = "#22d3ee";   // target
const AMBER = "#fbbf24";    // achieved
const GREEN = "#34d399";    // satisfied control point
const ROSE = "#fb7185";     // missed control point

// Upstream's FACE_JOINT_INDX = (r_hip, l_hip, r_shoulder, l_shoulder).
const FACE = [2, 1, 17, 16];

/** Body-frame forward at `frame`, by the same construction the dataset uses. */
function facingAt(joints, frame = 0) {
  if (!joints) return null;
  const p = jointsAt(joints, frame);
  const a = [0, 1, 2].map((k) => p[FACE[0]][k] - p[FACE[1]][k] + p[FACE[2]][k] - p[FACE[3]][k]);
  const n = Math.hypot(a[0], a[1], a[2]);
  if (n < 1e-6) return null;
  const across = a.map((v) => v / n);
  // forward = up × across, with up = +Y
  const f = [across[2], 0, -across[0]];
  const fn = Math.hypot(f[0], f[2]);
  return fn < 1e-6 ? null : [f[0] / fn, 0, f[2] / fn];
}

function jointsAt(joints, frame) {
  const [T, J] = joints.shape;
  const f = Math.min(Math.max(frame, 0), T - 1);
  const out = [];
  for (let j = 0; j < J; j++) {
    const o = (f * J + j) * 3;
    out.push([joints.data[o], joints.data[o + 1], joints.data[o + 2]]);
  }
  return out;
}

function Skeleton({ joints, frame }) {
  const pts = useMemo(() => (joints ? jointsAt(joints, frame) : null), [joints, frame]);
  if (!pts) return null;
  return (
    <group>
      {CHAINS.map((chain, i) => (
        <Line key={i} points={chain.map((j) => pts[j])} color="#dbe4f0" lineWidth={2.4} />
      ))}
      {pts.map((p, j) => (
        <mesh key={j} position={p}>
          <sphereGeometry args={[j === 0 ? 0.045 : 0.026, 10, 10]} />
          <meshStandardMaterial color={j === 0 ? AMBER : "#93a3b8"} />
        </mesh>
      ))}
    </group>
  );
}

function Playhead({ playing, fps, count, frame, setFrame }) {
  const acc = useRef(0);
  useFrame((_, dt) => {
    if (!playing || count < 2) return;
    acc.current += dt;
    const step = 1 / fps;
    if (acc.current >= step) {
      const n = Math.floor(acc.current / step);
      acc.current -= n * step;
      setFrame((f) => (f + n) % count);
    }
  });
  return null;
}

/** The room box: a wireframe volume sized to contain everything on show. */
function Box({ extent, height }) {
  const e = extent;
  return (
    <group>
      <Grid args={[e * 2, e * 2]} cellSize={0.5} cellThickness={0.5}
            sectionSize={1} sectionThickness={1} cellColor="#1e2a3a" sectionColor="#2c3d52"
            fadeDistance={e * 3.5} infiniteGrid={false} position={[0, 0, 0]} />
      <mesh position={[0, height / 2, 0]}>
        <boxGeometry args={[e * 2, height, e * 2]} />
        <meshBasicMaterial color="#22d3ee" wireframe transparent opacity={0.07} />
      </mesh>
    </group>
  );
}

export default function TrajectoryScene3D({
  joints,             // { shape:[T,J,3], data } — achieved motion (may be null)
  jointIdx = 0,       // the controlled joint
  frames = [],        // controlled frame indices
  points = [],        // world targets, one per controlled frame
  axes = "xyz",
  fps = 20,
  height = 400,
}) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const T = joints?.shape?.[0] ?? 0;

  useEffect(() => { setFrame(0); }, [joints]);

  // Achieved path of the controlled joint.
  const achieved = useMemo(() => {
    if (!joints) return [];
    const [t, J] = joints.shape;
    const out = [];
    for (let f = 0; f < t; f++) {
      const o = (f * J + jointIdx) * 3;
      out.push([joints.data[o], joints.data[o + 1], joints.data[o + 2]]);
    }
    return out;
  }, [joints, jointIdx]);

  // Where the joint actually was at each controlled frame, and by how much it
  // missed — measured on the controlled axes only, matching the reported error.
  const marks = useMemo(() => {
    const useY = axes === "xyz";
    return frames.map((f, k) => {
      const tgt = points[k];
      if (!joints || f >= T) return { target: tgt, got: null, err: null };
      const [, J] = joints.shape;
      const o = (f * J + jointIdx) * 3;
      const got = [joints.data[o], joints.data[o + 1], joints.data[o + 2]];
      const dx = got[0] - tgt[0];
      const dy = useY ? got[1] - tgt[1] : 0;
      const dz = got[2] - tgt[2];
      return { target: tgt, got, err: Math.hypot(dx, dy, dz), frame: f };
    });
  }, [frames, points, joints, jointIdx, T, axes]);

  // Fit the box to everything on show so nothing sits outside the volume.
  const { extent, boxH } = useMemo(() => {
    let m = 1.5, h = 2.0;
    const scan = (p) => {
      m = Math.max(m, Math.abs(p[0]), Math.abs(p[2]));
      h = Math.max(h, p[1]);
    };
    points.forEach(scan);
    achieved.forEach(scan);
    return { extent: Math.ceil(m * 1.15), boxH: Math.ceil(h * 1.2 * 2) / 2 };
  }, [points, achieved]);

  // A target path drawn only through the pinned points would imply the joint was
  // constrained between them, which it was not — so when the pins are sparse the
  // line is dashed to read as "interpolated, not enforced".
  const sparse = frames.length > 1 && T > 0 && frames.length < T * 0.5;

  // Frame the camera from the clip itself: look at the middle of the action from
  // three-quarters behind the body's own initial facing. Assuming a fixed world
  // heading is exactly what made this read wrong — the motion does not come with
  // one. Falls back to a fixed octant view when there is no clip to measure.
  const facing = useMemo(() => facingAt(joints, 0), [joints]);
  const { camPos, target } = useMemo(() => {
    const pts = achieved.length ? achieved : points;
    let cx = 0, cz = 0;
    if (pts.length) {
      for (const p of pts) { cx += p[0]; cz += p[2]; }
      cx /= pts.length; cz /= pts.length;
    }
    const tgt = [cx, 0.9, cz];
    const D = Math.max(extent * 1.5, 3.2);
    if (!facing) return { camPos: [D, D * 0.8, D * 1.15], target: tgt };
    // Swing 40° off the "directly behind" axis for a three-quarter view.
    const a = (40 * Math.PI) / 180;
    const bx = -facing[0], bz = -facing[2];
    const dx = bx * Math.cos(a) + bz * Math.sin(a);
    const dz = -bx * Math.sin(a) + bz * Math.cos(a);
    return { camPos: [cx + dx * D, D * 0.62, cz + dz * D], target: tgt };
  }, [facing, achieved, points, extent]);

  // The Canvas only reads `camera` on mount, so re-key it when the framing
  // changes — loading a different run should re-frame, not keep a stale angle.
  const camKey = camPos.map((v) => v.toFixed(2)).join(",");

  // Where the body faces at frame 0, drawn on the floor as a short arrow.
  const facingArrow = useMemo(() => {
    if (!facing || !joints) return null;
    const p = jointsAt(joints, 0)[0];
    const L = 0.7;
    return [[p[0], 0.01, p[2]], [p[0] + facing[0] * L, 0.01, p[2] + facing[2] * L]];
  }, [facing, joints]);

  const headingDeg = facing
    ? Math.round((Math.atan2(facing[0], facing[2]) * 180) / Math.PI)
    : null;

  return (
    <div className="space-y-2">
      <div className="overflow-hidden rounded-xl border border-[var(--hairline)] bg-black/50"
           style={{ height }}>
        <Canvas key={camKey} camera={{ position: camPos, fov: 45 }}>
          <ambientLight intensity={0.75} />
          <directionalLight position={[4, 8, 5]} intensity={1.1} />
          <Box extent={extent} height={boxH} />

          {/* Which way the body actually faces at frame 0 — shown, not assumed. */}
          {facingArrow && (
            <Line points={facingArrow} color="#93a3b8" lineWidth={2.5} transparent opacity={0.9} />
          )}

          {points.length > 1 && (
            <Line points={points} color={SIGNAL} lineWidth={2}
                  dashed={sparse} dashSize={0.14} gapSize={0.1} transparent opacity={0.85} />
          )}
          {achieved.length > 1 && (
            <Line points={achieved} color={AMBER} lineWidth={2.4} />
          )}

          {marks.map((m, k) => {
            const bad = m.err != null && m.err > 0.05;
            const c = m.err == null ? SIGNAL : bad ? ROSE : GREEN;
            return (
              <group key={k}>
                {/* drop-line to the floor: makes the point's height readable */}
                <Line points={[[m.target[0], 0, m.target[2]], m.target]}
                      color={c} lineWidth={1} transparent opacity={0.32} />
                <mesh position={m.target}>
                  <sphereGeometry args={[0.055, 14, 14]} />
                  <meshStandardMaterial color={c} emissive={c} emissiveIntensity={0.45} />
                </mesh>
                {/* the miss, drawn as the segment it is */}
                {bad && m.got && <Line points={[m.target, m.got]} color={ROSE} lineWidth={2} />}
              </group>
            );
          })}

          <Skeleton joints={joints} frame={frame} />
          <Playhead playing={playing} fps={fps} count={T} frame={frame} setFrame={setFrame} />
          {/* maxPolarAngle keeps the camera above the floor: orbiting underneath
              flips the whole scene visually and is the other half of why the
              orientation read wrong. Axis colours match MotionPlayer's gizmo so
              the two 3D views of the same world agree. */}
          <OrbitControls makeDefault enableDamping dampingFactor={0.1}
                         target={target} maxPolarAngle={Math.PI / 2.05} />
          <GizmoHelper alignment="bottom-right" margin={[58, 58]}>
            <GizmoViewport axisColors={["#ef6f6f", "#7bd88f", "#22d3ee"]} labelColor="#0a0f18" />
          </GizmoHelper>
        </Canvas>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button onClick={() => setPlaying((p) => !p)} disabled={!T}
                className="btn-ghost !px-3 !py-1 text-[11px]">
          {playing ? "pause" : "play"}
        </button>
        <input type="range" min={0} max={Math.max(0, T - 1)} value={frame} disabled={!T}
               onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }}
               className="min-w-[140px] flex-1" />
        <span className="font-mono text-[11px] text-[var(--muted)]">
          {T ? `${frame + 1}/${T}` : "no clip"}
        </span>
      </div>

      <div className="flex flex-wrap gap-4 text-[11px] text-[var(--muted)]">
        <Key color={SIGNAL} label={sparse ? "target path (interpolated)" : "target path"} />
        <Key color={GREEN} label="control point — met" dot />
        <Key color={ROSE} label="control point — missed >5 cm" dot />
        <Key color={AMBER} label="achieved path" />
        {headingDeg != null && (
          <Key color="#93a3b8" label={`body faces ${headingDeg}° at frame 0 (0° = +Z)`} />
        )}
      </div>
    </div>
  );
}

function Key({ color, label, dot }) {
  return (
    <span className="flex items-center gap-1.5">
      {dot ? (
        <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
      ) : (
        <span className="inline-block h-0.5 w-5" style={{ background: color }} />
      )}
      {label}
    </span>
  );
}
