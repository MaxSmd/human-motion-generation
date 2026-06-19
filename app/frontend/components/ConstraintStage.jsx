"use client";

// Interactive 3D constraint stage for the Studio tab. Unlike MotionPlayer (a
// read-only viewer), this is the *authoring surface*: every joint is a pickable
// target, the selected joint glows, and active constraints draw as gizmos right
// on the body — a pin as a swept angle arc, a hinge as a min↔max wedge. Click a
// joint to select it; the parent panel opens the contextual editor for it.
//
// Joints come from the same (T, J, 3) position .npy the rest of the app uses,
// already in HumanML3D world frame (X right, Y up, Z forward, floor y=0).

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, GizmoHelper, GizmoViewport, Line, OrbitControls } from "@react-three/drei";

// HumanML3D kinematic chains (same as the renderer / MotionPlayer).
const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];

const AXIS_VEC = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
// Orthonormal in-plane basis (u, v) for the disc perpendicular to each axis.
const AXIS_BASIS = {
  x: [[0, 1, 0], [0, 0, 1]],
  y: [[0, 0, 1], [1, 0, 0]],
  z: [[1, 0, 0], [0, 1, 0]],
};

function arcPoints(center, axis, fromDeg, toDeg, radius = 0.16, segs = 36) {
  const [u, v] = AXIS_BASIS[axis] || AXIS_BASIS.z;
  const a0 = (fromDeg * Math.PI) / 180;
  const a1 = (toDeg * Math.PI) / 180;
  const pts = [];
  for (let i = 0; i <= segs; i++) {
    const t = a0 + ((a1 - a0) * i) / segs;
    const c = Math.cos(t);
    const s = Math.sin(t);
    pts.push([
      center[0] + radius * (c * u[0] + s * v[0]),
      center[1] + radius * (c * u[1] + s * v[1]),
      center[2] + radius * (c * u[2] + s * v[2]),
    ]);
  }
  return pts;
}
function radial(center, axis, deg, radius = 0.16) {
  return arcPoints(center, axis, deg, deg, radius, 1).length
    ? [center, arcPoints(center, axis, deg, deg, radius, 1)[1]]
    : [center, center];
}

export default function ConstraintStage({
  joints,
  scene,
  fps = 20,
  pins = [],
  ranges = [],
  selected, // joint index or null
  onSelect,
  jointNames = [],
  onFrameChange,
}) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [hovered, setHovered] = useState(null);
  const T = joints?.shape?.[0] ?? 0;

  useEffect(() => {
    setFrame(0);
    setPlaying(true);
  }, [joints]);
  useEffect(() => {
    onFrameChange?.(frame);
  }, [frame, onFrameChange]);

  // Which joints carry a constraint active on the current frame (for the glow).
  const constrained = useMemo(() => {
    const m = new Map(); // jointIndex -> 'pin' | 'hinge'
    const inWin = (c) => {
      const s = Number(c.frame_start) || 0;
      const e = c.frame_end === "" || c.frame_end == null ? T : Number(c.frame_end);
      return frame >= s && frame < e;
    };
    const idxOf = (name) => jointNames.findIndex((n) => n === name);
    for (const p of pins) if (inWin(p)) m.set(idxOf(p.joint), "pin");
    for (const r of ranges) if (inWin(r)) m.set(idxOf(r.joint), m.get(idxOf(r.joint)) ? "both" : "hinge");
    m.delete(-1);
    return m;
  }, [pins, ranges, frame, T, jointNames]);

  if (!joints) {
    return (
      <div className="grid h-[460px] place-items-center rounded-lg border border-[var(--hairline)] text-[13px] text-[var(--muted)]">
        sample a clip to start authoring
      </div>
    );
  }

  const selName = selected != null ? jointNames[selected] : null;

  return (
    <div>
      <div className="relative overflow-hidden rounded-lg border border-[var(--hairline)]" style={{ height: 460 }}>
        {/* selection chip */}
        <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-md border border-[var(--hairline)] bg-black/55 px-2.5 py-1.5 backdrop-blur">
          <span className="label">selected</span>{" "}
          <span className="font-mono text-[12px] text-[var(--signal)]">{selName || "— click a joint"}</span>
        </div>
        {hovered != null && hovered !== selected && (
          <div className="pointer-events-none absolute right-3 top-3 z-10 rounded-md border border-[var(--hairline)] bg-black/55 px-2.5 py-1 font-mono text-[11px] text-slate-300 backdrop-blur">
            {jointNames[hovered]}
          </div>
        )}

        <Canvas camera={{ position: [4.5, 3.5, 5.5], fov: 48 }} style={{ height: 460, background: "transparent" }}>
          <hemisphereLight intensity={0.7} groundColor="#0a0f18" />
          <directionalLight position={[5, 8, 4]} intensity={1.0} />

          {scene && <StaticScene scene={scene} />}
          <Grid args={[scene?.room?.width ?? 6, scene?.room?.depth ?? 6]} cellSize={0.5} cellThickness={0.6}
            cellColor="#1d2738" sectionSize={1} sectionThickness={1} sectionColor="#2b3a52" fadeDistance={30}
            position={[0, 0.001, 0]} />

          <SkeletonFrame
            joints={joints}
            frame={frame}
            selected={selected}
            hovered={hovered}
            constrained={constrained}
            onSelect={onSelect}
            setHovered={setHovered}
          />
          <ConstraintGizmos joints={joints} frame={frame} pins={pins} ranges={ranges} jointNames={jointNames} T={T} />

          <PlaybackDriver playing={playing} speed={speed} fps={fps} count={T} frame={frame} setFrame={setFrame} />
          <OrbitControls makeDefault enableDamping dampingFactor={0.1} maxPolarAngle={Math.PI / 2.05} />
          <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
            <GizmoViewport axisColors={["#ef6f6f", "#7bd88f", "#22d3ee"]} labelColor="#0a0f18" />
          </GizmoHelper>
        </Canvas>
      </div>

      {/* transport */}
      <div className="mt-3 flex items-center gap-3">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="rounded-md border border-[var(--signal)]/50 bg-[var(--signal-dim)] px-3 py-1.5 text-[12px] font-semibold text-[var(--signal)]"
        >
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
        <input
          type="range" min={0} max={Math.max(0, T - 1)} value={frame}
          onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }}
          className="flex-1 accent-[var(--signal)]"
        />
        <span className="w-16 text-right font-mono text-[11px] text-[var(--muted)]">{frame + 1}/{T}</span>
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="field-input w-auto py-1 text-[11px]">
          {[0.25, 0.5, 1, 2].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
      </div>
    </div>
  );
}

function SkeletonFrame({ joints, frame, selected, hovered, constrained, onSelect, setHovered }) {
  const { shape, data } = joints;
  const [, J] = shape;
  const pos = (j) => {
    const o = (frame * J + j) * 3;
    return [data[o], data[o + 1], data[o + 2]];
  };
  const points = useMemo(() => CHAINS.map((c) => c.map(pos)), [frame, data, J]); // eslint-disable-line
  const all = useMemo(() => Array.from({ length: J }, (_, j) => pos(j)), [frame, data, J]); // eslint-disable-line

  return (
    <group>
      {points.map((pts, ci) => (
        <Line key={ci} points={pts} color="#22d3ee" lineWidth={3} />
      ))}
      {all.map((p, j) => {
        const isSel = j === selected;
        const kind = constrained.get(j);
        const isHover = j === hovered;
        const color = isSel ? "#fbbf24" : kind ? "#a78bfa" : isHover ? "#ffffff" : "#9be8f7";
        const emissive = isSel ? "#f59e0b" : kind ? "#7c3aed" : "#22d3ee";
        const r = isSel ? 0.05 : kind ? 0.042 : isHover ? 0.04 : 0.03;
        return (
          <group key={j}>
            <mesh
              position={p}
              onClick={(e) => { e.stopPropagation(); onSelect?.(j); }}
              onPointerOver={(e) => { e.stopPropagation(); setHovered(j); document.body.style.cursor = "pointer"; }}
              onPointerOut={() => { setHovered(null); document.body.style.cursor = "auto"; }}
            >
              <sphereGeometry args={[r, 14, 12]} />
              <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={isSel ? 0.9 : 0.45} />
            </mesh>
            {isSel && (
              <mesh position={p}>
                <ringGeometry args={[0.07, 0.085, 24]} />
                <meshBasicMaterial color="#fbbf24" transparent opacity={0.8} side={2} />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}

// Pin → swept arc (0 → angle) + a tick at the target. Hinge → min↔max wedge.
function ConstraintGizmos({ joints, frame, pins, ranges, jointNames, T }) {
  const { shape, data } = joints;
  const [, J] = shape;
  const pos = (name) => {
    const j = jointNames.findIndex((n) => n === name);
    if (j < 0) return null;
    const o = (frame * J + j) * 3;
    return [data[o], data[o + 1], data[o + 2]];
  };
  const active = (c) => {
    const s = Number(c.frame_start) || 0;
    const e = c.frame_end === "" || c.frame_end == null ? T : Number(c.frame_end);
    return frame >= s && frame < e;
  };
  return (
    <group>
      {pins.map((p) => {
        const c = pos(p.joint);
        if (!c || !active(p)) return null;
        const a = Number(p.angle_deg) || 0;
        return (
          <group key={`pin-${p.id}`}>
            <Line points={arcPoints(c, p.axis, 0, a, 0.16)} color="#fbbf24" lineWidth={2.5} />
            <Line points={radial(c, p.axis, a, 0.16)} color="#fbbf24" lineWidth={1.5} />
          </group>
        );
      })}
      {ranges.map((r) => {
        const c = pos(r.joint);
        if (!c || !active(r)) return null;
        const mn = Number(r.min_deg) || 0;
        const mx = Number(r.max_deg) || 0;
        return (
          <group key={`hinge-${r.id}`}>
            <Line points={arcPoints(c, r.axis, mn, mx, 0.18)} color="#34d399" lineWidth={3} />
            <Line points={radial(c, r.axis, mn, 0.18)} color="#34d399" lineWidth={1.5} />
            <Line points={radial(c, r.axis, mx, 0.18)} color="#34d399" lineWidth={1.5} />
          </group>
        );
      })}
    </group>
  );
}

function PlaybackDriver({ playing, speed, fps, count, frame, setFrame }) {
  const acc = useRef(frame);
  useEffect(() => { if (!playing) acc.current = frame; }, [frame, playing]);
  useFrame((_, delta) => {
    if (!playing || count <= 1) return;
    acc.current += delta * fps * speed;
    const idx = ((Math.floor(acc.current) % count) + count) % count;
    if (idx !== frame) setFrame(idx);
  });
  return null;
}

function StaticScene({ scene }) {
  const { width: w, depth: d, height: h } = scene.room;
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial color="#0c121d" roughness={1} />
      </mesh>
      {(scene.objects || []).map((o) => (
        <mesh key={o.id} position={[o.x, o.y, o.z]} rotation={[0, (o.rotation || 0) * Math.PI / 180, 0]}>
          {o.kind === "sphere" ? <sphereGeometry args={[o.radius, 24, 18]} /> :
            o.kind === "cylinder" ? <cylinderGeometry args={[o.radius, o.radius, o.height, 24]} /> :
            <boxGeometry args={[o.w, o.h, o.d]} />}
          <meshStandardMaterial color="#7c8aa5" transparent opacity={0.28} roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}
