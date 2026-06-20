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

// ── bend-gizmo geometry ───────────────────────────────────────────────────
// A constraint is the bend angle at a joint, drawn in the real plane of its two
// bones: 0° = straight (continuation of the incoming bone), sweeping toward the
// outgoing bone. Basis (u, v): u = incoming direction, v ⟂ u toward the child.
const _sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const _dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const _len = (a) => Math.hypot(a[0], a[1], a[2]);
const _norm = (a) => { const l = _len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function bendBasis(parentPos, jointPos, childPos) {
  const u = _norm(_sub(jointPos, parentPos));          // incoming bone dir
  const out = _norm(_sub(childPos, jointPos));         // outgoing bone dir
  let w = _sub(out, [u[0] * _dot(out, u), u[1] * _dot(out, u), u[2] * _dot(out, u)]);
  if (_len(w) < 1e-4) w = Math.abs(u[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]; // straight: any ⟂
  w = _norm(_sub(w, [u[0] * _dot(w, u), u[1] * _dot(w, u), u[2] * _dot(w, u)]));
  return [u, w];
}
function bendArc(center, u, v, fromDeg, toDeg, radius = 0.18, segs = 36) {
  const a0 = (fromDeg * Math.PI) / 180;
  const a1 = (toDeg * Math.PI) / 180;
  const pts = [];
  for (let i = 0; i <= segs; i++) {
    const t = a0 + ((a1 - a0) * i) / segs;
    const c = Math.cos(t), s = Math.sin(t);
    pts.push([
      center[0] + radius * (c * u[0] + s * v[0]),
      center[1] + radius * (c * u[1] + s * v[1]),
      center[2] + radius * (c * u[2] + s * v[2]),
    ]);
  }
  return pts;
}
function bendRadial(center, u, v, deg, radius = 0.18) {
  return [center, bendArc(center, u, v, deg, deg, radius, 1)[1]];
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
  parents = [],
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
          <ConstraintGizmos joints={joints} frame={frame} pins={pins} ranges={ranges} jointNames={jointNames} parents={parents} T={T} />

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

// Pin → swept bend arc (0 → bend) + tick at the target. Range → min↔max wedge.
// Both drawn in the joint's real bone plane (incoming bone = 0°, straight).
function ConstraintGizmos({ joints, frame, pins, ranges, jointNames, parents = [], T }) {
  const { shape, data } = joints;
  const [, J] = shape;
  const posIdx = (j) => {
    if (j == null || j < 0) return null;
    const o = (frame * J + j) * 3;
    return [data[o], data[o + 1], data[o + 2]];
  };
  const idxOf = (name) => jointNames.findIndex((n) => n === name);
  const childOf = (j) => parents.findIndex((p) => p === j);
  // Basis for the bend at the named joint; null if parent/child unavailable.
  const basisFor = (name) => {
    const j = idxOf(name);
    if (j < 0) return null;
    const c = posIdx(j), par = posIdx(parents[j]), ch = posIdx(childOf(j));
    if (!c || !par || !ch) return null;
    const [u, v] = bendBasis(par, c, ch);
    return { c, u, v };
  };
  const active = (con) => {
    const s = Number(con.frame_start) || 0;
    const e = con.frame_end === "" || con.frame_end == null ? T : Number(con.frame_end);
    return frame >= s && frame < e;
  };
  return (
    <group>
      {pins.map((p) => {
        const b = basisFor(p.joint);
        if (!b || !active(p)) return null;
        const a = Number(p.bend_deg) || 0;
        return (
          <group key={`pin-${p.id}`}>
            <Line points={bendArc(b.c, b.u, b.v, 0, a)} color="#fbbf24" lineWidth={2.5} />
            <Line points={bendRadial(b.c, b.u, b.v, a)} color="#fbbf24" lineWidth={1.5} />
          </group>
        );
      })}
      {ranges.map((r) => {
        const b = basisFor(r.joint);
        if (!b || !active(r)) return null;
        const mn = Number(r.bend_min) || 0;
        const mx = Number(r.bend_max) || 0;
        return (
          <group key={`range-${r.id}`}>
            <Line points={bendArc(b.c, b.u, b.v, mn, mx)} color="#34d399" lineWidth={3} />
            <Line points={bendRadial(b.c, b.u, b.v, mn)} color="#34d399" lineWidth={1.5} />
            <Line points={bendRadial(b.c, b.u, b.v, mx)} color="#34d399" lineWidth={1.5} />
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
