"use client";

// In-browser 3D motion player. Renders the generated skeleton (from a joints
// .npy) animating inside the room/scene, with free orbit camera + playback —
// replacing the serverside GIF for the Room tab so clips can be viewed from any
// angle alongside the constraints they were sampled under.
//
// Joints are already in the room's world frame (the backend applies spawn
// placement before dumping the .npy), and that frame is HumanML3D's: X right,
// Y up, Z forward, floor at y=0 — identical to RoomEditor. So scene + skeleton
// draw together with no extra transform.

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Edges, Grid, GizmoHelper, GizmoViewport, Line, OrbitControls } from "@react-three/drei";
import { loadNpy } from "@/lib/npy";
import { clipMetrics, frameJointsOf, frameViolations } from "@/lib/sceneMetrics";

// HumanML3D kinematic chains (same as the renderer / FK).
const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];

export default function MotionPlayer({ jointsUrl, scene, fps = 20 }) {
  const [joints, setJoints] = useState(null); // { shape:[T,J,3], data }
  const [error, setError] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [showAnalysis, setShowAnalysis] = useState(true);

  useEffect(() => {
    let alive = true;
    setJoints(null); setError(null); setFrame(0); setPlaying(true);
    loadNpy(jointsUrl)
      .then((d) => alive && setJoints(d))
      .catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, [jointsUrl]);

  const T = joints?.shape?.[0] ?? 0;

  // Constraint-violation analysis (only when a scene is present).
  const clip = useMemo(() => (joints && scene ? clipMetrics(joints, scene) : null), [joints, scene]);
  const fm = useMemo(
    () => (joints && scene ? frameViolations(frameJointsOf(joints, frame), scene) : null),
    [joints, scene, frame]
  );

  if (error) return <div className="grid h-[440px] place-items-center text-[13px] text-[var(--amber)]">{error}</div>;
  if (!joints) return <div className="grid h-[440px] place-items-center text-[13px] text-[var(--muted)]">loading motion…</div>;

  return (
    <div className="relative">
      <div className="relative overflow-hidden rounded-lg border border-[var(--hairline)]" style={{ height: 440 }}>
        {showAnalysis && fm && <ConstraintHUD fm={fm} clip={clip} />}
        {clip && (
          <button
            onClick={() => setShowAnalysis((s) => !s)}
            className={`absolute right-3 top-3 z-10 rounded-md border px-2.5 py-1 font-mono text-[11px] uppercase tracking-widest transition ${
              showAnalysis
                ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                : "border-[var(--hairline)] bg-black/50 text-[var(--muted)] hover:text-slate-200"
            }`}
          >
            analysis {showAnalysis ? "on" : "off"}
          </button>
        )}
        <Canvas camera={{ position: [4.5, 3.5, 5.5], fov: 48 }} style={{ height: 440, background: "transparent" }}>
          <hemisphereLight intensity={0.7} groundColor="#0a0f18" />
          <directionalLight position={[5, 8, 4]} intensity={1.0} />

          {scene && <StaticScene scene={scene} />}
          <Grid args={[scene?.room?.width ?? 6, scene?.room?.depth ?? 6]} cellSize={0.5} cellThickness={0.6}
            cellColor="#1d2738" sectionSize={1} sectionThickness={1} sectionColor="#2b3a52" fadeDistance={30}
            position={[0, 0.001, 0]} />

          <SkeletonFrame joints={joints} frame={frame} violating={showAnalysis ? fm?.violating : null} />
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
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}
          className="field-input w-auto py-1 text-[11px]">
          {[0.25, 0.5, 1, 2].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
      </div>

      {showAnalysis && clip && <ViolationTimeline clip={clip} frame={frame} onSeek={(f) => { setPlaying(false); setFrame(f); }} />}
    </div>
  );
}

// Live readout of the current frame's constraint violations + clip summary.
function ConstraintHUD({ fm, clip }) {
  const worst = fm.worst;
  const ok = worst <= 1e-3;
  const col = ok ? "var(--signal)" : worst > 0.1 ? "#f87171" : "var(--amber)";
  return (
    <div className="pointer-events-none absolute left-3 top-3 z-10 w-[210px] rounded-md border border-[var(--hairline)] bg-black/55 p-2.5 backdrop-blur">
      <div className="flex items-baseline justify-between">
        <span className="label">constraints</span>
        <span className="font-mono text-[12px] font-bold" style={{ color: col }}>
          {ok ? "✓ clear" : `${worst.toFixed(2)} m`}
        </span>
      </div>
      <dl className="mt-1.5 space-y-0.5 text-[11px]">
        <Row k="room" v={fm.roomOut > 1e-3 ? `${fm.roomOut.toFixed(2)} m out` : "inside"} bad={fm.roomOut > 1e-3} />
        <Row k="obstacle" v={fm.pen > 1e-3 ? `${fm.pen.toFixed(2)} m → ${fm.penLabel}` : "clear"} bad={fm.pen > 1e-3} />
        <Row k="joints" v={`${fm.nViol}/22`} bad={fm.nViol > 0} />
      </dl>
      {clip && (
        <div className="mt-1.5 border-t border-[var(--hairline)] pt-1.5 text-[10px] text-[var(--muted)]">
          peak <span className="font-mono text-slate-300">{clip.peak.toFixed(2)} m</span> ·
          {" "}<span className="font-mono text-slate-300">{Math.round(clip.pctViolating * 100)}%</span> of frames
        </div>
      )}
    </div>
  );
}

function Row({ k, v, bad }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-[var(--muted)]">{k}</dt>
      <dd className="font-mono" style={{ color: bad ? "#f87171" : "var(--text)" }}>{v}</dd>
    </div>
  );
}

// Per-frame violation strip; click to scrub. Bar height/colour ∝ violation depth.
function ViolationTimeline({ clip, frame, onSeek }) {
  const { series, peak, T } = clip;
  const norm = peak > 1e-6 ? peak : 1;
  return (
    <div className="mt-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="label">violation timeline</span>
        <span className="text-[10px] text-[var(--muted)]">{peak <= 1e-3 ? "no violations" : `peak ${peak.toFixed(2)} m`}</span>
      </div>
      <svg viewBox={`0 0 ${T} 12`} preserveAspectRatio="none" className="h-6 w-full cursor-pointer rounded bg-black/30"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(Math.max(0, Math.min(T - 1, Math.floor(((e.clientX - r.left) / r.width) * T))));
        }}
      >
        {Array.from(series, (v, f) =>
          v > 1e-3 ? (
            <rect key={f} x={f} y={12 - Math.max(1.5, (v / norm) * 12)} width={1.05}
              height={Math.max(1.5, (v / norm) * 12)} fill={v > 0.1 ? "#f87171" : "var(--amber)"} />
          ) : null
        )}
        <line x1={frame + 0.5} x2={frame + 0.5} y1={0} y2={12} stroke="var(--signal)" strokeWidth={0.6} />
      </svg>
    </div>
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

function SkeletonFrame({ joints, frame, violating }) {
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
        const bad = violating?.has(j);
        return (
          <mesh key={j} position={p}>
            <sphereGeometry args={[bad ? 0.04 : 0.028, 12, 10]} />
            <meshStandardMaterial
              color={bad ? "#f87171" : "#9be8f7"}
              emissive={bad ? "#ef4444" : "#22d3ee"}
              emissiveIntensity={bad ? 0.7 : 0.4}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// Static (non-editable) room + obstacles — mirrors RoomEditor's look.
function StaticScene({ scene }) {
  const { width: w, depth: d, height: h } = scene.room;
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial color="#0c121d" roughness={1} />
      </mesh>
      <mesh position={[0, h / 2, 0]}>
        <boxGeometry args={[w, h, d]} />
        <meshBasicMaterial transparent opacity={0} />
        <Edges color="#2b3a52" />
      </mesh>
      {(scene.objects || []).map((o) => <Obstacle key={o.id} o={o} />)}
    </group>
  );
}

function Obstacle({ o }) {
  const geom =
    o.kind === "sphere" ? <sphereGeometry args={[o.radius, 28, 20]} /> :
    o.kind === "cylinder" ? <cylinderGeometry args={[o.radius, o.radius, o.height, 28]} /> :
    <boxGeometry args={[o.w, o.h, o.d]} />;
  return (
    <mesh position={[o.x, o.y, o.z]} rotation={[0, (o.rotation || 0) * Math.PI / 180, 0]}>
      {geom}
      <meshStandardMaterial color="#7c8aa5" transparent opacity={0.28} roughness={0.6} />
      <Edges color="#46566f" />
    </mesh>
  );
}
