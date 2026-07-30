"use client";

// In-browser motion player. Loads (T, 22, 3) joint arrays and animates the
// HumanML3D skeleton on the visitor's GPU, so the server only ever ships a few
// tens of kilobytes per clip.
//
// Accepts either a single `url` or a list of `clips`, which are placed side by
// side in one canvas and played on a shared clock — that shared clock is the
// point of the compare view, since it is the only way to see the two motions
// diverge frame for frame.

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import { TOUCH } from "three";
import { loadNpy } from "@/lib/npy";

// Touch devices: one finger scrolls the page (the model is playing anyway),
// two fingers orbit/zoom. Without this a full-width canvas swallows the vertical
// swipe and the page gets stuck under your thumb.
function useCoarsePointer() {
  const [coarse, setCoarse] = useState(false);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia("(pointer: coarse)");
    const on = () => setCoarse(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return coarse;
}

// HumanML3D kinematic chains (root → limbs), same as the renderer and the FK.
const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];

// Joints belonging to a named limb, for highlighting a constrained chain.
export const LIMBS = {
  left_leg: [1, 4, 7, 10],
  right_leg: [2, 5, 8, 11],
  left_arm: [16, 18, 20],
  right_arm: [17, 19, 21],
};

export default function SkeletonPlayer({
  url,
  clips,
  fps = 20,
  height = 420,
  autoRotate = false,
  highlight = null, // key of LIMBS to paint in the accent colour
  accent = "#fbbf24",
  controls = true,
  follow = false, // re-centre on the root each frame (traveling hero clips)
  align = false, // rotate each clip about the vertical so they face one way
  spread = 1.7, // metres between side-by-side figures
}) {
  const coarse = useCoarsePointer();
  const list = useMemo(
    () => (clips?.length ? clips : url ? [{ url, color: "#22d3ee" }] : []),
    [clips, url],
  );
  const [loaded, setLoaded] = useState(null);
  const [error, setError] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoaded(null);
    setError(null);
    setFrame(0);
    setPlaying(true);
    Promise.all(list.map((c) => loadNpy(c.url)))
      .then((ds) => {
        if (!alive) return;
        setLoaded(
          ds.map((d, i) => {
            const yaw = align ? facingYaw(d) : 0;
            return { ...d, ...list[i], yaw, offset: groundOffset(d, yaw), travel: xzExtent(d) };
          }),
        );
      })
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [list, align]);

  const T = loaded ? Math.max(...loaded.map((c) => c.shape[0])) : 0;

  if (error)
    return (
      <div className="grid place-items-center text-[13px] text-[var(--warn)]" style={{ height }}>
        {error}
      </div>
    );
  if (!loaded)
    return (
      <div className="viewport grid place-items-center" style={{ height }}>
        <span className="animate-pulse-soft text-[12px] text-[var(--muted)]">loading motion…</span>
      </div>
    );

  const n = loaded.length;
  const span = (n - 1) * spread;
  // How far the widest clip travels across the floor. A clip that walks four
  // metres needs the camera further back than one that stands and waves, or the
  // figure leaves the frame halfway through.
  const travel = Math.max(...loaded.map((c) => c.travel ?? 0));

  return (
    <div>
      <div className="viewport" style={{ height, touchAction: coarse ? "pan-y" : undefined }}>
        <Canvas camera={{ position: [2.3, 1.9, 3.7], fov: 46 }} dpr={[1, coarse ? 1.5 : 1.8]} style={{ background: "transparent" }}>
          <hemisphereLight intensity={0.8} groundColor="#0a0f18" />
          <directionalLight position={[5, 8, 4]} intensity={1.05} />
          <Grid
            args={[10, 10]}
            cellSize={0.5}
            cellThickness={0.55}
            cellColor="#1b2434"
            sectionSize={1}
            sectionThickness={1}
            sectionColor="#28374e"
            fadeDistance={30}
          />
          {loaded.map((c, i) => (
            <Figure
              key={i}
              clip={c}
              frame={frame}
              x={-span / 2 + i * spread}
              highlight={highlight}
              accent={accent}
              follow={follow}
            />
          ))}
          <Driver playing={playing} fps={fps} count={T} frame={frame} setFrame={setFrame} />
          <Framer span={span} travel={follow ? 0 : travel} />
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.12}
            target={[0, 0.9, 0]}
            autoRotate={autoRotate}
            autoRotateSpeed={0.55}
            maxPolarAngle={Math.PI / 2.05}
            minDistance={2.2}
            maxDistance={14}
            // one finger scrolls the page on touch, two fingers orbit
            touches={coarse ? { ONE: undefined, TWO: TOUCH.DOLLY_ROTATE } : undefined}
          />
        </Canvas>

        {n > 1 && (
          <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center gap-8 p-3">
            {loaded.map((c, i) => (
              <span key={i} className="flex items-center gap-1.5 font-mono text-[10.5px]" style={{ color: c.color }}>
                <span className="dot" style={{ background: c.color, color: c.color }} />
                {c.label}
              </span>
            ))}
          </div>
        )}
      </div>

      {controls && (
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="rounded-md border border-[var(--signal)]/45 bg-[var(--signal-dim)] px-3 py-1.5 font-mono text-[11.5px] font-semibold text-[var(--signal)]"
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, T - 1)}
            value={frame}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
            className="flex-1 accent-[var(--signal)]"
            aria-label="frame"
          />
          <span className="w-20 text-right font-mono text-[10.5px] text-[var(--muted)]">
            {frame + 1}/{T} · {(frame / fps).toFixed(1)}s
          </span>
        </div>
      )}
    </div>
  );
}

// Fit the camera to the stage: a lone standing figure is framed close, extra
// figures widen it, and a clip that covers ground pushes it back further so the
// whole trajectory stays inside the viewport.
function Framer({ span, travel = 0 }) {
  const { camera } = useThree();
  useEffect(() => {
    const d = 3.7 + span * 1.05 + travel * 0.5;
    camera.position.set(d * 0.62, 1.9 + travel * 0.12, d);
    camera.updateProjectionMatrix();
  }, [span, travel, camera]);
  return null;
}

// Rotate (x, z) about the vertical by yaw.
function rot(x, z, yaw) {
  if (!yaw) return [x, z];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [x * c - z * s, x * s + z * c];
}

// Yaw that turns a clip so it faces +Z: by its travel direction when it covers
// ground, otherwise by the facing implied by the hip line at the first frame.
function facingYaw({ shape, data }) {
  const [T, J] = shape;
  const dx = data[(T - 1) * J * 3] - data[0];
  const dz = data[(T - 1) * J * 3 + 2] - data[2];
  let fx, fz;
  if (Math.hypot(dx, dz) > 0.4) {
    fx = dx;
    fz = dz;
  } else {
    const hx = data[2 * 3] - data[1 * 3]; // R_hip - L_hip, x
    const hz = data[2 * 3 + 2] - data[1 * 3 + 2]; // z
    fx = hz; // forward is perpendicular to the hip line
    fz = -hx;
  }
  return -Math.atan2(fx, fz);
}

// Offset that drops the clip's lowest point onto y = 0 and centres it in x/z on
// the whole-clip mean of the (yaw-rotated) root, so any clip is framed the same.
function groundOffset({ shape, data }, yaw = 0) {
  const [T, J] = shape;
  let minY = Infinity, sx = 0, sz = 0;
  for (let t = 0; t < T; t++) {
    const r = (t * J + 0) * 3;
    const [rx, rz] = rot(data[r], data[r + 2], yaw);
    sx += rx;
    sz += rz;
    for (let j = 0; j < J; j++) {
      const y = data[(t * J + j) * 3 + 1];
      if (y < minY) minY = y;
    }
  }
  return [-sx / T, -minY, -sz / T];
}

// Ground covered by the root over the clip, as the larger of the x and z spans.
function xzExtent({ shape, data }) {
  const [T, J] = shape;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let t = 0; t < T; t++) {
    const r = t * J * 3;
    x0 = Math.min(x0, data[r]); x1 = Math.max(x1, data[r]);
    z0 = Math.min(z0, data[r + 2]); z1 = Math.max(z1, data[r + 2]);
  }
  return Math.max(x1 - x0, z1 - z0);
}

function Figure({ clip, frame, x, highlight, accent, follow = false }) {
  const { shape, data, offset, color = "#22d3ee", yaw = 0 } = clip;
  const [T, J] = shape;
  const f = Math.min(frame, T - 1);
  const [ox, oy, oz] = offset;
  const hi = highlight ? new Set(LIMBS[highlight] ?? []) : null;

  // `follow` re-centres on the (rotated) root every frame, which keeps a clip
  // that covers ground in the middle of the viewport at the cost of hiding the
  // translation. Off everywhere the trajectory itself is the point.
  const rootRot = rot(data[f * J * 3], data[f * J * 3 + 2], yaw);
  const [cx, cz] = follow ? [-rootRot[0], -rootRot[1]] : [ox, oz];

  const pos = (j) => {
    const o = (f * J + j) * 3;
    const [rx, rz] = rot(data[o], data[o + 2], yaw);
    return [rx + cx + x, data[o + 1] + oy, rz + cz];
  };

  const bones = useMemo(() => CHAINS.map((c) => c.map(pos)), [f, data, J, x, follow, yaw]); // eslint-disable-line
  const joints = useMemo(() => Array.from({ length: J }, (_, j) => pos(j)), [f, data, J, x, follow, yaw]); // eslint-disable-line

  return (
    <group>
      {bones.map((pts, ci) => {
        const lit = hi && CHAINS[ci].some((j) => hi.has(j));
        return <Line key={ci} points={pts} color={lit ? accent : color} lineWidth={lit ? 4 : 3} />;
      })}
      {joints.map((p, j) => {
        const lit = hi?.has(j);
        return (
          <mesh key={j} position={p}>
            <sphereGeometry args={[lit ? 0.036 : 0.027, 12, 10]} />
            <meshStandardMaterial
              color={lit ? accent : "#cdeff8"}
              emissive={lit ? accent : color}
              emissiveIntensity={lit ? 0.9 : 0.4}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function Driver({ playing, fps, count, frame, setFrame }) {
  const acc = useRef(frame);
  useEffect(() => {
    if (!playing) acc.current = frame;
  }, [frame, playing]);
  useFrame((_, delta) => {
    if (!playing || count <= 1) return;
    acc.current += delta * fps;
    const i = ((Math.floor(acc.current) % count) + count) % count;
    if (i !== frame) setFrame(i);
  });
  return null;
}
