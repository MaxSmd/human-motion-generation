"use client";

// Scroll-scrubbed 3D view of the generation pipeline.
//
// The camera dollies from stage to stage as the section moves through the
// viewport, rather than orbiting a static diagram. Everything drawn is real:
// the caption becomes one pooled vector, that vector and the diffusion time are
// fused once and injected into all ten blocks as AdaLN-Zero modulation (there is
// no cross-attention anywhere), and the last two stages are the two operations
// that keep the sampler on the manifold.

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, Html } from "@react-three/drei";
import * as THREE from "three";
import { loadNpy } from "@/lib/npy";
import { useScrollProgress, smooth, lerp, phase } from "./motion";

const CYAN = "#22d3ee";
const VIOLET = "#a78bfa";
const GREEN = "#34d399";

// Each stage owns a camera viewpoint; scroll interpolates between them.
const STAGES = [
  {
    id: "text",
    name: "Caption",
    detail:
      "The prompt is encoded by Qwen3-Embedding-0.6B and mean-pooled to a single 1024-dimensional vector, which conditions the whole sequence.",
    cam: [-7.4, 2.0, 6.2],
    look: [-6.6, 0.5, 0],
  },
  {
    id: "fuse",
    name: "Conditioning",
    detail:
      "That vector is concatenated with a sinusoidal embedding of the flow time and passed through an MLP. The result modulates every block through AdaLN-Zero.",
    cam: [-4.6, 3.4, 5.6],
    look: [-4.2, 1.2, 0],
  },
  {
    id: "tokens",
    name: "Motion tokens",
    detail:
      "The state being denoised is a sequence of up to 196 poses, each a 91-dimensional point: three translation coordinates and 22 unit quaternions.",
    cam: [-3.0, 1.6, 6.4],
    look: [-2.6, 0.4, 0],
  },
  {
    id: "dit",
    name: "10 DiT blocks",
    detail:
      "768 wide, 12 heads. Each block is modulated self-attention over time followed by a feed-forward layer, with the conditioning setting the scale and shift at both.",
    cam: [0.4, 2.6, 6.6],
    look: [0.2, 0.6, 0],
  },
  {
    id: "proj",
    name: "Tangent projection",
    detail:
      "The network emits an ambient velocity. Subtracting its radial component leaves a vector in the tangent space of each sphere.",
    cam: [4.6, 1.7, 5.4],
    look: [4.6, 0.4, 0],
  },
  {
    id: "exp",
    name: "Exponential map",
    detail:
      "One Riemannian Euler step follows the geodesic in that direction. The result lies on the manifold, so no quaternion is renormalised.",
    cam: [6.2, 1.3, 4.4],
    look: [6.4, 0.5, 0],
  },
  {
    id: "out",
    name: "Pose sequence",
    detail:
      "After the last step the quaternions are resolved into joint positions by forward kinematics.",
    cam: [8.4, 1.9, 5.0],
    look: [8.2, 0.8, 0],
  },
];

export default function Architecture() {
  const wrap = useRef(null);
  const p = useScrollProgress(wrap);
  const prog = useRef(0);
  prog.current = smooth(phase(p, 0.08, 0.78));

  // Lighter render on phones — the scene animates continuously.
  const [coarse, setCoarse] = useState(false);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia("(pointer: coarse)");
    const on = () => setCoarse(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const [active, setActive] = useState(0);
  const pinned = useRef(null); // a clicked stage overrides the scroll for a while

  useEffect(() => {
    if (pinned.current != null) return;
    setActive(Math.min(STAGES.length - 1, Math.floor(prog.current * STAGES.length * 0.999)));
  }, [p]);

  const pick = (i) => {
    pinned.current = i;
    setActive(i);
    window.clearTimeout(pick._t);
    pick._t = window.setTimeout(() => {
      pinned.current = null;
    }, 4000);
  };

  return (
    <div ref={wrap}>
      <div className="viewport" style={{ height: 460 }}>
        <Canvas camera={{ position: [-7.4, 2, 6.2], fov: 40 }} dpr={[1, coarse ? 1.4 : 1.8]} style={{ background: "transparent" }}>
          <hemisphereLight intensity={0.62} groundColor="#050810" />
          <directionalLight position={[4, 8, 6]} intensity={1.05} />
          <pointLight position={[6, 2, 3]} intensity={18} color={CYAN} distance={9} />
          <Scene prog={prog} active={active} pinned={pinned} />
        </Canvas>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex gap-px p-3">
          {STAGES.map((s, i) => (
            <div
              key={s.id}
              className="h-[2px] flex-1 rounded-full transition-colors duration-300"
              style={{ background: i <= active ? CYAN : "var(--hairline-strong)" }}
            />
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        {STAGES.map((s, i) => (
          <button
            key={s.id}
            onClick={() => pick(i)}
            className="surface p-2.5 text-left transition"
            style={{ borderColor: i === active ? CYAN : undefined, opacity: i === active ? 1 : 0.5 }}
          >
            <div className="label mb-1.5" style={{ color: i === active ? CYAN : undefined }}>
              {String(i + 1).padStart(2, "0")}
            </div>
            <div className="text-[11.5px] font-semibold leading-tight text-[var(--text)]">{s.name}</div>
          </button>
        ))}
      </div>
      <p className="mt-3 min-h-[54px] max-w-3xl text-[12.5px] leading-relaxed text-[var(--muted)]">
        {STAGES[active].detail}
      </p>
    </div>
  );
}

function Scene({ prog, active, pinned }) {
  const target = useRef(new THREE.Vector3(-6.6, 0.5, 0));

  useFrame((state) => {
    // Blend between stage viewpoints. A click pins one stage; otherwise the
    // scroll position drives a continuous dolly down the pipeline.
    let camPos, look;
    if (pinned.current != null) {
      camPos = new THREE.Vector3(...STAGES[pinned.current].cam);
      look = new THREE.Vector3(...STAGES[pinned.current].look);
    } else {
      const f = prog.current * (STAGES.length - 1);
      const i = Math.min(STAGES.length - 2, Math.floor(f));
      const k = smooth(Math.min(1, Math.max(0, f - i)));
      camPos = new THREE.Vector3(...STAGES[i].cam).lerp(new THREE.Vector3(...STAGES[i + 1].cam), k);
      look = new THREE.Vector3(...STAGES[i].look).lerp(new THREE.Vector3(...STAGES[i + 1].look), k);
    }
    // A slow drift keeps the scene alive when the page is still.
    camPos.x += Math.sin(state.clock.elapsedTime * 0.22) * 0.14;
    camPos.y += Math.cos(state.clock.elapsedTime * 0.17) * 0.08;

    state.camera.position.lerp(camPos, 0.07);
    target.current.lerp(look, 0.07);
    state.camera.lookAt(target.current);
  });

  return (
    <group>
      <CaptionPlate />
      <TimeRing />
      <FusionNode />
      <TokenRibbon />
      <BlockStack active={active} />
      <ExplodedBlock active={active} />
      <TangentStage />
      <ManifoldGrid />
      <OutputFigure />
      <Floor />

      {/* In-world labels. Off-screen ones are clipped by the viewport, so as the
          camera settles on a stage only that stage's labels are on show. */}
      <Tag pos={[-7.15, 1.9, 0]} color={VIOLET} sub="Qwen3-Embedding · 1024-d">caption</Tag>
      <Tag pos={[-6.0, 2.5, 0]} color={VIOLET}>diffusion time t</Tag>
      <Tag pos={[-4.3, 2.5, 0]} color={VIOLET} sub="scale + shift per block">AdaLN-Zero</Tag>
      <Tag pos={[-2.7, 1.85, 0]} color={CYAN} sub="prior → data · 196 × 91">motion tokens</Tag>
      <Tag pos={[0.3, 2.0, 0]} color={CYAN} sub="768 wide · 12 heads">10 × DiT block</Tag>
      <Tag pos={[4.6, 1.95, 0]} color={CYAN} sub="ambient velocity → Tₓ𝑀">tangent projection</Tag>
      <Tag pos={[6.4, 1.95, 0]} color={CYAN} sub="one geodesic step">Exp map</Tag>
      <Tag pos={[8.4, 1.5, 0]} color={GREEN} sub="forward kinematics">pose sequence</Tag>
    </group>
  );
}

// A small billboarded label pinned to a point in the scene.
function Tag({ pos, color, sub, children }) {
  return (
    <Html position={pos} center pointerEvents="none" zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          whiteSpace: "nowrap",
          textAlign: "center",
          transform: "translateY(-50%)",
          userSelect: "none",
        }}
      >
        <div
          style={{
            fontSize: 10.5,
            fontWeight: 600,
            color,
            background: "rgba(7,10,17,0.72)",
            border: `1px solid ${color}44`,
            borderRadius: 5,
            padding: "2px 7px",
            backdropFilter: "blur(2px)",
          }}
        >
          {children}
        </div>
        {sub && (
          <div style={{ marginTop: 3, fontSize: 8.5, color: "rgba(219,228,240,0.62)", letterSpacing: "0.02em" }}>
            {sub}
          </div>
        )}
      </div>
    </Html>
  );
}

function Floor() {
  return <gridHelper args={[40, 40, "#132030", "#0c1522"]} position={[0.5, -1.7, 0]} />;
}

// ── 01 the caption, encoded to one pooled vector ──────────────────────────
function CaptionPlate() {
  const cells = useRef();
  useFrame((s) => {
    cells.current?.children.forEach((m, i) => {
      m.material.emissiveIntensity = 0.3 + 0.7 * Math.abs(Math.sin(s.clock.elapsedTime * 1.1 + i * 0.55));
    });
  });
  return (
    <group position={[-7.2, 0.4, 0]}>
      {/* the prompt, as a stack of word plates */}
      {[0, 1, 2].map((i) => (
        <mesh key={i} position={[0, 0.95 - i * 0.3, i * 0.06]}>
          <boxGeometry args={[1.5 - i * 0.22, 0.17, 0.05]} />
          <meshStandardMaterial color="#1a2b3a" emissive={VIOLET} emissiveIntensity={0.16} />
        </mesh>
      ))}
      {/* pooled 1024-d embedding */}
      <group ref={cells} position={[0.05, -0.35, 0]}>
        {Array.from({ length: 12 }, (_, i) => (
          <mesh key={i} position={[-0.66 + i * 0.12, 0, 0]}>
            <boxGeometry args={[0.09, 0.34, 0.09]} />
            <meshStandardMaterial color="#241b3d" emissive={VIOLET} emissiveIntensity={0.5} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

// ── 02 diffusion time, fused with the caption ─────────────────────────────
function TimeRing() {
  const ring = useRef();
  useFrame((s) => {
    if (ring.current) ring.current.rotation.z = s.clock.elapsedTime * 0.5;
  });
  return (
    <group position={[-6.0, 1.75, 0]}>
      <mesh ref={ring}>
        <torusGeometry args={[0.3, 0.028, 10, 40]} />
        <meshStandardMaterial color="#2a2340" emissive={VIOLET} emissiveIntensity={0.75} />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.07, 12, 10]} />
        <meshStandardMaterial color="#fff" emissive={VIOLET} emissiveIntensity={1.4} />
      </mesh>
    </group>
  );
}

// The MLP that fuses caption and time into the modulation signal.
function FusionNode() {
  const core = useRef();
  useFrame((s) => {
    const k = 1 + Math.sin(s.clock.elapsedTime * 2.1) * 0.06;
    core.current?.scale.setScalar(k);
  });
  return (
    <group position={[-4.3, 1.5, 0]}>
      <mesh ref={core}>
        <octahedronGeometry args={[0.36, 0]} />
        <meshStandardMaterial color="#2b2350" emissive={VIOLET} emissiveIntensity={0.8} transparent opacity={0.92} />
      </mesh>
      <Line points={[[-2.7, -1.05, 0], [-0.4, 0, 0]]} color={VIOLET} lineWidth={1.4} transparent opacity={0.5} />
      <Line points={[[-1.6, 0.3, 0], [-0.4, 0.05, 0]]} color={VIOLET} lineWidth={1.4} transparent opacity={0.5} />
    </group>
  );
}

// ── 03 the motion token sequence entering the stack ───────────────────────
function TokenRibbon() {
  const g = useRef();
  const cells = useMemo(
    () => Array.from({ length: 16 * 5 }, (_, k) => [k % 16, Math.floor(k / 16), Math.random()]),
    [],
  );
  useFrame((s) => {
    g.current?.children.forEach((m, k) => {
      const r = cells[k][2];
      m.material.emissiveIntensity = 0.18 + 0.5 * Math.abs(Math.sin(s.clock.elapsedTime * 0.8 + r * 7));
    });
  });
  return (
    <group ref={g} position={[-2.7, 0.35, 0]}>
      {cells.map(([i, j], k) => (
        <mesh key={k} position={[0, 0.55 - j * 0.26, -1.95 + i * 0.26]}>
          <boxGeometry args={[0.08, 0.2, 0.2]} />
          <meshStandardMaterial color="#0e3b47" emissive={CYAN} emissiveIntensity={0.35} />
        </mesh>
      ))}
    </group>
  );
}

// ── 04 the ten transformer blocks ─────────────────────────────────────────
const N_BLOCKS = 10;
const BLOCK_X = (i) => -1.9 + i * 0.5;

function BlockStack({ active }) {
  const g = useRef();
  useFrame((s) => {
    const wave = (s.clock.elapsedTime * 1.7) % (N_BLOCKS + 5);
    g.current?.children.forEach((m, i) => {
      const d = Math.abs(wave - i);
      const lit = Math.max(0, 1 - d / 2.4);
      const on = active >= 3 ? 1 : 0.45;
      m.material.emissiveIntensity = (0.12 + lit * 1.35) * on;
      m.material.opacity = (0.24 + lit * 0.42) * on;
    });
  });
  return (
    <group>
      <group ref={g}>
        {Array.from({ length: N_BLOCKS }, (_, i) => (
          <mesh key={i} position={[BLOCK_X(i), 0.35, 0]}>
            <boxGeometry args={[0.3, 1.9, 1.9]} />
            <meshStandardMaterial color="#123543" emissive={CYAN} emissiveIntensity={0.12} transparent opacity={0.3} />
          </mesh>
        ))}
      </group>
      {/* AdaLN-Zero: the one conditioning vector reaching every block */}
      {Array.from({ length: N_BLOCKS }, (_, i) => (
        <Line
          key={i}
          points={[[-3.95, 1.5, 0], [BLOCK_X(i), 1.35, 0]]}
          color={VIOLET}
          lineWidth={1}
          transparent
          opacity={0.28}
        />
      ))}
    </group>
  );
}

// One block opened up above the stack: modulated attention, then feed-forward.
function ExplodedBlock({ active }) {
  const g = useRef();
  const show = active === 3;
  useFrame(() => {
    if (!g.current) return;
    const want = show ? 1 : 0;
    g.current.userData.k = lerp(g.current.userData.k ?? 0, want, 0.08);
    const k = g.current.userData.k;
    g.current.position.y = 2.15 + (1 - k) * -1.2;
    g.current.children.forEach((c) => {
      if (c.material) c.material.opacity = k * (c.userData.base ?? 0.8);
      c.visible = k > 0.02;
    });
  });
  const parts = [
    { y: 0, w: 1.0, c: VIOLET, base: 0.75 }, // AdaLN-Zero
    { y: 0.42, w: 1.5, c: CYAN, base: 0.8 }, // self-attention
    { y: 0.84, w: 1.0, c: VIOLET, base: 0.75 }, // AdaLN-Zero
    { y: 1.26, w: 1.3, c: CYAN, base: 0.8 }, // feed-forward
  ];
  return (
    <group ref={g} position={[0.9, 2.15, 0]}>
      {parts.map((p, i) => (
        <mesh key={i} position={[0, p.y, 0]} userData={{ base: p.base }}>
          <boxGeometry args={[p.w, 0.2, 1.1]} />
          <meshStandardMaterial color="#16323f" emissive={p.c} emissiveIntensity={0.5} transparent opacity={0.8} />
        </mesh>
      ))}
    </group>
  );
}

// ── 05 ambient velocity, projected into the tangent space ─────────────────
function TangentStage() {
  const arrow = useRef();
  const plane = useRef();
  useFrame((s) => {
    const k = 0.5 + 0.5 * Math.sin(s.clock.elapsedTime * 1.3);
    if (arrow.current) arrow.current.rotation.z = -0.5 + k * 0.35;
    if (plane.current) plane.current.material.opacity = 0.16 + k * 0.14;
  });
  return (
    <group position={[4.6, 0.5, 0]}>
      <mesh>
        <sphereGeometry args={[0.75, 32, 22]} />
        <meshStandardMaterial color="#0d2b38" emissive="#0e7490" emissiveIntensity={0.24} wireframe transparent opacity={0.4} />
      </mesh>
      {/* the point on the sphere and its tangent plane */}
      <mesh position={[0, 0.75, 0]}>
        <sphereGeometry args={[0.075, 14, 12]} />
        <meshStandardMaterial color="#e7fbff" emissive={CYAN} emissiveIntensity={1.4} />
      </mesh>
      <mesh ref={plane} position={[0, 0.78, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[1.5, 1.5]} />
        <meshBasicMaterial color={VIOLET} transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>
      {/* ambient velocity (swinging out of the plane) vs its projection */}
      <group ref={arrow} position={[0, 0.75, 0]}>
        <mesh position={[0.34, 0.18, 0]} rotation={[0, 0, -0.5]}>
          <cylinderGeometry args={[0.018, 0.018, 0.75, 6]} />
          <meshBasicMaterial color="#fb7185" />
        </mesh>
      </group>
      <mesh position={[0.38, 0.78, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[0.022, 0.022, 0.78, 6]} />
        <meshBasicMaterial color={VIOLET} />
      </mesh>
    </group>
  );
}

// ── 06 the geodesic step, and the 22 spheres it happens on ────────────────
function ManifoldGrid() {
  const dot = useRef();
  const arc = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= 40; i++) {
      const a = -0.5 + (i / 40) * 1.5;
      pts.push(new THREE.Vector3(Math.sin(a) * 0.78, Math.cos(a) * 0.78, Math.sin(a * 1.7) * 0.22));
    }
    return pts;
  }, []);
  useFrame((s) => {
    const k = (s.clock.elapsedTime * 0.35) % 1;
    const i = Math.floor(k * (arc.length - 1));
    dot.current?.position.copy(arc[i]);
  });
  return (
    <group position={[6.4, 0.5, 0]}>
      <mesh>
        <sphereGeometry args={[0.78, 34, 24]} />
        <meshStandardMaterial color="#0d2b38" emissive="#0e7490" emissiveIntensity={0.26} wireframe transparent opacity={0.45} />
      </mesh>
      <Line points={arc} color={CYAN} lineWidth={2.6} />
      <mesh ref={dot}>
        <sphereGeometry args={[0.085, 14, 12]} />
        <meshStandardMaterial color="#e7fbff" emissive={CYAN} emissiveIntensity={1.6} />
      </mesh>
      {/* the other 21 factors, one sphere per joint */}
      {Array.from({ length: 21 }, (_, i) => {
        const c = i % 7;
        const r = Math.floor(i / 7);
        return (
          <mesh key={i} position={[-0.5 + c * 0.32, -1.35 - r * 0.32, 0.4]}>
            <sphereGeometry args={[0.1, 10, 8]} />
            <meshStandardMaterial color="#0d2b38" emissive="#0e7490" emissiveIntensity={0.3} wireframe />
          </mesh>
        );
      })}
    </group>
  );
}

// ── 07 the resolved pose sequence ─────────────────────────────────────────
const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];

function OutputFigure() {
  const [clip, setClip] = useState(null);
  const [f, setF] = useState(0);
  const acc = useRef(0);

  useEffect(() => {
    let alive = true;
    loadNpy("/data/gt_dance.npy")
      .then((d) => alive && setClip(d))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useFrame((_, dt) => {
    if (!clip) return;
    acc.current += dt * 20;
    const i = Math.floor(acc.current) % clip.shape[0];
    if (i !== f) setF(i);
  });

  const bones = useMemo(() => {
    if (!clip) return null;
    const [, J] = clip.shape;
    // drop the figure onto the floor and stand it beside the manifold stage
    let minY = Infinity;
    for (let j = 0; j < J; j++) minY = Math.min(minY, clip.data[(f * J + j) * 3 + 1]);
    const at = (j) => {
      const o = (f * J + j) * 3;
      return [clip.data[o] * 0.85 + 8.4, (clip.data[o + 1] - minY) * 0.85 - 1.7, clip.data[o + 2] * 0.85];
    };
    return CHAINS.map((c) => c.map(at));
  }, [clip, f]);

  if (!bones) return null;
  return (
    <group>
      {bones.map((pts, i) => (
        <Line key={i} points={pts} color={GREEN} lineWidth={2.4} />
      ))}
    </group>
  );
}
