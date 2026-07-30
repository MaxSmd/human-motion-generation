"use client";

// 3D room / scene editor for euclidean (room) constraints. FRONTEND-ONLY for
// now — no backend dispatch yet; this is the modeling surface we'll later turn
// into a constraint spec (feasible region = room interior minus obstacles, plus
// a skeleton spawn pose).
//
// Scene model (metres, y-up, floor at y=0, room centred at origin in x/z):
//   room    : { width, depth, height }
//   objects : [{ id, kind: 'sphere'|'cylinder'|'box', x, y, z, ...dims, rotation }]
//   spawn   : { x, z, rotation }   // skeleton start pose on the floor
//
// Primitives cover the requested set: sphere, round column (cylinder),
// rectangular column / wall / rectangle (box).

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import {
  Edges,
  Grid,
  GizmoHelper,
  GizmoViewport,
  OrbitControls,
  TransformControls,
} from "@react-three/drei";
import { api, mediaUrl } from "@/lib/api";
import { sdfBox, sdfSphere, sdfCylinder } from "@/lib/sceneMetrics";
import MediaViewer from "./MediaViewer";
import MotionPlayer from "./MotionPlayer";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import PhraseAttestation from "./PhraseAttestation";
import { useVizJob } from "@/lib/useVizJob";

const DEG = Math.PI / 180;

let _oid = 0;
const uid = () => `obj-${++_oid}`;
let _cid = 0;
const cuid = () => `c-${++_cid}`;

// SMPL 22-joint names (matches shared.geometry.skeleton.JOINT_NAMES) — used for
// the contact joint picker. End-effectors (feet/wrists/head) are the useful
// contact joints, so the full list is offered.
const JOINTS = [
  "pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
  "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
  "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
  "L_Wrist", "R_Wrist",
];

const CONTACT_TARGETS = [
  ["obstacle_top", "on top of object", "sit / stand / step on the object's top face"],
  ["obstacle", "touch object surface", "hand-on-wall: nearest point on the object"],
  ["floor", "on the floor", "plant a foot / hand on the ground"],
  ["point", "fixed point", "reach a specific xyz"],
];

const DEFAULT_SCENE = {
  room: { width: 4, depth: 4, height: 2.5 },
  objects: [],
  spawn: { x: 0, z: 0, rotation: 0 },
  padding: 0.1, // standoff margin (m): guidance brakes this far from surfaces
  contacts: [], // [{ id, joint, target, object_id, x,y,z, tol, weight, frame_start, frame_end }]
  foot_skate_weight: 0, // >0 ⇒ penalise sliding planted feet during sampling
};

// Rebuild an editable scene from a job's persisted `params.scene`. Jobs carry
// whatever the editor sent at submit time, which may predate later fields, so
// every default is re-applied underneath. Ids come back too — the uid counters
// are pushed past them so a later "+ Sphere" can't collide with a restored id
// and silently drive the wrong object's gizmo.
function absorbIds(raw) {
  for (const o of raw?.objects || []) {
    const m = /^obj-(\d+)$/.exec(o?.id || "");
    if (m) _oid = Math.max(_oid, Number(m[1]));
  }
  for (const c of raw?.contacts || []) {
    const m = /^c-(\d+)$/.exec(c?.id || "");
    if (m) _cid = Math.max(_cid, Number(m[1]));
  }
}

function hydrateScene(raw) {
  if (!raw) return DEFAULT_SCENE;
  absorbIds(raw);
  return {
    ...DEFAULT_SCENE,
    ...raw,
    room: { ...DEFAULT_SCENE.room, ...(raw.room || {}) },
    spawn: { ...DEFAULT_SCENE.spawn, ...(raw.spawn || {}) },
    objects: (raw.objects || []).map((o) => ({ ...o, id: o.id || uid(), label: o.label ?? "" })),
    contacts: (raw.contacts || []).map((c) => ({
      ...c, id: c.id || cuid(), frame_end: c.frame_end == null ? "" : c.frame_end,
    })),
  };
}

function makeContact(scene) {
  const firstObj = scene.objects[0];
  return {
    id: cuid(),
    joint: firstObj ? "pelvis" : "L_Foot",
    target: firstObj ? "obstacle_top" : "floor",
    object_id: firstObj?.id ?? null,
    x: 0, y: 0.9, z: 0,
    tol: 0.08,
    weight: 1.0,
    frame_start: 0,
    frame_end: "", // "" ⇒ to last frame
  };
}

// Add-button presets. `kind` is the geometry; walls are just a thin tall box.
function makeObject(preset) {
  const base = { id: uid(), x: 0, z: 0, rotation: 0, label: "" };
  switch (preset) {
    case "sphere":
      return { ...base, kind: "sphere", radius: 0.3, y: 0.3, label: "sphere" };
    case "cylinder":
      return { ...base, kind: "cylinder", radius: 0.25, height: 2.0, y: 1.0, label: "round column" };
    case "wall":
      return { ...base, kind: "box", w: 2.0, h: 2.0, d: 0.1, y: 1.0, label: "wall" };
    case "box":
    default:
      return { ...base, kind: "box", w: 0.5, h: 1.0, d: 0.5, y: 0.5, label: "box / column" };
  }
}

const C = {
  signal: "#22d3ee",
  amber: "#fbbf24",
  obj: "#7c8aa5",
  room: "#2b3a52",
};

// Preset scenes. Each builder returns a fresh scene (new object ids). The
// geometry ones (low ceiling / narrow path / stairs) only bite when room
// guidance > 0; spawn placement is always exact.
function presetEmpty() {
  return { room: { width: 4, depth: 4, height: 2.5 }, objects: [], spawn: { x: 0, z: 0, rotation: 0 } };
}
function presetLowCeiling() {
  // 1.3 m ceiling — a ~1.7 m skeleton must crouch (bones are fixed, can't shrink).
  return { room: { width: 3, depth: 4, height: 1.3 }, objects: [], spawn: { x: 0, z: -1.4, rotation: 0 } };
}
function presetNarrowPath() {
  const wall = (x, label) => ({ id: uid(), kind: "box", x, y: 0.75, z: 0, w: 0.2, h: 1.5, d: 5, rotation: 0, label });
  return {
    room: { width: 4, depth: 6, height: 2.5 },
    objects: [wall(-0.6, "left wall"), wall(0.6, "right wall")], // ~1.0 m gap down +z
    spawn: { x: 0, z: -2.6, rotation: 0 },
  };
}
function _stairs(z0 = 0.8, n = 4, w = 1.6) {
  const steps = [];
  for (let i = 0; i < n; i++) {
    const h = (i + 1) * 0.2;
    steps.push({ id: uid(), kind: "box", x: 0, y: h / 2, z: z0 + i * 0.45, w, h, d: 0.45, rotation: 0, label: `step ${i + 1}` });
  }
  return steps;
}
function presetStairs() {
  // Solid blocks of increasing height ahead of the spawn (avoid-penetration).
  return { room: { width: 4, depth: 6, height: 3 }, objects: _stairs(0.8), spawn: { x: 0, z: -2.2, rotation: 0 } };
}
function presetPathStairs() {
  // Narrow corridor leading INTO a staircase — the combined test scene.
  const wall = (x, label) => ({ id: uid(), kind: "box", x, y: 0.75, z: -1.5, w: 0.2, h: 1.5, d: 3, rotation: 0, label });
  return {
    room: { width: 4, depth: 7, height: 3 },
    objects: [wall(-0.6, "left wall"), wall(0.6, "right wall"), ..._stairs(1.0, 4, 1.0)],
    spawn: { x: 0, z: -3.0, rotation: 0 },
  };
}
const PRESETS = [
  { key: "empty", label: "Empty", build: presetEmpty },
  { key: "low", label: "Low ceiling", build: presetLowCeiling },
  { key: "narrow", label: "Narrow path", build: presetNarrowPath },
  { key: "stairs", label: "Stairs ahead", build: presetStairs },
  { key: "pathstairs", label: "Path + stairs", build: presetPathStairs },
];

export default function RoomEditor({ clusterMode = false, checkpoints = [], restore = null }) {
  const [scene, setScene] = useState(DEFAULT_SCENE);
  const [selectedId, setSelectedId] = useState(null); // object id | 'spawn' | null
  const [gizmo, setGizmo] = useState("translate"); // 'translate' | 'rotate'

  const setRoom = (patch) => setScene((s) => ({ ...s, room: { ...s.room, ...patch } }));
  const setSpawn = (patch) => setScene((s) => ({ ...s, spawn: { ...s.spawn, ...patch } }));
  const addObject = (preset) => {
    const o = makeObject(preset);
    setScene((s) => ({ ...s, objects: [...s.objects, o] }));
    setSelectedId(o.id);
  };
  const updateObject = (id, patch) =>
    setScene((s) => ({ ...s, objects: s.objects.map((o) => (o.id === id ? { ...o, ...patch } : o)) }));
  const removeObject = (id) => {
    setScene((s) => ({
      ...s,
      objects: s.objects.filter((o) => o.id !== id),
      // drop contacts that referenced the removed object
      contacts: (s.contacts || []).filter((c) => c.object_id !== id),
    }));
    setSelectedId((cur) => (cur === id ? null : cur));
  };

  const addContact = () => setScene((s) => ({ ...s, contacts: [...(s.contacts || []), makeContact(s)] }));
  // Replace the contact list with auto-extracted ones (from a draft's foot
  // plants). Each server spec is wrapped with a local id + point/height defaults
  // so it slots into the same editor rows as a hand-authored contact.
  const setAutoContacts = (specs) =>
    setScene((s) => ({
      ...s,
      contacts: (specs || []).map((c) => ({
        id: cuid(), x: 0, y: 0.9, z: 0, object_id: null,
        ...c,
        frame_end: c.frame_end == null ? "" : c.frame_end,
      })),
    }));
  const updateContact = (id, patch) =>
    setScene((s) => ({ ...s, contacts: (s.contacts || []).map((c) => (c.id === id ? { ...c, ...patch } : c)) }));
  const removeContact = (id) =>
    setScene((s) => ({ ...s, contacts: (s.contacts || []).filter((c) => c.id !== id) }));

  const selectedObj = scene.objects.find((o) => o.id === selectedId) || null;

  const [preset, setPreset] = useState("empty");
  // Presets replace room geometry but keep the user's solver knobs (padding,
  // foot-skate) and reset contacts (they reference the old objects).
  const applyPreset = (p) => {
    setScene((s) => ({
      padding: s.padding ?? 0.1, foot_skate_weight: s.foot_skate_weight ?? 0, contacts: [], ...p.build(),
    }));
    setSelectedId(null);
    setPreset(p.key);
  };

  // Pull a past scene generation back onto the editing surface. The room stops
  // matching any preset chip once it's someone's authored geometry, so clear the
  // highlight rather than lie about which preset is active.
  useEffect(() => {
    if (!restore?.params?.scene) return;
    setScene(hydrateScene(restore.params.scene));
    setSelectedId(null);
    setPreset(null);
  }, [restore?.at]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-5">
    {/* ───────────── presets ───────────── */}
    <div className="surface flex flex-wrap items-center gap-2 p-3">
      <span className="label mr-1">presets</span>
      {PRESETS.map((p) => (
        <button
          key={p.key}
          onClick={() => applyPreset(p)}
          className={`rounded-md border px-3 py-1.5 text-[12px] transition ${
            preset === p.key
              ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
              : "border-[var(--hairline)] text-slate-300 hover:border-[var(--hairline-strong)]"
          }`}
        >
          {p.label}
        </button>
      ))}
      <span className="ml-auto text-[11px] text-[var(--muted)]">
        low ceiling / narrow path / stairs need room guidance &gt; 0 to take effect
      </span>
    </div>
    <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
      {/* ───────────── 3D viewport ───────────── */}
      <div className="surface relative overflow-hidden p-0" style={{ minHeight: 540 }}>
        <div className="pointer-events-none absolute left-4 top-4 z-10 flex flex-col gap-1">
          <span className="label">room · {scene.room.width}×{scene.room.depth}×{scene.room.height} m</span>
          <span className="text-[11px] text-[var(--muted)]">drag gizmo to place · click to select · drag empty to orbit</span>
        </div>
        <div className="absolute right-4 top-4 z-10 flex gap-1.5">
          {["translate", "rotate"].map((m) => (
            <button
              key={m}
              onClick={() => setGizmo(m)}
              className={`rounded-md border px-2.5 py-1 font-mono text-[11px] uppercase tracking-widest transition ${
                gizmo === m
                  ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                  : "border-[var(--hairline)] text-[var(--muted)] hover:text-slate-200"
              }`}
            >
              {m === "translate" ? "move" : "rotate"}
            </button>
          ))}
        </div>

        <Canvas
          shadows
          camera={{ position: [5, 4.5, 6], fov: 48 }}
          onPointerMissed={() => setSelectedId(null)}
          style={{ height: 540, background: "transparent" }}
        >
          <hemisphereLight intensity={0.6} groundColor="#0a0f18" />
          <directionalLight position={[5, 8, 4]} intensity={1.1} castShadow />

          <Room room={scene.room} />

          <Grid
            args={[scene.room.width, scene.room.depth]}
            cellSize={0.5}
            cellThickness={0.6}
            cellColor="#1d2738"
            sectionSize={1}
            sectionThickness={1}
            sectionColor="#2b3a52"
            fadeDistance={28}
            infiniteGrid={false}
            position={[0, 0.001, 0]}
          />

          {scene.objects.map((o) => (
            <SceneObject
              key={o.id}
              obj={o}
              selected={selectedId === o.id}
              gizmo={gizmo}
              onSelect={() => setSelectedId(o.id)}
              onChange={(patch) => updateObject(o.id, patch)}
            />
          ))}

          {(scene.contacts || []).map((c) => (
            <ContactMarker key={c.id} contact={c} scene={scene} />
          ))}

          <Spawn
            spawn={scene.spawn}
            selected={selectedId === "spawn"}
            gizmo={gizmo}
            onSelect={() => setSelectedId("spawn")}
            onChange={(patch) => setSpawn(patch)}
          />

          <OrbitControls makeDefault enableDamping dampingFactor={0.1} maxPolarAngle={Math.PI / 2.05} />
          <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
            <GizmoViewport axisColors={["#ef6f6f", "#7bd88f", "#22d3ee"]} labelColor="#0a0f18" />
          </GizmoHelper>
        </Canvas>
      </div>

      {/* ───────────── side panel ───────────── */}
      <div className="space-y-4">
        <Section title="Room" sub="rectangular bounds (m)">
          <div className="grid grid-cols-3 gap-2">
            <Num label="width" value={scene.room.width} min={0.5} step={0.1} onChange={(v) => setRoom({ width: v })} />
            <Num label="depth" value={scene.room.depth} min={0.5} step={0.1} onChange={(v) => setRoom({ depth: v })} />
            <Num label="height" value={scene.room.height} min={0.5} step={0.1} onChange={(v) => setRoom({ height: v })} />
          </div>
          <div className="mt-2">
            <Num label="padding / standoff (m)" value={scene.padding ?? 0} min={0} step={0.05}
              onChange={(v) => setScene((s) => ({ ...s, padding: v }))} />
            <p className="label mt-1">brake this far before walls & objects (0 = touch allowed)</p>
          </div>
          <label className="mt-3 flex items-center justify-between gap-2 rounded-md border border-[var(--hairline)] px-2.5 py-2">
            <span>
              <span className="label block">foot anti-skate</span>
              <span className="text-[10px] text-[var(--muted)]">penalise planted feet sliding</span>
            </span>
            <input type="checkbox" className="accent-[var(--signal)]"
              checked={(scene.foot_skate_weight ?? 0) > 0}
              onChange={(e) => setScene((s) => ({ ...s, foot_skate_weight: e.target.checked ? 1.0 : 0 }))} />
          </label>
        </Section>

        <Section title="Objects" sub="obstacles & structure">
          <div className="grid grid-cols-2 gap-1.5">
            {[
              ["sphere", "+ Sphere"],
              ["cylinder", "+ Round column"],
              ["box", "+ Box / column"],
              ["wall", "+ Wall"],
            ].map(([preset, label]) => (
              <button
                key={preset}
                onClick={() => addObject(preset)}
                className="rounded-md border border-[var(--hairline)] px-2 py-1.5 text-[11px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]"
              >
                {label}
              </button>
            ))}
          </div>

          {scene.objects.length > 0 && (
            <ul className="mt-3 space-y-1">
              {scene.objects.map((o) => (
                <li key={o.id}>
                  <button
                    onClick={() => setSelectedId(o.id)}
                    className={`flex w-full items-center justify-between rounded-md border px-2.5 py-1.5 text-left text-[12px] ${
                      selectedId === o.id
                        ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                        : "border-[var(--hairline)] text-slate-300 hover:border-[var(--hairline-strong)]"
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <KindDot kind={o.kind} />
                      {o.label || o.kind}
                    </span>
                    <span
                      onClick={(e) => { e.stopPropagation(); removeObject(o.id); }}
                      className="text-[var(--muted)] hover:text-[var(--amber)]"
                    >
                      ✕
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Section>

        {selectedObj && (
          <Section title="Selected" sub={selectedObj.label || selectedObj.kind}>
            <ObjectProps obj={selectedObj} onChange={(patch) => updateObject(selectedObj.id, patch)} />
          </Section>
        )}

        <Section title="Spawn" sub="skeleton start pose">
          <div className="grid grid-cols-3 gap-2">
            <Num label="x" value={scene.spawn.x} step={0.1} onChange={(v) => setSpawn({ x: v })} />
            <Num label="z" value={scene.spawn.z} step={0.1} onChange={(v) => setSpawn({ z: v })} />
            <Num label="facing°" value={scene.spawn.rotation} step={5} onChange={(v) => setSpawn({ rotation: v })} />
          </div>
          <button
            onClick={() => setSelectedId("spawn")}
            className={`mt-2 w-full rounded-md border px-2 py-1.5 text-[11px] ${
              selectedId === "spawn"
                ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                : "border-[var(--hairline)] text-slate-300 hover:border-[var(--hairline-strong)]"
            }`}
          >
            select spawn in viewport
          </button>
          <SpawnFeasibility scene={scene} />
        </Section>

        <ContactsSection
          scene={scene}
          onAdd={addContact}
          onUpdate={updateContact}
          onRemove={removeContact}
        />

        <SceneJSON scene={scene} />
      </div>
    </div>

      <RoomDispatch scene={scene} clusterMode={clusterMode} checkpoints={checkpoints} restore={restore} onAutoContacts={setAutoContacts} />
    </div>
  );
}

// ───────────────────────────────────────────── sample-in-room dispatch

function RoomDispatch({ scene, clusterMode, checkpoints, restore, onAutoContacts }) {
  const [form, setForm] = useState({
    text: "a person walks forward", guidance: 6.5, num_steps: 50, num_frames: 120, seed: 0, room_guidance: 0.75,
  });
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);
  // local-mode result
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // a clip pulled back out of history — takes over the preview until you sample
  const [loaded, setLoaded] = useState(null);
  // cluster-mode job
  const { job, error: jobError, submitting, run } = useVizJob();

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "number" || e.target.type === "range" ? Number(e.target.value) : e.target.value }));

  // Restore the sampling settings alongside the room (RoomEditor handles the
  // geometry), and show the clip that job actually produced.
  useEffect(() => {
    const p = restore?.params;
    if (!p?.scene) return;
    setForm((f) => ({
      ...f,
      text: p.prompts ?? p.text ?? f.text,
      guidance: p.guidance ?? f.guidance,
      num_steps: p.num_steps ?? f.num_steps,
      num_frames: p.num_frames ?? f.num_frames,
      seed: p.seed ?? f.seed,
      room_guidance: p.room_guidance ?? f.room_guidance,
    }));
    if (p.checkpoint) setCheckpoint(p.checkpoint);
    const outs = restore.job?.outputs || [];
    const out = outs.find((o) => o.npy_url) || outs[0] || null;
    setLoaded(out ? { ...out, caption: out.caption || p.prompts || p.text || "" } : null);
    setError(null);
  }, [restore?.at]); // eslint-disable-line react-hooks/exhaustive-deps

  async function go(e) {
    e.preventDefault();
    setLoaded(null); // sampling replaces the history clip in the preview
    if (clusterMode) {
      run({
        mode: "prompt", checkpoint, prompts: form.text, guidance: form.guidance,
        num_steps: form.num_steps, num_frames: form.num_frames,
        model_preset: presets?.model_preset, train_preset: presets?.train_preset,
        scene, room_guidance: form.room_guidance,
      });
      return;
    }
    setLoading(true); setError(null);
    try {
      const body = {
        text: form.text, guidance: form.guidance, num_steps: form.num_steps,
        num_frames: form.num_frames, seed: form.seed, scene, room_guidance: form.room_guidance,
      };
      if (checkpoint) body.checkpoint = checkpoint;
      const res = await api.generate(body);
      setResult({ ...res, caption: form.text });
    } catch (err) { setError(err.message); setResult(null); }
    finally { setLoading(false); }
  }

  // Joints .npy for the in-browser 3D player (placement already applied server
  // side, so it's in the room's world frame). A clip loaded from history wins
  // until the next sample; otherwise local → joints_npy_url, cluster → first
  // finished output's npy_url.
  const clusterOut = clusterMode && job?.state === "done" ? (job.outputs || []).find((o) => o.npy_url) : null;

  // Pass A → Pass B bootstrap: read the just-sampled draft's foot plants and turn
  // them into contacts (floor, or onto an obstacle top by xz). Cluster-only —
  // the endpoint reads the pulled .npy locally.
  const [extracting, setExtracting] = useState(false);
  const [extractMsg, setExtractMsg] = useState(null);
  const canAutoContact = !!(clusterMode && clusterOut && job?.id);
  async function autoContact() {
    if (!canAutoContact) return;
    setExtracting(true); setExtractMsg(null);
    try {
      const name = clusterOut.npy_url.split("/").pop();
      const res = await api.autoContacts(job.id, name, scene, scene.fps || 20);
      onAutoContacts?.(res.contacts);
      setExtractMsg(
        `${res.count} contact${res.count === 1 ? "" : "s"} — ${res.on_obstacle} on obstacles, ${res.on_floor} on floor. ` +
        (form.room_guidance > 0 ? "Re-sample to apply." : "Raise strength > 0, then re-sample."),
      );
    } catch (err) { setExtractMsg(`extraction failed: ${err.message}`); }
    finally { setExtracting(false); }
  }

  const freshNpy = clusterMode ? clusterOut?.npy_url : result?.joints_npy_url;
  const npyUrl = loaded ? loaded.npy_url : freshNpy;
  const jointsUrl = npyUrl ? mediaUrl(npyUrl) : null;
  const gifUrl = loaded ? loaded.media_url : (clusterOut?.media_url || result?.media_url);

  return (
    <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
      <form className="surface space-y-4 p-5" onSubmit={go}>
        <div className="flex items-center justify-between">
          <div className="display text-sm font-bold text-white">Sample in this room</div>
          <span className="label">{clusterMode ? "cluster job" : "local"}</span>
        </div>

        <label className="block"><span className="label mb-1.5 block">prompt</span>
          <textarea rows={2} className="field-input resize-none" value={form.text} onChange={set("text")} />
          <PhraseAttestation phrase={form.text} /></label>

        {clusterMode ? (
          <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
        ) : (
          <label className="block"><span className="label mb-1.5 block">checkpoint</span>
            <select className="field-input" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
              <option value="">server default</option>
              {checkpoints.map((c) => <option key={c.path} value={c.path}>{c.run} / {c.name}</option>)}
            </select>
          </label>
        )}

        <div className="grid grid-cols-2 gap-3">
          <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
          <Field label="ODE steps"><input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} /></Field>
          <Field label="frames"><input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} /></Field>
          {!clusterMode && <Field label="seed"><input type="number" className="field-input" value={form.seed} onChange={set("seed")} /></Field>}
        </div>

        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <span className="label">soft constraint strength</span>
            <span className="font-mono text-sm text-[var(--signal)]">{guidanceLabel(form.room_guidance)}</span>
          </div>
          <input type="range" min="0" max="2" step="0.05" value={form.room_guidance} onChange={set("room_guidance")} className="w-full accent-[var(--signal)]" />
          <p className="label mt-1">
            drives avoidance, contacts &amp; anti-skate (relative to the motion). 0 = exact spawn placement only —
            walls, contacts &amp; foot-skate are ignored.
          </p>
          {form.room_guidance === 0 && ((scene.contacts || []).length > 0 || (scene.foot_skate_weight ?? 0) > 0) && (
            <p className="mt-1 text-[11px] text-[var(--amber)]">
              ⚠ contacts / anti-skate set but strength is 0 — raise it for them to take effect.
            </p>
          )}
        </div>

        <button type="submit" className="btn-signal w-full" disabled={(clusterMode && (submitting || !checkpoint)) || (!clusterMode && loading)}>
          {clusterMode ? (submitting ? "SUBMITTING…" : "▶  SAMPLE ON CLUSTER") : (loading ? "Sampling…" : "▶  SAMPLE IN ROOM")}
        </button>
        {clusterMode && !checkpoint && <p className="label text-center">pick a remote checkpoint first</p>}
      </form>

      <div className="surface p-5">
        {loaded && (
          <div className="mb-3 flex items-center justify-between gap-2 rounded-md border border-[var(--accent2)]/50 bg-[var(--accent2)]/10 px-2.5 py-1.5">
            <span className="min-w-0 truncate text-[11px] text-slate-300">
              <span className="font-semibold text-[var(--accent2)]">from history</span>
              {loaded.caption ? <span className="text-[var(--muted)]"> · {loaded.caption}</span> : null}
            </span>
            <button onClick={() => setLoaded(null)} className="shrink-0 text-[11px] text-[var(--muted)] hover:text-slate-200">
              ✕ clear
            </button>
          </div>
        )}
        {jointsUrl ? (
          <>
            <div className="mb-3 flex items-center justify-between">
              <span className="label">3d preview · orbit to view from any angle</span>
              <div className="flex items-center gap-3">
                {canAutoContact && (
                  <button
                    onClick={autoContact}
                    disabled={extracting}
                    title="Read this draft's foot plants and turn them into contacts (floor / obstacle top)"
                    className="text-[11px] text-[var(--signal)] hover:underline disabled:opacity-50"
                  >
                    {extracting ? "extracting…" : "⇊ auto-contacts from draft"}
                  </button>
                )}
                {gifUrl && (
                  <a
                    href={mediaUrl(gifUrl)}
                    target="_blank" rel="noreferrer"
                    className="text-[11px] text-[var(--muted)] hover:text-[var(--signal)]"
                  >
                    open gif ↗
                  </a>
                )}
              </div>
            </div>
            {extractMsg && (
              <p className="mb-2 rounded-md border border-[var(--hairline)] bg-[var(--signal-dim)] px-2 py-1 text-[11px] text-slate-300">
                {extractMsg}
              </p>
            )}
            <MotionPlayer jointsUrl={jointsUrl} scene={scene} />
          </>
        ) : loaded ? (
          // pre-dates the npy dump (or the job only pulled a gif) — the flat
          // render is all there is to show, but the room is still restored
          <MediaViewer url={loaded.media_url} caption={loaded.caption} />
        ) : clusterMode ? (
          <VizJobResult job={job} error={jobError} submitting={submitting} emptyHint="Room-constrained clips will appear here." />
        ) : (
          <MediaViewer url={result?.media_url} caption={result?.caption} loading={loading} error={error} />
        )}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────── 3D pieces

function Room({ room }) {
  const { width: w, depth: d, height: h } = room;
  return (
    <group>
      {/* floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial color="#0c121d" roughness={1} metalness={0} />
      </mesh>
      {/* wireframe shell */}
      <mesh position={[0, h / 2, 0]}>
        <boxGeometry args={[w, h, d]} />
        <meshBasicMaterial transparent opacity={0} />
        <Edges color={C.room} />
      </mesh>
    </group>
  );
}

function SceneObject({ obj, selected, gizmo, onSelect, onChange }) {
  // `setNode` (ref-as-state) forces a re-render once the mesh mounts so
  // TransformControls has a real object3D to attach to.
  const [node, setNode] = useState(null);

  const geometry = useMemo(() => {
    if (obj.kind === "sphere") return <sphereGeometry args={[obj.radius, 32, 24]} />;
    if (obj.kind === "cylinder") return <cylinderGeometry args={[obj.radius, obj.radius, obj.height, 32]} />;
    return <boxGeometry args={[obj.w, obj.h, obj.d]} />;
  }, [obj.kind, obj.radius, obj.height, obj.w, obj.h, obj.d]);

  const sync = (o) => {
    if (!o) return;
    onChange({
      x: round(o.position.x), y: round(o.position.y), z: round(o.position.z),
      rotation: round(o.rotation.y / DEG, 0),
    });
  };

  return (
    <>
      <mesh
        ref={setNode}
        position={[obj.x, obj.y, obj.z]}
        rotation={[0, (obj.rotation || 0) * DEG, 0]}
        castShadow
        onClick={(e) => { e.stopPropagation(); onSelect(); }}
      >
        {geometry}
        <meshStandardMaterial
          color={selected ? C.signal : C.obj}
          transparent
          opacity={selected ? 0.42 : 0.3}
          roughness={0.6}
          emissive={selected ? C.signal : "#000000"}
          emissiveIntensity={selected ? 0.25 : 0}
        />
        <Edges color={selected ? C.signal : "#46566f"} />
      </mesh>
      {selected && node && (
        <TransformControls
          object={node}
          mode={gizmo}
          showY={gizmo === "translate"}
          onObjectChange={(e) => sync(e?.target?.object ?? node)}
        />
      )}
    </>
  );
}

function Spawn({ spawn, selected, gizmo, onSelect, onChange }) {
  const [node, setNode] = useState(null);
  const col = selected ? C.signal : C.amber;

  const sync = (o) => {
    if (!o) return;
    onChange({ x: round(o.position.x), z: round(o.position.z), rotation: round(o.rotation.y / DEG, 0) });
  };

  return (
    <>
      <group
        ref={setNode}
        position={[spawn.x, 0, spawn.z]}
        rotation={[0, (spawn.rotation || 0) * DEG, 0]}
        onClick={(e) => { e.stopPropagation(); onSelect(); }}
      >
        {/* body */}
        <mesh position={[0, 0.62, 0]} castShadow>
          <capsuleGeometry args={[0.16, 0.62, 6, 12]} />
          <meshStandardMaterial color={col} transparent opacity={0.6} emissive={col} emissiveIntensity={0.3} />
        </mesh>
        {/* facing arrow (+z) */}
        <mesh position={[0, 0.12, 0.34]} rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.1, 0.26, 16]} />
          <meshStandardMaterial color={col} emissive={col} emissiveIntensity={0.4} />
        </mesh>
        {/* floor ring */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
          <ringGeometry args={[0.24, 0.3, 24]} />
          <meshBasicMaterial color={col} transparent opacity={0.8} />
        </mesh>
      </group>
      {selected && node && (
        <TransformControls
          object={node}
          mode={gizmo}
          showY={false}
          onObjectChange={(e) => sync(e?.target?.object ?? node)}
        />
      )}
    </>
  );
}

// ───────────────────────────────────────────── panel widgets

function ObjectProps({ obj, onChange }) {
  return (
    <div className="space-y-2.5">
      <div className="grid grid-cols-3 gap-2">
        <Num label="x" value={obj.x} step={0.1} onChange={(v) => onChange({ x: v })} />
        <Num label="y" value={obj.y} step={0.1} onChange={(v) => onChange({ y: v })} />
        <Num label="z" value={obj.z} step={0.1} onChange={(v) => onChange({ z: v })} />
      </div>
      {obj.kind === "sphere" && (
        <Num label="radius" value={obj.radius} min={0.02} step={0.05} onChange={(v) => onChange({ radius: v })} />
      )}
      {obj.kind === "cylinder" && (
        <div className="grid grid-cols-2 gap-2">
          <Num label="radius" value={obj.radius} min={0.02} step={0.05} onChange={(v) => onChange({ radius: v })} />
          <Num label="height" value={obj.height} min={0.05} step={0.1} onChange={(v) => onChange({ height: v })} />
        </div>
      )}
      {obj.kind === "box" && (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Num label="w" value={obj.w} min={0.02} step={0.1} onChange={(v) => onChange({ w: v })} />
            <Num label="h" value={obj.h} min={0.02} step={0.1} onChange={(v) => onChange({ h: v })} />
            <Num label="d" value={obj.d} min={0.02} step={0.1} onChange={(v) => onChange({ d: v })} />
          </div>
          <Num label="rotation°" value={obj.rotation} step={5} onChange={(v) => onChange({ rotation: v })} />
        </>
      )}
      <label className="block">
        <span className="label mb-1 block">label</span>
        <input className="field-input" value={obj.label} onChange={(e) => onChange({ label: e.target.value })} />
      </label>
    </div>
  );
}

function Section({ title, sub, children }) {
  return (
    <section className="surface p-4">
      <div className="mb-3">
        <div className="display text-sm font-bold text-white">{title}</div>
        {sub && <div className="label mt-0.5">{sub}</div>}
      </div>
      {children}
    </section>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}

function Num({ label, value, onChange, step = 0.1, min }) {
  return (
    <label className="block">
      <span className="label mb-1 block">{label}</span>
      <input
        type="number"
        className="field-input"
        value={value}
        step={step}
        min={min}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isNaN(v)) return;
          onChange(min != null ? Math.max(min, v) : v);
        }}
      />
    </label>
  );
}

function KindDot({ kind }) {
  const shape = kind === "sphere" ? "●" : kind === "cylinder" ? "▮" : "▆";
  return <span className="font-mono text-[var(--muted)]">{shape}</span>;
}

function SceneJSON({ scene }) {
  const [open, setOpen] = useState(false);
  const text = useMemo(() => JSON.stringify(scene, (k, v) => (k === "id" || k === "label" ? v : typeof v === "number" ? round(v) : v), 2), [scene]);
  return (
    <section className="surface p-4">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center justify-between">
        <span className="display text-sm font-bold text-white">Scene JSON</span>
        <span className="label">{open ? "hide" : "show"} · {scene.objects.length} obj</span>
      </button>
      {open && (
        <>
          <pre className="mt-3 max-h-64 overflow-auto rounded-md border border-[var(--hairline)] bg-black/30 p-3 font-mono text-[10px] leading-relaxed text-slate-300">
            {text}
          </pre>
          <button
            onClick={() => navigator.clipboard?.writeText(text)}
            className="mt-2 w-full rounded-md border border-[var(--hairline)] px-2 py-1.5 text-[11px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]"
          >
            copy
          </button>
        </>
      )}
    </section>
  );
}

function round(v, p = 3) {
  const f = 10 ** p;
  return Math.round(v * f) / f;
}

// ───────────────────────────────────────────── contacts

function guidanceLabel(v) {
  if (v === 0) return "off (place only)";
  if (v < 0.6) return `gentle · ${v}`;
  if (v < 1.2) return `firm · ${v}`;
  return `hard · ${v}`;
}

// Where the contact target sits, for the authoring marker. Mirrors the backend
// _contact_target for the static (no-clip) case: object top-centre, the object
// centre, a fixed point, or the floor under the spawn.
function contactTargetPos(contact, scene) {
  const o = scene.objects.find((x) => x.id === contact.object_id);
  if (contact.target === "point") return [contact.x, contact.y, contact.z];
  if (contact.target === "floor") return [scene.spawn.x, 0, scene.spawn.z];
  if (!o) return null;
  if (contact.target === "obstacle_top") {
    if (o.kind === "box") return [o.x, o.y + o.h / 2, o.z];
    if (o.kind === "cylinder") return [o.x, o.y + o.height / 2, o.z];
    if (o.kind === "sphere") return [o.x, o.y + o.radius, o.z];
  }
  return [o.x, o.y, o.z]; // obstacle surface → centre as the indicative marker
}

function ContactMarker({ contact, scene }) {
  const p = contactTargetPos(contact, scene);
  if (!p) return null;
  return (
    <group position={p}>
      <mesh>
        <sphereGeometry args={[0.07, 16, 12]} />
        <meshBasicMaterial color="#34d399" transparent opacity={0.85} />
      </mesh>
      <mesh>
        <sphereGeometry args={[Math.max(0.02, contact.tol), 16, 12]} />
        <meshBasicMaterial color="#34d399" transparent opacity={0.12} />
      </mesh>
    </group>
  );
}

function ContactsSection({ scene, onAdd, onUpdate, onRemove }) {
  const contacts = scene.contacts || [];
  const objects = scene.objects || [];
  return (
    <Section title="Contacts" sub="pull a joint onto a surface (sit · step · touch)">
      <button
        onClick={onAdd}
        className="w-full rounded-md border border-[var(--signal)]/50 bg-[var(--signal-dim)] px-3 py-1.5 text-[12px] font-semibold text-[var(--signal)] hover:bg-[var(--signal)]/15"
      >
        + add contact
      </button>
      {contacts.length === 0 && (
        <p className="label mt-3">No contacts — the body only avoids obstacles. Add one to make it rest on / touch a surface.</p>
      )}
      <div className="mt-3 space-y-3">
        {contacts.map((c, i) => {
          const needsObj = c.target === "obstacle_top" || c.target === "obstacle";
          return (
            <div key={c.id} className="rounded-lg border border-[var(--hairline)] p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="font-mono text-[11px] tracking-widest text-[var(--muted)]">CONTACT {String(i + 1).padStart(2, "0")}</span>
                <button onClick={() => onRemove(c.id)} className="text-[11px] text-[var(--muted)] hover:text-[var(--amber)]">remove</button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <label className="block">
                  <span className="label mb-1 block">joint</span>
                  <select className="field-input" value={c.joint} onChange={(e) => onUpdate(c.id, { joint: e.target.value })}>
                    {JOINTS.map((j) => <option key={j} value={j}>{j}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="label mb-1 block">target</span>
                  <select className="field-input" value={c.target}
                    onChange={(e) => onUpdate(c.id, { target: e.target.value, object_id: e.target.value.startsWith("obstacle") ? (c.object_id ?? objects[0]?.id ?? null) : c.object_id })}>
                    {CONTACT_TARGETS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                  </select>
                </label>
              </div>
              <p className="label mt-1.5">{CONTACT_TARGETS.find(([v]) => v === c.target)?.[2]}</p>

              {needsObj && (
                <label className="mt-2 block">
                  <span className="label mb-1 block">object</span>
                  <select className="field-input" value={c.object_id ?? ""} onChange={(e) => onUpdate(c.id, { object_id: e.target.value })}>
                    {objects.length === 0 && <option value="">— add an object first —</option>}
                    {objects.map((o) => <option key={o.id} value={o.id}>{o.label || o.kind}</option>)}
                  </select>
                </label>
              )}
              {c.target === "point" && (
                <div className="mt-2 grid grid-cols-3 gap-2">
                  <Num label="x" value={c.x} step={0.1} onChange={(v) => onUpdate(c.id, { x: v })} />
                  <Num label="y" value={c.y} step={0.1} onChange={(v) => onUpdate(c.id, { y: v })} />
                  <Num label="z" value={c.z} step={0.1} onChange={(v) => onUpdate(c.id, { z: v })} />
                </div>
              )}

              <div className="mt-2 grid grid-cols-2 gap-2">
                <Num label="tolerance (m)" value={c.tol} min={0.01} step={0.01} onChange={(v) => onUpdate(c.id, { tol: v })} />
                <Num label="weight" value={c.weight} min={0} step={0.5} onChange={(v) => onUpdate(c.id, { weight: v })} />
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <Num label="frame start" value={c.frame_start} min={0} step={1} onChange={(v) => onUpdate(c.id, { frame_start: v })} />
                <label className="block">
                  <span className="label mb-1 block">frame end <span className="text-[var(--muted)]">(blank = all)</span></span>
                  <input type="number" min="0" className="field-input" placeholder="all" value={c.frame_end}
                    onChange={(e) => onUpdate(c.id, { frame_end: e.target.value })} />
                </label>
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ───────────────────────────────────────────── spawn feasibility check

function SpawnFeasibility({ scene }) {
  const msg = useMemo(() => {
    const { width: w, depth: d } = scene.room;
    const p = [scene.spawn.x, 0.9, scene.spawn.z]; // mid-body height probe
    if (Math.abs(scene.spawn.x) > w / 2 || Math.abs(scene.spawn.z) > d / 2)
      return "spawn is outside the room footprint";
    for (const o of scene.objects || []) {
      let sdf;
      if (o.kind === "sphere") sdf = sdfSphere(p, [o.x, o.y, o.z], o.radius);
      else if (o.kind === "cylinder") sdf = sdfCylinder(p, [o.x, o.z], o.radius, o.y, o.height / 2);
      else sdf = sdfBox(p, [o.x, o.y, o.z], [o.w / 2, o.h / 2, o.d / 2], ((o.rotation || 0) * Math.PI) / 180);
      if (sdf < 0) return `spawn is inside "${o.label || o.kind}" — move it clear`;
    }
    return null;
  }, [scene]);
  if (!msg) return null;
  return <p className="mt-2 text-[11px] text-[var(--amber)]">⚠ {msg}</p>;
}
