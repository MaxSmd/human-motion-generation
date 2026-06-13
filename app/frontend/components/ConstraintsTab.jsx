"use client";

// Functional constraint editor for RMG. V1 supports FIXED JOINT ANGLES applied
// during sampling only (no fine-tuning): each pinned joint's quaternion is held
// at an axis+angle target over a frame range. Because RMG lives on
// R^3 × (S^3)^J, a pin is just inpainting that joint's S^3 factor — the backend
// (RiemannianEulerSampler) overwrites it after every ODE step.
//
// Mode-aware like GenerateTab: local mode calls POST /generate directly; cluster
// mode submits a mode=prompt viz job (constraints forwarded to visualize.py).

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import MediaViewer from "./MediaViewer";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import { useVizJob } from "@/lib/useVizJob";

// Fallback joint list (SMPL 22) if /meta/joints can't be reached.
const FALLBACK_JOINTS = [
  "pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
  "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
  "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
  "L_Wrist", "R_Wrist",
].map((name, index) => ({ index, name }));

const AXES = ["x", "y", "z"];
let _pid = 0;
const newPin = () => ({
  id: ++_pid,
  joint: "L_Elbow",
  axis: "z",
  angle_deg: 90,
  frame_start: 0,
  frame_end: "", // "" ⇒ to last frame
});

function pinsToConstraints(pins) {
  return pins.map((p) => ({
    joint: p.joint,
    axis: p.axis,
    angle_deg: Number(p.angle_deg),
    frame_start: Number(p.frame_start) || 0,
    frame_end: p.frame_end === "" ? null : Number(p.frame_end),
  }));
}

export default function ConstraintsTab({ checkpoints = [], clusterMode }) {
  const [joints, setJoints] = useState(FALLBACK_JOINTS);
  const [pins, setPins] = useState([newPin()]);

  useEffect(() => {
    api.metaJoints().then((m) => m?.joints?.length && setJoints(m.joints)).catch(() => {});
  }, []);

  const editor = (
    <ConstraintEditor joints={joints} pins={pins} setPins={setPins} />
  );

  return clusterMode ? (
    <ConstraintsCluster checkpoints={checkpoints} pins={pins} editor={editor} />
  ) : (
    <ConstraintsLocal checkpoints={checkpoints} pins={pins} editor={editor} />
  );
}

// ───────────────────────────────────────────── the shared pin editor

function ConstraintEditor({ joints, pins, setPins }) {
  const set = (id, k, v) => setPins((ps) => ps.map((p) => (p.id === id ? { ...p, [k]: v } : p)));
  const remove = (id) => setPins((ps) => ps.filter((p) => p.id !== id));

  return (
    <section className="surface p-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="display text-base font-bold text-white">Fixed joint angles</div>
          <div className="label mt-0.5">held during sampling · axis + angle · per frame range</div>
        </div>
        <button
          type="button"
          onClick={() => setPins((ps) => [...ps, newPin()])}
          className="rounded-md border border-[var(--signal)]/50 bg-[var(--signal-dim)] px-3 py-1.5 text-[12px] font-semibold text-[var(--signal)] hover:bg-[var(--signal)]/15"
        >
          + add pin
        </button>
      </div>

      {pins.length === 0 && (
        <p className="label mt-4">No pins — the model samples freely. Add a pin to constrain a joint.</p>
      )}

      <div className="mt-4 space-y-3">
        {pins.map((p, i) => (
          <div key={p.id} className="rounded-lg border border-[var(--hairline)] p-3">
            <div className="mb-2.5 flex items-center justify-between">
              <span className="font-mono text-[11px] tracking-widest text-[var(--muted)]">
                PIN {String(i + 1).padStart(2, "0")}
              </span>
              <button
                type="button"
                onClick={() => remove(p.id)}
                className="text-[11px] text-[var(--muted)] hover:text-[var(--amber)]"
              >
                remove
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="label mb-1 block">joint</span>
                <select className="field-input" value={p.joint} onChange={(e) => set(p.id, "joint", e.target.value)}>
                  {joints.map((j) => (
                    <option key={j.name} value={j.name}>{j.name}</option>
                  ))}
                </select>
              </label>

              <div>
                <span className="label mb-1 block">axis</span>
                <div className="flex gap-1.5">
                  {AXES.map((a) => (
                    <button
                      key={a}
                      type="button"
                      onClick={() => set(p.id, "axis", a)}
                      className={`flex-1 rounded-md border px-2 py-1.5 font-mono text-[12px] uppercase transition ${
                        p.axis === a
                          ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                          : "border-[var(--hairline)] text-[var(--muted)] hover:text-slate-200"
                      }`}
                    >
                      {a}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between">
                <span className="label">angle</span>
                <span className="font-mono text-sm text-[var(--signal)]">{Number(p.angle_deg)}°</span>
              </div>
              <input
                type="range" min="-180" max="180" step="5"
                value={p.angle_deg}
                onChange={(e) => set(p.id, "angle_deg", Number(e.target.value))}
                className="w-full accent-[var(--signal)]"
              />
            </div>

            <div className="mt-3 grid grid-cols-2 gap-3">
              <label className="block">
                <span className="label mb-1 block">frame start</span>
                <input type="number" min="0" className="field-input" value={p.frame_start}
                  onChange={(e) => set(p.id, "frame_start", e.target.value)} />
              </label>
              <label className="block">
                <span className="label mb-1 block">frame end <span className="text-[var(--muted)]">(blank = all)</span></span>
                <input type="number" min="0" className="field-input" placeholder="all" value={p.frame_end}
                  onChange={(e) => set(p.id, "frame_end", e.target.value)} />
              </label>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ───────────────────────────────────────────── local mode (in-process /generate)

function ConstraintsLocal({ checkpoints, pins, editor }) {
  const [form, setForm] = useState({ text: "a person waves their right hand", guidance: 6.5, num_steps: 50, seed: 0, num_frames: 100 });
  const [checkpoint, setCheckpoint] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  async function onResample(e) {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      const body = { ...form, constraints: pinsToConstraints(pins) };
      if (checkpoint) body.checkpoint = checkpoint;
      const res = await api.generate(body);
      setResult({ ...res, caption: form.text });
    } catch (err) { setError(err.message); setResult(null); }
    finally { setLoading(false); }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      <form className="space-y-5" onSubmit={onResample}>
        <div className="surface space-y-4 p-5">
          <label className="block"><span className="label mb-1.5 block">prompt</span>
            <textarea rows={2} className="field-input resize-none" value={form.text} onChange={set("text")} /></label>
          <div className="grid grid-cols-2 gap-3">
            <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
            <Field label="ODE steps"><input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} /></Field>
            <Field label="frames"><input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} /></Field>
            <Field label="seed"><input type="number" className="field-input" value={form.seed} onChange={set("seed")} /></Field>
          </div>
          <Field label="checkpoint">
            <select className="field-input" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
              <option value="">server default</option>
              {checkpoints.map((c) => <option key={c.path} value={c.path}>{c.run} / {c.name}</option>)}
            </select>
          </Field>
        </div>
        {editor}
        <button type="submit" className="btn-signal w-full" disabled={loading}>
          {loading ? "Resampling…" : "▶  APPLY & RESAMPLE"}
        </button>
      </form>
      <div className="surface p-6">
        <MediaViewer url={result?.media_url} caption={result?.caption} loading={loading} error={error} />
      </div>
    </div>
  );
}

// ───────────────────────────────────────────── cluster mode (mode=prompt viz job)

function ConstraintsCluster({ pins, editor }) {
  const [form, setForm] = useState({ prompts: "a person waves their right hand", guidance: 6.5, num_steps: 50, num_frames: 100 });
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);
  const { job, error, submitting, run } = useVizJob();
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" || e.target.type === "range" ? Number(e.target.value) : e.target.value }));

  function go(e) {
    e.preventDefault();
    run({
      mode: "prompt", checkpoint, prompts: form.prompts, guidance: form.guidance,
      num_steps: form.num_steps, num_frames: form.num_frames,
      model_preset: presets?.model_preset, train_preset: presets?.train_preset,
      constraints: pinsToConstraints(pins),
    });
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      <form className="space-y-5" onSubmit={go}>
        <div className="surface space-y-4 p-5">
          <label className="block"><span className="label mb-1.5 block">prompt</span>
            <textarea rows={2} className="field-input resize-none" value={form.prompts} onChange={set("prompts")} /></label>
          <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
          <div className="grid grid-cols-3 gap-3">
            <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
            <Field label="ODE steps"><input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} /></Field>
            <Field label="frames"><input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} /></Field>
          </div>
        </div>
        {editor}
        <button type="submit" className="btn-signal w-full" disabled={submitting || !checkpoint}>
          {submitting ? "SUBMITTING…" : "▶  APPLY & RESAMPLE ON CLUSTER"}
        </button>
        {!checkpoint && <p className="label text-center">pick a remote checkpoint first</p>}
      </form>
      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Constrained clips will appear here." />
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}
