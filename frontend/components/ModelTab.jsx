"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import { useVizJob } from "@/lib/useVizJob";

const SUB = [
  { id: "train", label: "Train" },
  { id: "eval", label: "Evaluate" },
];

export default function ModelTab() {
  const [sub, setSub] = useState("train");
  return (
    <div className="space-y-5">
      <div className="flex gap-1.5">
        {SUB.map((s) => (
          <button key={s.id} onClick={() => setSub(s.id)}
            className={`rounded-md px-4 py-2 text-[13px] font-medium transition ${sub === s.id ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
            {s.label}
          </button>
        ))}
      </div>
      {sub === "train" ? <Train /> : <Evaluate />}
    </div>
  );
}

function Train() {
  const [form, setForm] = useState({ model_preset: "dit_base", train_preset: "rmg_base", subset_n: 16, max_steps: 15000, overrides: "" });
  const [preview, setPreview] = useState(null);
  const [pErr, setPErr] = useState(null);
  const { job, error, submitting, run } = useVizJob(api.submitTrain);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  async function doPreview() {
    setPErr(null);
    try { setPreview((await api.previewTrain(form)).command); }
    catch (e) { setPErr(e.message); }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <form className="surface space-y-4 p-6" onSubmit={(e) => { e.preventDefault(); run(form); }}>
        <div className="grid grid-cols-2 gap-3">
          <Field label="model"><select className="field-input" value={form.model_preset} onChange={set("model_preset")}><option>dit_base</option><option>dit_large</option></select></Field>
          <Field label="train preset"><select className="field-input" value={form.train_preset} onChange={set("train_preset")}><option>rmg_base</option><option>rmg_large</option></select></Field>
          <Field label="subset_n"><input type="number" className="field-input" value={form.subset_n} onChange={set("subset_n")} /></Field>
          <Field label="max_steps"><input type="number" className="field-input" value={form.max_steps} onChange={set("max_steps")} /></Field>
        </div>
        <Field label="extra overrides (hydra, space-sep)">
          <input className="field-input" placeholder="train.optimizer.lr=1e-4 train.precision=bf16" value={form.overrides} onChange={set("overrides")} />
        </Field>
        {preview && <pre className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink p-3 text-[11px] text-[var(--signal)]">{preview}</pre>}
        {pErr && <p className="text-[12px] text-rose-300">⚠ {pErr}</p>}
        <div className="flex gap-2">
          <button type="button" onClick={doPreview} className="btn-ghost flex-1 text-[12px]">PREVIEW COMMAND</button>
          <button type="submit" className="btn-signal flex-1" disabled={submitting}>{submitting ? "SUBMITTING…" : "▶  LAUNCH TRAINING"}</button>
        </div>
        <p className="label text-center">launches a real GPU training run — confirm the preview first</p>
      </form>
      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Training job status appears here — full progress in the Cluster tab." />
      </div>
    </div>
  );
}

function Evaluate() {
  const [form, setForm] = useState({ max_clips: 256, guidance_scales: "[6.5]" });
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null); // model/train preset from the run config
  const { job, error, submitting, run } = useVizJob(api.submitEval);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <form className="surface space-y-4 p-6" onSubmit={(e) => { e.preventDefault(); run({ ...form, checkpoint, model_preset: presets?.model_preset, train_preset: presets?.train_preset }); }}>
        <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
        <div className="grid grid-cols-2 gap-3">
          <Field label="max_clips"><input type="number" className="field-input" value={form.max_clips} onChange={set("max_clips")} /></Field>
          <Field label="guidance_scales"><input className="field-input" value={form.guidance_scales} onChange={set("guidance_scales")} /></Field>
        </div>
        <button type="submit" className="btn-signal w-full" disabled={submitting || !checkpoint}>
          {submitting ? "SUBMITTING…" : "▶  LAUNCH EVAL"}
        </button>
        <p className="label text-center">results land under the eval run; view them in Analysis → Eval comparison</p>
      </form>
      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Eval job status appears here." />
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}
