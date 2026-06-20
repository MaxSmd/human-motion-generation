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

export default function ModelTab({ model = "rmg" }) {
  const [sub, setSub] = useState("train");
  const isMardm = model === "mardm";
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-1.5">
        {SUB.map((s) => (
          <button key={s.id} onClick={() => setSub(s.id)}
            className={`rounded-md px-4 py-2 text-[13px] font-medium transition ${sub === s.id ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
            {s.label}
          </button>
        ))}
        <span className="ml-auto font-mono text-[11px] uppercase tracking-widest text-[var(--muted)]">
          model · <span className="text-[var(--signal)]">{model}</span>
        </span>
      </div>
      {sub === "train"
        ? (isMardm ? <MardmTrain /> : <RmgTrain />)
        : (isMardm ? <MardmEvaluate /> : <RmgEvaluate />)}
    </div>
  );
}

// ───────────────────────────────────────────────────────────────── rmg: train

// One-click training recipes. `subset_n: 0` = FULL dataset (the form default of
// 16 is a tiny smoke subset that would otherwise silently sabotage a real run).
// Step counts scale with model size: bigger models reach lower FID / smoother
// motion but cost more per step (base 150k → small 200k → mid 300k; the paper's
// 462M ran 600k). All ride the 24h walltime via auto-resubmit, so you can also
// eval FID at checkpoints and stop early once it plateaus.
const TRAIN_RECIPES = [
  { key: "smoke",  label: "smoke test",      hint: "16 clips · 300 steps · ~minutes",
    cfg: { model_preset: "dit_base",  train_preset: "rmg_base",  subset_n: 16, max_steps: 300,    sample_every: 100,   ckpt_every: 200 } },
  { key: "base",   label: "base · 25M full", hint: "full data · 150k · ~3.5 days",
    cfg: { model_preset: "dit_base",  train_preset: "rmg_base",  subset_n: 0,  max_steps: 150000, sample_every: 10000, ckpt_every: 10000 } },
  { key: "small",  label: "small · 50M full", hint: "full data · 200k",
    cfg: { model_preset: "dit_small", train_preset: "rmg_small", subset_n: 0,  max_steps: 200000, sample_every: 10000, ckpt_every: 10000 } },
  { key: "mid",    label: "mid · 112M full", hint: "full data · 300k · recommended",
    cfg: { model_preset: "dit_mid",   train_preset: "rmg_mid",   subset_n: 0,  max_steps: 300000, sample_every: 10000, ckpt_every: 10000 } },
];

function RmgTrain() {
  const [form, setForm] = useState({ model_preset: "dit_base", train_preset: "rmg_base", subset_n: 16, max_steps: 15000, sample_every: 1000, ckpt_every: 5000, overrides: "" });
  const [preview, setPreview] = useState(null);
  const [pErr, setPErr] = useState(null);
  const { job, error, submitting, run } = useVizJob(api.submitTrain);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));
  const applyRecipe = (cfg) => setForm((f) => ({ ...f, ...cfg }));
  const activeRecipe = TRAIN_RECIPES.find((r) =>
    Object.entries(r.cfg).every(([k, v]) => form[k] === v)
  )?.key;

  async function doPreview() {
    setPErr(null);
    try { setPreview((await api.previewTrain(form)).command); }
    catch (e) { setPErr(e.message); }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <form className="surface space-y-4 p-6" onSubmit={(e) => { e.preventDefault(); run(form); }}>
        <div>
          <div className="label mb-1.5">recipe</div>
          <div className="flex flex-wrap gap-2">
            {TRAIN_RECIPES.map((r) => (
              <button key={r.key} type="button" onClick={() => applyRecipe(r.cfg)} title={r.hint}
                className={`rounded-md border px-2.5 py-1 text-[11px] transition ${activeRecipe === r.key ? "border-[var(--signal)] bg-[var(--signal)]/10 text-[var(--signal)]" : "border-[var(--hairline-strong)] text-slate-300 hover:border-[var(--signal)]"}`}>
                {r.label}
              </button>
            ))}
          </div>
          <p className="label mt-1.5">{TRAIN_RECIPES.find((r) => r.key === activeRecipe)?.hint || "custom config — tweak the fields below"}</p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="model"><select className="field-input" value={form.model_preset} onChange={set("model_preset")}><option>dit_base</option><option>dit_small</option><option>dit_mid</option><option>dit_large</option></select></Field>
          <Field label="train preset"><select className="field-input" value={form.train_preset} onChange={set("train_preset")}><option>rmg_base</option><option>rmg_small</option><option>rmg_mid</option><option>rmg_large</option></select></Field>
          <Field label="subset_n (0 = full dataset)"><input type="number" className="field-input" value={form.subset_n} onChange={set("subset_n")} /></Field>
          <Field label="max_steps"><input type="number" className="field-input" value={form.max_steps} onChange={set("max_steps")} /></Field>
          <Field label="sample_every (steps/sample)"><input type="number" className="field-input" value={form.sample_every} onChange={set("sample_every")} /></Field>
          <Field label="ckpt_every (steps/save)"><input type="number" className="field-input" value={form.ckpt_every} onChange={set("ckpt_every")} /></Field>
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

// ─────────────────────────────────────────────────────────────── mardm: train
//
// Two-stage: stage-1 AutoEncoder, then a stage-2 generator that loads the AE
// checkpoint. The gen stage requires picking an AE checkpoint from a prior AE run.

function MardmTrain() {
  const [stage, setStage] = useState("ae");
  const [form, setForm] = useState({ subset_fraction: 0.01, max_steps: 50000, ckpt_every: 10000, overrides: "" });
  const [aeCkpt, setAeCkpt] = useState("");
  const [preview, setPreview] = useState(null);
  const [pErr, setPErr] = useState(null);
  const { job, error, submitting, run } = useVizJob(api.submitTrain);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const body = () => ({ ...form, stage, ...(stage === "gen" ? { ae_checkpoint: aeCkpt } : {}) });
  const genBlocked = stage === "gen" && !aeCkpt;

  async function doPreview() {
    setPErr(null);
    try { setPreview((await api.previewTrain(body())).command); }
    catch (e) { setPErr(e.message); }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <form className="surface space-y-4 p-6" onSubmit={(e) => { e.preventDefault(); run(body()); }}>
        <Field label="stage">
          <div className="flex gap-1.5">
            {[["ae", "AutoEncoder (stage 1)"], ["gen", "Generator (stage 2)"]].map(([id, lbl]) => (
              <button type="button" key={id} onClick={() => setStage(id)}
                className={`flex-1 rounded-md px-3 py-2 text-[12px] font-medium transition ${stage === id ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                {lbl}
              </button>
            ))}
          </div>
        </Field>
        {stage === "gen" && (
          <div>
            <span className="label mb-1.5 block">AE checkpoint (from a stage-1 run)</span>
            <RemoteCheckpointPicker value={aeCkpt} onChange={setAeCkpt} />
          </div>
        )}
        <div className="grid grid-cols-2 gap-3">
          <Field label="subset_fraction"><input type="number" step="0.001" className="field-input" value={form.subset_fraction} onChange={set("subset_fraction")} /></Field>
          <Field label="max_steps"><input type="number" className="field-input" value={form.max_steps} onChange={set("max_steps")} /></Field>
          <Field label="ckpt_every (steps/save)"><input type="number" className="field-input" value={form.ckpt_every} onChange={set("ckpt_every")} /></Field>
        </div>
        <Field label="extra overrides (hydra, space-sep)">
          <input className="field-input" placeholder="train.optimizer.lr=1e-4 train.micro_batch_size=64" value={form.overrides} onChange={set("overrides")} />
        </Field>
        {preview && <pre className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink p-3 text-[11px] text-[var(--signal)]">{preview}</pre>}
        {pErr && <p className="text-[12px] text-rose-300">⚠ {pErr}</p>}
        <div className="flex gap-2">
          <button type="button" onClick={doPreview} className="btn-ghost flex-1 text-[12px]" disabled={genBlocked}>PREVIEW COMMAND</button>
          <button type="submit" className="btn-signal flex-1" disabled={submitting || genBlocked}>{submitting ? "SUBMITTING…" : `▶  LAUNCH ${stage === "ae" ? "AE" : "GEN"}`}</button>
        </div>
        <p className="label text-center">
          {stage === "gen" ? "stage 2 loads the AE checkpoint above — train the AE first" : "stage 1 — train this, then use its checkpoint for the generator"}
        </p>
      </form>
      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Training job status appears here — full progress in the Cluster tab." />
      </div>
    </div>
  );
}

// The full guidance sweep (matches eval.sbatch's default GUIDANCE_SCALES) — one
// eval job evaluates every ω, so Analysis can plot the FID/R-precision curve.
const ALL_GUIDANCE_SCALES = "[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]";

// ────────────────────────────────────────────────────────────────── rmg: eval

function RmgEvaluate() {
  const [form, setForm] = useState({ max_clips: 256, guidance_scales: "[6.5]" });
  const [allScales, setAllScales] = useState(false);
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null); // model/train preset from the run config
  const { job, error, submitting, run } = useVizJob(api.submitEval);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const guidance_scales = allScales ? ALL_GUIDANCE_SCALES : form.guidance_scales;

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <form className="surface space-y-4 p-6" onSubmit={(e) => { e.preventDefault(); run({ ...form, guidance_scales, checkpoint, model_preset: presets?.model_preset, train_preset: presets?.train_preset }); }}>
        <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
        <div className="grid grid-cols-2 gap-3">
          <Field label="max_clips"><input type="number" className="field-input" value={form.max_clips} onChange={set("max_clips")} /></Field>
          <Field label="guidance_scales">
            <input className="field-input disabled:opacity-50" value={guidance_scales} onChange={set("guidance_scales")} disabled={allScales} />
          </Field>
        </div>
        <label className="flex items-center gap-2 text-[12px] text-slate-300">
          <input type="checkbox" checked={allScales} onChange={(e) => setAllScales(e.target.checked)} />
          all guidance scales <span className="font-mono text-[11px] text-[var(--muted)]">{ALL_GUIDANCE_SCALES}</span>
        </label>
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

// ──────────────────────────────────────────────────────────────── mardm: eval
//
// MARDM eval scores a stage-2 generator checkpoint and needs the matching
// stage-1 AE checkpoint to decode its latents → motion → 263-D features.

function MardmEvaluate() {
  const [form, setForm] = useState({ guidance_scales: "[6.5]" });
  const [allScales, setAllScales] = useState(false);
  const [aeCkpt, setAeCkpt] = useState("");
  const [genCkpt, setGenCkpt] = useState("");
  const { job, error, submitting, run } = useVizJob(api.submitEval);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const guidance_scales = allScales ? ALL_GUIDANCE_SCALES : form.guidance_scales;
  const blocked = !aeCkpt || !genCkpt;

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <form className="surface space-y-4 p-6" onSubmit={(e) => { e.preventDefault(); run({ checkpoint: genCkpt, ae_checkpoint: aeCkpt, guidance_scales }); }}>
        <div>
          <span className="label mb-1.5 block">AE checkpoint (stage 1)</span>
          <RemoteCheckpointPicker value={aeCkpt} onChange={setAeCkpt} />
        </div>
        <div>
          <span className="label mb-1.5 block">generator checkpoint (stage 2)</span>
          <RemoteCheckpointPicker value={genCkpt} onChange={setGenCkpt} />
        </div>
        <Field label="guidance_scales">
          <input className="field-input disabled:opacity-50" value={guidance_scales} onChange={set("guidance_scales")} disabled={allScales} />
        </Field>
        <label className="flex items-center gap-2 text-[12px] text-slate-300">
          <input type="checkbox" checked={allScales} onChange={(e) => setAllScales(e.target.checked)} />
          all guidance scales <span className="font-mono text-[11px] text-[var(--muted)]">{ALL_GUIDANCE_SCALES}</span>
        </label>
        <button type="submit" className="btn-signal w-full" disabled={submitting || blocked}>
          {submitting ? "SUBMITTING…" : "▶  LAUNCH EVAL"}
        </button>
        <p className="label text-center">pick both the AE and generator checkpoints; results appear in Analysis → Eval comparison</p>
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
