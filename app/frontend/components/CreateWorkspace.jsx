"use client";

// CREATE workspace — the single place you author a motion.
//
// Folds the old Generate, Constraints, Room and Studio tabs into one surface:
//   • Prompt  — text → motion, with collapsible Joint-constraint / Joint-range
//               sections that attach to the SAME generate call (was Generate +
//               Constraints, now merged).
//   • Scene   — the 3D room editor (RoomEditor).
//   • Studio  — click-a-joint authoring + live evaluation (StudioTab).
//
// A shared History rail (right / below) lists every generation from all three
// modes — filtered to the create categories — so constrained gens live with
// constrained gens, scene gens with scene gens, none of it leaking to a global
// job list. The rail is a read-only view of /cluster/jobs; the modes keep their
// existing submit pipelines untouched.

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, mediaUrl } from "@/lib/api";
import MediaViewer from "./MediaViewer";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import HistoryRail from "./HistoryRail";
import HistoryPreview from "./HistoryPreview";
import JobProgress from "./JobProgress";
import { useVizJob } from "@/lib/useVizJob";
import { useJobHistory } from "@/lib/useJobHistory";
import { WORKSPACE_CATEGORIES } from "@/lib/classifyJob";
import {
  ConstraintEditor, RangeEditor, FALLBACK_JOINTS,
  pinsToConstraints, rangesToPayload,
} from "./ConstraintsTab";

const RoomEditor = dynamic(() => import("./RoomEditor"), { ssr: false });
const StudioTab = dynamic(() => import("./StudioTab"), { ssr: false });

const MODES = [
  { id: "prompt", label: "Prompt", sub: "text → motion · + constraints" },
  { id: "scene", label: "Scene", sub: "3D room · placement · guidance" },
  { id: "studio", label: "Studio", sub: "click-joint authoring · live eval" },
];

const PRESETS = [
  "a person walks forward",
  "a person waves their right hand",
  "a person sits down on the floor",
  "a person does jumping jacks",
];

export default function CreateWorkspace({ clusterMode, checkpoints = [] }) {
  const [mode, setMode] = useState("prompt");
  const [showRail, setShowRail] = useState(true);
  const [restore, setRestore] = useState(null); // {params, at} pushed from history
  const [preview, setPreview] = useState(null); // history job loaded into the centre
  const { jobs, all, refresh } = useJobHistory(WORKSPACE_CATEGORIES.create);

  // Restoring a job's settings jumps to the right mode and prefills the editor.
  function onRestore(params, cat) {
    setMode(cat === "scene" ? "scene" : "prompt");
    setRestore({ params, at: Date.now() });
    setShowRail(true);
  }

  const main = (
    <div className="min-w-0 space-y-5">
      {clusterMode && <JobProgress jobs={all} onChange={refresh} />}
      {preview && <HistoryPreview job={preview} onClose={() => setPreview(null)} />}
      {mode === "prompt" && (
        <CreatePrompt clusterMode={clusterMode} checkpoints={checkpoints} restore={restore} />
      )}
      {mode === "scene" && <RoomEditor clusterMode={clusterMode} checkpoints={checkpoints} />}
      {mode === "studio" && <StudioTab clusterMode={clusterMode} checkpoints={checkpoints} />}
    </div>
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap gap-1.5">
          {MODES.map((m) => (
            <button key={m.id} onClick={() => setMode(m.id)}
              className={`rounded-md border px-3.5 py-2 text-left transition ${mode === m.id ? "border-[var(--signal)] bg-[var(--signal-dim)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"}`}>
              <div className={`text-[13px] font-bold ${mode === m.id ? "text-[var(--signal)]" : "text-slate-300"}`}>{m.label}</div>
              <div className="label mt-0.5 normal-case">{m.sub}</div>
            </button>
          ))}
        </div>
        <button onClick={() => setShowRail((s) => !s)}
          className="ml-auto rounded-md border border-[var(--hairline)] px-3 py-1.5 text-[12px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]">
          {showRail ? "hide" : "show"} history · {jobs.length}
        </button>
      </div>

      {showRail ? (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px] 2xl:grid-cols-[minmax(0,1fr)_380px]">
          {main}
          <div className="xl:sticky xl:top-6 xl:self-start">
            <HistoryRail jobs={jobs} categories={WORKSPACE_CATEGORIES.create} onRestore={onRestore}
              onOpen={setPreview} emptyHint="Generations you launch here collect by type." />
          </div>
        </div>
      ) : (
        main
      )}
    </div>
  );
}

// ───────────────────────────────────────── Prompt = Generate + Constraints merged

function fromConstraints(arr = []) {
  let id = 0;
  return arr.map((c) => ({
    id: ++id, joint: c.joint, bend_deg: c.bend_deg ?? 90,
    frame_start: c.frame_start ?? 0, frame_end: c.frame_end == null ? "" : c.frame_end,
  }));
}
function fromRanges(arr = []) {
  let id = 0;
  return arr.map((r) => ({
    id: ++id, joint: r.joint, bend_min: r.bend_min ?? 0, bend_max: r.bend_max ?? 90,
    frame_start: r.frame_start ?? 0, frame_end: r.frame_end == null ? "" : r.frame_end,
  }));
}

function CreatePrompt({ clusterMode, checkpoints, restore }) {
  const [joints, setJoints] = useState(FALLBACK_JOINTS);
  const [form, setForm] = useState({ text: "a person walks forward", guidance: 6.5, num_steps: 50, seed: 0, num_frames: 100 });
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);
  const [pins, setPins] = useState([]);
  const [ranges, setRanges] = useState([]);

  // local-mode result + history strip
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [localHist, setLocalHist] = useState([]);
  // cluster-mode job
  const { job, error: jobError, submitting, run } = useVizJob();

  useEffect(() => {
    api.metaJoints().then((m) => m?.joints?.length && setJoints(m.joints)).catch(() => {});
  }, []);

  // Pull settings in from a history "use these settings" click.
  useEffect(() => {
    if (!restore?.params) return;
    const p = restore.params;
    setForm((f) => ({
      ...f,
      text: p.prompts ?? p.text ?? f.text,
      guidance: p.guidance ?? f.guidance,
      num_steps: p.num_steps ?? f.num_steps,
      num_frames: p.num_frames ?? f.num_frames,
    }));
    if (p.checkpoint) setCheckpoint(p.checkpoint);
    setPins(fromConstraints(p.constraints));
    setRanges(fromRanges(p.ranges));
  }, [restore?.at]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "number" || e.target.type === "range" ? Number(e.target.value) : e.target.value }));

  const nActive = pins.length + ranges.length;

  async function go(e) {
    e.preventDefault();
    const constraints = pinsToConstraints(pins);
    const rngs = rangesToPayload(ranges);
    if (clusterMode) {
      run({
        mode: "prompt", checkpoint, prompts: form.text, guidance: form.guidance,
        num_steps: form.num_steps, num_frames: form.num_frames,
        model_preset: presets?.model_preset, train_preset: presets?.train_preset,
        constraints, ranges: rngs,
      });
      return;
    }
    setLoading(true); setError(null);
    try {
      const body = { ...form, constraints, ranges: rngs };
      if (checkpoint) body.checkpoint = checkpoint;
      const res = await api.generate(body);
      const item = { ...res, caption: form.text };
      setResult(item);
      setLocalHist((h) => [item, ...h].slice(0, 12));
    } catch (err) { setError(err.message); setResult(null); }
    finally { setLoading(false); }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      <form className="space-y-5" onSubmit={go}>
        <div className="surface space-y-4 p-5">
          <div>
            <div className="label mb-1.5">prompt</div>
            <textarea rows={2} className="field-input resize-none" value={form.text} onChange={set("text")} />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button key={p} type="button" onClick={() => setForm((f) => ({ ...f, text: p }))}
                  className="rounded-full border border-[var(--hairline)] px-2.5 py-1 text-[11px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">
                  {p}
                </button>
              ))}
            </div>
          </div>

          {clusterMode ? (
            <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
          ) : (
            <Field label="checkpoint">
              <select className="field-input" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
                <option value="">server default</option>
                {checkpoints.map((c) => <option key={c.path} value={c.path}>{c.run} / {c.name}</option>)}
              </select>
            </Field>
          )}

          <div className="grid grid-cols-2 gap-3">
            <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
            <Field label="ODE steps"><input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} /></Field>
            <Field label="frames"><input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} /></Field>
            {!clusterMode && <Field label="seed"><input type="number" className="field-input" value={form.seed} onChange={set("seed")} /></Field>}
          </div>
        </div>

        {/* collapsible constraint authoring — attaches to the same generate call */}
        <Collapsible title="Joint constraints" sub="fixed bend angles + hinge ranges" badge={nActive || null}>
          <div className="space-y-5">
            <ConstraintEditor joints={joints} pins={pins} setPins={setPins} />
            <RangeEditor joints={joints} ranges={ranges} setRanges={setRanges} />
          </div>
        </Collapsible>

        <button type="submit" className="btn-signal w-full" disabled={(clusterMode && (submitting || !checkpoint)) || (!clusterMode && loading)}>
          {clusterMode
            ? (submitting ? "SUBMITTING…" : nActive ? "▶  GENERATE (CONSTRAINED) ON CLUSTER" : "▶  GENERATE ON CLUSTER")
            : (loading ? "Generating…" : nActive ? "▶  GENERATE (CONSTRAINED)" : "▶  GENERATE MOTION")}
        </button>
        {clusterMode && !checkpoint && <p className="label text-center">pick a remote checkpoint first</p>}
      </form>

      <div className="surface space-y-4 p-6">
        {clusterMode ? (
          <VizJobResult job={job} error={jobError} submitting={submitting} emptyHint="Generated clips will appear here." />
        ) : (
          <>
            <MediaViewer url={result?.media_url} caption={result?.caption} loading={loading} error={error} />
            {localHist.length > 0 && (
              <div className="space-y-2">
                <h4 className="label">session history</h4>
                <div className="flex gap-2 overflow-x-auto pb-2">
                  {localHist.map((h, i) => (
                    <button key={i} type="button" onClick={() => setResult(h)} className="shrink-0 overflow-hidden rounded-md border border-[var(--hairline)] hover:border-[var(--signal)]">
                      {h.media_url.toLowerCase().endsWith(".mp4")
                        ? <video src={mediaUrl(h.media_url)} className="h-20 w-20 object-cover" muted />
                        : <img src={mediaUrl(h.media_url)} alt={h.caption} className="h-20 w-20 object-cover" />}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ───────────────────────────────────────── collapsible section with active badge

function Collapsible({ title, sub, badge, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="surface p-5">
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-center justify-between text-left">
        <div className="flex items-center gap-2.5">
          <span className="text-[var(--muted)]">{open ? "▾" : "▸"}</span>
          <div>
            <div className="display text-base font-bold text-white">{title}</div>
            {sub && <div className="label mt-0.5">{sub}</div>}
          </div>
        </div>
        {badge ? (
          <span className="rounded-full border border-[var(--signal)]/50 bg-[var(--signal-dim)] px-2 py-0.5 text-[11px] font-semibold text-[var(--signal)]">
            {badge} active
          </span>
        ) : (
          <span className="label">{open ? "hide" : "add"}</span>
        )}
      </button>
      {open && <div className="mt-4">{children}</div>}
    </section>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}
