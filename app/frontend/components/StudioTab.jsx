"use client";

// ── Constraint Studio (prototype / experimental tab) ───────────────────────
// A single authoring surface that fuses what today lives across the Constraints
// and Room tabs, plus a fast local evaluation loop — WITHOUT touching either
// existing tab. The vision (per the brainstorm):
//
//   • one 3D viewport where you CLICK a joint to select it (ConstraintStage),
//   • a contextual editor for the selected joint — pin (fixed angle) or hinge
//     (range) — authored with a drag DIAL of the relative angle, not raw axis
//     numbers,
//   • constraints kept compact (at most one pin + one hinge per joint) so the
//     view never gets packed,
//   • evaluation as inline FIELDS, not a tab: jitter / foot-skate / accel with
//     a live delta vs an unconstrained baseline, so you can see immediately
//     whether a constraint wrecked the motion.
//
// Euclidean (room) constraints are scaffolded as a collapsed section to merge
// in next; the headline of this prototype is joint authoring + live eval.

import { useEffect, useMemo, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import { ACTIVE } from "@/lib/useVizJob";
import { loadNpy } from "@/lib/npy";
import { analyzeMotion, jointHoldStability, childrenFromParents } from "@/lib/motionMetrics";
import { classifyJob, categoryMeta } from "@/lib/classifyJob";
import ConstraintStage from "./ConstraintStage";
import ConstraintAnalysis from "./ConstraintAnalysis";
import CourseLab from "./CourseLab";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import PhraseAttestation from "./PhraseAttestation";

const FALLBACK = [
  "pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
  "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
  "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
  "L_Wrist", "R_Wrist",
].map((name, index) => ({ index, name, parent: index === 0 ? -1 : 0 }));

let _pid = 0;
let _rid = 0;

// Approximate healthy ranges of motion for the bend (flexion) at each joint,
// in degrees. Used by the "anatomical limit" preset so "realistic knee" is one
// click. 0° = straight; values are deliberately generous.
const ANATOMICAL_BEND = {
  L_Knee: [0, 150], R_Knee: [0, 150],
  L_Elbow: [0, 145], R_Elbow: [0, 145],
  L_Hip: [0, 120], R_Hip: [0, 120],
  L_Shoulder: [0, 170], R_Shoulder: [0, 170],
  L_Ankle: [0, 50], R_Ankle: [0, 50],
  Neck: [0, 45], Spine1: [0, 30], Spine2: [0, 30], Spine3: [0, 30],
  L_Collar: [0, 30], R_Collar: [0, 30],
};

// A constraint is the BEND ANGLE at a joint (angle between its two bones, 0 =
// straight) — axis-free. A pin fixes it exactly; a range limits it to [min,max].
// `strength` (0..1) softens the hold; `ease_frames` ramps a windowed hold in/out.
function toConstraints(pins) {
  return pins.map((p) => ({
    joint: p.joint, bend_deg: Number(p.bend_deg),
    strength: p.strength == null ? 1 : Number(p.strength),
    ease_frames: Number(p.ease_frames) || 0,
    frame_start: Number(p.frame_start) || 0,
    frame_end: p.frame_end === "" ? null : Number(p.frame_end),
  }));
}
function toRanges(ranges) {
  return ranges.map((r) => ({
    joint: r.joint, bend_min: Number(r.bend_min), bend_max: Number(r.bend_max),
    strength: r.strength == null ? 1 : Number(r.strength),
    ease_frames: Number(r.ease_frames) || 0,
    frame_start: Number(r.frame_start) || 0,
    frame_end: r.frame_end === "" ? null : Number(r.frame_end),
  }));
}

// Inverse of toConstraints/toRanges — a loaded job's payload back into editor
// rows, so the constraints a clip was sampled WITH land in the editor with it
// (null frame_end is the "" blank = runs to the end).
function fromConstraints(arr = []) {
  return arr.map((c) => ({
    id: ++_pid, joint: c.joint, bend_deg: c.bend_deg ?? 90,
    strength: c.strength == null ? 1 : c.strength,
    ease_frames: c.ease_frames ?? 0,
    frame_start: c.frame_start ?? 0,
    frame_end: c.frame_end == null ? "" : c.frame_end,
  }));
}
function fromRanges(arr = []) {
  return arr.map((r) => ({
    id: ++_rid, joint: r.joint, bend_min: r.bend_min ?? 0, bend_max: r.bend_max ?? 90,
    strength: r.strength == null ? 1 : r.strength,
    ease_frames: r.ease_frames ?? 0,
    frame_start: r.frame_start ?? 0,
    frame_end: r.frame_end == null ? "" : r.frame_end,
  }));
}

function agoLabel(ts) {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export default function StudioTab({ checkpoints = [], clusterMode }) {
  const [meta, setMeta] = useState(FALLBACK);
  const [form, setForm] = useState({
    text: "a person waves their right hand", guidance: 6.5, num_steps: 50, seed: 0, num_frames: 100,
  });
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null); // remote run config (cluster mode)
  const [pins, setPins] = useState([]);
  const [ranges, setRanges] = useState([]);
  const [selected, setSelected] = useState(null);

  const [joints, setJoints] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [baseline, setBaseline] = useState(null); // metrics of last unconstrained sample
  const [baselineJoints, setBaselineJoints] = useState(null); // joints of last ◇ baseline
  const [vizJobs, setVizJobs] = useState([]);   // finished viz jobs with a joints .npy — loadable into the stage
  const [query, setQuery] = useState("");
  const [catFilter, setCatFilter] = useState("all");
  const [loadBusy, setLoadBusy] = useState(null); // key of the clip being loaded
  const [loaded, setLoaded] = useState(null);     // key of the clip in the stage
  const [, setFrame] = useState(0);
  const [busy, setBusy] = useState(false);
  const [busyState, setBusyState] = useState(null); // cluster job state while sampling
  const [error, setError] = useState(null);

  const jointNames = useMemo(() => meta.map((m) => m.name), [meta]);
  const parents = useMemo(() => meta.map((m) => m.parent ?? -1), [meta]);
  const children = useMemo(() => childrenFromParents(parents), [parents]);

  useEffect(() => {
    api.metaJoints().then((m) => m?.joints?.length && setMeta(m.joints)).catch(() => {});
    refreshVizJobs();
  }, []);

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  // ── load a finished viz render into the stage ─────────────────────────────
  // Any done viz job that pulled a joints .npy is loadable; compare renders
  // pair real-<cid>/gen-<cid> so gen goes into the viewport and GT becomes the
  // ◇ baseline overlay (same slots the local sampling loop uses).
  function refreshVizJobs() {
    api.jobs()
      .then((js) => setVizJobs(js.filter((j) => j.kind === "viz" && j.state === "done" && (j.outputs || []).some((o) => o.npy_url))))
      .catch(() => {});
  }

  const loadables = useMemo(() => vizJobs.flatMap((j) => {
    const p = j.params || {};
    const cons = p.constraints || [];
    const rngs = p.ranges || [];
    const byId = {};
    (j.outputs || []).filter((o) => o.npy_url).forEach((o) => {
      const name = o.npy_url.split("/").pop();
      const m = name.match(/^(real|gen)-(.+)\.npy$/);
      const role = m ? m[1] : "gen";
      const cid = m ? m[2] : name.replace(/\.npy$/, "");
      const g = (byId[cid] = byId[cid] || { key: `${j.id}::${cid}`, cid, caption: "", outs: {} });
      g.outs[role] = o;
      if (o.caption) g.caption = o.caption;
    });
    return Object.values(byId).map((g) => ({
      ...g,
      job: j,
      params: p,
      cat: classifyJob(j),
      at: j.submitted_at,
      constraints: cons,
      ranges: rngs,
      jointsTouched: [...new Set([...cons, ...rngs].map((c) => c.joint))],
      // everything the search box matches on, lowercased once
      haystack: [g.caption, p.prompts, p.text, j.mode, j.id, p.checkpoint,
        ...cons.map((c) => c.joint), ...rngs.map((r) => r.joint)]
        .filter(Boolean).join(" ").toLowerCase(),
    }));
  }), [vizJobs]);

  // filters: free-text over prompt/joints/checkpoint + a category chip row
  const cats = useMemo(() => [...new Set(loadables.map((g) => g.cat))], [loadables]);
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return loadables
      .filter((g) => (catFilter === "all" || g.cat === catFilter) && (!q || g.haystack.includes(q)))
      .sort((a, b) => (b.at || 0) - (a.at || 0));
  }, [loadables, query, catFilter]);

  // Loading a clip restores the whole authoring context, not just the motion:
  // the prompt + sampling settings it was made with, and — the point of the
  // studio — the constraints it was sampled UNDER, so you can tweak one pin and
  // resample instead of re-authoring from scratch.
  async function loadFromJob(g) {
    if (!g) return;
    setLoadBusy(g.key);
    setError(null);
    try {
      const main = g.outs.gen || Object.values(g.outs)[0];
      const npy = await loadNpy(mediaUrl(main.npy_url));
      setJoints(npy);
      setMetrics(analyzeMotion(npy, 20));
      if (g.outs.real && g.outs.gen) {
        const rnpy = await loadNpy(mediaUrl(g.outs.real.npy_url));
        setBaselineJoints(rnpy);
        setBaseline(analyzeMotion(rnpy, 20));
      } else {
        setBaselineJoints(null);
        setBaseline(null);
      }
      const p = g.params;
      setForm((f) => ({
        ...f,
        text: g.caption || (p.prompts ?? p.text ?? f.text).split("|")[0],
        guidance: p.guidance ?? f.guidance,
        num_steps: p.num_steps ?? f.num_steps,
        num_frames: p.num_frames ?? npy.shape?.[0] ?? f.num_frames,
        seed: p.seed ?? f.seed,
      }));
      if (p.checkpoint) setCheckpoint(p.checkpoint);
      setPins(fromConstraints(g.constraints));
      setRanges(fromRanges(g.ranges));
      setLoaded(g.key);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadBusy(null);
    }
  }

  // Cluster mode has no local model/checkpoints, so the studio samples through
  // a cluster viz job (mode=prompt): submit → poll to completion → load the
  // pulled joints .npy. Slower than the local loop (queue + GPU + rsync), but
  // it is the only sampling path that exists against the cluster.
  async function sampleClusterNpy(body) {
    let job = await api.submitViz({
      mode: "prompt", checkpoint, prompts: body.text,
      guidance: body.guidance, num_steps: body.num_steps, num_frames: body.num_frames,
      model_preset: presets?.model_preset, train_preset: presets?.train_preset,
      constraints: body.constraints, ranges: body.ranges,
    });
    setBusyState(job.state);
    while (ACTIVE.has(job.state)) {
      await new Promise((r) => setTimeout(r, 3000));
      job = await api.job(job.id);
      setBusyState(job.state);
    }
    if (job.state !== "done") throw new Error(job.error || `job ${job.state}`);
    const out = (job.outputs || []).find((o) => o.npy_url);
    if (!out) throw new Error("job finished but no joints .npy was pulled");
    return loadNpy(mediaUrl(out.npy_url));
  }

  async function sample({ asBaseline = false } = {}) {
    setBusy(true);
    setError(null);
    try {
      const body = {
        ...form,
        constraints: asBaseline ? [] : toConstraints(pins),
        ranges: asBaseline ? [] : toRanges(ranges),
      };
      let npy;
      if (clusterMode) {
        npy = await sampleClusterNpy(body);
      } else {
        if (checkpoint) body.checkpoint = checkpoint;
        const res = await api.generate(body);
        const npyUrl = res.joints_npy_url?.startsWith("http")
          ? res.joints_npy_url
          : `${apiBase()}${res.joints_npy_url}`;
        npy = await loadNpy(npyUrl);
      }
      const m = analyzeMotion(npy, 20);
      setMetrics(m);
      if (asBaseline) {
        // a baseline run is the unconstrained reference — keep its joints for the
        // free-vs-constrained overlay, but don't replace the constrained clip in
        // the viewport unless there's nothing shown yet.
        setBaseline(m);
        setBaselineJoints(npy);
        setJoints((cur) => cur ?? npy);
      } else {
        setJoints(npy);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setBusyState(null);
    }
  }

  // ── per-joint constraint helpers (≤1 pin + ≤1 hinge per joint) ────────────
  const selName = selected != null ? jointNames[selected] : null;
  const pin = pins.find((p) => p.joint === selName) || null;
  const hinge = ranges.find((r) => r.joint === selName) || null;

  const upsertPin = (patch) =>
    setPins((ps) => {
      const i = ps.findIndex((p) => p.joint === selName);
      if (i < 0) return [...ps, { id: ++_pid, joint: selName, bend_deg: 90, frame_start: 0, frame_end: "", ...patch }];
      const n = [...ps]; n[i] = { ...n[i], ...patch }; return n;
    });
  const removePin = () => setPins((ps) => ps.filter((p) => p.joint !== selName));
  const upsertHinge = (patch) =>
    setRanges((rs) => {
      const i = rs.findIndex((r) => r.joint === selName);
      if (i < 0) return [...rs, { id: ++_rid, joint: selName, bend_min: 0, bend_max: 90, frame_start: 0, frame_end: "", ...patch }];
      const n = [...rs]; n[i] = { ...n[i], ...patch }; return n;
    });
  const removeHinge = () => setRanges((rs) => rs.filter((r) => r.joint !== selName));

  return (
    <div className="space-y-5">
    <div className="grid gap-5 lg:grid-cols-[1.15fr_1fr]">
      {/* ── LEFT: viewport + eval fields ───────────────────────────────── */}
      <div className="space-y-4">
        <div className="surface p-4">
          <ConstraintStage
            joints={joints}
            fps={20}
            pins={pins}
            ranges={ranges}
            selected={selected}
            onSelect={setSelected}
            jointNames={jointNames}
            parents={parents}
            onFrameChange={setFrame}
          />
        </div>
        <EvalFields metrics={metrics} baseline={baseline} selected={selected} selName={selName}
          joints={joints} parents={parents} children={children} pin={pin} hinge={hinge} />
      </div>

      {/* ── RIGHT: sample controls + contextual editor + constraint list ── */}
      <div className="space-y-4">
        <section className="surface space-y-3 p-5">
          <label className="block">
            <span className="label mb-1.5 block">prompt</span>
            <textarea rows={2} className="field-input resize-none" value={form.text} onChange={set("text")} />
            <PhraseAttestation phrase={form.text} />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
            <Field label="ODE steps"><input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} /></Field>
            <Field label="frames"><input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} /></Field>
            <Field label="seed"><input type="number" className="field-input" value={form.seed} onChange={set("seed")} /></Field>
          </div>
          {clusterMode ? (
            <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
          ) : (
            <LocalCheckpointPicker checkpoints={checkpoints} value={checkpoint} onChange={setCheckpoint} />
          )}
          <div className="flex gap-2">
            <button onClick={() => sample()} disabled={busy || (clusterMode && !checkpoint)} className="btn-signal flex-1">
              {busy ? (busyState ? `${busyState} on cluster…` : "sampling…") : "▶ apply & resample"}
            </button>
            <button
              onClick={() => sample({ asBaseline: true })}
              disabled={busy || (clusterMode && !checkpoint)}
              title="Sample with NO constraints and store as the comparison baseline"
              className="rounded-md border border-[var(--hairline)] px-3 py-2 text-[12px] text-[var(--muted)] hover:border-[var(--hairline-strong)] hover:text-slate-200"
            >
              ◇ baseline
            </button>
          </div>
          {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
          {clusterMode && (
            <p className="label normal-case tracking-normal">
              {checkpoint
                ? "samples run as a cluster viz job (queue → GPU → pull) — expect a couple of minutes per sample"
                : "pick a training run + checkpoint above — the studio needs a trained model to sample from"}
            </p>
          )}
        </section>

        {/* load an existing clip (any finished viz job with joints) into the stage */}
        <ClipLibrary
          clips={shown}
          total={loadables.length}
          cats={cats}
          query={query}
          setQuery={setQuery}
          catFilter={catFilter}
          setCatFilter={setCatFilter}
          onLoad={loadFromJob}
          loadBusy={loadBusy}
          loaded={loaded}
          onRefresh={refreshVizJobs}
        />

        {/* contextual joint editor */}
        <JointEditor
          selName={selName}
          pin={pin}
          hinge={hinge}
          numFrames={form.num_frames}
          upsertPin={upsertPin}
          removePin={removePin}
          upsertHinge={upsertHinge}
          removeHinge={removeHinge}
        />

        {/* compact constraint list */}
        <ConstraintList pins={pins} ranges={ranges} jointNames={jointNames}
          onPick={(name) => setSelected(jointNames.findIndex((n) => n === name))}
          onRemovePin={(name) => setPins((ps) => ps.filter((p) => p.joint !== name))}
          onRemoveHinge={(name) => setRanges((rs) => rs.filter((r) => r.joint !== name))} />
      </div>
    </div>

    {/* ── constraint analysis: realized bend vs target, over time ───────── */}
    <ConstraintAnalysis
      joints={joints}
      baselineJoints={baselineJoints}
      pins={pins}
      ranges={ranges}
      parents={parents}
      children={children}
      jointNames={jointNames}
      fps={20}
    />

    {/* ── euclidean obstacle courses: room + objects + trajectory, scored ── */}
    <CourseLab clusterMode={clusterMode} />
    </div>
  );
}

function apiBase() {
  // mirror lib/api API_BASE without re-importing the symbol name
  return (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");
}

// ───────────────────────────────────────── local run → checkpoint picker
// Local /generate takes any training checkpoint (<run>/checkpoints/*.pt under
// RMG_RUNS_DIR). Grouping by run makes "which model am I sampling?" explicit —
// "server default" is just the newest checkpoint on disk (or RMG_CHECKPOINT).
function LocalCheckpointPicker({ checkpoints = [], value, onChange }) {
  const byRun = useMemo(() => {
    const m = {};
    for (const c of checkpoints) (m[c.run] = m[c.run] || []).push(c);
    return m;
  }, [checkpoints]);
  const runs = Object.keys(byRun);
  const [run, setRun] = useState("");

  // keep the run dropdown in sync when the checkpoint is set from outside
  useEffect(() => {
    if (!value) return;
    const c = checkpoints.find((x) => x.path === value);
    if (c && c.run !== run) setRun(c.run);
  }, [value, checkpoints]); // eslint-disable-line react-hooks/exhaustive-deps

  if (runs.length === 0) {
    return (
      <div>
        <div className="label mb-1.5">checkpoint</div>
        <p className="rounded-md border border-dashed border-[var(--hairline)] px-3 py-2 text-[11px] text-[var(--muted)]">
          no local checkpoints found — sampling uses the server default
          (RMG_CHECKPOINT). Copy a run with <span className="font-mono">checkpoints/</span> into
          RMG_RUNS_DIR to pick one here.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3">
      <Field label="training run">
        <select className="field-input" value={run}
          onChange={(e) => { setRun(e.target.value); onChange(""); }}>
          <option value="">server default</option>
          {runs.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="checkpoint">
        <select className="field-input" value={value} disabled={!run}
          onChange={(e) => onChange(e.target.value)}>
          <option value="">{run ? "select…" : "latest on disk"}</option>
          {(byRun[run] || []).map((c) => <option key={c.path} value={c.path}>{c.name}</option>)}
        </select>
      </Field>
    </div>
  );
}

// ───────────────────────────────────────── clip library (load into stage)
// Every finished viz output with joints is a clip you can pull into the studio.
// Each row states what the motion IS — prompt, category, ω/steps/frames/seed,
// and the constraints it was sampled under — so you pick by motion, not by an
// opaque job id. Search matches prompt text, joint names and checkpoint.
function ClipLibrary({ clips, total, cats, query, setQuery, catFilter, setCatFilter, onLoad, loadBusy, loaded, onRefresh }) {
  return (
    <section className="surface space-y-3 p-5">
      <div className="flex items-center justify-between">
        <span className="label">clip library · {clips.length}{clips.length !== total ? ` / ${total}` : ""}</span>
        <button onClick={onRefresh} className="btn-ghost text-[11px]">↻ REFRESH</button>
      </div>

      <input
        className="field-input"
        placeholder="search prompt, joint, checkpoint…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {cats.length > 1 && (
        <div className="flex flex-wrap gap-1.5">
          {["all", ...cats].map((c) => {
            const meta = c === "all" ? { label: "all", color: "var(--muted)" } : categoryMeta(c);
            const on = catFilter === c;
            return (
              <button key={c} onClick={() => setCatFilter(c)}
                className={`rounded-full border px-2.5 py-0.5 text-[11px] transition ${on ? "text-slate-100" : "border-[var(--hairline)] text-[var(--muted)] hover:text-slate-300"}`}
                style={on ? { borderColor: meta.color, background: "rgba(255,255,255,0.04)" } : undefined}>
                {meta.label}
              </button>
            );
          })}
        </div>
      )}

      {clips.length === 0 ? (
        <p className="text-[12px] text-[var(--muted)]">
          {total === 0
            ? "No finished renders with joints yet — generate a clip (here or in Prompt/Scene) and it shows up."
            : "No clips match this filter."}
        </p>
      ) : (
        <ul className="max-h-[22rem] space-y-1.5 overflow-y-auto pr-1">
          {clips.map((g) => <ClipRow key={g.key} g={g} onLoad={onLoad} busy={loadBusy === g.key} active={loaded === g.key} />)}
        </ul>
      )}
      <p className="label normal-case tracking-normal">
        loading restores the prompt, sampling settings and the clip's constraints; compare renders put GT on the ◇ baseline overlay
      </p>
    </section>
  );
}

function ClipRow({ g, onLoad, busy, active }) {
  const p = g.params;
  const meta = categoryMeta(g.cat);
  const nCon = g.constraints.length + g.ranges.length;
  const facts = [
    p.guidance != null && `ω ${p.guidance}`,
    p.num_steps != null && `${p.num_steps} steps`,
    p.num_frames != null && `${p.num_frames}f`,
    p.seed != null && `seed ${p.seed}`,
    g.outs.real && "GT + gen",
  ].filter(Boolean);

  return (
    <li>
      <button onClick={() => onLoad(g)} disabled={busy}
        className={`w-full space-y-1.5 rounded-md border px-3 py-2 text-left transition ${
          active ? "border-[var(--signal)] bg-[var(--signal-dim)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"
        }`}>
        <div className="flex items-start justify-between gap-2">
          <span className="text-[12px] leading-snug text-slate-200">
            {g.caption || <span className="font-mono text-[var(--muted)]">{g.cid}</span>}
          </span>
          <span className="shrink-0 rounded border px-1.5 text-[9px] uppercase tracking-wider"
            style={{ borderColor: meta.color, color: meta.color }}>
            {meta.label}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[10px] text-[var(--muted)]">
          {facts.map((f) => <span key={f}>{f}</span>)}
          <span className="ml-auto">{busy ? "loading…" : agoLabel(g.at)}</span>
        </div>
        {nCon > 0 && (
          <div className="flex flex-wrap gap-1">
            <span className="text-[10px] text-[var(--amber)]">{nCon} constraint{nCon > 1 ? "s" : ""}</span>
            {g.jointsTouched.slice(0, 4).map((j) => (
              <span key={j} className="rounded border border-[var(--hairline)] px-1 font-mono text-[9px] text-[var(--muted)]">{j}</span>
            ))}
            {g.jointsTouched.length > 4 && <span className="text-[9px] text-[var(--muted)]">+{g.jointsTouched.length - 4}</span>}
          </div>
        )}
      </button>
    </li>
  );
}

// ───────────────────────────────────────── contextual per-joint editor
function JointEditor({ selName, pin, hinge, numFrames, upsertPin, removePin, upsertHinge, removeHinge }) {
  const [tab, setTab] = useState("pin");
  if (!selName) {
    return (
      <section className="surface p-5">
        <div className="label">joint editor</div>
        <p className="mt-3 text-[13px] text-[var(--muted)]">Click a joint in the viewport to pin its bend or set a bend range.</p>
      </section>
    );
  }
  return (
    <section className="surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="label">editing joint</div>
          <div className="display text-base font-bold text-[var(--signal)]">{selName}</div>
        </div>
        <div className="flex gap-1">
          {["pin", "range"].map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`rounded-md border px-3 py-1 text-[11px] font-semibold uppercase tracking-widest transition ${
                tab === t ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border-[var(--hairline)] text-[var(--muted)] hover:text-slate-200"
              }`}>
              {t}{((t === "pin" && pin) || (t === "range" && hinge)) ? " •" : ""}
            </button>
          ))}
        </div>
      </div>

      {tab === "pin" ? (
        <PinEditor pin={pin} numFrames={numFrames} upsertPin={upsertPin} removePin={removePin} />
      ) : (
        <HingeEditor selName={selName} hinge={hinge} numFrames={numFrames} upsertHinge={upsertHinge} removeHinge={removeHinge} />
      )}
    </section>
  );
}

function PinEditor({ pin, numFrames, upsertPin, removePin }) {
  const bend = pin?.bend_deg ?? 90;
  return (
    <div className="space-y-4">
      <p className="text-[12px] text-[var(--muted)]">
        Holds this joint at a fixed bend angle each sampling step — the angle between its two bones
        (0° = straight). Twist and bend direction stay free.
      </p>
      <div className="flex items-center gap-5">
        <AngleDial value={Number(bend)} onChange={(v) => upsertPin({ bend_deg: v })} color="#fbbf24" min={0} max={180} />
        <div className="flex-1 space-y-3">
          <div>
            <div className="mb-1 flex justify-between"><span className="label">bend angle</span>
              <span className="font-mono text-sm text-[var(--amber)]">{Number(bend)}°</span></div>
            <input type="range" min="0" max="180" step="5" value={bend}
              onChange={(e) => upsertPin({ bend_deg: Number(e.target.value) })} className="w-full accent-[var(--amber)]" />
          </div>
        </div>
      </div>
      <HoldStrength c={pin} onChange={(patch) => upsertPin(patch)} />
      <FrameWindow c={pin} numFrames={numFrames} onChange={(patch) => upsertPin(patch)} />
      {pin && (
        <button onClick={removePin} className="text-[11px] text-[var(--muted)] hover:text-[var(--amber)]">remove pin</button>
      )}
    </div>
  );
}

// Soft-hold controls shared by pin & hinge: stiffness (0..1) + window ease ramp.
function HoldStrength({ c, onChange }) {
  const strength = c?.strength == null ? 1 : Number(c.strength);
  const ease = Number(c?.ease_frames) || 0;
  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <div className="mb-1 flex justify-between">
          <span className="label">hold strength</span>
          <span className="font-mono text-[12px] text-[var(--signal)]">{strength >= 1 ? "hard" : strength.toFixed(2)}</span>
        </div>
        <input type="range" min="0" max="1" step="0.05" value={strength}
          onChange={(e) => onChange({ strength: Number(e.target.value) })} className="w-full accent-[var(--signal)]" />
        <p className="label mt-1">1 = snap to target · &lt;1 lets the prompt fight back</p>
      </div>
      <label className="block">
        <span className="label mb-1 block">ease frames</span>
        <input type="number" min="0" className="field-input" value={ease}
          onChange={(e) => onChange({ ease_frames: Number(e.target.value) })} />
        <p className="label mt-1">ramp the hold in/out at the window edges</p>
      </label>
    </div>
  );
}

function HingeEditor({ selName, hinge, numFrames, upsertHinge, removeHinge }) {
  const mn = hinge?.bend_min ?? 0;
  const mx = hinge?.bend_max ?? 90;
  const anat = ANATOMICAL_BEND[selName];
  return (
    <div className="space-y-4">
      <p className="text-[12px] text-[var(--muted)]">
        Limits the joint's bend angle to [min, max] each sampling step (0° = straight). The motion
        bends freely within the range; anything outside is projected back in.
      </p>
      {anat && (
        <button
          onClick={() => upsertHinge({ bend_min: anat[0], bend_max: anat[1] })}
          className="rounded-md border border-[var(--signal)]/50 bg-[var(--signal-dim)] px-3 py-1.5 text-[11px] font-semibold text-[var(--signal)] hover:bg-[var(--signal)]/15"
        >
          use anatomical limit ({anat[0]}–{anat[1]}°)
        </button>
      )}
      <div className="grid grid-cols-2 gap-3">
        <label className="block"><span className="label mb-1 block">min bend (°)</span>
          <input type="number" min="0" max="180" className="field-input" value={mn} onChange={(e) => upsertHinge({ bend_min: Number(e.target.value) })} /></label>
        <label className="block"><span className="label mb-1 block">max bend (°)</span>
          <input type="number" min="0" max="180" className="field-input" value={mx} onChange={(e) => upsertHinge({ bend_max: Number(e.target.value) })} /></label>
      </div>
      <HoldStrength c={hinge} onChange={(patch) => upsertHinge(patch)} />
      <FrameWindow c={hinge} numFrames={numFrames} onChange={(patch) => upsertHinge(patch)} />
      {hinge && (
        <button onClick={removeHinge} className="text-[11px] text-[var(--muted)] hover:text-[var(--amber)]">remove hinge</button>
      )}
    </div>
  );
}

function FrameWindow({ c, numFrames, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <label className="block"><span className="label mb-1 block">frame start</span>
        <input type="number" min="0" max={numFrames} className="field-input"
          value={c?.frame_start ?? 0} onChange={(e) => onChange({ frame_start: e.target.value })} /></label>
      <label className="block"><span className="label mb-1 block">frame end <span className="text-[var(--muted)]">(blank = all)</span></span>
        <input type="number" min="0" max={numFrames} className="field-input" placeholder="all"
          value={c?.frame_end ?? ""} onChange={(e) => onChange({ frame_end: e.target.value })} /></label>
    </div>
  );
}

// ───────────────────────────────────────── drag-dial for "relative angle"
function AngleDial({ value, onChange, color = "#22d3ee", size = 96, min = -180, max = 180 }) {
  const r = size / 2;
  const rad = (Number(value) * Math.PI) / 180;
  // 0° points up; clockwise positive. Convert to screen coords.
  const hx = r + (r - 12) * Math.sin(rad);
  const hy = r - (r - 12) * Math.cos(rad);

  function fromEvent(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = e.clientX - rect.left - r;
    const cy = e.clientY - rect.top - r;
    let deg = (Math.atan2(cx, -cy) * 180) / Math.PI; // up = 0, clockwise +
    deg = Math.max(min, Math.min(max, Math.round(deg / 5) * 5));
    onChange(deg);
  }
  return (
    <svg width={size} height={size} className="cursor-pointer select-none"
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); fromEvent(e); }}
      onPointerMove={(e) => { if (e.buttons) fromEvent(e); }}>
      <circle cx={r} cy={r} r={r - 4} fill="rgba(255,255,255,0.02)" stroke="var(--hairline)" strokeWidth="1.5" />
      {[0, 90, 180, 270].map((t) => {
        const tr = (t * Math.PI) / 180;
        return <line key={t} x1={r + (r - 8) * Math.sin(tr)} y1={r - (r - 8) * Math.cos(tr)}
          x2={r + (r - 4) * Math.sin(tr)} y2={r - (r - 4) * Math.cos(tr)} stroke="var(--muted)" strokeWidth="1" />;
      })}
      <line x1={r} y1={r} x2={hx} y2={hy} stroke={color} strokeWidth="2.5" />
      <circle cx={hx} cy={hy} r="6" fill={color} />
      <circle cx={r} cy={r} r="3" fill={color} />
      <text x={r} y={r + 4} textAnchor="middle" className="font-mono" fontSize="12" fill={color}>{Math.round(value)}°</text>
    </svg>
  );
}

// ───────────────────────────────────────── compact constraint list
function ConstraintList({ pins, ranges, jointNames, onPick, onRemovePin, onRemoveHinge }) {
  const rows = [
    ...pins.map((p) => ({ key: `p${p.id}`, joint: p.joint, kind: "pin", color: "#fbbf24",
      desc: `bend ${p.bend_deg}° · ${winLabel(p)}`, remove: () => onRemovePin(p.joint) })),
    ...ranges.map((r) => ({ key: `r${r.id}`, joint: r.joint, kind: "range", color: "#34d399",
      desc: `bend [${r.bend_min},${r.bend_max}]° · ${winLabel(r)}`, remove: () => onRemoveHinge(r.joint) })),
  ];
  return (
    <section className="surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="label">active constraints · {rows.length}</span>
      </div>
      {rows.length === 0 ? (
        <p className="text-[12px] text-[var(--muted)]">None yet — select a joint and add a pin or hinge.</p>
      ) : (
        <ul className="space-y-1.5">
          {rows.map((row) => (
            <li key={row.key} className="flex items-center justify-between rounded-md border border-[var(--hairline)] px-3 py-2">
              <button onClick={() => onPick(row.joint)} className="flex items-center gap-2 text-left">
                <span className="h-2 w-2 rounded-full" style={{ background: row.color }} />
                <span className="font-mono text-[12px] text-slate-200">{row.joint}</span>
                <span className="rounded border border-[var(--hairline)] px-1.5 text-[9px] uppercase tracking-wider text-[var(--muted)]">{row.kind}</span>
                <span className="font-mono text-[11px] text-[var(--muted)]">{row.desc}</span>
              </button>
              <button onClick={row.remove} className="text-[11px] text-[var(--muted)] hover:text-[var(--amber)]">✕</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
function winLabel(c) {
  const s = Number(c.frame_start) || 0;
  return c.frame_end === "" || c.frame_end == null ? `${s}→end` : `${s}→${c.frame_end}`;
}

// ───────────────────────────────────────── evaluation FIELDS (not a tab)
function EvalFields({ metrics, baseline, selected, selName, joints, parents, children, pin, hinge }) {
  if (!metrics) {
    return (
      <section className="surface p-5">
        <div className="label">evaluation</div>
        <p className="mt-3 text-[13px] text-[var(--muted)]">Sample a clip to see live quality metrics. Hit ◇ baseline first to capture an unconstrained reference, then compare.</p>
      </section>
    );
  }
  const win = selName && (pin || hinge)
    ? { start: Number((pin || hinge).frame_start) || 0,
        end: (pin || hinge).frame_end === "" || (pin || hinge).frame_end == null ? metrics.T : Number((pin || hinge).frame_end) }
    : null;
  const hold = selected != null && win ? jointHoldStability(joints, selected, parents, children, win) : null;
  const selJerk = selected != null ? metrics.perJointJerk[selected] : null;
  const baseSelJerk = selected != null && baseline ? baseline.perJointJerk[selected] : null;

  return (
    <section className="surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="label">evaluation · live</span>
        <span className="text-[10px] text-[var(--muted)]">{baseline ? "Δ vs baseline" : "no baseline — hit ◇"}</span>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Metric label="jitter" unit="m/s³" value={metrics.jitter} base={baseline?.jitter} lowerBetter />
        <Metric label="foot-skate" unit="cm/s" value={metrics.footSkate} base={baseline?.footSkate} lowerBetter />
        <Metric label="accel" unit="m/s²" value={metrics.accel} base={baseline?.accel} lowerBetter />
      </div>

      <div className="mt-4">
        <div className="mb-1 flex items-center justify-between">
          <span className="label">global jerk over time</span>
          <span className="text-[10px] text-[var(--muted)]">spikes ≈ constraint snaps</span>
        </div>
        <Sparkline series={metrics.jerkSeries} />
      </div>

      {selName && (
        <div className="mt-4 rounded-md border border-[var(--hairline)] p-3">
          <div className="label mb-2">selected · {selName}</div>
          <div className="grid grid-cols-2 gap-3">
            <Metric label="joint jitter" unit="m/s³" value={selJerk} base={baseSelJerk} lowerBetter small />
            <div>
              <div className="label mb-1">hold stability</div>
              {hold ? (
                <div className="font-mono text-[13px]" style={{ color: hold.stdDeg < 1 ? "var(--signal)" : hold.stdDeg < 5 ? "var(--amber)" : "#f87171" }}>
                  ±{hold.stdDeg.toFixed(2)}°
                  <span className="ml-1 text-[10px] text-[var(--muted)]">{hold.stdDeg < 1 ? "held" : "drifting"}</span>
                </div>
              ) : (
                <div className="font-mono text-[12px] text-[var(--muted)]">{pin || hinge ? "end joint — n/a" : "no constraint"}</div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({ label, unit, value, base, lowerBetter, small }) {
  const v = value ?? 0;
  let delta = null;
  if (base != null && base > 1e-9) delta = ((v - base) / base) * 100;
  const worse = delta != null && (lowerBetter ? delta > 0 : delta < 0);
  const dcol = delta == null ? "var(--muted)" : Math.abs(delta) < 2 ? "var(--muted)" : worse ? "#f87171" : "var(--signal)";
  return (
    <div>
      <div className="label mb-1">{label}</div>
      <div className={`font-mono ${small ? "text-[13px]" : "text-lg"} text-slate-100`}>
        {v.toFixed(small ? 2 : 1)}<span className="ml-1 text-[10px] text-[var(--muted)]">{unit}</span>
      </div>
      {delta != null && (
        <div className="font-mono text-[11px]" style={{ color: dcol }}>
          {delta > 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(0)}%
        </div>
      )}
    </div>
  );
}

function Sparkline({ series, height = 36 }) {
  const n = series.length;
  let peak = 1e-6;
  for (let i = 0; i < n; i++) peak = Math.max(peak, series[i]);
  const pts = Array.from(series, (v, i) => `${(i / Math.max(1, n - 1)) * 100},${height - (v / peak) * height}`).join(" ");
  return (
    <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="h-9 w-full rounded bg-black/30">
      <polyline points={pts} fill="none" stroke="var(--signal)" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}
