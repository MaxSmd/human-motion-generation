"use client";

// ── Constraint analysis ────────────────────────────────────────────────────
//
// Everything about "we asked the sampler to hold a joint — what did that cost?"
// in one place. Two questions, two sections:
//
//   01 FIDELITY — did the constraint hold, and what did the motion pay for it?
//      Realized bend vs target, plus jerk / foot-skate / accel against the GT
//      reference. This is the diagnosis.
//
//   02 TEXT ABLATION — the treatment. A pin is enforced geometrically but never
//      spoken, so the text prior keeps proposing an unconstrained motion and the
//      projection keeps overwriting it. The clip shakes because the sampler is
//      fighting itself. If we SAY the constraint in the prompt — in language the
//      encoder knows — the prior should propose it and the fight should stop.
//      Run as a 4×2 factorial so the mechanism (does text move the prior?) is
//      separable from the outcome (does enforcement get cheaper?).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import LineChart from "./LineChart";
import BarChart from "./BarChart";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import {
  FALLBACK_JOINTS, ConstraintEditor, RangeEditor, pinsToConstraints, rangesToPayload, newRange,
} from "./ConstraintsTab";
import { loadNpy } from "@/lib/npy";
import { bendSeries, constraintSatisfaction, constraintReport, childrenFromParents } from "@/lib/motionMetrics";
import { TEXT_MODES, CELLS, cellId, buildTextModes, normalizeSpec } from "@/lib/constraintText";
import { Section, Chart, EmptyHint, SIGNAL, AMBER, GREEN, ROSE, SLATE } from "./AnalysisKit";
import PhraseAttestation from "./PhraseAttestation";
import { TableTools } from "./ExportButtons";
import { useLightbox } from "./Lightbox";

// Mean ‖Δ³x‖ jerk of HumanML3D ground-truth clips under the backend's metric
// (analysis/joints.py). Not a pass/fail threshold — the floor a generated clip
// is doing well to approach, drawn on the charts so "0.14" has a scale.
const GT_JERK = 0.0092;

const TOL_DEG = 5; // a frame is "held" within ±5° of the pin / hinge band

export default function ConstraintAnalysisTab() {
  return (
    <div className="space-y-8">
      <Section n="01" title="Constraint fidelity"
        sub="did the pin/hinge hold? — realized bend vs target + what the motion paid">
        <ConstraintFidelity />
      </Section>
      <Section n="02" title="Constraint → text ablation"
        sub="does saying the constraint in the prompt make it cheaper to enforce?">
        <TextAblation />
      </Section>
    </div>
  );
}

// ── shared: joint metadata ──────────────────────────────────────────────────

function useJointMeta() {
  const [meta, setMeta] = useState(FALLBACK_JOINTS);
  useEffect(() => {
    api.metaJoints().then((m) => m?.joints?.length && setMeta(m.joints)).catch(() => {});
  }, []);
  const nameToIdx = useMemo(() => {
    const m = {};
    meta.forEach((j) => { m[j.name] = j.index; });
    return m;
  }, [meta]);
  const parents = useMemo(() => meta.map((j) => j.parent ?? -1), [meta]);
  const children = useMemo(() => childrenFromParents(parents), [parents]);
  return { meta, nameToIdx, parents, children };
}

// Specs off a viz job's params → the normalized shape the verbalizer/analysis use.
function specsOf(params = {}) {
  return [...(params.constraints || []), ...(params.ranges || [])];
}

// The backend analysis endpoint resolves an .npy by (owner job, name). For a
// fused viz job the media lives under the LEAD's dir, so the owner is whatever
// `/media/jobs/<owner>/<file>.npy` encodes — NOT the follower's own job id.
function npyRef(url = "") {
  const parts = url.split("/");
  return { name: parts.pop(), owner: parts.pop() };
}

// The PREDICTED clip's output — the sampled `gen-*` npy. A job dir can also carry
// a `gt`/`real-*` reference file (the real motion for the base prompt, banked
// once per study), and it often sorts FIRST — so `.find(o => o.npy_url)` would
// analyse the wrong clip and, since that reference is identical across cells,
// show one result duplicated everywhere. Pick the prediction explicitly:
// kind==="pred", else the `gen-` file, else fall back to the first npy so nothing
// regresses on old jobs.
function predOutput(outputs = []) {
  const withNpy = outputs.filter((o) => o.npy_url);
  return (
    withNpy.find((o) => o.kind === "pred") ||
    withNpy.find((o) => (o.npy_url.split("/").pop() || "").startsWith("gen-")) ||
    withNpy.find((o) => o.kind !== "gt") ||
    withNpy[0] ||
    null
  );
}

// The ground-truth reference banked alongside a clip — the real `real-*` motion
// the generation is being compared against. One per study (identical across the
// cells), so any cell's copy is the reference.
function gtOutput(outputs = []) {
  const withNpy = outputs.filter((o) => o.npy_url);
  return (
    withNpy.find((o) => o.kind === "gt") ||
    withNpy.find((o) => (o.npy_url.split("/").pop() || "").startsWith("real-")) ||
    null
  );
}

// GT reference gets its own bright, near-white line so it reads as "reality" and
// never collides with a text-mode hue (slate/amber/cyan/green) in the overlays.
const GT_COLOR = "#f8fafc";

// A bend series (Float array, NaN where unrecoverable) → [frame, value] points,
// NaN frames dropped, for the LineChart.
function seriesToPoints(series) {
  return series
    ? Array.from(series, (v, f) => [f, Number.isNaN(v) ? null : v]).filter(([, v]) => v != null)
    : [];
}

// Realized bend vs requested target for one clip.
function measureSpec(joints, spec, { nameToIdx, parents, children }) {
  const T = joints.shape[0];
  const n = normalizeSpec(spec, T);
  const idx = nameToIdx[n.joint];
  const raw = idx == null ? null : bendSeries(joints, idx, parents, children);
  if (!raw) return { ...n, series: null, sat: null, report: null };
  const target = n.kind === "pin" ? { bend: n.lo } : { min: n.lo, max: n.hi };
  const win = { start: n.start, end: n.end };
  return {
    ...n,
    target,
    series: raw,
    sat: constraintSatisfaction(raw, win, target, TOL_DEG),
    report: constraintReport(raw, win, target, TOL_DEG),
  };
}

// ── 01 · fidelity ───────────────────────────────────────────────────────────

function ConstraintFidelity() {
  const { nameToIdx, parents, children } = useJointMeta();
  const [jobs, setJobs] = useState([]);
  const [sel, setSel] = useState("");
  const [traces, setTraces] = useState(null);
  const [cost, setCost] = useState(null);
  // GT reference banked with the clip: bend at each constrained joint (so the
  // chart shows what the real motion did there) + its cost metrics.
  const [gt, setGt] = useState(null); // { bendByJoint: {joint: points}, cost }
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.jobs().then((js) => setJobs(js.filter((j) =>
      j.state === "done" &&
      (j.params?.constraints?.length || j.params?.ranges?.length) &&
      (j.outputs || []).some((o) => o.npy_url)
    ))).catch(() => {});
  }, []);

  const selJob = jobs.find((j) => j.id === sel);

  useEffect(() => {
    if (!selJob) { setTraces(null); setCost(null); setGt(null); return; }
    const out = predOutput(selJob.outputs);
    if (!out) { setTraces(null); setCost(null); setGt(null); return; }
    setLoading(true); setError(null);
    (async () => {
      try {
        const { name, owner } = npyRef(out.npy_url);
        const joints = await loadNpy(mediaUrl(out.npy_url));
        const specs = specsOf(selJob.params);
        setTraces(specs.map((s) => measureSpec(joints, s, { nameToIdx, parents, children })));
        // jerk / foot-skate from the backend so the numbers are in the same
        // units as the GT reference (client-side jerk is dt-scaled — different scale)
        try { setCost(await api.analysisNpy(owner, name)); } catch { setCost(null); }

        // The GT the clip was compared against — overlay its bend at each
        // constrained joint and its cost, so "did the constraint hold?" is read
        // against what the real motion actually did there.
        const gout = gtOutput(selJob.outputs);
        if (gout) {
          try {
            const gjoints = await loadNpy(mediaUrl(gout.npy_url));
            const bendByJoint = {};
            for (const s of specs) {
              const m = measureSpec(gjoints, s, { nameToIdx, parents, children });
              if (m.series) bendByJoint[m.joint] = seriesToPoints(m.series);
            }
            const g = npyRef(gout.npy_url);
            let gcost = null;
            try { gcost = await api.analysisNpy(g.owner, g.name); } catch { /* older */ }
            setGt({ bendByJoint, cost: gcost });
          } catch { setGt(null); }
        } else { setGt(null); }
      } catch (e) { setError(e.message); setTraces(null); setCost(null); setGt(null); }
      finally { setLoading(false); }
    })();
  }, [sel, nameToIdx, parents, children]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="surface p-5">
      <label className="mb-4 block max-w-2xl">
        <span className="label mb-1.5 block">constrained clip (a Create ▸ prompt job with pins / hinge ranges)</span>
        <select className="field-input" value={sel} onChange={(e) => setSel(e.target.value)}>
          <option value="">select constrained clip…</option>
          {jobs.map((j) => {
            const n = specsOf(j.params).length;
            const cap = j.outputs?.[0]?.caption || j.params?.prompts || j.run_name;
            return <option key={j.id} value={j.id}>{`${cap?.slice(0, 46)} · ${n} constraint${n === 1 ? "" : "s"}`}</option>;
          })}
        </select>
      </label>

      {jobs.length === 0 && (
        <EmptyHint>
          No constrained clips yet — author pins / hinge ranges in <span className="text-slate-300">Create ▸ Prompt</span> and generate one.
        </EmptyHint>
      )}
      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {loading && <p className="text-[12px] text-[var(--muted)]">recovering bend angles…</p>}

      {cost && <CostStrip cost={cost} gt={gt?.cost} />}

      {traces && traces.length > 0 && (
        <div className="mt-5 grid gap-6 lg:grid-cols-2">
          {traces.map((t, i) => <BendTrace key={i} t={t} gtPoints={gt?.bendByJoint?.[t.joint]} />)}
        </div>
      )}
      {traces && (
        <p className="label mt-3 normal-case tracking-normal">
          shaded band = active frame window · dashed line = pin target · green band = hinge range
          {gt ? " · white line = ground truth" : ""}.
          A trace that rides its target while the jerk above stays near GT means the constraint was cheap;
          held-but-jittery means the prior was fighting it — see §02.
        </p>
      )}
    </div>
  );
}

// jerk / foot-skate / accel against the GT reference — the cost side. When the
// clip's own GT is loaded, show it alongside so the ratio is against THIS motion's
// real dynamics, not just the corpus-wide GT_JERK floor.
function CostStrip({ cost, gt }) {
  const ratio = cost.jerk_mean != null ? cost.jerk_mean / GT_JERK : null;
  const color = ratio == null ? "var(--muted)" : ratio <= 2.5 ? GREEN : ratio <= 6 ? AMBER : ROSE;
  const f4 = (v) => (v == null ? "—" : v.toFixed(4));
  return (
    <div className="rounded-lg border border-[var(--hairline)] bg-ink px-4 py-3">
      <div className="flex flex-wrap items-baseline gap-x-8 gap-y-3">
        <Metric label="jerk ‖Δ³x‖" value={cost.jerk_mean?.toFixed(4) ?? "—"} color={color}
          note={ratio ? `${ratio.toFixed(1)}× ground truth (${GT_JERK})` : null} />
        <Metric label="foot-skate" value={cost.foot_skate_mean != null ? `${cost.foot_skate_mean.toFixed(3)} m/s` : "—"} />
        <Metric label="accel ‖Δ²x‖" value={cost.accel_mean?.toFixed(4) ?? "—"} />
        <Metric label="frames" value={cost.frames ?? "—"} />
        {gt && (
          <div className="ml-auto flex flex-wrap items-baseline gap-x-6 gap-y-1 border-l border-[var(--hairline)] pl-6">
            <div className="label" style={{ color: GT_COLOR }}>GT reference</div>
            <Metric label="jerk" value={f4(gt.jerk_mean)} color={GT_COLOR} />
            <Metric label="foot-skate" value={gt.foot_skate_mean != null ? `${gt.foot_skate_mean.toFixed(3)}` : "—"} color={GT_COLOR} />
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value, color = "#f1f5f9", note }) {
  return (
    <div>
      <div className="label mb-0.5">{label}</div>
      <div className="font-mono text-[15px]" style={{ color }}>{value}</div>
      {note && <div className="text-[10px] text-[var(--muted)]">{note}</div>}
    </div>
  );
}

function BendTrace({ t, gtPoints }) {
  const points = seriesToPoints(t.series);
  const heldColor = !t.sat ? SLATE : t.sat.frac > 0.8 ? GREEN : t.sat.frac > 0.5 ? AMBER : ROSE;
  const label = `${t.joint} · ${t.kind === "pin" ? `pin ${t.lo}°` : `${t.lo}–${t.hi}°`}`;
  const series = [{ label: "realized bend", color: SIGNAL, points }];
  if (gtPoints?.length) series.push({ label: "GT", color: GT_COLOR, points: gtPoints });

  return (
    <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
      <figcaption className="mb-2 flex items-center justify-between">
        <span className="label">{label}</span>
        {t.sat && (
          <span className="font-mono text-[11px]" style={{ color: heldColor }}>
            {(t.sat.frac * 100).toFixed(0)}% held
          </span>
        )}
      </figcaption>
      {points.length === 0 ? (
        <p className="px-3 py-8 text-center text-[11px] text-[var(--muted)]">
          {t.joint} has no child bone — bend can't be recovered from positions (end-effector).
        </p>
      ) : (
        <>
          <LineChart
            series={series}
            width={420} height={240} xLabel="frame" yLabel="bend °" yZero
            hlines={t.kind === "pin" ? [{ y: t.lo, color: AMBER, dash: "5 4" }] : []}
            bands={t.kind === "hinge" ? [{ y0: t.lo, y1: t.hi, color: "rgba(52,211,153,0.12)" }] : []}
            regions={[{ x0: t.start, x1: t.end, color: "rgba(148,163,184,0.10)" }]}
            exportName={`fidelity-${t.joint}`}
          />
          {t.sat && (
            <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-slate-400">
              <span>realized <span className="text-slate-200">{t.sat.meanBend.toFixed(1)}°</span></span>
              <span>mean viol <span className="text-slate-200">{t.sat.meanViol.toFixed(1)}°</span></span>
              <span>max viol <span className="text-slate-200">{t.sat.maxViol.toFixed(1)}°</span></span>
            </div>
          )}
        </>
      )}
    </figure>
  );
}

// ── 02 · the ablation ───────────────────────────────────────────────────────

function TextAblation() {
  const [tab, setTab] = useState("setup");
  return (
    <div className="space-y-4">
      <div className="flex gap-1.5">
        {[["setup", "Design & run"], ["results", "Results"]].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)}
            className={`rounded-md border px-3 py-1.5 text-[12px] transition ${tab === id ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
            {label}
          </button>
        ))}
      </div>
      {tab === "setup" ? <AblationSetup /> : <AblationResults />}
    </div>
  );
}

const DEFAULT_SEEDS = "0,1,2";

function parseSeeds(s) {
  return [...new Set(String(s).split(/[,\s]+/).map((x) => parseInt(x, 10)).filter((n) => Number.isFinite(n)))];
}

// Rank every archetype's candidate phrasings against the caption corpus and pick
// the best-supported one per archetype.
//
// This is the archetype map's whole point. Hand-picking a phrasing encodes my
// taste, and my taste is measurably wrong: the corpus prefers "squatting down"
// (551 clips) over "crouching down" (328), and "with their arms straight" (369)
// over "with both arms stretched out straight" (18) — a 20× miss. Support is what
// predicts whether the prior moves, so the corpus chooses, not the lexicon.
//
// `manual` (the user's overrides) wins over the ranking — a pick is a suggestion.
function useVariantRanking(archHits, manual = {}) {
  const [rankMap, setRankMap] = useState({}); // phrase → stats
  const [ranking, setRanking] = useState(false);
  // stable identity for the effect: the variant set, not the array object
  const key = useMemo(() => archHits.flatMap((h) => h.variants).join("|"), [archHits]);

  useEffect(() => {
    const variants = key ? key.split("|") : [];
    if (variants.length === 0) { setRankMap({}); return undefined; }
    let cancelled = false;
    setRanking(true);
    const t = setTimeout(() => {
      api.corpusRank(variants)
        .then((r) => {
          if (cancelled) return;
          const m = {};
          for (const s of r.ranked || []) m[s.query] = s;
          setRankMap(m);
        })
        .catch(() => { /* corpus offline → fall back to lexicon order */ })
        .finally(() => !cancelled && setRanking(false));
    }, 400); // the editor retypes the constraint on every keystroke
    return () => { cancelled = true; clearTimeout(t); setRanking(false); };
  }, [key]);

  // best variant per archetype; unranked (corpus down) keeps the lexicon's order
  const picks = useMemo(() => {
    const p = {};
    for (const h of archHits) {
      if (manual[h.id]) { p[h.id] = manual[h.id]; continue; }
      const best = h.variants
        .slice()
        .sort((a, b) => (rankMap[b]?.all_clips ?? -1) - (rankMap[a]?.all_clips ?? -1))[0];
      if (best) p[h.id] = best;
    }
    return p;
  }, [archHits, rankMap, manual]);

  return { rankMap, picks, ranking };
}

function AblationSetup() {
  const { meta } = useJointMeta();
  const [base, setBase] = useState("a person is walking forward");
  const [pins, setPins] = useState([]);
  // Seeded with the knee-clamp-on-a-walk case: the constraint most likely to
  // fight its prior, which is what this study is for. Built via newRange() so the
  // id comes from the shared counter and can't collide with an added row.
  const [ranges, setRanges] = useState(() => [{ ...newRange(), joint: "L_Knee", bend_min: 0, bend_max: 10 }]);
  const [form, setForm] = useState({ guidance: 6.5, num_steps: 50, num_frames: 100 });
  const [seedText, setSeedText] = useState(DEFAULT_SEEDS);
  const [checkpoint, setCheckpoint] = useState("");
  const [presets, setPresets] = useState(null);
  const [active, setActive] = useState(() => new Set(CELLS.map((c) => c.id)));
  const [overrides, setOverrides] = useState({});   // text-mode id → hand-edited prompt
  const [variantPicks, setVariantPicks] = useState({}); // archetype id → chosen variant
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(null);
  const [error, setError] = useState(null);

  const specs = useMemo(
    () => [...pinsToConstraints(pins), ...rangesToPayload(ranges)],
    [pins, ranges]
  );

  // Which archetypes the constraint set fires, and what each could say. Probed
  // with no picks so we can discover the variants, rank them, then rebuild.
  const archHits = useMemo(
    () => buildTextModes(base, specs, { numFrames: form.num_frames }).find((t) => t.id === "archetype")?.archetypes || [],
    [base, specs, form.num_frames]
  );
  // variantPicks is keyed by ARCHETYPE id (limp, crouch…), overrides by TEXT MODE
  // id (none, literal…) — separate keyspaces, deliberately separate state.
  const { rankMap, picks, ranking } = useVariantRanking(archHits, variantPicks);

  const textModes = useMemo(
    () => buildTextModes(base, specs, { numFrames: form.num_frames, overrides, picks }),
    [base, specs, form.num_frames, overrides, picks]
  );
  const seeds = useMemo(() => parseSeeds(seedText), [seedText]);
  const activeCells = useMemo(() => CELLS.filter((c) => active.has(c.id)), [active]);
  const nJobs = activeCells.length * seeds.length;

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const toggle = (id) =>
    setActive((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  async function launch() {
    setBusy(true); setError(null); setDone(null);
    const study = `st-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;
    const cons = pinsToConstraints(pins);
    const rngs = rangesToPayload(ranges);
    let n = 0;
    try {
      // Serial submits: the backend runs one cluster job at a time, so this just
      // fills its local queue in a deterministic order (seed-major, so the first
      // complete seed lands before the second starts).
      for (const seed of seeds) {
        for (const cell of activeCells) {
          const tm = textModes.find((t) => t.id === cell.text);
          await api.submitViz({
            mode: "prompt",
            checkpoint,
            prompts: tm.prompt,
            guidance: form.guidance,
            num_steps: form.num_steps,
            num_frames: form.num_frames,
            seed,
            model_preset: presets?.model_preset,
            train_preset: presets?.train_preset,
            // projection-off cells carry NO constraint — that's the whole point
            constraints: cell.projected ? cons : [],
            ranges: cell.projected ? rngs : [],
            // the tag that lets Results regroup these into one study. The specs
            // ride along even for projection-off cells: measuring agreement needs
            // to know which bend we were asking about, and the job itself has none.
            ablation: {
              study, arm: cell.id, text_mode: cell.text, projected: cell.projected,
              seed, base_prompt: base, prompt: tm.prompt, added: tm.added,
              constraints: cons, ranges: rngs,
            },
          });
          n++;
        }
      }
      setDone({ study, n });
    } catch (e) {
      setError(`${e.message}${n ? ` — ${n}/${nJobs} submitted before the failure` : ""}`);
    } finally { setBusy(false); }
  }

  const noSpecs = specs.length === 0;

  return (
    <div className="space-y-5">
      <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
        <div className="space-y-5">
          <div className="surface space-y-4 p-5">
            <label className="block">
              <span className="label mb-1.5 block">base prompt (the motion, without the constraint)</span>
              <textarea rows={2} className="field-input resize-none" value={base}
                onChange={(e) => setBase(e.target.value)} />
            </label>
            <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setPresets} />
            <div className="grid grid-cols-3 gap-3">
              <Field label="guidance ω"><input type="number" step="0.5" className="field-input" value={form.guidance} onChange={set("guidance")} /></Field>
              <Field label="ODE steps"><input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} /></Field>
              <Field label="frames"><input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} /></Field>
            </div>
            <Field label="seeds (comma-separated — conflict severity swings ~4× across seeds, so one seed proves nothing)">
              <input className="field-input" value={seedText} onChange={(e) => setSeedText(e.target.value)} />
            </Field>
          </div>
          <ConstraintEditor joints={meta} pins={pins} setPins={setPins} numFrames={form.num_frames} />
          <RangeEditor joints={meta} ranges={ranges} setRanges={setRanges} numFrames={form.num_frames} />
        </div>

        <div className="space-y-5">
          <PromptArms textModes={textModes} noSpecs={noSpecs}
            rankMap={rankMap} ranking={ranking} picks={picks}
            onPickVariant={(archId, v) => setVariantPicks((p) => ({ ...p, [archId]: v }))}
            onEdit={(id, v) => setOverrides((o) => ({ ...o, [id]: v }))}
            onReset={(id) => setOverrides((o) => { const n = { ...o }; delete n[id]; return n; })} />
          <CellMatrix active={active} toggle={toggle} textModes={textModes} />
        </div>
      </div>

      <div className="surface flex flex-wrap items-center gap-4 p-5">
        <div>
          <div className="label">runs to submit</div>
          <div className="font-mono text-[15px] text-slate-100">
            {activeCells.length} cells × {seeds.length} seed{seeds.length === 1 ? "" : "s"} ={" "}
            <span className="text-[var(--signal)]">{nJobs}</span> jobs
          </div>
        </div>
        <p className="max-w-md text-[11px] leading-relaxed text-[var(--muted)]">
          Each is one prompt render. They queue on the backend and run one at a time — the cluster
          queue is never stacked directly.
        </p>
        <button onClick={launch} disabled={busy || !checkpoint || noSpecs || nJobs === 0}
          className="btn-signal ml-auto disabled:opacity-40">
          {busy ? "SUBMITTING…" : `▶  RUN ABLATION (${nJobs})`}
        </button>
      </div>
      {!checkpoint && <p className="label text-center">pick a remote checkpoint first</p>}
      {noSpecs && <p className="label text-center">add a pin or hinge — there's no constraint to formalize yet</p>}
      {error && <p className="text-[12px] text-rose-300">⚠ {error}</p>}
      {done && (
        <p className="rounded-lg border border-[var(--signal)]/40 bg-[var(--signal-dim)] px-4 py-3 text-[12px] text-[var(--signal)]">
          Submitted {done.n} jobs as study <span className="font-mono">{done.study}</span>. They'll appear under
          <span className="font-semibold"> Results</span> as they finish.
        </p>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return <label className="block"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}

// The prompts, editable, each with its corpus-support badge.
function PromptArms({ textModes, noSpecs, onEdit, onReset, rankMap, ranking, picks, onPickVariant }) {
  return (
    <section className="surface p-5">
      <div className="mb-1 flex items-baseline justify-between">
        <div className="display text-base font-bold text-white">Formalized prompts</div>
        <span className="label">lexicon proposes · corpus disposes</span>
      </div>
      <p className="mb-4 text-[11px] leading-relaxed text-[var(--muted)]">
        Generated from the constraint above. Edit any of them — the lexicon is a starting point, not an
        authority. The badge is real support in the HumanML3D captions the encoder trained on.
      </p>
      <div className="space-y-4">
        {textModes.map((t) => (
          <div key={t.id} className="rounded-lg border border-[var(--hairline)] p-3">
            <div className="mb-2 flex items-center gap-2">
              <span className="font-mono text-[11px] uppercase tracking-widest text-slate-300">{t.label}</span>
              {t.edited && (
                <button onClick={() => onReset(t.id)}
                  className="rounded border border-[var(--hairline)] px-1.5 text-[9px] uppercase tracking-wider text-[var(--amber)] hover:border-[var(--amber)]">
                  edited · reset
                </button>
              )}
              {t.empty && (
                <span className="rounded border border-rose-400/40 px-1.5 text-[9px] uppercase tracking-wider text-rose-300">
                  no phrase — same as “no text”
                </span>
              )}
            </div>
            <p className="mb-2 text-[11px] leading-relaxed text-[var(--muted)]">{t.describe}</p>
            {/* "no text" IS the base prompt — editing it here would desync it from
                the base the other two arms are built from, so it's read-only and
                the base-prompt field on the left stays the single source. */}
            <textarea rows={2} className="field-input resize-none text-[12px] disabled:opacity-60"
              value={t.prompt} onChange={(e) => onEdit(t.id, e.target.value)}
              readOnly={t.id === "none"}
              disabled={noSpecs && t.id !== "none"} />
            {t.id === "none" && (
              <p className="mt-1.5 text-[11px] text-[var(--muted)]">mirrors the base prompt — edit it there.</p>
            )}
            {t.meta?.some((m) => !m.reliable) && (
              <p className="mt-1.5 text-[11px] text-[var(--amber)]">
                ⚠ bend at this joint has no clean anatomical reading — the phrase is a guess, check it.
              </p>
            )}
            {t.id === "archetype" && (
              <ArchetypeDetail hits={t.archetypes} uncovered={t.uncovered} picks={picks}
                rankMap={rankMap} ranking={ranking} onPick={onPickVariant} />
            )}
            {t.added ? <PhraseAttestation phrase={t.added} /> : null}
          </div>
        ))}
      </div>
    </section>
  );
}

// Which archetypes matched, and the variants each one weighed. Shown because the
// ranking is a finding in its own right: the phrasing a human would reach for is
// often not the phrasing the corpus supports, and you should be able to see the
// margin and overrule it.
function ArchetypeDetail({ hits, uncovered, picks, rankMap, ranking, onPick }) {
  const [open, setOpen] = useState(false);
  if (!hits || hits.length === 0) return null;
  return (
    <div className="mt-2 rounded border border-[var(--hairline)] bg-ink/60 p-2">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 text-left">
        <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--signal)]">
          {hits.map((h) => h.label).join(" + ")}
        </span>
        <span className="text-[10px] text-[var(--muted)]">
          {ranking ? "ranking variants…" : `${hits.length} archetype${hits.length === 1 ? "" : "s"} matched`}
        </span>
        <span className="ml-auto text-[10px] text-[var(--muted)]">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-3 border-t border-[var(--hairline)] pt-2">
          {hits.map((h) => {
            const ranked = h.variants
              .slice()
              .sort((a, b) => (rankMap[b]?.all_clips ?? -1) - (rankMap[a]?.all_clips ?? -1));
            return (
              <div key={h.id}>
                <div className="label mb-1">{h.label} · variants by corpus support</div>
                <div className="space-y-0.5">
                  {ranked.map((v) => {
                    const n = rankMap[v]?.all_clips;
                    const chosen = picks[h.id] === v;
                    return (
                      <button key={v} onClick={() => onPick(h.id, v)}
                        className={`flex w-full items-baseline gap-2 rounded px-1.5 py-0.5 text-left transition hover:bg-[var(--hairline)]/40 ${chosen ? "bg-[var(--signal-dim)]" : ""}`}>
                        <span className="font-mono text-[10px]" style={{ color: chosen ? "var(--signal)" : "var(--muted)" }}>
                          {chosen ? "★" : "·"} {n == null ? "—" : n}
                        </span>
                        <span className="text-[11px]" style={{ color: chosen ? "#e2e8f0" : "var(--muted)" }}>{v}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
          {uncovered?.length > 0 && (
            <p className="text-[11px] text-[var(--amber)]">
              ⚠ no archetype covers {uncovered.map((u) => u.joint).join(", ")} — those fall back to the
              per-joint lexicon so the prompt still names every enforced constraint.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// The 3×2 grid of cells to run.
function CellMatrix({ active, toggle, textModes }) {
  return (
    <section className="surface p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <div className="display text-base font-bold text-white">Cells</div>
        <span className="label">{TEXT_MODES.length} text modes × projection off/on</span>
      </div>
      <table className="w-full border-collapse text-left text-[12px]">
        <thead>
          <tr className="border-b border-[var(--hairline-strong)]">
            <th className="py-1.5 pr-3 font-medium text-slate-300">text</th>
            <th className="py-1.5 pr-3 font-medium text-slate-300">projection off<div className="label normal-case tracking-normal">does text move the prior?</div></th>
            <th className="py-1.5 font-medium text-slate-300">projection on<div className="label normal-case tracking-normal">what does enforcing cost?</div></th>
          </tr>
        </thead>
        <tbody>
          {TEXT_MODES.map((t) => {
            const mode = textModes.find((m) => m.id === t.id);
            return (
              <tr key={t.id} className="border-b border-[var(--hairline)]">
                <td className="py-2 pr-3">
                  <span className="text-slate-200">{t.label}</span>
                  {mode?.empty && <span className="ml-1 text-[10px] text-rose-300">∅</span>}
                </td>
                {[false, true].map((projected) => {
                  const id = cellId(t.id, projected);
                  const on = active.has(id);
                  const ref = t.id === "none" && !projected;
                  return (
                    <td key={id} className="py-2 pr-3">
                      <button onClick={() => toggle(id)}
                        className={`rounded-md border px-2.5 py-1 text-[11px] transition ${on ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border-[var(--hairline)] text-slate-500 hover:text-slate-300"}`}>
                        {on ? "run" : "skip"}{ref ? " · reference" : ""}
                      </button>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {!active.has("none-off") && (
        <p className="mt-3 text-[11px] text-[var(--amber)]">
          ⚠ the <span className="font-mono">no text · projection off</span> cell is the reference every jerk
          ratio is measured against — without it the results table can't compute a constraint tax.
        </p>
      )}
    </section>
  );
}

// ── 02b · results ───────────────────────────────────────────────────────────

function AblationResults() {
  const { nameToIdx, parents, children } = useJointMeta();
  const [jobs, setJobs] = useState([]);
  const [sel, setSel] = useState("");
  const [rows, setRows] = useState(null);
  // The study's GT reference (one real clip, shared across the cells): its
  // per-frame series, bend at each constrained joint, cost, and render.
  const [gt, setGt] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(() => {
    api.jobs().then((js) => setJobs(js.filter((j) => j.params?.ablation?.study))).catch(() => {});
  }, []);
  useEffect(refresh, [refresh]);

  // group every tagged job into its study
  const studies = useMemo(() => {
    const by = {};
    for (const j of jobs) {
      const a = j.params.ablation;
      const s = (by[a.study] = by[a.study] || {
        id: a.study, base: a.base_prompt, constraints: a.constraints || [], ranges: a.ranges || [],
        jobs: [], created: j.created_at || j.created || 0,
      });
      s.jobs.push(j);
    }
    return Object.values(by)
      .map((s) => {
        // A cancelled cell was pulled from the study on purpose (e.g. re-queued to
        // fold it into a fused submission) — it will never land, so it must not
        // count toward the total or the progress bar stalls at done/(total) forever.
        const live = s.jobs.filter((j) => j.state !== "cancelled");
        return {
          ...s,
          jobs: live,
          done: live.filter((j) => j.state === "done").length,
          total: live.length,
          seeds: [...new Set(live.map((j) => j.params.ablation.seed))].sort((a, b) => a - b),
        };
      })
      .sort((a, b) => String(b.id).localeCompare(String(a.id)));
  }, [jobs]);

  const study = studies.find((s) => s.id === sel);

  useEffect(() => {
    if (!study) { setRows(null); setProgress(null); setGt(null); return; }
    const ready = study.jobs.filter((j) => j.state === "done" && predOutput(j.outputs));
    if (ready.length === 0) { setRows([]); setProgress(null); setGt(null); return; }
    let cancelled = false;
    setRows(null); // otherwise the previous study's table lingers while this one loads
    setGt(null);
    setError(null);
    (async () => {
      // Load the GT reference once for the whole study — it's the same real clip
      // banked with every cell, so the first cell that carries it is enough.
      const gjob = ready.find((j) => gtOutput(j.outputs));
      const gout = gjob && gtOutput(gjob.outputs);
      if (gout) {
        try {
          const gjoints = await loadNpy(mediaUrl(gout.npy_url));
          const gspecs = [...(study.constraints || []), ...(study.ranges || [])];
          const gbends = gspecs
            .map((s) => measureSpec(gjoints, s, { nameToIdx, parents, children }))
            .filter((m) => m.series)
            .map((m) => [m.joint, seriesToPoints(m.series)]);
          const gref = npyRef(gout.npy_url);
          let gcost = null;
          try { gcost = await api.analysisNpy(gref.owner, gref.name); } catch { /* older */ }
          if (!cancelled) {
            setGt({
              media: gout.media_url,
              bendByJoint: Object.fromEntries(gbends),
              cost: gcost,
              series: gcost ? {
                trajectory: gcost.trajectory, jitter: gcost.jitter, speed: gcost.speed,
                foot_height: gcost.foot_height, foot_skate: gcost.foot_skate,
              } : null,
            });
          }
        } catch { /* GT is a reference overlay — never fail the table over it */ }
      }
      const out = [];
      // Sequential: the backend analyses one .npy at a time anyway, and a study
      // is ~18 clips — parallel would just queue behind itself with worse feedback.
      for (let i = 0; i < ready.length; i++) {
        if (cancelled) return;
        setProgress({ i, n: ready.length });
        const j = ready[i];
        const a = j.params.ablation;
        const o = predOutput(j.outputs);
        const { name, owner } = npyRef(o.npy_url);
        try {
          const joints = await loadNpy(mediaUrl(o.npy_url));
          const specs = [...(a.constraints || []), ...(a.ranges || [])];
          const measured = specs.map((s) => measureSpec(joints, s, { nameToIdx, parents, children }));
          const sats = measured.map((m) => m.report).filter(Boolean);
          let cost = null;
          try { cost = await api.analysisNpy(owner, name); } catch { /* older job → no cost */ }
          const avg = (k) => {
            const v = sats.map((s) => s[k]).filter((x) => x != null);
            return v.length ? v.reduce((a2, b2) => a2 + b2, 0) / v.length : null;
          };
          // Jerk AT the constrained joints vs the rest of the body: a clip-level
          // mean spreads the constraint's cost over 22 joints and dilutes it, so
          // this localises where the fight actually shows up.
          const idxs = measured.map((m) => nameToIdx[m.joint]).filter((x) => x != null);
          const pj = cost?.per_joint_jerk;
          const jointJerk = pj && idxs.length
            ? idxs.reduce((s, i2) => s + (pj[i2] ?? 0), 0) / idxs.length : null;
          const restJerk = pj && idxs.length
            ? pj.filter((_, i2) => !idxs.includes(i2)).reduce((s, v) => s + v, 0) / Math.max(1, pj.length - idxs.length)
            : null;
          out.push({
            arm: a.arm, text: a.text_mode, projected: a.projected, seed: a.seed,
            prompt: a.prompt, media: o.media_url,
            jerk: cost?.jerk_mean ?? null,
            skate: cost?.foot_skate_mean ?? null,
            accel: cost?.accel_mean ?? null,
            jointJerk, restJerk,
            held: avg("frac"),
            demand: avg("demand"),
            violIntegral: avg("violIntegral"),
            boundaryOcc: avg("boundaryOcc"),
            longestHold: avg("longestHold"),
            // full per-frame series — already in the payload we fetched for jerk,
            // so the overlay charts below cost no extra requests
            series: cost ? {
              trajectory: cost.trajectory, jitter: cost.jitter, speed: cost.speed,
              foot_height: cost.foot_height, foot_skate: cost.foot_skate,
            } : null,
            bends: measured.map((m) => ({
              joint: m.joint, kind: m.kind, lo: m.lo, hi: m.hi, start: m.start, end: m.end,
              points: seriesToPoints(m.series),
            })),
          });
        } catch (e) {
          if (!cancelled) setError(e.message);
        }
      }
      if (!cancelled) { setRows(out); setProgress(null); }
    })();
    return () => { cancelled = true; };
    // `study` is rebuilt on every poll, so depending on the object itself would
    // re-analyse every clip on each refresh. Its done-count is what actually
    // changes the table — re-run when a new job lands, not when the array is new.
  }, [sel, study?.done, study?.total, nameToIdx, parents, children]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-5">
      <div className="surface p-5">
        <div className="flex flex-wrap items-end gap-3">
          <label className="block min-w-[320px] flex-1">
            <span className="label mb-1.5 block">study</span>
            <select className="field-input" value={sel} onChange={(e) => setSel(e.target.value)}>
              <option value="">select a study…</option>
              {studies.map((s) => (
                <option key={s.id} value={s.id}>
                  {`${s.id} · “${String(s.base).slice(0, 38)}” · ${s.done}/${s.total} done · ${s.seeds.length} seed${s.seeds.length === 1 ? "" : "s"}`}
                </option>
              ))}
            </select>
          </label>
          <button onClick={refresh}
            className="rounded-md border border-[var(--hairline)] px-3 py-2 text-[12px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]">
            refresh
          </button>
        </div>
        {studies.length === 0 && (
          <div className="mt-4"><EmptyHint>No ablation studies yet — design one under <span className="text-slate-300">Design &amp; run</span>.</EmptyHint></div>
        )}
        {study && study.done < study.total && (
          <p className="mt-3 text-[12px] text-[var(--amber)]">
            {study.done}/{study.total} runs finished — the table fills in as the rest land.
          </p>
        )}
        {progress && (
          <p className="mt-3 text-[12px] text-[var(--muted)]">analysing clip {progress.i + 1}/{progress.n}…</p>
        )}
        {error && <p className="mt-3 text-[12px] text-rose-300">⚠ {error}</p>}
      </div>

      {study && rows && rows.length > 0 && (
        <AblationTable study={study} rows={rows} gt={gt} />
      )}
      {study && rows && rows.length === 0 && (
        <EmptyHint>No finished runs in this study yet.</EmptyHint>
      )}
    </div>
  );
}

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

function AblationTable({ study, rows, gt }) {
  // aggregate the seeds within each cell — a single seed swings ~4× on a
  // conflicting constraint, so the per-cell number only means something as a mean
  const cells = useMemo(() => {
    const by = {};
    for (const r of rows) (by[r.arm] = by[r.arm] || { arm: r.arm, text: r.text, projected: r.projected, runs: [] }).runs.push(r);
    return Object.fromEntries(
      Object.values(by).map((c) => {
        // job order isn't guaranteed, so sort before calling runs[0] "the first
        // seed" — the render strip labels it that way
        c.runs.sort((a, b) => a.seed - b.seed);
        const num = (k) => c.runs.map((r) => r[k]).filter((v) => v != null);
        return [c.arm, {
          ...c,
          n: c.runs.length,
          jerk: mean(num("jerk")),
          jerkSpread: num("jerk").length > 1 ? [Math.min(...num("jerk")), Math.max(...num("jerk"))] : null,
          skate: mean(num("skate")),
          accel: mean(num("accel")),
          held: mean(num("held")),
          demand: mean(num("demand")),
          violIntegral: mean(num("violIntegral")),
          boundaryOcc: mean(num("boundaryOcc")),
          longestHold: mean(num("longestHold")),
          jointJerk: mean(num("jointJerk")),
          restJerk: mean(num("restJerk")),
          prompt: c.runs[0]?.prompt,
          media: c.runs[0]?.media,
        }];
      })
    );
  }, [rows]);

  const ref = cells["none-off"];
  const tax = (c) => (ref?.jerk && c?.jerk != null && ref.jerk > 1e-9 ? c.jerk / ref.jerk : null);
  const factorialRef = useRef(null);

  const bars = useMemo(
    () => CELLS.filter((c) => cells[c.id]).map((c) => ({
      // the y-axis gutter is narrow — "semantic · on" reads, the long form doesn't
      label: `${c.text} · ${c.projected ? "on" : "off"}`,
      // hue = text mode (matching the overlay charts); unenforced cells are greyed
      // since they're the reference, not a result
      values: [{ v: cells[c.id].jerk ?? 0, color: c.projected ? (TEXT_COLOR[c.text] || AMBER) : SLATE }],
    })),
    [cells]
  );

  // The two numbers the study exists to produce. With three text treatments the
  // winner is whichever actually cut the cost most — naming a favourite in advance
  // would be assuming the result the study is meant to establish.
  const onNone = cells["none-on"];
  const offNone = cells["none-off"];
  const contenders = TEXT_MODES.filter((t) => t.id !== "none")
    .map((t) => ({ mode: t, on: cells[cellId(t.id, true)], off: cells[cellId(t.id, false)] }))
    .filter((c) => c.on?.jerk > 1e-9);
  const best = contenders.sort((a, b) => a.on.jerk - b.on.jerk)[0];
  // both jerks must be real and non-zero — dividing by an all-zero clip's jerk
  // would report an Infinity× "improvement"
  const verdict =
    onNone?.jerk > 1e-9 && best
      ? {
          drop: onNone.jerk / best.on.jerk,
          label: best.mode.label,
          on: best.on,
          off: best.off,
          prior: best.off?.held != null && offNone?.held != null ? best.off.held - offNone.held : null,
          // is the winner actually distinguishable from the runner-up, or is this
          // a coin flip dressed up as a finding?
          runnerUp: contenders[1] || null,
        }
      : null;

  return (
    <div className="space-y-5">
      {verdict && <Verdict v={verdict} onNone={onNone} offNone={offNone} />}

      <div className="surface p-5">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="label">the factorial · seeds averaged</span>
          <span className="flex items-center gap-2 font-mono text-[10px] text-[var(--muted)]">
            <span>“{String(study.base).slice(0, 60)}” · {study.seeds.length} seed{study.seeds.length === 1 ? "" : "s"}</span>
            <TableTools getTable={() => factorialRef.current} name={`ablation-${study.id}`}
              caption="Constraint→text factorial: fidelity metrics, seeds averaged." label={`tab:ablation-${study.id}`} />
          </span>
        </div>
        <div className="overflow-x-auto">
          <table ref={factorialRef} className="w-full border-collapse text-left text-[12px]">
            <thead>
              <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
                <th className="py-2 pr-4 font-medium">text</th>
                <th className="py-2 pr-4 font-medium" colSpan={2}>
                  projection OFF <span className="label normal-case tracking-normal">— the prior's own opinion</span>
                </th>
                <th className="py-2 font-medium" colSpan={4}>
                  projection ON <span className="label normal-case tracking-normal">— enforced</span>
                </th>
              </tr>
              <tr className="border-b border-[var(--hairline)] text-[10px] uppercase tracking-wider text-[var(--muted)]">
                <th className="py-1.5 pr-4 font-normal" />
                <th className="py-1.5 pr-4 font-normal">agreement</th>
                <th className="py-1.5 pr-4 font-normal">smoothness</th>
                <th className="py-1.5 pr-4 font-normal">held</th>
                <th className="py-1.5 pr-4 font-normal">smoothness</th>
                <th className="py-1.5 pr-4 font-normal">tax vs free</th>
                <th className="py-1.5 font-normal">foot-skate</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {TEXT_MODES.map((t) => {
                const off = cells[cellId(t.id, false)];
                const on = cells[cellId(t.id, true)];
                const taxOn = tax(on);
                return (
                  <tr key={t.id} className="border-b border-[var(--hairline)]">
                    <td className="py-2 pr-4 font-sans text-slate-200">{t.label}</td>
                    <td className="py-2 pr-4"><Agreement c={off} /></td>
                    <td className="py-2 pr-4 text-slate-400">{off?.jerk != null ? off.jerk.toFixed(4) : "—"}</td>
                    <td className="py-2 pr-4"><Agreement c={on} held /></td>
                    <td className="py-2 pr-4 text-slate-300">{on?.jerk != null ? on.jerk.toFixed(4) : "—"}</td>
                    <td className="py-2 pr-4" style={{ color: taxOn == null ? "var(--muted)" : taxOn <= 2.5 ? GREEN : taxOn <= 6 ? AMBER : ROSE }}>
                      {taxOn == null ? "—" : `${taxOn.toFixed(1)}×`}
                    </td>
                    <td className="py-2 text-slate-400">{on?.skate != null ? on.skate.toFixed(3) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="mt-3 space-y-1 text-[11px] leading-relaxed text-[var(--muted)]">
          <p>
            <span className="text-slate-300">agreement</span> = frames the realized bend already satisfies the
            constraint with nothing enforcing it — the prior's own opinion, which is what the text is supposed to
            move. <span className="text-slate-300">smoothness</span> = mean jerk ‖Δ³x‖ (lower = smoother);{" "}
            <span className="text-slate-300">tax</span> = that ÷ the free cell ({ref?.jerk?.toFixed(4) ?? "—"}).
            GT clips sit at {GT_JERK}.
          </p>
        </div>
      </div>

      <Chart title="smoothness by cell — jerk ‖Δ³x‖, lower is smoother (dashed = ground truth)">
        <BarChart bars={bars} width={720} height={320} horizontal labelWidth={104}
          hlines={gt?.cost?.jerk_mean != null
            ? [{ v: gt.cost.jerk_mean, color: GT_COLOR, label: "GT" }]
            : [{ v: GT_JERK, color: GREEN, label: "GT" }]}
          legend={[...TEXT_MODES.map((t) => ({ label: `enforced · ${t.label}`, color: TEXT_COLOR[t.id] })),
                   { label: "unenforced", color: SLATE }]}
          exportName={`ablation-jerk-${study.id}`} />
      </Chart>

      <RespectTable cells={cells} />
      <CellOverlay rows={rows} study={study} gt={gt} />
      <CellClips cells={cells} gt={gt} />
    </div>
  );
}

function Agreement({ c, held }) {
  if (!c || c.held == null) return <span className="text-[var(--muted)]">—</span>;
  const pct = c.held * 100;
  // for the unenforced row, high agreement is the interesting result; for the
  // enforced row it's just a sanity check that the projection did its job
  const color = held
    ? pct >= 95 ? GREEN : pct >= 80 ? AMBER : ROSE
    : pct >= 60 ? GREEN : pct >= 30 ? AMBER : "var(--muted)";
  return <span style={{ color }}>{pct.toFixed(0)}%</span>;
}

// The headline, in words — a table of numbers isn't a finding.
function Verdict({ v, onNone, offNone }) {
  const worked = v.drop >= 1.5;
  // A 5% gap between two text modes across a handful of seeds is not a ranking.
  const close = v.runnerUp?.on?.jerk != null && v.runnerUp.on.jerk / v.on.jerk < 1.15;
  return (
    <div className="rounded-lg border p-4"
      style={{ borderColor: worked ? "rgba(52,211,153,0.4)" : "var(--hairline)", background: worked ? "rgba(52,211,153,0.06)" : "transparent" }}>
      <div className="label mb-2">verdict</div>
      <p className="text-[14px] leading-relaxed text-slate-200">
        Best text treatment: <span className="font-semibold text-white">{v.label}</span>.{" "}
        {worked ? (
          <>It cut the enforcement cost <span className="font-mono text-[var(--signal)]">{v.drop.toFixed(1)}×</span> —
            jerk <span className="font-mono">{onNone.jerk.toFixed(4)}</span> →{" "}
            <span className="font-mono" style={{ color: GREEN }}>{v.on.jerk.toFixed(4)}</span>.</>
        ) : (
          <>No text treatment meaningfully cut the enforcement cost
            ({onNone.jerk.toFixed(4)} → {v.on.jerk.toFixed(4)}, {v.drop.toFixed(2)}×).</>
        )}
      </p>
      {v.prior != null && (
        <p className="mt-2 font-mono text-[12px] text-slate-300">
          prior agreement · {v.label} <span>{(v.off.held * 100).toFixed(0)}%</span> vs bare{" "}
          <span>{(offNone.held * 100).toFixed(0)}%</span> ({v.prior >= 0 ? "+" : ""}{(v.prior * 100).toFixed(0)} pts)
        </p>
      )}
      {close && (
        <p className="mt-2 text-[12px] leading-relaxed text-[var(--amber)]">
          ⚠ {v.runnerUp.mode.label} within {((v.runnerUp.on.jerk / v.on.jerk - 1) * 100).toFixed(0)}% — too close to call at this seed count.
        </p>
      )}
    </div>
  );
}

// ── cell overlay ────────────────────────────────────────────────────────────
//
// The same plots the Clip comparison tab draws, but with every cell of the study
// overlaid instead of GT-vs-gen. This is where the factorial stops being a table:
// the bend trace shows all arms against the target band at once, so you can watch
// the unenforced arms wander (and see which text pulled the prior toward the band)
// while the enforced arms ride it.
//
// Identity is two channels so eight lines stay readable: HUE = text mode,
// DASH = projection off. Everything is per-seed — averaging per-frame series
// across seeds would smear out exactly the structure you're looking for, and
// conflict severity swings ~4× seed to seed.

const TEXT_COLOR = { none: SLATE, literal: AMBER, semantic: SIGNAL, archetype: GREEN };

function CellOverlay({ rows, study, gt }) {
  const seeds = useMemo(() => [...new Set(rows.map((r) => r.seed))].sort((a, b) => a - b), [rows]);
  const [seed, setSeed] = useState(seeds[0]);
  const [showText, setShowText] = useState(() => new Set(TEXT_MODES.map((t) => t.id)));
  const [showProj, setShowProj] = useState(() => new Set(["on", "off"]));
  const [showGt, setShowGt] = useState(true);

  useEffect(() => {
    if (!seeds.includes(seed)) setSeed(seeds[0]);
  }, [seeds, seed]);

  const visible = useMemo(
    () => rows.filter((r) => r.seed === seed && showText.has(r.text) && showProj.has(r.projected ? "on" : "off")),
    [rows, seed, showText, showProj]
  );

  // GT is a single reference clip for the whole study — draw it as one bright,
  // solid line on top of the per-seed cells (it doesn't vary with seed).
  const gtSeries = (field) =>
    showGt && Array.isArray(gt?.series?.[field])
      ? [{ label: "GT", color: GT_COLOR, points: gt.series[field] }]
      : [];

  const mk = (field) => [
    ...visible
      .filter((r) => Array.isArray(r.series?.[field]))
      .map((r) => ({
        label: `${r.text} · ${r.projected ? "on" : "off"}`,
        color: TEXT_COLOR[r.text] || SIGNAL,
        dash: r.projected ? undefined : "3 3",
        opacity: r.projected ? 1 : 0.75,
        points: r.series[field],
      })),
    ...gtSeries(field),
  ];

  // bend traces, grouped by constrained joint — the plot this study is about
  const bendCharts = useMemo(() => {
    const byJoint = {};
    for (const r of visible)
      for (const b of r.bends || []) {
        if (!b.points.length) continue;
        (byJoint[b.joint] = byJoint[b.joint] || { spec: b, series: [] }).series.push({
          label: `${r.text} · ${r.projected ? "on" : "off"}`,
          color: TEXT_COLOR[r.text] || SIGNAL,
          dash: r.projected ? undefined : "3 3",
          opacity: r.projected ? 1 : 0.75,
          points: b.points,
        });
      }
    // overlay the GT bend at each constrained joint
    if (showGt)
      for (const [joint, entry] of Object.entries(byJoint)) {
        const pts = gt?.bendByJoint?.[joint];
        if (pts?.length) entry.series.push({ label: "GT", color: GT_COLOR, points: pts });
      }
    return Object.entries(byJoint);
  }, [visible, gt, showGt]);

  const toggle = (set, fn) => (v) =>
    fn((s) => {
      const n = new Set(s);
      n.has(v) ? n.delete(v) : n.add(v);
      return n;
    });

  if (rows.length === 0) return null;

  return (
    <div className="surface p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="label">per-frame overlay · one seed at a time</span>
        <span className="font-mono text-[10px] text-[var(--muted)]">{visible.length} cells shown</span>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="flex items-center gap-1.5">
          <span className="label">seed</span>
          {seeds.map((s) => (
            <button key={s} onClick={() => setSeed(s)}
              className={`rounded-md px-2.5 py-1 font-mono text-[11px] transition ${s === seed ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
              {s}
            </button>
          ))}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="label">text</span>
          {TEXT_MODES.map((t) => (
            <button key={t.id} onClick={() => toggle(showText, setShowText)(t.id)}
              className={`rounded-md border px-2.5 py-1 text-[11px] transition ${showText.has(t.id) ? "text-slate-100" : "border-[var(--hairline)] text-slate-600"}`}
              style={showText.has(t.id) ? { borderColor: TEXT_COLOR[t.id], background: `${TEXT_COLOR[t.id]}1a` } : undefined}>
              {t.label}
            </button>
          ))}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="label">projection</span>
          {["off", "on"].map((p) => (
            <button key={p} onClick={() => toggle(showProj, setShowProj)(p)}
              className={`rounded-md border px-2.5 py-1 text-[11px] transition ${showProj.has(p) ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border-[var(--hairline)] text-slate-600"}`}>
              {p}
            </button>
          ))}
        </span>
        {gt && (
          <span className="flex items-center gap-1.5">
            <span className="label">GT</span>
            <button onClick={() => setShowGt((v) => !v)}
              className="rounded-md border px-2.5 py-1 text-[11px] transition"
              style={showGt
                ? { borderColor: GT_COLOR, background: `${GT_COLOR}1a`, color: GT_COLOR }
                : { borderColor: "var(--hairline)", color: "#475569" }}>
              reference
            </button>
          </span>
        )}
      </div>
      <p className="mb-4 text-[11px] text-[var(--muted)]">
        hue = text mode · dashed = projection off (the prior unopposed) · solid = enforced
        {gt ? " · white = ground truth" : ""}
      </p>

      {bendCharts.map(([joint, { spec, series }]) => (
        <div key={joint} className="mb-6">
          <Chart title={`realized bend · ${joint} — ${spec.kind === "pin" ? `pin ${spec.lo}°` : `clamp ${spec.lo}–${spec.hi}°`} · dashed lines are what the prior wanted`}>
            <LineChart series={series} width={760} height={300} xLabel="frame" yLabel="bend °" yZero
              hlines={spec.kind === "pin" ? [{ y: spec.lo, color: "#fbbf24", label: "target" }] : []}
              bands={spec.kind === "hinge" ? [{ y0: spec.lo, y1: spec.hi, color: "rgba(52,211,153,0.12)" }] : []}
              regions={[{ x0: spec.start, x1: spec.end, color: "rgba(148,163,184,0.08)" }]}
              exportName={`overlay-bend-${joint}-seed${seed}`} />
          </Chart>
        </div>
      ))}

      <div className="grid gap-6 lg:grid-cols-2">
        <Chart title="jitter — ‖Δ³x‖ per frame (lower = smoother)">
          <LineChart series={mk("jitter")} width={420} height={280} xLabel="frame" yLabel="jerk" yZero
            exportName={`overlay-jitter-seed${seed}`} />
        </Chart>
        <Chart title="root trajectory (top-down) — did the constraint derail the path?">
          <LineChart series={mk("trajectory")} equal width={420} height={280} xLabel="x" yLabel="z"
            exportName={`overlay-traj-seed${seed}`} />
        </Chart>
        <Chart title="root speed (m/s)">
          <LineChart series={mk("speed")} width={420} height={260} xLabel="frame" yLabel="m/s" yZero
            exportName={`overlay-speed-seed${seed}`} />
        </Chart>
        <Chart title="foot-skate — planted-foot drift (m/s)">
          <LineChart series={mk("foot_skate")} width={420} height={260} xLabel="frame" yLabel="m/s" yZero
            exportName={`overlay-skate-seed${seed}`} />
        </Chart>
      </div>
    </div>
  );
}

// ── constraint respect ──────────────────────────────────────────────────────
//
// "Held %" saturates at ~100% for every projected clip, so on its own it says
// nothing about whether the constraint was cheap or a fight. These are the
// measures that separate those.
function RespectTable({ cells }) {
  const respectRef = useRef(null);
  const rows = CELLS.filter((c) => cells[c.id]);
  if (rows.length === 0) return null;
  return (
    <div className="surface p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="label">constraint respect · seeds averaged</span>
        <span className="flex items-center gap-2 font-mono text-[10px] text-[var(--muted)]">
          <span>measured on the constrained joint itself</span>
          <TableTools getTable={() => respectRef.current} name="constraint-respect"
            caption="Constraint respect on the constrained joint, seeds averaged." label="tab:constraint-respect" />
        </span>
      </div>
      <div className="overflow-x-auto">
        <table ref={respectRef} className="w-full border-collapse text-left text-[12px]">
          <thead>
            <tr className="border-b border-[var(--hairline-strong)] text-[10px] uppercase tracking-wider text-[var(--muted)]">
              <th className="py-1.5 pr-4 font-normal">cell</th>
              <th className="py-1.5 pr-4 font-normal">held</th>
              <th className="py-1.5 pr-4 font-normal">demand °</th>
              <th className="py-1.5 pr-4 font-normal">viol ∫ °·s</th>
              <th className="py-1.5 pr-4 font-normal">on limit</th>
              <th className="py-1.5 pr-4 font-normal">longest hold</th>
              <th className="py-1.5 pr-4 font-normal">jerk @ joint</th>
              <th className="py-1.5 font-normal">jerk elsewhere</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((c) => {
              const x = cells[c.id];
              const f = (v, d = 2, suf = "") => (v == null ? "—" : v.toFixed(d) + suf);
              return (
                <tr key={c.id} className="border-b border-[var(--hairline)]">
                  <td className="py-2 pr-4 font-sans" style={{ color: c.projected ? "#e2e8f0" : "var(--muted)" }}>
                    {c.text} · {c.projected ? "on" : "off"}
                  </td>
                  <td className="py-2 pr-4 text-slate-300">{x.held == null ? "—" : `${(x.held * 100).toFixed(0)}%`}</td>
                  <td className="py-2 pr-4" style={{ color: !c.projected && x.demand != null ? (x.demand > 20 ? ROSE : x.demand > 8 ? AMBER : GREEN) : "var(--muted)" }}>
                    {f(x.demand, 1, "°")}
                  </td>
                  <td className="py-2 pr-4 text-slate-400">{f(x.violIntegral, 2)}</td>
                  <td className="py-2 pr-4 text-slate-400">{x.boundaryOcc == null ? "—" : `${(x.boundaryOcc * 100).toFixed(0)}%`}</td>
                  <td className="py-2 pr-4 text-slate-400">{f(x.longestHold, 1, "s")}</td>
                  <td className="py-2 pr-4 text-slate-300">{f(x.jointJerk, 4)}</td>
                  <td className="py-2 text-slate-500">{f(x.restJerk, 4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 space-y-1 text-[11px] leading-relaxed text-[var(--muted)]">
        <p>
          <span className="text-slate-300">demand</span> — mean violation in degrees on the{" "}
          <span className="text-slate-300">projection-off</span> rows: how far the prior wants to be from the
          constraint, measured on the constraint itself rather than inferred from jerk. This is the conflict
          severity, and the number the text has to shrink. (On projection-on rows it's just residual error.)
        </p>
        <p>
          <span className="text-slate-300">on limit</span> — clamps only: frames resting within 1° of a limit.
          A trace parked on the boundary means the prior is pushing through and the projection is holding it
          back — strain that “100% held” hides.
        </p>
        <p>
          <span className="text-slate-300">jerk @ joint</span> vs <span className="text-slate-300">elsewhere</span> —
          where the cost landed. Roughness concentrated at the constrained joint is a local fight; spread across
          the body means the constraint dragged the whole motion off-manifold.
        </p>
      </div>
    </div>
  );
}

// The actual renders, one per cell — numbers are the claim, clips are the evidence.
// The GT reference clip leads the grid so every cell is read against the real
// motion it's being compared to.
function CellClips({ cells, gt }) {
  const { open } = useLightbox();
  const list = CELLS.filter((c) => cells[c.id]?.media);
  if (list.length === 0 && !gt?.media) return null;

  // One lightbox group in the same order the grid renders — GT leads, then the
  // cells — so ←/→ walks GT ⇄ each treatment and the jerk claim above every clip
  // is one keypress from its evidence. Items are job-output shaped (media_url,
  // caption, kind, clip_id) for the shared Overlay.
  const items = [
    ...(gt?.media
      ? [{ media_url: gt.media, kind: "gt", clip_id: "ground truth · dataset",
           caption: "the real motion each cell is compared against" }]
      : []),
    ...list.map((c) => ({
      media_url: cells[c.id].media,
      kind: c.projected ? "pred" : "sample",
      clip_id: c.label,
      caption: cells[c.id].prompt,
    })),
  ];
  const gtOffset = gt?.media ? 1 : 0;

  return (
    <div className="surface p-5">
      <div className="label mb-3">
        renders · first seed of each cell{gt?.media ? " · GT leads" : ""} · click any clip to preview fullscreen
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {gt?.media && (
          <figure className="overflow-hidden rounded-lg border bg-black" style={{ borderColor: GT_COLOR }}>
            <ClipButton media={gt.media} label="ground truth" onOpen={() => open(items, 0)} />
            <figcaption className="space-y-1 px-2 py-1.5">
              <div className="font-mono text-[10px]" style={{ color: GT_COLOR }}>ground truth · dataset</div>
              <div className="text-[10px] leading-snug text-[var(--muted)]">the real motion each cell is compared against</div>
              <div className="font-mono text-[10px]" style={{ color: GT_COLOR }}>
                jerk {gt.cost?.jerk_mean?.toFixed(4) ?? "—"}
              </div>
            </figcaption>
          </figure>
        )}
        {list.map((c, i) => {
          const cell = cells[c.id];
          return (
            <figure key={c.id} className="overflow-hidden rounded-lg border bg-black"
              style={{ borderColor: c.projected ? (c.text === "semantic" ? "rgba(52,211,153,0.5)" : "var(--hairline)") : "var(--hairline)" }}>
              <ClipButton media={cell.media} label={c.label} onOpen={() => open(items, gtOffset + i)} />
              <figcaption className="space-y-1 px-2 py-1.5">
                <div className="font-mono text-[10px] text-slate-300">{c.label}</div>
                <div className="text-[10px] leading-snug text-[var(--muted)]">“{cell.prompt}”</div>
                <div className="font-mono text-[10px]" style={{ color: cell.jerk > GT_JERK * 6 ? ROSE : cell.jerk > GT_JERK * 2.5 ? AMBER : GREEN }}>
                  jerk {cell.jerk?.toFixed(4) ?? "—"}
                </div>
              </figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}

// The clickable clip itself — a button so it's keyboard-reachable, with a hover
// affordance (the whole app opens clips into the shared Lightbox rather than a raw
// browser tab). autoPlay/muted keeps the grid thumbnails looping as before.
function ClipButton({ media, label, onOpen }) {
  const isVideo = media?.toLowerCase().endsWith(".mp4");
  return (
    <button type="button" onClick={onOpen} title="click to preview fullscreen"
      className="group relative block w-full cursor-zoom-in">
      {isVideo ? (
        <video src={mediaUrl(media)} className="w-full" autoPlay loop muted />
      ) : (
        <img src={mediaUrl(media)} alt={label} className="w-full" />
      )}
      <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/0 text-transparent transition group-hover:bg-black/30 group-hover:text-white">
        <span className="rounded-full border border-white/60 px-2.5 py-1 text-[11px] font-medium backdrop-blur-sm">⤢ fullscreen</span>
      </span>
    </button>
  );
}
