// Shared eval-results plumbing for the Lab's Analysis tab.
//
// §02 (Eval comparison) and §04 (Model head-to-head) read exactly the same
// payload — every eval run's per-ω metrics plus the job registry that resolves a
// run to its (model config, ODE steps, checkpoint). Both the parsing helpers and
// the fetch itself live here so the two sections can never disagree about what a
// run IS, and so mounting the tab costs ONE batched request: the head node
// serializes SSH, and `analysisTable` is a per-run `cat` fan-out on the far side.

import { api } from "@/lib/api";

// ─────────────────────────────────────────────────────── run → (config, steps)

// Run → ODE step count. Resolution order: the run dir's own hydra dump
// (backend `meta` — authoritative, survives renames/merges), the run's job
// record, then an `odeNNN` hint in the run name. Null = unknown, never guessed.
export const stepsOf = (jobs, run, meta) => {
  const ms = meta?.[run]?.steps;
  if (ms != null) return ms;
  const s = jobs.find((x) => x.run_name === run)?.params?.num_sample_steps;
  if (s != null) return s;
  const m = run.match(/ode(\d+)/i);
  return m ? parseInt(m[1], 10) : null;
};

// Run → training checkpoint step (the step the evaluated .pt was saved at).
// From the eval job's checkpoint path (ckpt-000300000.pt), else a ckptNNN hint
// in the run name. Null = unknown. This is the x-axis of the "FID over training"
// view — distinct from ODE steps (a sampling knob).
export const ckptStepOf = (jobs, run) => {
  const p = jobs.find((x) => x.run_name === run)?.params?.checkpoint;
  const m = p && p.match(/ckpt-0*(\d+)\.pt/i);
  if (m) return parseInt(m[1], 10);
  const r = run.match(/ckpt-?0*(\d+)/i);
  return r ? parseInt(r[1], 10) : null;
};

// Run → model config, same resolution order (hydra dump, registry, name).
export const modelOf = (jobs, run, meta) =>
  meta?.[run]?.model ||
  jobs.find((x) => x.run_name === run)?.params?.model_preset ||
  (/base/i.test(run) ? "dit_base" : /mid/i.test(run) ? "dit_mid" : "other");

// ───────────────────────────────────────────────────────────── chart identity

// Chart colors: identity = CONFIG (hue family), runs within a family take
// ordinal lightness steps (dim → bright). Ramps validated on the app surface
// (#070a11): monotone OKLab L, adjacent ΔL ≥ 0.06, dark end ≥ ~2:1. This keeps
// many runs plottable without cycling hues; past a family's step count,
// neighbors start sharing a step — collapse runs rather than adding colors.
export const FAMILY_RAMPS = {
  dit_mid: ["#155e75", "#0e7490", "#0891b2", "#06b6d4", "#22d3ee", "#67e8f9"],
  dit_base: ["#92400e", "#b45309", "#d97706", "#f59e0b", "#fbbf24"],
  other: ["#5b21b6", "#7c3aed", "#a78bfa", "#c4b5fd"],
};

// A config's headline color (one step below the brightest — full brightness is
// reserved for the many-runs case where the ramp is actually spread out).
export const familyColor = (config) => {
  const ramp = FAMILY_RAMPS[config] || FAMILY_RAMPS.other;
  return ramp[ramp.length - 2];
};

// One marker shape per ODE step count, shared across configs (secondary
// identity channel on top of the lightness ramp).
export const MARKERS = ["circle", "square", "triangle", "diamond", "cross"];

export const stepsVal = (s) => (s == null ? Infinity : s); // unknown steps sort last

// ────────────────────────────────────────────────────── (config, steps, ω) grid

// Replicates at the same cell collapse to the best-FID run's metrics, so every
// view built on this grid shows the same numbers. Run identity is kept only for
// tooltips. ω keys are float-normalized so "6.5" and "6.50" can't split a cell.
export function gridCells(sweeps, jobs, meta) {
  const cells = new Map(); // `${config}|${steps ?? "?"}|${ω}` → {…metrics, run}
  const rowSet = new Map(); // `${config}|${steps ?? "?"}` → {config, steps}
  const omegaSet = new Set();
  for (const [run, byW] of Object.entries(sweeps || {})) {
    const config = modelOf(jobs, run, meta), steps = stepsOf(jobs, run, meta);
    const rowKey = `${config}|${steps ?? "?"}`;
    for (const [w, m] of Object.entries(byW)) {
      if (m?.fid == null) continue;
      rowSet.set(rowKey, { config, steps });
      const omega = parseFloat(w);
      omegaSet.add(omega);
      const key = `${rowKey}|${omega}`;
      const cur = cells.get(key);
      if (!cur || m.fid < cur.fid) cells.set(key, { ...m, run });
    }
  }
  return {
    cells,
    rows: [...rowSet.values()].sort((a, b) =>
      a.config !== b.config ? a.config.localeCompare(b.config) : stepsVal(a.steps) - stepsVal(b.steps)),
    omegas: [...omegaSet].sort((a, b) => a - b),
  };
}

export const cellKey = (r, omega) => `${r.config}|${r.steps ?? "?"}|${omega}`;

// ───────────────────────────────────────────────────────────── shared fetch

// Module-level cache of the last fetch. Two sections mount together and both
// want the same payload; the first one to ask starts the request and the second
// awaits the SAME promise (in-flight or settled) instead of opening a second
// SSH fan-out. `force` (the ↻ button) drops it and refetches for everyone.
let _pending = null;

export function loadEvalData({ force = false } = {}) {
  if (!force && _pending) return _pending;
  _pending = (async () => {
    const [runs, jobs] = await Promise.all([api.evalRuns(), api.jobs().catch(() => [])]);
    const cmp = runs.length ? await api.analysisTable(runs.map((r) => r.run)) : null;
    return { runs, jobs, cmp };
  })().catch((e) => {
    _pending = null; // a failure must not be cached — the next mount retries
    throw e;
  });
  return _pending;
}
