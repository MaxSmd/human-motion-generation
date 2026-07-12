"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";

const SIGNAL = "#22d3ee";
const AMBER = "#fbbf24";
const SLATE = "#94a3b8";

// ── constants (not live) ────────────────────────────────────────────────────
// The measurement floor and the paper's headline are fixed reference points; the
// mid/base FID are pulled live from /analysis/table and fall back to these when
// no eval run is available. Numbers from docs/rmg_mid_validation.md §1.
const GAP = {
  gtgt: 0.0019, // GT-vs-GT measurement floor (published ≈ 0.002)
  paper: 0.043, // RMG-main, 460M / 600k
  midFallback: 0.96, // ours — mid, 112M / 300k (prefer live)
  baseFallback: 8.19, // base, 25M
};

// docs/rmg_mid_validation.md §2 — every pipeline stage was audited and is faithful.
const SCORECARD = [
  ["Data processing", "Bit-exact to the upstream HumanML3D pipeline (X-flip, uniform_skeleton, IK)"],
  ["Container", "Same rmg.sqsh (torch 2.9, py 3.11) for train + eval — no drift"],
  ["Data loading", "Upper-hemisphere quats, valid-frame masking, crop, min-40"],
  ["Representation", "(T,R) → FK → 263-D bit-exact to official new_joint_vecs (0.0 % rel.)"],
  ["Training", "Riemannian CFM, tangent projection, EMA 0.9999, effective BS 256"],
  ["Sampling", "Riemannian Euler + Exp-map stepping, CFG combined in ambient then projected"],
  ["Evaluation", "Loads the evaluator's OWN mean/std — GT-GT 0.0019 ≈ published 0.002"],
];

// Two evaluation-harness bugs found + fixed during the audit ("harness bugs").
const BUGS = [
  {
    title: "Motion-embedding pairing",
    delta: "R@1 0.34 → 0.513",
    tag: "FID unchanged",
    mechanism:
      "encode_motion mis-paired text↔motion on length ties, depressing R-precision / MM-Dist. Fixed R@1 now matches the published 0.511.",
  },
  {
    title: "Missing length mask at generation",
    delta: "FID 1.10 → 0.822",
    tag: "≈ 25 % better",
    mechanism:
      "sampling without a per-clip validity mask let short clips attend across the padded batch. Masking is now the default.",
  },
];

// docs/rmg_mid_validation.md — "Exact numbers to hardcode". Overridden live by
// /analysis/forensics when a run has written eval/forensics.json (else static).
const FORENSICS_STATIC = {
  velocity: {
    quat_norm: { gen: 1.0, real: 1.0 },
    angular_vel: { gen: 0.0875, real: 0.0645 }, // rad/frame
    translation_vel: { gen: 0.0241, real: 0.0171 }, // m/frame
    translation_abs: { gen: 0.514, real: 0.646 }, // m
  },
  functional: {
    r1: { gen: 0.45, gt: 0.513, published: 0.511 },
    diversity: { gen: 8.61, gt: 9.79, published: 9.5 },
  },
};

const VEL_ROWS = [
  ["angular_vel", "angular velocity", "rad/frame"],
  ["translation_vel", "translation velocity", "m/frame"],
  ["translation_abs", "|translation|", "m"],
];

// Pick a run name from the eval-runs list. Prefer the corrected `-masked` sweep
// (the true, better numbers) over the pre-mask run when both are present.
function pickRun(runs, kind) {
  const names = runs.map((r) => r.run);
  const masked = names.find((n) => n.includes(kind) && n.toLowerCase().includes("mask"));
  return masked || names.find((n) => n.includes(kind)) || null;
}

export default function ValidationPanel() {
  const [runs, setRuns] = useState([]);
  const [table, setTable] = useState(null);
  const [forensics, setForensics] = useState(null); // live payload or null → static
  const [error, setError] = useState(null);

  // Resolve the mid / base eval runs, then fetch their best-FID rows + forensics.
  useEffect(() => {
    let alive = true;
    (async () => {
      let evalRuns = [];
      try {
        evalRuns = await api.evalRuns();
      } catch (e) {
        if (alive) setError(e.message);
      }
      if (!alive) return;
      setRuns(evalRuns);

      const midRun = pickRun(evalRuns, "mid");
      const baseRun = pickRun(evalRuns, "base");
      const wanted = [midRun, baseRun].filter(Boolean);
      if (wanted.length) {
        try {
          const t = await api.analysisTable(wanted);
          if (alive) setTable({ ...t, midRun, baseRun });
        } catch {
          /* keep static fallbacks */
        }
      }
      if (midRun) {
        try {
          const f = await api.forensics(midRun);
          if (alive) setForensics(f);
        } catch {
          /* 404 / offline → static values with a TODO note */
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Live best-FID per run from the comparison rows (undefined → fallback).
  const fidOf = (run) => table?.rows?.find((r) => r.run === run)?.fid;
  const midFid = fidOf(table?.midRun) ?? GAP.midFallback;
  const baseFid = fidOf(table?.baseRun) ?? GAP.baseFallback;
  const midLive = fidOf(table?.midRun) != null;
  const baseLive = fidOf(table?.baseRun) != null;

  const fx = forensics || FORENSICS_STATIC;
  const fxLive = !!forensics;

  return (
    <div className="surface space-y-8 p-5">
      {error && (
        <p className="text-[12px] text-[var(--amber)]">
          eval runs unavailable ({error}) — showing validated reference values from the write-up.
        </p>
      )}

      <GapHeadline midFid={midFid} baseFid={baseFid} midLive={midLive} baseLive={baseLive} />
      <Scorecard />
      <Bugs />
      <Forensics fx={fx} live={fxLive} />
    </div>
  );
}

// ── block 1 · the gap (log-scale lollipop) ──────────────────────────────────

function GapHeadline({ midFid, baseFid, midLive, baseLive }) {
  const bars = [
    { label: "GT vs GT (floor)", v: GAP.gtgt, color: SLATE, note: "measurement floor" },
    { label: "Paper — RMG-main", v: GAP.paper, color: AMBER, note: "460M / 600k" },
    { label: "Ours — mid", v: midFid, color: SIGNAL, note: `112M / 300k${midLive ? " · live" : " · doc"}`, hi: true },
    { label: "Base", v: baseFid, color: SLATE, note: `25M${baseLive ? " · live" : " · doc"}` },
  ];
  const lo = Math.min(...bars.map((b) => b.v));
  const hi = Math.max(...bars.map((b) => b.v));
  const L = Math.log10, span = L(hi) - L(lo) || 1;
  const frac = (v) => 0.02 + 0.98 * ((L(v) - L(lo)) / span); // keep a stub for the floor
  const ratio = Math.round(midFid / GAP.paper);

  return (
    <Block title="The gap" sub="FID vs the paper — is a 22× gap a bug?">
      <div className="space-y-3">
        {bars.map((b) => (
          <div key={b.label} className="flex items-center gap-3">
            <div className={`w-36 shrink-0 text-right text-[12px] ${b.hi ? "text-white" : "text-slate-400"}`}>
              {b.label}
              <div className="label mt-0.5 normal-case tracking-normal">{b.note}</div>
            </div>
            <div className="relative h-6 flex-1 rounded bg-ink">
              <div
                className="absolute inset-y-0 left-0 rounded"
                style={{ width: `${frac(b.v) * 100}%`, background: b.color, opacity: b.hi ? 1 : 0.55 }}
              />
              <span
                className="absolute inset-y-0 flex items-center pl-2 font-mono text-[12px]"
                style={{ left: `${frac(b.v) * 100}%`, color: b.color }}
              >
                {b.v < 0.01 ? b.v.toFixed(4) : b.v.toFixed(b.v < 1 ? 3 : 2)}
              </span>
            </div>
          </div>
        ))}
      </div>
      <p className="label mt-3 normal-case tracking-normal">
        log scale · mid sits <span className="font-mono text-[var(--signal)]">≈ {ratio}×</span> above the paper's{" "}
        {GAP.paper} — the gap decomposes into ¼ the parameters and ½ the training steps, not a defect.
      </p>
    </Block>
  );
}

// ── block 2 · pipeline scorecard ────────────────────────────────────────────

function Scorecard() {
  return (
    <Block title="Pipeline scorecard" sub="every stage audited — each is faithful">
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
        {SCORECARD.map(([stage, note]) => (
          <div key={stage} className="flex gap-2.5 rounded-lg border border-[var(--hairline)] bg-ink px-3 py-2.5">
            <span className="mt-0.5 shrink-0 font-mono text-[13px] text-[var(--signal)]">✓</span>
            <div>
              <div className="text-[13px] font-medium text-slate-200">{stage}</div>
              <div className="label mt-0.5 normal-case tracking-normal leading-snug">{note}</div>
            </div>
          </div>
        ))}
      </div>
    </Block>
  );
}

// ── block 3 · two harness bugs found + fixed ────────────────────────────────

function Bugs() {
  return (
    <Block title="Harness bugs found + fixed" sub="two evaluation-side defects, corrected">
      <div className="grid gap-3 md:grid-cols-2">
        {BUGS.map((b) => (
          <div key={b.title} className="rounded-lg border border-[var(--hairline)] bg-ink p-3.5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-[13px] font-medium text-white">{b.title}</span>
              <span className="rounded border border-[var(--hairline-strong)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--muted)]">
                {b.tag}
              </span>
            </div>
            <div className="mt-1.5 font-mono text-[13px] text-[var(--signal)]">{b.delta}</div>
            <p className="mt-1.5 text-[12px] leading-snug text-slate-400">{b.mechanism}</p>
          </div>
        ))}
      </div>
    </Block>
  );
}

// ── block 4 · generation forensics ──────────────────────────────────────────

function Forensics({ fx, live }) {
  const qn = fx.velocity?.quat_norm;
  return (
    <Block
      title="Generation forensics"
      sub={`gen vs real motion — valid, conditioned, diverse${live ? "" : " · static (TODO: forensics endpoint)"}`}
    >
      <div className="grid gap-5 lg:grid-cols-2">
        {/* velocity / dynamics — gen vs real grouped bars */}
        <div>
          <div className="label mb-2">motion dynamics · gen vs real</div>
          {qn && (
            <div className="mb-3 flex items-center gap-2 text-[12px] text-slate-400">
              <span className="font-mono text-[var(--signal)]">✓</span>
              quaternion norm {qn.gen.toFixed(4)} / {qn.real.toFixed(4)} — rotations perfectly valid
            </div>
          )}
          <div className="space-y-3">
            {VEL_ROWS.map(([key, label, unit]) => (
              <GenRealBar key={key} label={label} unit={unit} d={fx.velocity?.[key]} />
            ))}
          </div>
          <p className="label mt-3 normal-case tracking-normal">
            ~36 % jitterier (angular velocity) and conservative (smaller |translation|) — the fingerprint of a small,
            half-trained model.
          </p>
        </div>

        {/* functional — gen vs GT ceiling vs published */}
        <div>
          <div className="label mb-2">functional · gen vs GT ceiling vs published</div>
          <div className="space-y-3">
            <TripleBar label="R@1 ↑" d={fx.functional?.r1} fmt={(v) => v.toFixed(3)} />
            <TripleBar label="Diversity" d={fx.functional?.diversity} fmt={(v) => v.toFixed(2)} />
          </div>
          <p className="label mt-3 normal-case tracking-normal">
            R@1 is 6× above chance (text conditioning genuinely works) and diversity is 88 % of GT (not collapsed).
          </p>
        </div>
      </div>
    </Block>
  );
}

// Two-series (gen amber vs real cyan) horizontal bars, scaled to the pair max.
function GenRealBar({ label, unit, d }) {
  if (!d) return null;
  const max = Math.max(d.gen, d.real) || 1;
  const rows = [
    ["gen", d.gen, AMBER],
    ["real", d.real, SIGNAL],
  ];
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[12px]">
        <span className="text-slate-300">{label}</span>
        <span className="label normal-case tracking-normal">{unit}</span>
      </div>
      {rows.map(([role, v, color]) => (
        <div key={role} className="mb-1 flex items-center gap-2">
          <span className="w-8 shrink-0 font-mono text-[10px] text-[var(--muted)]">{role}</span>
          <div className="relative h-4 flex-1 rounded bg-ink">
            <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${(v / max) * 100}%`, background: color }} />
          </div>
          <span className="w-14 shrink-0 text-right font-mono text-[11px]" style={{ color }}>
            {v.toFixed(v < 1 ? 4 : 3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// Three-series (gen / GT ceiling / published) horizontal bars, scaled to group max.
function TripleBar({ label, d, fmt }) {
  if (!d) return null;
  const max = Math.max(d.gen, d.gt, d.published) || 1;
  const rows = [
    ["gen", d.gen, AMBER],
    ["GT ceiling", d.gt, SIGNAL],
    ["published", d.published, SLATE],
  ];
  return (
    <div>
      <div className="mb-1 text-[12px] text-slate-300">{label}</div>
      {rows.map(([role, v, color]) => (
        <div key={role} className="mb-1 flex items-center gap-2">
          <span className="w-20 shrink-0 font-mono text-[10px] text-[var(--muted)]">{role}</span>
          <div className="relative h-4 flex-1 rounded bg-ink">
            <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${(v / max) * 100}%`, background: color }} />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-[11px]" style={{ color }}>
            {fmt(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── shared block chrome ─────────────────────────────────────────────────────

function Block({ title, sub, children }) {
  return (
    <div>
      <div className="mb-3 flex items-baseline gap-2.5">
        <h3 className="text-[13px] font-semibold text-white">{title}</h3>
        <span className="label">{sub}</span>
      </div>
      {children}
    </div>
  );
}
