"use client";

import { useRef } from "react";
import { TableTools } from "./ExportButtons";

// Harness calibration — the falsifiable evidence that the eval instrument can
// be trusted: on ground-truth motions it must reproduce the published
// ground-truth statistics of the Guo evaluator, and generated rotations must
// be valid by construction. These are dated, one-time calibration experiments
// (eval_sanity on the full test split, lemma-token protocol) — a calibration
// certificate, not a live dashboard; re-run eval_sanity if the harness changes.
//
// Everything else the old panel showed (pipeline audit prose, harness-bug
// history) is documentation, not measurement — kept under a collapsed audit
// trail so the section leads with evidence.

const CHECKS = [
  {
    check: "FID(GT, GT) — measurement floor",
    measured: "0.0019",
    reference: "≈ 0.002",
    note: "two-pass unit-length crop protocol, full split",
  },
  {
    check: "GT retrieval R@1 — pairing ceiling",
    measured: "0.513",
    reference: "0.511",
    note: "text↔motion matching on real clips",
  },
  {
    check: "GT MM-Dist",
    measured: "3.10",
    reference: "≈ 2.97",
    note: "text–motion embedding distance, real clips",
  },
  {
    check: "GT Diversity",
    measured: "9.79",
    reference: "9.50",
    note: "embedding spread, real clips",
  },
  {
    check: "Generated quaternion norm",
    measured: "1.0000",
    reference: "1.0 by construction",
    note: "manifold sampling yields valid rotations",
  },
];

// Audit trail (collapsed): the 2026-07 end-to-end audit + the two harness bugs
// found and fixed by it. History, kept for provenance.
const AUDIT = [
  ["Data processing", "bit-exact to upstream HumanML3D (X-flip, uniform_skeleton, IK)"],
  ["Representation", "(T,R) → FK → 263-D features bit-equal to official new_joint_vecs (0.0% rel. diff)"],
  ["Training", "Riemannian CFM, tangent projection, EMA 0.9999, effective BS 256 (= paper)"],
  ["Sampling", "Riemannian Euler + Exp-map stepping, CFG combined in ambient then projected"],
  ["Evaluation", "loads the Guo evaluator's own mean/std; seeded-shuffled subsets"],
];

const BUGS = [
  [
    "Motion-embedding pairing (fixed 2026-07-10)",
    "encode_motion mis-paired text↔motion on length ties → R-precision under-reported. GT R@1 0.34 → 0.513 (= published); FID unaffected (permutation-invariant).",
  ],
  [
    "Missing length mask at generation (fixed 2026-07-12, now default)",
    "sampling without per-clip validity masks let short clips attend across batch padding. Mid full-split FID 0.96 → 0.607.",
  ],
];

export default function ValidationPanel() {
  const calibRef = useRef(null);
  return (
    <div className="surface p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <span className="label">
          on ground-truth inputs, the harness must reproduce the published GT statistics — it does
        </span>
        <span className="flex items-center gap-2 font-mono text-[10px] text-[var(--muted)]">
          <span>measured 2026-07-09/10 · eval_sanity · full test split</span>
          <TableTools getTable={() => calibRef.current} name="harness-calibration"
            caption="Harness calibration on ground-truth inputs vs published statistics." label="tab:harness-calibration" />
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink px-3 py-2">
        <table ref={calibRef} className="w-full border-collapse text-left text-[12px]">
          <thead>
            <tr className="border-b border-[var(--hairline-strong)] text-slate-300">
              <th className="py-2 pr-4 font-medium">check</th>
              <th className="py-2 pr-4 font-medium">measured</th>
              <th className="py-2 pr-4 font-medium">published / expected</th>
              <th className="py-2 pr-4 font-medium" />
            </tr>
          </thead>
          <tbody>
            {CHECKS.map((c) => (
              <tr key={c.check} className="border-b border-[var(--hairline)]">
                <td className="py-2 pr-4 text-slate-300">
                  {c.check}
                  <div className="label mt-0.5 normal-case tracking-normal">{c.note}</div>
                </td>
                <td className="py-2 pr-4 font-mono text-[var(--signal)]">{c.measured}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{c.reference}</td>
                <td className="py-2 pr-4 font-mono text-[var(--signal)]">✓</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="label mt-3 normal-case tracking-normal">
        implication: the metrics in 02 are on the official scale — the remaining gap to the paper's 0.043
        (460M / 600k steps vs our 112M / 300k) decomposes into capacity and training budget, not measurement error.
      </p>

      <details className="mt-4">
        <summary className="label cursor-pointer">audit trail · pipeline audit + harness bugs found &amp; fixed</summary>
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          <div>
            <div className="label mb-2">end-to-end audit (2026-07, no defect found)</div>
            <ul className="space-y-1.5 text-[12px]">
              {AUDIT.map(([stage, note]) => (
                <li key={stage} className="flex gap-2">
                  <span className="shrink-0 font-mono text-[var(--signal)]">✓</span>
                  <span className="text-slate-400"><span className="text-slate-200">{stage}</span> — {note}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="label mb-2">harness bugs found by the audit</div>
            <ul className="space-y-2 text-[12px]">
              {BUGS.map(([title, note]) => (
                <li key={title} className="rounded-lg border border-[var(--hairline)] bg-ink px-3 py-2 text-slate-400">
                  <span className="text-slate-200">{title}</span> — {note}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </details>
    </div>
  );
}
