"use client";

// ── Constraint Analysis ────────────────────────────────────────────────────
// Turns the "did the constraint actually take?" question into a chart, the way
// the temp matplotlib figures did offline. For every active pin / hinge we
// recover the joint's realized BEND ANGLE per frame straight from the position
// .npy (bendSeries) and plot it against the target the user asked for:
//
//   • pin   → a dashed target line; the trace should snap to it,
//   • hinge → a shaded [min,max] band; the trace should stay inside it,
//   • the active frame window is shaded so partial-window constraints read,
//   • the unconstrained baseline (◇) is overlaid so you can see what the model
//     *wanted* to do vs what the projection forced.
//
// A satisfaction readout (held %, mean bend, worst violation) sits beside each
// chart — the numbers behind the curve.

import { useMemo } from "react";
import LineChart from "./LineChart";
import { bendSeries, constraintSatisfaction } from "@/lib/motionMetrics";

const TOL_DEG = 5; // a frame counts as "held" within ±5° of the target / band

export default function ConstraintAnalysis({
  joints,
  baselineJoints,
  pins = [],
  ranges = [],
  parents,
  children,
  jointNames,
  fps = 20,
}) {
  const items = useMemo(() => {
    const out = [];
    for (const p of pins) out.push({ kind: "pin", c: p, target: { bend: Number(p.bend_deg) } });
    for (const r of ranges)
      out.push({ kind: "hinge", c: r, target: { min: Number(r.bend_min), max: Number(r.bend_max) } });
    return out;
  }, [pins, ranges]);

  if (!joints) {
    return (
      <section className="surface p-5">
        <div className="label">constraint analysis</div>
        <p className="mt-3 text-[13px] text-[var(--muted)]">
          Sample a constrained clip to chart each joint's realized bend angle against its
          target. Hit ◇ baseline first to overlay the unconstrained motion.
        </p>
      </section>
    );
  }
  if (items.length === 0) {
    return (
      <section className="surface p-5">
        <div className="label">constraint analysis</div>
        <p className="mt-3 text-[13px] text-[var(--muted)]">
          No active constraints — add a pin or hinge to see whether the projection holds it
          across the clip.
        </p>
      </section>
    );
  }

  return (
    <section className="surface p-5">
      <div className="mb-1 flex items-center justify-between">
        <span className="label">constraint analysis · {items.length}</span>
        <span className="text-[10px] text-[var(--muted)]">
          realized bend vs target {baselineJoints ? "· vs free baseline" : "· ◇ for baseline"}
        </span>
      </div>
      <p className="mb-4 text-[11px] leading-relaxed text-[var(--muted)]">
        Bend angle recovered from joint positions (0° = straight). The trace should ride the
        amber pin line / green hinge band inside the shaded window.
      </p>
      <div className="space-y-5">
        {items.map((it, i) => (
          <ConstraintCard
            key={`${it.kind}-${it.c.joint}-${it.c.id ?? i}`}
            item={it}
            joints={joints}
            baselineJoints={baselineJoints}
            parents={parents}
            children={children}
            jointNames={jointNames}
            fps={fps}
          />
        ))}
      </div>
    </section>
  );
}

function ConstraintCard({ item, joints, baselineJoints, parents, children, jointNames, fps }) {
  const { kind, c, target } = item;
  const jointIdx = jointNames.indexOf(c.joint);
  const T = joints.shape[0];
  const win = {
    start: Number(c.frame_start) || 0,
    end: c.frame_end === "" || c.frame_end == null ? T : Number(c.frame_end),
  };

  const series = jointIdx >= 0 ? bendSeries(joints, jointIdx, parents, children) : null;
  const baseSeries =
    baselineJoints && jointIdx >= 0 ? bendSeries(baselineJoints, jointIdx, parents, children) : null;

  if (!series) {
    return (
      <div className="rounded-lg border border-[var(--hairline)] p-3">
        <CardHead kind={kind} c={c} target={target} />
        <p className="mt-2 text-[12px] text-[var(--muted)]">
          {c.joint} is an end-effector (no child bone) — its bend can't be read from positions.
        </p>
      </div>
    );
  }

  const sat = constraintSatisfaction(series, win, target, TOL_DEG);
  const toPts = (s) =>
    Array.from(s, (v, f) => [f, v]).filter(([, v]) => !Number.isNaN(v));

  const chartSeries = [];
  if (baseSeries) chartSeries.push({ label: "free (baseline)", color: "#64748b", points: toPts(baseSeries) });
  chartSeries.push({ label: "constrained", color: "var(--signal)", points: toPts(series) });

  const hlines = kind === "pin" ? [{ y: target.bend, color: "#fbbf24", label: "target" }] : [];
  const bands = kind === "hinge" ? [{ y0: target.min, y1: target.max, color: "rgba(52,211,153,0.14)" }] : [];
  const regions =
    win.start > 0 || win.end < T ? [{ x0: win.start, x1: win.end, color: "rgba(148,163,184,0.10)" }] : [];

  const held = sat ? sat.frac >= 0.95 : false;
  const heldColor = !sat ? "var(--muted)" : sat.frac >= 0.95 ? "var(--signal)" : sat.frac >= 0.7 ? "var(--amber)" : "#f87171";

  return (
    <div className="rounded-lg border border-[var(--hairline)] p-3">
      <CardHead kind={kind} c={c} target={target} />
      <div className="mt-3 overflow-x-auto">
        <LineChart
          series={chartSeries}
          bands={bands}
          hlines={hlines}
          regions={regions}
          width={520}
          height={200}
          xLabel="frame"
          yLabel="bend (°)"
        />
      </div>
      {sat && (
        <div className="mt-2 grid grid-cols-3 gap-3">
          <Stat label="held" value={`${(sat.frac * 100).toFixed(0)}%`} color={heldColor}
            note={held ? "within ±5°" : "drifting"} />
          <Stat label="mean bend" value={`${sat.meanBend.toFixed(1)}°`} />
          <Stat label="worst miss" value={`${sat.maxViol.toFixed(1)}°`}
            color={sat.maxViol <= TOL_DEG ? "var(--signal)" : sat.maxViol <= 15 ? "var(--amber)" : "#f87171"} />
        </div>
      )}
    </div>
  );
}

function CardHead({ kind, c, target }) {
  const desc =
    kind === "pin" ? `pin ${target.bend}°` : `hinge [${target.min}, ${target.max}]°`;
  const color = kind === "pin" ? "#fbbf24" : "#34d399";
  const win =
    c.frame_end === "" || c.frame_end == null
      ? `${Number(c.frame_start) || 0}→end`
      : `${Number(c.frame_start) || 0}→${c.frame_end}`;
  return (
    <div className="flex items-center gap-2">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      <span className="font-mono text-[12px] text-slate-200">{c.joint}</span>
      <span className="rounded border border-[var(--hairline)] px-1.5 text-[9px] uppercase tracking-wider text-[var(--muted)]">
        {kind}
      </span>
      <span className="font-mono text-[11px] text-[var(--muted)]">{desc}</span>
      <span className="ml-auto font-mono text-[10px] text-[var(--muted)]">frames {win}</span>
    </div>
  );
}

function Stat({ label, value, color = "var(--slate-100)", note }) {
  return (
    <div>
      <div className="label mb-0.5">{label}</div>
      <div className="font-mono text-[14px]" style={{ color: color === "var(--slate-100)" ? "#f1f5f9" : color }}>
        {value}
      </div>
      {note && <div className="text-[10px] text-[var(--muted)]">{note}</div>}
    </div>
  );
}
