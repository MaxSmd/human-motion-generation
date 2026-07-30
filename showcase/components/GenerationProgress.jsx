"use client";

// The loading panel shown while a stored clip is fetched. Given a staged plan
// naming the steps the real RMG sampler took (encode text → Riemannian-Euler ODE
// → constraint projection → FK render), it animates a progress bar and a
// scrolling log, then calls onDone. No compute happens here: the clip was
// sampled on the cluster, and the panel states the settings it was sampled at.

import { useEffect, useRef, useState } from "react";

// Build the stage plan for a given active-constraint list. Durations (ms) get a
// little per-run jitter so repeated generations don't feel canned.
export function buildPlan(constraints, steps = 800) {
  const j = () => 0.88 + Math.random() * 0.24;
  const plan = [
    { key: "text", label: "encoding caption · Qwen3-Embedding-0.6B", dur: 460 * j() },
    { key: "ode", label: "sampling · Riemannian Euler on ℝ³×(S³)²²", dur: 1850 * j(), steps },
  ];
  for (const c of constraints) plan.push({ key: `c:${c.id}`, label: c.stage, dur: 520 * j() });
  plan.push({ key: "fk", label: "forward kinematics · rasterising skeleton", dur: 470 * j() });
  return plan;
}

export default function GenerationProgress({ plan, onDone }) {
  const [pct, setPct] = useState(0);
  const [stageIdx, setStageIdx] = useState(0);
  const [log, setLog] = useState([]);
  const doneRef = useRef(false);

  useEffect(() => {
    const total = plan.reduce((s, st) => s + st.dur, 0);
    const bounds = []; // cumulative end time per stage
    let acc = 0;
    for (const st of plan) { acc += st.dur; bounds.push(acc); }

    const t0 = performance.now();
    let seenStage = -1;
    let raf;

    const tick = (now) => {
      const el = now - t0;
      setPct(Math.min(100, (el / total) * 100));

      let idx = bounds.findIndex((b) => el < b);
      if (idx === -1) idx = plan.length - 1;
      setStageIdx(idx);

      // append a log line each time we enter a new stage
      if (idx > seenStage) {
        for (let k = seenStage + 1; k <= idx; k++) {
          const st = plan[k];
          setLog((L) => [...L, { label: st.label, done: false }]);
        }
        // mark all previous stages done
        setLog((L) => L.map((e, i) => (i < idx ? { ...e, done: true } : e)));
        seenStage = idx;
      }

      if (el >= total) {
        if (!doneRef.current) {
          doneRef.current = true;
          setPct(100);
          setLog((L) => L.map((e) => ({ ...e, done: true })));
          setTimeout(onDone, 180);
        }
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [plan, onDone]);

  const stage = plan[stageIdx] || plan[plan.length - 1];

  return (
    <div className="surface animate-fade-up p-6" style={{ minHeight: 460 }}>
      <div className="flex items-center gap-3">
        <span className="dot animate-pulse-soft" style={{ color: "var(--signal)", background: "var(--signal)" }} />
        <span className="label" style={{ color: "var(--signal)" }}>loading</span>
        <span className="font-mono text-[9.5px] uppercase tracking-[0.16em] text-[var(--warn)]">pre-computed</span>
        <span className="ml-auto font-mono text-[12px] text-[var(--muted)]">{Math.round(pct)}%</span>
      </div>

      <div className="mt-4 font-mono text-[13px] text-[var(--text)]">{stage?.label}</div>

      {/* main progress bar */}
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-black/40">
        <div className="h-full rounded-full transition-[width] duration-100"
          style={{ width: `${pct}%`, background: "linear-gradient(90deg,#2fdcf0,#18b6d4)" }} />
      </div>

      {/* The settings the stored clip was sampled at. A ticking step counter sat
          here once, which animated a computation that is not happening. */}
      {stage?.key === "ode" && (
        <div className="mt-3 flex items-baseline justify-between font-mono text-[12px]">
          <span className="text-[var(--muted)]">sampled at</span>
          <span className="text-[var(--signal)]">ω 6.5 · {stage.steps} ODE steps</span>
        </div>
      )}

      {/* scrolling log */}
      <div className="mt-5 space-y-1.5 rounded-lg border border-[var(--hairline)] bg-black/25 p-3 font-mono text-[11.5px]">
        {log.map((e, i) => (
          <div key={i} className="flex items-center gap-2">
            <span style={{ color: e.done ? "var(--signal)" : "var(--muted)" }}>
              {e.done ? "✓" : "▸"}
            </span>
            <span style={{ color: e.done ? "var(--text)" : "var(--muted)" }}>{e.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
