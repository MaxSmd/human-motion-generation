"use client";

// The Studio: pick a motion, optionally add a hard joint constraint and/or an
// extra phrase, "generate" (a choreographed illusion over a pre-rendered clip),
// then inspect the result and flip through the whole study — ground truth, the
// plain sample, and each combination of constraint and text — with live
// analysis. Everything is already rendered; the flips are instant.

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import GenerationProgress, { buildPlan } from "@/components/GenerationProgress";
import ClipStats from "@/components/ClipStats";
import Viewport from "@/components/Viewport";
import { SAMPLER, SCENARIOS, comboKey, resolveVariant, dataUrl } from "@/lib/catalog";

const SkeletonPlayer = dynamic(() => import("@/components/SkeletonPlayer"), { ssr: false });

export default function ConstraintStudio() {
  const [sid, setSid] = useState(SCENARIOS[0].id);
  const [constraint, setConstraint] = useState(false);
  const [text, setText] = useState(false);
  const [phase, setPhase] = useState("idle"); // idle | generating | result
  const [plan, setPlan] = useState([]);
  const [view, setView] = useState(null); // active combo key shown in result, or "gt"

  const scenario = useMemo(() => SCENARIOS.find((s) => s.id === sid), [sid]);
  const hasC = !!scenario.constraint;
  const hasT = !!scenario.text;
  const variant = resolveVariant(scenario, constraint, text);

  const pick = (id) => {
    const s = SCENARIOS.find((x) => x.id === id);
    setSid(id);
    setConstraint(false);
    setText(false);
    setPhase("idle");
    setView(null);
  };

  const generate = () => {
    if (!variant) return;
    const cs = constraint ? [{ id: scenario.id, stage: `projecting constraint · ${scenario.joint}` }] : [];
    setPlan(buildPlan(cs, SAMPLER.steps));
    setPhase("generating");
  };

  const onDone = () => {
    setView(comboKey(constraint, text));
    setPhase("result");
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      {/* control panel */}
      <div className="surface h-fit p-6">
        <div className="label mb-2">motion</div>
        <div className="grid grid-cols-2 gap-2">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              onClick={() => pick(s.id)}
              disabled={phase === "generating"}
              className="rounded-lg border px-2.5 py-2 text-left text-[12.5px] font-semibold transition"
              style={{
                borderColor: s.id === sid ? "var(--signal)" : "var(--hairline)",
                background: s.id === sid ? "var(--signal-dim)" : "rgba(0,0,0,0.2)",
                color: s.id === sid ? "var(--signal)" : "var(--text)",
              }}
            >
              {s.title}
            </button>
          ))}
        </div>

        <div className="mt-5 rounded-lg border border-[var(--hairline)] bg-black/20 p-3 font-mono text-[12px] leading-snug text-[var(--text)]">
          “{scenario.prompt}
          {text && scenario.text ? ` ${scenario.text.phrase}` : ""}”
        </div>

        {(hasC || hasT) && <div className="label mb-2 mt-6">options</div>}
        {hasC && (
          <Toggle
            on={constraint}
            set={setConstraint}
            disabled={phase === "generating"}
            label={scenario.constraint.label}
            detail={scenario.constraint.detail}
          />
        )}
        {hasT && (
          <Toggle
            on={text}
            set={setText}
            disabled={phase === "generating"}
            label={scenario.text.label}
            detail="an added phrase in the caption, no constraint of its own"
          />
        )}

        <button onClick={generate} disabled={phase === "generating" || !variant} className="btn-signal mt-6 w-full">
          {phase === "generating" ? "loading…" : "Show clip"}
        </button>
        {!variant && (
          <div className="mt-2 text-center text-[11px] text-[var(--warn)]">this combination was not rendered</div>
        )}
        <div className="mt-3 text-center font-mono text-[10px] text-[var(--muted)]">
          {SAMPLER.model} · ω {SAMPLER.omega} · {SAMPLER.steps} ODE steps
        </div>
      </div>

      {/* stage */}
      <div>
        {phase === "idle" && <Idle scenario={scenario} />}
        {phase === "generating" && <GenerationProgress plan={plan} onDone={onDone} />}
        {phase === "result" && <Result scenario={scenario} view={view} setView={setView} onRegen={generate} />}
      </div>
    </div>
  );
}

function Toggle({ on, set, disabled, label, detail }) {
  return (
    <button
      onClick={() => set((v) => !v)}
      disabled={disabled}
      className="mb-2 w-full rounded-lg border p-3 text-left transition"
      style={{
        borderColor: on ? "var(--signal)" : "var(--hairline)",
        background: on ? "var(--signal-dim)" : "rgba(0,0,0,0.2)",
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[12.5px] font-semibold" style={{ color: on ? "var(--signal)" : "var(--text)" }}>{label}</span>
        <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: on ? "var(--signal)" : "var(--muted)" }}>
          {on ? "on" : "off"}
        </span>
      </div>
      <div className="mt-1 font-mono text-[10.5px] leading-snug text-[var(--muted)]">{detail}</div>
    </button>
  );
}

function Idle({ scenario }) {
  return (
    <div className="surface grid place-items-center p-6 text-center" style={{ minHeight: 460 }}>
      <div>
        <div className="display text-[22px] text-[var(--text)]">Select a configuration</div>
        <p className="mx-auto mt-3 max-w-sm text-[13px] leading-relaxed text-[var(--muted)]">
Samples are generated on ℝ³ × (S³)²². A constraint is applied by projecting one joint back
          onto its allowed set after each of the {SAMPLER.steps} integration steps, without changing the
          weights.
        </p>
      </div>
    </div>
  );
}

// Build the list of study variants that exist for a scenario, in study order.
function studyViews(scenario) {
  const order = [
    { key: "gt", label: "ground truth", tag: "real" },
    { key: "", label: "generated", tag: "plain" },
    { key: "c", label: "+ constraint", tag: "constraint" },
    { key: "t", label: "+ text", tag: "text" },
    { key: "ct", label: "+ both", tag: "constraint + text" },
  ];
  return order.filter((o) => (o.key === "gt" ? scenario.gt : scenario.variants[o.key]));
}

function Result({ scenario, view, setView, onRegen }) {
  const views = studyViews(scenario);
  const isGt = view === "gt";
  const file = isGt ? `${scenario.gt}.npy` : scenario.variants[view].file;
  const clip = [{ url: dataUrl(file), color: isGt ? "var(--real)" : "#22d3ee", label: isGt ? "ground truth" : "generated" }];
  const v = isGt ? null : scenario.variants[view];
  const showLimb = !isGt && (view === "c" || view === "ct") && scenario.limb;

  return (
    <div className="animate-fade-up space-y-4">
      {views.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {views.map((o) => (
            <button
              key={o.key}
              onClick={() => setView(o.key)}
              className="rounded-lg border px-3 py-1.5 text-[12px] font-semibold transition"
              style={{
                borderColor: o.key === view ? "var(--signal)" : "var(--hairline)",
                background: o.key === view ? "var(--signal-dim)" : "rgba(0,0,0,0.2)",
                color: o.key === view ? "var(--signal)" : o.key === "gt" ? "var(--real)" : "var(--text)",
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}

      <div className="surface p-4">
        <Viewport height={420}>
          <SkeletonPlayer key={view} clips={clip} height={420} align highlight={showLimb ? scenario.limb : null} accent="#fbbf24" />
        </Viewport>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            {v?.held && (
              <Chip k={scenario.joint || "joint"} v={v.held} tone={view === "c" || view === "ct" ? "good" : "text"} />
            )}
            {v?.jerk != null && <Chip k="jerk" v={v.jerk.toFixed(4)} tone={v.jerk > 0.05 ? "warn" : "text"} />}
          </div>
          <button onClick={onRegen} className="btn-ghost text-[12px]" title="replays the same stored clip">↻ replay</button>
        </div>
      </div>

      <div className="surface p-4">
        <div className="label mb-3">analysis</div>
        <ClipStats key={view} clips={clip} />
      </div>

      {scenario.note && (
        <p className="text-[12.5px] leading-relaxed text-[var(--muted)]">{scenario.note}</p>
      )}
    </div>
  );
}

function Chip({ k, v, tone }) {
  return (
    <div className="rounded-lg border border-[var(--hairline)] bg-black/20 px-2.5 py-1.5">
      <div className="label mb-1">{k}</div>
      <div className="font-mono text-[12px]" style={{ color: tone === "warn" ? "var(--warn)" : tone === "good" ? "var(--real)" : "var(--text)" }}>{v}</div>
    </div>
  );
}
