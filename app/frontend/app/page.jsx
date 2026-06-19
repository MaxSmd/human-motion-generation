"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import GenerateTab from "@/components/GenerateTab";
import GTBrowserTab from "@/components/GTBrowserTab";
import TrainingViewerTab from "@/components/TrainingViewerTab";
import ConstraintsTab from "@/components/ConstraintsTab";
import ClusterTab from "@/components/ClusterTab";
import VisualizeTab from "@/components/VisualizeTab";
import ModelTab from "@/components/ModelTab";
import AnalysisTab from "@/components/AnalysisTab";
import ClusterGate from "@/components/ClusterGate";

// three.js / r3f canvas — client-only (no SSR).
const RoomEditor = dynamic(() => import("@/components/RoomEditor"), { ssr: false });
const StudioTab = dynamic(() => import("@/components/StudioTab"), { ssr: false });

export default function Home() {
  const [health, setHealth] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [connError, setConnError] = useState(null);
  const [clusterStatus, setClusterStatus] = useState(null);
  const [queue, setQueue] = useState(null);
  const [modelInfo, setModelInfo] = useState(null); // { active, models, tasks }

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setConnError(e.message));
    api.checkpoints().then(setCheckpoints).catch(() => {});
    api.getModel().then(setModelInfo).catch(() => {});
  }, []);

  const model = modelInfo?.active || "rmg";

  // Switch the global active model, then refetch checkpoints (they're listed per
  // active model on the backend) so downstream pickers reflect the new model.
  async function changeModel(next) {
    if (next === model) return;
    try {
      const info = await api.setModel(next);
      setModelInfo((prev) => ({ ...(prev || {}), ...info }));
      api.checkpoints().then(setCheckpoints).catch(() => {});
    } catch {
      /* ignore — toggle stays on the current model */
    }
  }

  const clusterMode = !!health?.cluster_mode;

  useEffect(() => {
    if (!clusterMode) return;
    const poll = () => {
      api.clusterStatus().then(setClusterStatus).catch(() => {});
      api.queue().then(setQueue).catch(() => {}); // local, no SSH
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, [clusterMode]);

  // Constraints (fixed joint angles) is implemented for RMG; still roadmap-only
  // for MARDM, so show the V2 badge when MARDM is the active model.
  const constraintsSoon = model === "mardm";
  const tabs = useMemo(() => {
    if (clusterMode)
      return [
        { id: "cluster", n: "00", label: "Cluster", sub: "jobs · squeue" },
        { id: "generate", n: "01", label: "Generate", sub: "text → motion" },
        { id: "visualize", n: "02", label: "Visualize", sub: "GT · compare · samples" },
        { id: "model", n: "03", label: "Model", sub: "train · eval" },
        { id: "analysis", n: "04", label: "Analysis", sub: "metrics · plots" },
        { id: "constraints", n: "05", label: "Constraints", sub: "fixed joint angles", soon: constraintsSoon },
        { id: "room", n: "06", label: "Room", sub: "scene · objects · spawn", soon: constraintsSoon },
        { id: "studio", n: "07", label: "Studio", sub: "constraints · live eval", soon: constraintsSoon },
      ];
    return [
      { id: "generate", n: "01", label: "Generate", sub: "text → motion" },
      { id: "gt", n: "02", label: "Ground Truth", sub: "dataset browser" },
      { id: "training", n: "03", label: "Training", sub: "sample scrubber" },
      { id: "constraints", n: "04", label: "Constraints", sub: "fixed joint angles", soon: constraintsSoon },
      { id: "room", n: "05", label: "Room", sub: "scene · objects · spawn", soon: constraintsSoon },
      { id: "studio", n: "06", label: "Studio", sub: "constraints · live eval", soon: constraintsSoon },
    ];
  }, [clusterMode, constraintsSoon]);

  const [tab, setTab] = useState("generate");
  useEffect(() => {
    if (clusterMode) setTab("cluster");
  }, [clusterMode]);

  const panel = (
    <>
      <nav className="mb-8 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {tabs.map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`group relative overflow-hidden rounded-xl border px-4 py-3 text-left transition ${
                active
                  ? "border-[var(--signal)] bg-[var(--signal-dim)]"
                  : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`font-mono text-[11px] tracking-widest ${active ? "text-[var(--signal)]" : "text-[var(--muted)]"}`}>
                  {t.n}
                </span>
                {t.soon && (
                  <span className="rounded-full border border-[var(--amber)]/40 px-1.5 py-0.5 text-[9px] font-semibold tracking-wider text-[var(--amber)]">
                    V2
                  </span>
                )}
              </div>
              <div className={`display mt-1.5 text-base font-bold ${active ? "text-white" : "text-slate-300"}`}>
                {t.label}
              </div>
              <div className="label mt-0.5">{t.sub}</div>
              {active && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-[var(--signal)]" />}
            </button>
          );
        })}
      </nav>

      <div key={tab} className="animate-fade-up">
        {tab === "cluster" && <ClusterTab status={clusterStatus} />}
        {tab === "generate" &&
          (model === "mardm" ? (
            <MardmUnsupported feature="Generate" />
          ) : (
            <GenerateTab checkpoints={checkpoints} clusterMode={clusterMode} />
          ))}
        {tab === "visualize" &&
          (model === "mardm" ? <MardmUnsupported feature="Visualize" /> : <VisualizeTab />)}
        {tab === "model" && <ModelTab model={model} />}
        {tab === "analysis" && <AnalysisTab />}
        {tab === "gt" && <GTBrowserTab clusterMode={clusterMode} />}
        {tab === "training" && <TrainingViewerTab clusterMode={clusterMode} />}
        {tab === "constraints" &&
          (model === "mardm" ? (
            <MardmUnsupported feature="Constraints" />
          ) : (
            <ConstraintsTab checkpoints={checkpoints} clusterMode={clusterMode} />
          ))}
        {tab === "room" &&
          (model === "mardm" ? (
            <MardmUnsupported feature="Room" />
          ) : (
            <RoomEditor clusterMode={clusterMode} checkpoints={checkpoints} />
          ))}
        {tab === "studio" &&
          (model === "mardm" ? (
            <MardmUnsupported feature="Studio" />
          ) : (
            <StudioTab clusterMode={clusterMode} checkpoints={checkpoints} />
          ))}
      </div>
    </>
  );

  return (
    <div className="min-h-screen">
      <div className="mx-auto max-w-6xl px-6 pb-20 pt-10">
        <header className="mb-9 flex flex-wrap items-end justify-between gap-6 border-b border-[var(--hairline)] pb-6">
          <div className="animate-fade-up">
            <div className="label mb-2 text-[var(--signal)]">Riemannian Motion Generation</div>
            <h1 className="display text-5xl font-extrabold leading-none text-white">
              RMG<span className="text-[var(--signal)]">.</span>
              <span className="ml-3 align-middle text-lg font-semibold text-[var(--muted)]">motion instrument</span>
            </h1>
            <p className="mt-3 max-w-md text-[13px] leading-relaxed text-[var(--muted)]">
              Drive the SLURM cluster: prompt the model, browse ground-truth clips,
              scrub training samples, and launch viz / train / eval jobs.
            </p>
          </div>
          <div className="flex flex-col items-end gap-3">
            {clusterMode && (
              <ModelToggle model={model} models={modelInfo?.models} onChange={changeModel} />
            )}
            <Telemetry health={health} error={connError} cluster={clusterStatus} clusterMode={clusterMode} queue={queue} />
          </div>
        </header>

        {clusterMode ? <ClusterGate>{panel}</ClusterGate> : panel}
      </div>
    </div>
  );
}

function MardmUnsupported({ feature }) {
  return (
    <div className="surface flex flex-col items-center justify-center gap-3 p-12 text-center">
      <span className="rounded-full border border-[var(--amber)]/40 px-2.5 py-0.5 text-[10px] font-semibold tracking-wider text-[var(--amber)]">
        MARDM
      </span>
      <h3 className="display text-lg font-bold text-slate-200">
        {feature} is not available for MARDM
      </h3>
      <p className="max-w-sm text-[13px] leading-relaxed text-[var(--muted)]">
        MARDM has no standalone viz/generate pipeline yet — its only generation
        path runs inside evaluation. Use{" "}
        <span className="text-[var(--signal)]">Model ▸ Evaluate</span> to score a
        MARDM checkpoint, or switch the model back to{" "}
        <span className="text-[var(--signal)]">rmg</span>.
      </p>
    </div>
  );
}

function ModelToggle({ model, models, onChange }) {
  const opts = models && models.length ? models : ["rmg", "mardm"];
  return (
    <div className="surface flex items-center gap-1.5 px-3 py-2">
      <span className="label mr-1">model</span>
      {opts.map((m) => {
        const active = m === model;
        return (
          <button
            key={m}
            onClick={() => onChange(m)}
            className={`rounded-md px-3 py-1 font-mono text-[11px] uppercase tracking-widest transition ${
              active
                ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]"
                : "border border-[var(--hairline)] text-[var(--muted)] hover:text-slate-200"
            }`}
          >
            {m}
          </button>
        );
      })}
    </div>
  );
}

function Telemetry({ health, error, cluster, clusterMode, queue }) {
  const active = queue?.active;
  const jobLabel = active
    ? `${active.kind}${active.mode ? `/${active.mode}` : ""} · ${active.state}`
    : queue?.queued
    ? "queued"
    : "idle";
  const jobColor = active
    ? active.state === "running"
      ? "var(--signal)"
      : "var(--amber)"
    : queue?.queued
    ? "var(--amber)"
    : "var(--muted)";

  const rows = error
    ? [["link", "OFFLINE", "var(--amber)"]]
    : health
    ? [
        clusterMode
          ? ["cluster", cluster?.state || "…", cluster?.state === "online" ? "var(--signal)" : "var(--amber)"]
          : ["device", health.device.toUpperCase(), "var(--signal)"],
        clusterMode && cluster?.latency_ms != null
          ? ["latency", `${cluster.latency_ms} ms`, "var(--signal)"]
          : ["repr", health.representation, "var(--signal)"],
        ...(clusterMode
          ? [
              ["job", jobLabel, jobColor],
              ...(queue?.queued ? [["queue", `${queue.queued} waiting`, "var(--amber)"]] : []),
            ]
          : [["host", health.media_format, "var(--signal)"]]),
      ]
    : [["link", "connecting…", "var(--muted)"]];

  const busy = !!(active || queue?.queued);
  return (
    <div className="surface min-w-[230px] animate-fade-up p-4">
      <div className="label mb-3 flex items-center gap-2">
        <span
          className={`dot ${busy || error ? "animate-pulse-soft" : ""}`}
          style={{ color: error ? "var(--amber)" : busy ? "var(--amber)" : "var(--signal)" }}
        />
        telemetry{busy ? " · busy" : ""}
      </div>
      <dl className="space-y-1.5">
        {rows.map(([k, v, c]) => (
          <div key={k} className="flex items-center justify-between gap-6 text-[12px]">
            <dt className="text-[var(--muted)]">{k}</dt>
            <dd className="font-mono lowercase" style={{ color: c }}>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
