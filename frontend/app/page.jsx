"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import GenerateTab from "@/components/GenerateTab";
import GTBrowserTab from "@/components/GTBrowserTab";
import TrainingViewerTab from "@/components/TrainingViewerTab";
import ConstraintsTab from "@/components/ConstraintsTab";
import ClusterTab from "@/components/ClusterTab";
import ClusterGate from "@/components/ClusterGate";

export default function Home() {
  const [health, setHealth] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [connError, setConnError] = useState(null);
  const [clusterStatus, setClusterStatus] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setConnError(e.message));
    api.checkpoints().then(setCheckpoints).catch(() => {});
  }, []);

  const clusterMode = !!health?.cluster_mode;

  useEffect(() => {
    if (!clusterMode) return;
    const poll = () => api.clusterStatus().then(setClusterStatus).catch(() => {});
    poll();
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, [clusterMode]);

  const tabs = useMemo(() => {
    const base = [
      { id: "generate", n: "01", label: "Generate", sub: "text → motion" },
      { id: "gt", n: "02", label: "Ground Truth", sub: "dataset browser" },
      { id: "training", n: "03", label: "Training", sub: "sample scrubber" },
      { id: "constraints", n: "04", label: "Constraints", sub: "pins · limits", soon: true },
    ];
    if (clusterMode)
      base.unshift({ id: "cluster", n: "00", label: "Cluster", sub: "jobs · squeue" });
    return base;
  }, [clusterMode]);

  const [tab, setTab] = useState("generate");
  useEffect(() => {
    if (clusterMode) setTab("cluster");
  }, [clusterMode]);

  const panel = (
    <>
      <nav className="mb-8 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
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
        {tab === "generate" && <GenerateTab checkpoints={checkpoints} />}
        {tab === "gt" && <GTBrowserTab />}
        {tab === "training" && <TrainingViewerTab />}
        {tab === "constraints" && <ConstraintsTab />}
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
          <Telemetry health={health} error={connError} cluster={clusterStatus} clusterMode={clusterMode} />
        </header>

        {clusterMode ? <ClusterGate>{panel}</ClusterGate> : panel}
      </div>
    </div>
  );
}

function Telemetry({ health, error, cluster, clusterMode }) {
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
        ["host", clusterMode ? health.cluster_host : health.media_format, "var(--signal)"],
      ]
    : [["link", "connecting…", "var(--muted)"]];

  return (
    <div className="surface min-w-[230px] animate-fade-up p-4">
      <div className="label mb-3 flex items-center gap-2">
        <span className="dot animate-pulse-soft" style={{ color: error ? "var(--amber)" : "var(--signal)" }} />
        telemetry
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
