"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import CreateWorkspace from "@/components/CreateWorkspace";
import LibraryWorkspace from "@/components/LibraryWorkspace";
import LabWorkspace from "@/components/LabWorkspace";
import TrajectoryTab from "@/components/TrajectoryTab";
import SystemDrawer from "@/components/SystemDrawer";
import ClusterGate from "@/components/ClusterGate";
import { LightboxProvider } from "@/components/Lightbox";
import ErrorBoundary from "@/components/ErrorBoundary";

export default function Home() {
  const [health, setHealth] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [connError, setConnError] = useState(null);
  const [clusterStatus, setClusterStatus] = useState(null);
  const [queue, setQueue] = useState(null);
  const [modelInfo, setModelInfo] = useState(null); // { active, models, tasks }
  const [systemOpen, setSystemOpen] = useState(false);

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

  // Three intent-based workspaces. Lab (train/eval/analysis) is cluster-only;
  // local mode keeps Create + Library.
  const tabs = useMemo(() => {
    const base = [
      { id: "create", n: "01", label: "Create", sub: "prompt · constraints · scene" },
      { id: "library", n: "02", label: "Library", sub: "render · ground truth · samples" },
    ];
    // Lab isn't integrated for MARDM yet — cluster-only and RMG-only.
    if (clusterMode && model !== "mardm") base.push({ id: "lab", n: "03", label: "Lab", sub: "train · eval · analysis" });
    // Trajectory (spatial mask-control) constraints. Deliberately its own tab
    // while the feature is being evaluated, rather than folded into Create —
    // it is the only surface where a joint is driven to world positions.
    if (model !== "mardm") base.push({ id: "trajectory", n: "04", label: "Trajectory", sub: "spatial control · path targets" });
    return base;
  }, [clusterMode, model]);

  const [tab, setTab] = useState("create");

  // If the selected tab disappears (e.g. switching to MARDM while on Lab),
  // fall back to the first available tab.
  useEffect(() => {
    if (!tabs.some((t) => t.id === tab)) setTab(tabs[0].id);
  }, [tabs, tab]);

  const panel = (
    <>
      {/* flex (not a fixed grid) so all tabs stay on ONE row as the count grows —
          they share the width evenly and only wrap on narrow viewports */}
      <nav className="mb-8 flex max-w-5xl flex-wrap gap-2">
        {tabs.map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`group relative min-w-[190px] flex-1 overflow-hidden rounded-xl border px-4 py-3 text-left transition ${
                active
                  ? "border-[var(--signal)] bg-[var(--signal-dim)]"
                  : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"
              }`}
            >
              <span className={`font-mono text-[11px] tracking-widest ${active ? "text-[var(--signal)]" : "text-[var(--muted)]"}`}>
                {t.n}
              </span>
              <div className={`display mt-1.5 text-base font-bold ${active ? "text-white" : "text-slate-300"}`}>
                {t.label}
              </div>
              <div className="label mt-0.5">{t.sub}</div>
              {active && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-[var(--signal)]" />}
            </button>
          );
        })}
      </nav>

      {clusterMode && <SystemDrawer status={clusterStatus} open={systemOpen} onClose={() => setSystemOpen(false)} />}

      {/* One boundary per tab: a crash in one workspace shows its own error and
          leaves the rest of the app usable, instead of blanking the page with
          Next's generic client-side-exception screen (no console on Safari). */}
      <div key={tab} className="animate-fade-up">
        <ErrorBoundary label={tabs.find((t) => t.id === tab)?.label || tab}>
        {tab === "create" &&
          (model === "mardm" ? <MardmUnsupported feature="Create" /> : <CreateWorkspace clusterMode={clusterMode} checkpoints={checkpoints} />)}
        {tab === "library" &&
          (model === "mardm" ? <MardmUnsupported feature="Library" /> : <LibraryWorkspace clusterMode={clusterMode} />)}
        {tab === "lab" && model !== "mardm" && <LabWorkspace model={model} />}
        {tab === "trajectory" && model !== "mardm" && (
          <TrajectoryTab clusterMode={clusterMode} checkpoints={checkpoints} />
        )}
        </ErrorBoundary>
      </div>
    </>
  );

  return (
    <LightboxProvider>
    <div className="min-h-screen">
      <div className="mx-auto w-full max-w-[1920px] px-6 pb-20 pt-10 lg:px-10">
        <header className="mb-9 flex flex-wrap items-end justify-between gap-6 border-b border-[var(--hairline)] pb-6">
          <div className="animate-fade-up">
            <div className="label mb-2 text-[var(--signal)]">Riemannian Motion Generation</div>
            <h1 className="display text-5xl font-extrabold leading-none text-white">
              RMG<span className="text-[var(--signal)]">.</span>
              <span className="ml-3 align-middle text-lg font-semibold text-[var(--muted)]">motion instrument</span>
            </h1>
            <p className="mt-3 max-w-md text-[13px] leading-relaxed text-[var(--muted)]">
              Author motion with constraints &amp; scenes, browse the dataset, and run
              training / eval on the SLURM cluster — each with its own history.
            </p>
          </div>
          <div className="flex flex-col items-end gap-3">
            <div className="flex items-center gap-2">
              {clusterMode && (
                <>
                  <ModelToggle model={model} models={modelInfo?.models} onChange={changeModel} />
                  {/* live job progress renders ONLY inside this drawer — the
                      button pulses while anything runs so it stays findable */}
                  <button
                    onClick={() => setSystemOpen((o) => !o)}
                    className={`surface flex items-center gap-1.5 px-3 py-2 text-[11px] uppercase tracking-widest transition ${systemOpen ? "border-[var(--signal)] text-[var(--signal)]" : "text-[var(--muted)] hover:text-slate-200"}`}
                    title={queue?.active || queue?.queued ? "cluster monitor — a job is live: progress bars are in here" : "cluster monitor: status · squeue · jobs · live progress"}
                  >
                    <span
                      className={`dot ${queue?.active || queue?.queued ? "animate-pulse-soft" : ""}`}
                      style={{ color: queue?.active || queue?.queued ? "var(--amber)" : clusterStatus?.state === "online" ? "var(--signal)" : "var(--amber)" }}
                    />
                    system{queue?.active ? " · busy" : queue?.queued ? " · queued" : ""}
                  </button>
                </>
              )}
            </div>
            <Telemetry health={health} error={connError} cluster={clusterStatus} clusterMode={clusterMode} queue={queue} />
          </div>
        </header>

        {clusterMode ? <ClusterGate>{panel}</ClusterGate> : panel}
      </div>
    </div>
    </LightboxProvider>
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
        <span className="text-[var(--signal)]">Lab ▸ Evaluate</span> to score a
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
