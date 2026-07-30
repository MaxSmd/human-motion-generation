"use client";

// A history job loaded into the workspace's main preview area. Clicking a row
// in the HistoryRail lands here, and the body adapts to the job kind so every
// async product stays viewable from history at any time:
//   • viz   — the pulled clips/GIFs (same VizJobResult as a fresh job)
//   • eval  — results.json as a per-ω metric table + FID sweep chart (the
//             backend salvages partial files, so a running sweep shows the
//             levels finished so far)
//   • train — the run's loss curve + headline facts from metrics.csv
// Results refetch when the job's state changes (the workspaces hand in the
// live-polled job object) or via the ↻ button; there is deliberately no poll
// loop here — the head node serializes SSH.

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import VizJobResult from "./VizJobResult";
import LineChart from "./LineChart";
import { TableTools } from "./ExportButtons";
import { api, mediaUrl } from "@/lib/api";
import { classifyJob, categoryMeta } from "@/lib/classifyJob";
import { ACTIVE } from "@/lib/useVizJob";

// three.js player is client-only (same as the RoomEditor viewport).
const MotionPlayer = dynamic(() => import("./MotionPlayer"), { ssr: false });

const SIGNAL = "#22d3ee";

function ago(sec) {
  if (!sec) return "";
  const d = Math.max(0, Date.now() / 1000 - sec);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

const fmt = (v, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : "—");

// Per-ω results of one eval run: compact metric table (best-FID row marked)
// plus a FID-vs-ω chart when the run swept more than one guidance level.
// results.json is written incrementally (one ω at a time) and the backend
// salvages partial files, so a still-running sweep fills in on each open.
function EvalResults({ job }) {
  const resultsRef = useRef(null);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0); // manual ↻ — long sweeps sit in "running" for hours
  const run = job.run_name;
  const active = ACTIVE.has(job.state);

  useEffect(() => {
    if (!run) return;
    let alive = true;
    setLoading(true);
    setErr(null);
    api.evalResults(run)
      .then((r) => alive && setRes(r))
      .catch((e) => alive && setErr(e.message))
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [run, job.state, tick]);

  const rows = res?.results
    ? Object.entries(res.results)
        .map(([w, m]) => ({ w: parseFloat(w), ...m }))
        .filter((r) => Number.isFinite(r.w))
        .sort((a, b) => a.w - b.w)
    : [];
  const fids = rows.map((r) => r.fid).filter(Number.isFinite);
  const bestFid = fids.length ? Math.min(...fids) : null;
  const p = job.params || {};

  return (
    <div>
      <div className="label mb-2 flex items-center normal-case tracking-normal">
        <span>
          {p.checkpoint ? `${p.checkpoint.split("/").slice(-2).join("/")} · ` : ""}
          {p.num_sample_steps != null ? `${p.num_sample_steps} ODE steps · ` : ""}
          {p.guidance_scales ? `ω ${p.guidance_scales} · ` : ""}
          {rows.length} ω level{rows.length === 1 ? "" : "s"}
          {active ? " · running — partial results" : ""}
          {loading ? " · loading…" : ""}
        </span>
        <span className="ml-auto flex items-center gap-1">
          {rows.length > 0 && (
            <TableTools getTable={() => resultsRef.current} name={`results-${job.run_name || job.id || ""}`}
              caption="Guidance ω sweep — eval metrics." label="tab:omega-sweep" />
          )}
          <button onClick={() => setTick((t) => t + 1)} title="refetch results.json"
            className="rounded border border-[var(--hairline)] px-2 py-0.5 text-[11px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">
            ↻ refresh
          </button>
        </span>
      </div>

      {err && rows.length === 0 && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-8 text-center text-[12px] text-[var(--muted)]">
          {active ? "No results.json yet — still sampling; re-open once the first ω completes." : `No readable results for this run (${err}).`}
        </p>
      )}

      {rows.length > 0 && (
        <>
          <div className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink">
            <table ref={resultsRef} className="w-full text-[12px]">
              <thead>
                <tr className="label text-left">
                  <th className="px-3 py-1.5">ω</th>
                  <th className="px-3 py-1.5">FID ↓</th>
                  <th className="px-3 py-1.5">R@1 ↑</th>
                  <th className="px-3 py-1.5">R@3 ↑</th>
                  <th className="px-3 py-1.5">MM-Dist ↓</th>
                  <th className="px-3 py-1.5">Diversity</th>
                  <th className="px-3 py-1.5">MModality</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {rows.map((r) => {
                  const best = bestFid != null && r.fid === bestFid;
                  return (
                    <tr key={r.w} className="border-t border-[var(--hairline)]"
                      style={best ? { color: SIGNAL } : undefined}>
                      <td className="px-3 py-1.5">{r.w}{best ? " ★" : ""}</td>
                      <td className="px-3 py-1.5">{fmt(r.fid)}</td>
                      <td className="px-3 py-1.5">{fmt(r.r1)}</td>
                      <td className="px-3 py-1.5">{fmt(r.r3)}</td>
                      <td className="px-3 py-1.5">{fmt(r.mm_dist, 2)}</td>
                      <td className="px-3 py-1.5">{fmt(r.diversity, 2)}</td>
                      <td className="px-3 py-1.5">{fmt(r.multimodality, 2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {rows.length >= 2 && (
            <figure className="mt-4 rounded-lg border border-[var(--hairline)] bg-ink p-3">
              <figcaption className="label mb-2">FID ↓ vs guidance ω</figcaption>
              <LineChart
                series={[{ label: run, color: SIGNAL, points: rows.filter((r) => Number.isFinite(r.fid)).map((r) => [r.w, r.fid]) }]}
                width={560} height={240} xLabel="guidance ω" yLabel="FID" yZero
                xTicks={rows.map((r) => r.w)}
              />
            </figure>
          )}
        </>
      )}
    </div>
  );
}

// A train run opened from history: the loss curve + where the run got to,
// straight from metrics.csv. Column switching, config and duration stay in
// Lab ▸ Analysis ▸ Training; live step/ETA bars live in the System drawer.
function TrainRun({ job }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0); // manual ↻ — training runs for days in one state
  const run = job.run_name;
  const active = ACTIVE.has(job.state) || job.state === "paused";

  useEffect(() => {
    if (!run) return;
    let alive = true;
    setLoading(true);
    setErr(null);
    api.runMetrics(run)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(e.message))
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [run, job.state, tick]);

  const si = data ? data.columns.indexOf("step") : -1;
  const li = data ? data.columns.indexOf("loss") : -1;
  const points = data && li >= 0
    ? data.rows.map((r) => [si >= 0 ? r[si] : 0, r[li]]).filter(([x, y]) => x != null && y != null)
    : [];
  const lastStep = points.length ? points[points.length - 1][0] : null;
  const tail = points.slice(-20).map(([, y]) => y);
  const tailLoss = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : null;
  const p = job.params || {};

  return (
    <div>
      <div className="label mb-2 flex items-center normal-case tracking-normal">
        <span>
          {p.model_preset ? `${p.model_preset} · ` : ""}
          {p.max_steps ? `${Number(p.max_steps).toLocaleString()} steps planned · ` : ""}
          {lastStep != null ? `at step ${lastStep.toLocaleString()} · ` : ""}
          {tailLoss != null ? `loss ${fmt(tailLoss, 4)} (tail avg)` : ""}
          {active ? " · live — bars in header ▸ system" : ""}
          {loading ? " · loading…" : ""}
        </span>
        <button onClick={() => setTick((t) => t + 1)} title="refetch metrics.csv"
          className="ml-auto rounded border border-[var(--hairline)] px-2 py-0.5 text-[11px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">
          ↻ refresh
        </button>
      </div>

      {points.length > 0 ? (
        <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
          <figcaption className="label mb-2">loss vs step · full curves in Lab ▸ Analysis</figcaption>
          <LineChart series={[{ label: "loss", color: SIGNAL, points }]}
            width={560} height={240} xLabel="step" yLabel="loss" yZero />
        </figure>
      ) : (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-8 text-center text-[12px] text-[var(--muted)]">
          {active
            ? "No metrics.csv rows yet — the run hasn't logged its first steps."
            : err
            ? `No readable metrics for this run (${err}).`
            : "No metrics found for this run."}
        </p>
      )}
    </div>
  );
}

// A scene job opened from history: the generated clip animating *inside* the
// room it was sampled in — the same interactive 3D viewer as Create ▸ Scene, so
// clicking a scene in history lands you in the room viewer (not a flat GIF).
// The clip is already in the room's world frame (placement applied server side),
// so scene + skeleton draw together. Jobs that predate the .npy dump (gif only)
// fall back to the flat clip grid, with the room still described in the header.
function SceneResult({ job }) {
  const scene = job.params?.scene || null;
  const outs = job.outputs || [];
  const withNpy = outs.find((o) => o.npy_url);
  const jointsUrl = withNpy ? mediaUrl(withNpy.npy_url) : null;
  const busy = ACTIVE.has(job.state);

  if (jointsUrl && scene) {
    const obs = scene.objects?.length ?? 0;
    return (
      <div>
        <div className="label mb-2 normal-case tracking-normal">
          room {scene.room?.width}×{scene.room?.depth}×{scene.room?.height} m ·{" "}
          {obs} obstacle{obs === 1 ? "" : "s"}
          {scene.contacts?.length ? ` · ${scene.contacts.length} contact${scene.contacts.length === 1 ? "" : "s"}` : ""}
          {" · orbit to view from any angle"}
        </div>
        <MotionPlayer jointsUrl={jointsUrl} scene={scene} fps={scene.fps || 20} />
      </div>
    );
  }
  // no .npy to animate (still rendering, or an older gif-only job) — show the
  // flat clips; the header already names the room this was sampled in.
  return (
    <VizJobResult
      job={job}
      emptyHint={busy ? "Rendering in the room on GPU…" : "This scene job has no pulled outputs."}
    />
  );
}

export default function HistoryPreview({ job, onClose }) {
  if (!job) return null;
  const cat = classifyJob(job);
  const meta = categoryMeta(cat);
  const caption = job.outputs?.[0]?.caption || job.params?.prompts || job.params?.text || job.run_name || "";

  return (
    <div className="surface animate-fade-up p-5" style={{ borderColor: "var(--hairline-strong)" }}>
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="label flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
            history preview · {meta.label}
            {job.submitted_at ? <span className="normal-case tracking-normal">· {ago(job.submitted_at)}</span> : null}
          </div>
          {caption && <div className="mt-1 truncate text-[13px] text-slate-300">{caption}</div>}
        </div>
        <button onClick={onClose}
          className="shrink-0 rounded-md border border-[var(--hairline)] px-2.5 py-1 text-[11px] text-[var(--muted)] hover:border-[var(--signal)] hover:text-[var(--signal)]">
          ✕ close
        </button>
      </div>
      {job.kind === "eval"
        ? <EvalResults job={job} />
        : job.kind === "train"
        ? <TrainRun job={job} />
        : cat === "scene"
        ? <SceneResult job={job} />
        : <VizJobResult job={job} emptyHint="This job has no pulled outputs." />}
    </div>
  );
}
