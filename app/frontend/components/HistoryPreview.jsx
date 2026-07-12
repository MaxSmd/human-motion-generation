"use client";

// A history job loaded into the workspace's main preview area. Clicking a row
// in the HistoryRail lands here: viz-like jobs render through the same
// VizJobResult used for freshly completed jobs; eval jobs render their
// results.json (per-ω metric table + FID sweep chart) instead — an eval has no
// media outputs, so VizJobResult would only ever say "no pulled outputs".

import { useEffect, useState } from "react";
import VizJobResult from "./VizJobResult";
import LineChart from "./LineChart";
import { api } from "@/lib/api";
import { classifyJob, categoryMeta } from "@/lib/classifyJob";
import { ACTIVE } from "@/lib/useVizJob";

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
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
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
  }, [run, job.state]);

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
      <div className="label mb-2 normal-case tracking-normal">
        {p.checkpoint ? `${p.checkpoint.split("/").slice(-2).join("/")} · ` : ""}
        {p.num_sample_steps != null ? `${p.num_sample_steps} ODE steps · ` : ""}
        {p.guidance_scales ? `ω ${p.guidance_scales} · ` : ""}
        {rows.length} ω level{rows.length === 1 ? "" : "s"}
        {active ? " · running — partial results" : ""}
        {loading ? " · loading…" : ""}
      </div>

      {err && rows.length === 0 && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-8 text-center text-[12px] text-[var(--muted)]">
          {active ? "No results.json yet — still sampling; re-open once the first ω completes." : `No readable results for this run (${err}).`}
        </p>
      )}

      {rows.length > 0 && (
        <>
          <div className="overflow-x-auto rounded-lg border border-[var(--hairline)] bg-ink">
            <table className="w-full text-[12px]">
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
        : <VizJobResult job={job} emptyHint="This job has no pulled outputs." />}
    </div>
  );
}
