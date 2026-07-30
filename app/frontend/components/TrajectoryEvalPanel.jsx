"use client";

// Benchmark results for trajectory-controlled generation, read straight off the
// cluster's eval runs — no numbers are stored in the frontend.
//
// Why this table is laid out the way it is: control fidelity and motion quality
// have to be read TOGETHER. Every controlled arm hits its targets exactly, so a
// column of zeros in "avg err" is not the story — the story is what the run
// costs in FID / R-precision / foot-skate to get those zeros, and against which
// unconstrained baseline. So the uncontrolled run is pinned to the top as the
// reference, and the control columns sit beside the quality columns rather than
// in a separate view.

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";

const OMEGA = "6.5";

/** Human label for one run's control settings. */
function describe(t) {
  if (!t) return { label: "unconstrained", detail: "no spatial control — the reference" };
  const dens = t.density === "all" ? "every frame" : `${t.density} keyframes`;
  const bits = [t.mode, t.blend].filter(Boolean).join(" · ");
  return {
    label: `${t.joints} × ${dens}`,
    detail: [bits, t.face_path ? "face_path" : null, t.axes && t.axes !== "xyz" ? t.axes : null]
      .filter(Boolean).join(" · "),
  };
}

function fmt(v, digits = 3) {
  return v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);
}

// Hard ceiling on the fetch. The backend talks to the cluster login node over
// SSH, which can stall; without this the panel would sit on "loading…" for as
// long as the browser is willing to wait, which reads as a hung UI.
const TIMEOUT_MS = 45_000;

export default function TrajectoryEvalPanel() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function load(signal) {
    setBusy(true);
    setError(null);
    // One request for every traj-* run — the whole table is a single SSH round
    // trip on the backend, so a slow cluster costs one timeout, not one per run.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    const onAbort = () => ctrl.abort();
    signal?.addEventListener("abort", onAbort);
    try {
      const { runs = {} } = await api.evalResultsBulk("traj", { signal: ctrl.signal });
      const out = [];
      for (const [name, per] of Object.entries(runs)) {
        const key = per[OMEGA] ? OMEGA : Object.keys(per)[0];
        if (!key) continue; // a run still in flight has no ω block yet
        out.push({ run: name, omega: key, ...per[key] });
      }
      // Reference first, then by how much control costs.
      out.sort((a, b) => {
        if (!a.trajectory !== !b.trajectory) return a.trajectory ? 1 : -1;
        return (a.fid ?? 0) - (b.fid ?? 0);
      });
      if (!signal?.aborted) setRows(out);
    } catch (e) {
      if (signal?.aborted) return;               // unmounted — nothing to report
      setError(
        e.name === "AbortError"
          ? `timed out after ${TIMEOUT_MS / 1000}s — the cluster login node is slow or unreachable`
          : e.message
      );
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      if (!signal?.aborted) setBusy(false);
    }
  }

  useEffect(() => {
    const ctrl = new AbortController();
    load(ctrl.signal);
    return () => ctrl.abort();
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const ref = useMemo(() => rows?.find((r) => !r.trajectory) || null, [rows]);

  return (
    <div className="surface p-5">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="label">benchmark · HumanML3D test split</div>
          <h3 className="display mt-1 text-base font-bold text-white">
            Controlled generation — what the control costs
          </h3>
        </div>
        <button onClick={() => load()} disabled={busy} className="btn-ghost !px-3 !py-1.5 text-[11px]">
          {busy ? "loading…" : "refresh"}
        </button>
      </div>
      <p className="mb-4 max-w-3xl text-[12px] leading-relaxed text-[var(--muted)]">
        Targets are read from each test clip&apos;s <em>reference</em> motion, so a perfect
        method reproduces the real trajectory. Every controlled row hits its targets
        exactly — so read the <span className="font-mono">err</span> columns as a check,
        and FID / R@3 / skate as the price paid for them.
      </p>

      {/* The fetch is a cluster round trip, so the wait is visible: say so in the
          body rather than only on the refresh button, which looked like a hang. */}
      {busy && !rows && (
        <div className="space-y-2" aria-busy="true">
          <p className="text-[11px] text-[var(--muted)]">
            reading results.json off the cluster…
          </p>
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-7 animate-pulse-soft rounded bg-[var(--hairline)]" />
          ))}
        </div>
      )}

      {error && (
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-[12px] text-[var(--amber)]">{error}</p>
          <button onClick={() => load()} disabled={busy} className="btn-ghost !px-3 !py-1.5 text-[11px]">
            retry
          </button>
        </div>
      )}

      {rows && rows.length === 0 && !busy && (
        <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-6 text-center text-[11px] text-[var(--muted)]">
          No trajectory eval runs found yet. Submit one from the Lab tab with
          <span className="font-mono"> eval.control_joints</span> set.
        </p>
      )}

      {rows && rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] border-collapse text-[12px]">
            <thead>
              <tr className="border-b border-[var(--hairline)] text-left">
                <th className="py-2 pr-3 font-normal text-[var(--muted)]">config</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">FID ↓</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">R@3 ↑</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">avg err ↓</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">loc@50 ↓</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">traj@50 ↓</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">skate ↓</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">jerk ↓</th>
                <th className="px-2 py-2 text-right font-normal text-[var(--muted)]">slots</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {rows.map((r) => {
                const d = describe(r.trajectory);
                const c = r.trajectory?.control || {};
                // Runs made before quality became a top-level block carry it
                // nested under `trajectory` — fall back so older rows still show.
                const q = r.quality || r.trajectory?.quality || {};
                const isRef = !r.trajectory;
                const worse = ref && r.fid != null && ref.fid != null && r.fid > ref.fid * 2;
                return (
                  <tr key={r.run}
                      className={`border-b border-[var(--hairline)] ${isRef ? "bg-[var(--signal-dim)]" : ""}`}>
                    <td className="py-2 pr-3">
                      <div className={isRef ? "font-semibold text-[var(--signal)]" : "text-slate-200"}>
                        {d.label}
                      </div>
                      <div className="font-sans text-[10px] text-[var(--muted)]">{d.detail}</div>
                    </td>
                    <td className={`px-2 py-2 text-right ${worse ? "text-[var(--amber)]" : "text-slate-200"}`}>
                      {fmt(r.fid)}
                    </td>
                    <td className="px-2 py-2 text-right text-slate-200">{fmt(r.r3)}</td>
                    <td className="px-2 py-2 text-right text-[#34d399]">
                      {isRef ? "—" : fmt(c.avg_err, 5)}
                    </td>
                    <td className="px-2 py-2 text-right text-[#34d399]">
                      {isRef ? "—" : fmt(c["loc_err_0.5"], 4)}
                    </td>
                    <td className="px-2 py-2 text-right text-[#34d399]">
                      {isRef ? "—" : fmt(c["traj_err_0.5"], 4)}
                    </td>
                    <td className="px-2 py-2 text-right text-slate-200">{fmt(q.foot_skate_ratio)}</td>
                    <td className="px-2 py-2 text-right text-slate-200">{fmt(q.jerk, 1)}</td>
                    <td className="px-2 py-2 text-right text-[var(--muted)]">
                      {isRef ? "—" : (c.n_locations ?? "—")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-3 text-[11px] leading-relaxed text-[var(--muted)]">
            ω = {rows[0]?.omega}. <span className="text-[#34d399]">Green</span> columns are
            control fidelity (metres, and the share of missed keyframe locations / failed
            clips at 50 cm, following OmniControl&apos;s definitions).
            {ref && (
              <> Reference row is the same model with no control:
                {" "}FID {fmt(ref.fid)}, skate {fmt(ref.quality?.foot_skate_ratio ?? ref.trajectory?.quality?.foot_skate_ratio)}.</>
            )}
          </p>
        </div>
      )}
    </div>
  );
}
