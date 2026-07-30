"use client";

// Small analysis panels for the clips currently on stage, computed in the
// browser from the same joint arrays the player loads. The measures mirror the
// ones in the desktop app's clip comparison: root path (top view), jerk over
// time (mean ‖Δ³x‖), and root speed.

import { useEffect, useMemo, useState } from "react";
import { loadNpy } from "@/lib/npy";

const FOOT = [7, 10, 8, 11];

// Per-frame metrics for one (T, 22, 3) clip.
function metrics(shape, data) {
  const [T, J] = shape;
  const at = (t, j, k) => data[(t * J + j) * 3 + k];

  const traj = [];
  for (let t = 0; t < T; t++) traj.push([at(t, 0, 0), at(t, 0, 2)]);

  // jerk: mean over joints of ‖third difference‖, per frame
  const jerk = [];
  for (let t = 3; t < T; t++) {
    let s = 0;
    for (let j = 0; j < J; j++) {
      let m = 0;
      for (let k = 0; k < 3; k++) {
        const d3 = at(t, j, k) - 3 * at(t - 1, j, k) + 3 * at(t - 2, j, k) - at(t - 3, j, k);
        m += d3 * d3;
      }
      s += Math.sqrt(m);
    }
    jerk.push([t, s / J]);
  }

  // root speed (m/s), fps 20
  const speed = [];
  for (let t = 1; t < T; t++) {
    const dx = at(t, 0, 0) - at(t - 1, 0, 0);
    const dz = at(t, 0, 2) - at(t - 1, 0, 2);
    speed.push([t, Math.hypot(dx, dz) * 20]);
  }

  const meanJerk = jerk.reduce((a, p) => a + p[1], 0) / Math.max(1, jerk.length);
  return { traj, jerk, speed, meanJerk, T };
}

export default function ClipStats({ clips }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    Promise.all(clips.map((c) => loadNpy(c.url)))
      .then((ds) => alive && setData(ds.map((d, i) => ({ ...clips[i], m: metrics(d.shape, d.data) }))))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [clips]);

  if (!data) return <div className="h-[150px]" />;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <Panel title="root path" note="top view · metres">
        <TrajPlot data={data} />
      </Panel>
      <Panel title="jerk" note="‖Δ³x‖ per frame">
        <SeriesPlot data={data} pick={(m) => m.jerk} />
      </Panel>
      <Panel title="root speed" note="m/s">
        <SeriesPlot data={data} pick={(m) => m.speed} />
      </Panel>
    </div>
  );
}

function Panel({ title, note, children }) {
  return (
    <div className="rounded-lg border border-[var(--hairline)] bg-black/20 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="label">{title}</span>
        <span className="font-mono text-[9px] text-[var(--muted)]">{note}</span>
      </div>
      {children}
    </div>
  );
}

// Top-view root trajectory, equal aspect, each clip its own colour.
function TrajPlot({ data }) {
  const W = 200, H = 132, m = 12;
  const dom = useMemo(() => {
    const pts = data.flatMap((d) => d.m.traj);
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (const [x, z] of pts) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      z0 = Math.min(z0, z); z1 = Math.max(z1, z);
    }
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    const half = Math.max(x1 - x0, z1 - z0, 0.5) / 2 * 1.15;
    return { x0: cx - half, x1: cx + half, z0: cz - half, z1: cz + half };
  }, [data]);
  const sx = (x) => m + ((x - dom.x0) / (dom.x1 - dom.x0)) * (W - 2 * m);
  const sz = (z) => m + ((z - dom.z0) / (dom.z1 - dom.z0)) * (H - 2 * m);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {data.map((d, i) => {
        const p = d.m.traj;
        return (
          <g key={i}>
            <polyline fill="none" stroke={d.color} strokeWidth="1.6" opacity="0.9"
              points={p.map(([x, z]) => `${sx(x)},${sz(z)}`).join(" ")} />
            <circle cx={sx(p[0][0])} cy={sz(p[0][1])} r="2.4" fill="none" stroke={d.color} strokeWidth="1.4" />
            <circle cx={sx(p[p.length - 1][0])} cy={sz(p[p.length - 1][1])} r="2.6" fill={d.color} />
          </g>
        );
      })}
    </svg>
  );
}

// Overlaid per-frame series (jerk or speed), one line per clip.
function SeriesPlot({ data, pick }) {
  const W = 200, H = 132, m = { l: 8, r: 8, t: 10, b: 12 };
  const series = data.map((d) => ({ color: d.color, pts: pick(d.m) }));
  const dom = useMemo(() => {
    let xmax = 0, ymax = 0;
    for (const s of series) for (const [x, y] of s.pts) { xmax = Math.max(xmax, x); ymax = Math.max(ymax, y); }
    return { xmax: xmax || 1, ymax: ymax * 1.15 || 1 };
  }, [series]);
  const sx = (x) => m.l + (x / dom.xmax) * (W - m.l - m.r);
  const sy = (y) => H - m.b - (y / dom.ymax) * (H - m.t - m.b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="var(--hairline)" />
      {series.map((s, i) => (
        <polyline key={i} fill="none" stroke={s.color} strokeWidth="1.4" opacity="0.9"
          points={s.pts.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
      ))}
    </svg>
  );
}
