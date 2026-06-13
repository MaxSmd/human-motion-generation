"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

// run (with checkpoints) → checkpoint dropdown, both from the cluster. When a run
// is picked, reads its stored config.json and reports the model/train presets via
// `onConfig`, so the caller submits with dims that MATCH the checkpoint (no more
// dit_base/dit_large mismatches).
export default function RemoteCheckpointPicker({ value, onChange, onConfig }) {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [ckpts, setCkpts] = useState([]);
  const [presets, setPresets] = useState(null);

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_checkpoints))).catch(() => {});
  }, []);

  useEffect(() => {
    if (!run) { setCkpts([]); setPresets(null); onConfig?.(null); return; }
    api.clusterCheckpoints(run).then(setCkpts).catch(() => setCkpts([]));
    api.runInfo(run).then((info) => {
      const c = info.config || {};
      const d = c.data || {};
      const p = {
        run,
        model_preset: c.model?.name,
        train_preset: c.train?.preset,
        representation: c.representation?.name,
        // training subset, so callers can mark seen vs unseen GT clips
        subset_fraction: d.subset_fraction ?? 1.0,
        subset_seed: d.subset_seed ?? 0,
        subset_n: d.subset_n ?? 0,
      };
      setPresets(p);
      onConfig?.(p);
    }).catch(() => { setPresets(null); onConfig?.(null); });
  }, [run]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="label mb-1.5 block">run (remote)</span>
          <select className="field-input" value={run} onChange={(e) => { setRun(e.target.value); onChange(""); }}>
            <option value="">select run…</option>
            {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="label mb-1.5 block">checkpoint</span>
          <select className="field-input" value={value} onChange={(e) => onChange(e.target.value)} disabled={!run}>
            <option value="">select…</option>
            {ckpts.map((c) => <option key={c} value={c}>{c.split("/").pop()}</option>)}
          </select>
        </label>
      </div>
      {presets && (
        <p className="text-[11px] text-[var(--muted)]">
          from config ·{" "}
          <span className="font-mono text-slate-300">{presets.model_preset || "?"}</span> /{" "}
          <span className="font-mono text-slate-300">{presets.train_preset || "?"}</span>
          {presets.representation ? <> · <span className="font-mono text-slate-300">{presets.representation}</span></> : null}
        </p>
      )}
    </div>
  );
}
