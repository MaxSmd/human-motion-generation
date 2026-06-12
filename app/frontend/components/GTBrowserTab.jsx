"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import MediaViewer from "./MediaViewer";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import { useVizJob } from "@/lib/useVizJob";

export default function GTBrowserTab({ clusterMode }) {
  return clusterMode ? <GTCluster /> : <GTLocal />;
}

// ───────────────────────────────────────────── cluster: render GT clips on the GPU

const EXAMPLES = ["000021", "000019", "000022", "000026"];

function GTCluster() {
  const [clips, setClips] = useState("000021,000019,000022");
  const [mode, setMode] = useState("clip"); // clip | compare
  const [checkpoint, setCheckpoint] = useState("");
  const { job, error, submitting, run } = useVizJob();

  function go(e) {
    e.preventDefault();
    const body = mode === "compare"
      ? { mode: "compare", clips, checkpoint }
      : { mode: "clip", clips };
    run(body);
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      <form className="surface space-y-4 p-6" onSubmit={go}>
        <div className="flex gap-1.5">
          {["clip", "compare"].map((m) => (
            <button key={m} type="button" onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1.5 text-[12px] font-medium transition ${mode === m ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
              {m === "clip" ? "GT clips" : "GT vs prediction"}
            </button>
          ))}
        </div>

        <label className="block">
          <span className="label mb-1.5 block">clip ids (comma-sep{mode === "compare" ? ", or 'auto'" : ""})</span>
          <input className="field-input" value={clips} onChange={(e) => setClips(e.target.value)} />
        </label>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map((c) => (
            <button key={c} type="button" onClick={() => setClips(c)}
              className="rounded-full border border-[var(--hairline)] px-2.5 py-1 font-mono text-[11px] text-slate-400 hover:border-[var(--signal)] hover:text-[var(--signal)]">
              {c}
            </button>
          ))}
        </div>

        {mode === "compare" && <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} />}

        <button type="submit" className="btn-signal w-full" disabled={submitting || (mode === "compare" && !checkpoint)}>
          {submitting ? "SUBMITTING…" : `▶  RENDER ${mode === "compare" ? "COMPARE" : "CLIPS"} ON CLUSTER`}
        </button>
        <p className="label text-center">
          renders the stored GT motion{mode === "compare" ? " + the model's prediction for each clip's caption" : ""} via forward kinematics
        </p>
      </form>

      <div className="surface p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Rendered GT clips will appear here." />
      </div>
    </div>
  );
}

// ───────────────────────────────────────────── local mode (RMG_CLUSTER_MODE=0)

function GTLocal() {
  const [params, setParams] = useState({ subset_n: 0, subset_fraction: 0.01, subset_seed: 0 });
  const [clips, setClips] = useState([]);
  const [listError, setListError] = useState(null);
  const [listing, setListing] = useState(false);
  const [selected, setSelected] = useState(null);
  const [media, setMedia] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setParams((p) => ({ ...p, [k]: Number(e.target.value) }));

  async function loadList(e) {
    e.preventDefault(); setListing(true); setListError(null);
    try { setClips(await api.gtList(params)); }
    catch (err) { setListError(err.message); setClips([]); }
    finally { setListing(false); }
  }
  async function loadClip(cid) {
    setSelected(cid); setLoading(true); setError(null);
    try { setMedia(await api.gtClip(cid)); }
    catch (err) { setError(err.message); setMedia(null); }
    finally { setLoading(false); }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      <div className="surface flex flex-col p-6">
        <form className="flex flex-wrap items-end gap-3" onSubmit={loadList}>
          <Field label="subset_n"><input type="number" className="field-input" value={params.subset_n} onChange={set("subset_n")} /></Field>
          <Field label="fraction"><input type="number" step="0.01" className="field-input" value={params.subset_fraction} onChange={set("subset_fraction")} /></Field>
          <Field label="seed"><input type="number" className="field-input" value={params.subset_seed} onChange={set("subset_seed")} /></Field>
          <button type="submit" className="btn-ghost" disabled={listing}>{listing ? "…" : "LIST"}</button>
        </form>
        {listError && <p className="mt-4 text-[12px] text-rose-300">⚠ {listError}</p>}
        <ul className="mt-4 max-h-[58vh] space-y-1 overflow-y-auto pr-1">
          {clips.map((c) => (
            <li key={c.cid}>
              <button onClick={() => loadClip(c.cid)} className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition ${selected === c.cid ? "border-[var(--signal)] bg-[var(--signal-dim)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"}`}>
                <span className="font-mono text-[12px] text-[var(--signal)]">{c.cid}</span>
                <span className="truncate text-[12px] text-slate-400">{c.caption}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <MediaViewer url={media?.media_url} caption={media?.caption || selected} loading={loading} error={error} />
    </div>
  );
}

function Field({ label, children }) {
  return <label className="block w-24"><span className="label mb-1.5 block">{label}</span>{children}</label>;
}
