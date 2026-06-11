"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import MediaViewer from "./MediaViewer";

export default function GTBrowserTab() {
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
    e.preventDefault();
    setListing(true);
    setListError(null);
    try {
      setClips(await api.gtList(params));
    } catch (err) {
      setListError(err.message);
      setClips([]);
    } finally {
      setListing(false);
    }
  }

  async function loadClip(cid) {
    setSelected(cid);
    setLoading(true);
    setError(null);
    try {
      setMedia(await api.gtClip(cid));
    } catch (err) {
      setError(err.message);
      setMedia(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      <div className="surface flex flex-col p-6">
        <form className="flex flex-wrap items-end gap-3" onSubmit={loadList}>
          <Field label="subset_n">
            <input type="number" className="field-input" value={params.subset_n} onChange={set("subset_n")} />
          </Field>
          <Field label="fraction">
            <input type="number" step="0.01" className="field-input" value={params.subset_fraction} onChange={set("subset_fraction")} />
          </Field>
          <Field label="seed">
            <input type="number" className="field-input" value={params.subset_seed} onChange={set("subset_seed")} />
          </Field>
          <button type="submit" className="btn-ghost" disabled={listing}>
            {listing ? "…" : "LIST"}
          </button>
        </form>

        {listError && (
          <p className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/5 px-3 py-2 text-[12px] text-rose-300">
            ⚠ {listError}
          </p>
        )}

        <div className="mt-4 flex-1">
          <div className="label mb-2">
            {clips.length > 0 ? `${clips.length} clips · model trained on these` : "subset"}
          </div>
          <ul className="max-h-[58vh] space-y-1 overflow-y-auto pr-1">
            {clips.map((c) => (
              <li key={c.cid}>
                <button
                  onClick={() => loadClip(c.cid)}
                  className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition ${
                    selected === c.cid
                      ? "border-[var(--signal)] bg-[var(--signal-dim)]"
                      : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"
                  }`}
                >
                  <span className="font-mono text-[12px] text-[var(--signal)]">{c.cid}</span>
                  <span className="truncate text-[12px] text-slate-400">{c.caption}</span>
                </button>
              </li>
            ))}
            {clips.length === 0 && !listError && (
              <li className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-6 text-center text-[12px] text-[var(--muted)]">
                Run a subset query to browse ground-truth clips.
              </li>
            )}
          </ul>
        </div>
      </div>

      <MediaViewer
        url={media?.media_url}
        caption={media?.caption || selected}
        loading={loading}
        error={error}
        status={selected ? `clip ${selected}` : null}
      />
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block w-24">
      <span className="label mb-1.5 block">{label}</span>
      {children}
    </label>
  );
}
