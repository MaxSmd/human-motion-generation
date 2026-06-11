"use client";

import { useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import MediaViewer from "./MediaViewer";

const DEFAULTS = {
  text: "a person walks forward",
  guidance: 6.5,
  num_steps: 50,
  seed: 0,
  num_frames: 100,
};

const PRESETS = [
  "a person walks forward",
  "a person waves their right hand",
  "a person sits down on the floor",
  "a person does jumping jacks",
];

export default function GenerateTab({ checkpoints }) {
  const [form, setForm] = useState(DEFAULTS);
  const [checkpoint, setCheckpoint] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const set = (k) => (e) =>
    setForm((f) => ({
      ...f,
      [k]: e.target.type === "number" || e.target.type === "range" ? Number(e.target.value) : e.target.value,
    }));

  async function onGenerate(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const body = { ...form };
      if (checkpoint) body.checkpoint = checkpoint;
      const res = await api.generate(body);
      const item = { ...res, caption: form.text };
      setResult(item);
      setHistory((h) => [item, ...h].slice(0, 12));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr]">
      {/* ── control panel ─────────────────────────────────────────── */}
      <form className="surface space-y-5 p-6" onSubmit={onGenerate}>
        <div>
          <div className="label mb-2">prompt</div>
          <textarea
            rows={3}
            className="field-input resize-none"
            value={form.text}
            onChange={set("text")}
          />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setForm((f) => ({ ...f, text: p }))}
                className="rounded-full border border-[var(--hairline)] px-2.5 py-1 text-[11px] text-slate-400 transition hover:border-[var(--signal)] hover:text-[var(--signal)]"
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {/* guidance slider */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <span className="label">guidance ω</span>
            <span className="font-mono text-sm text-[var(--signal)]">{form.guidance.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min="1"
            max="12"
            step="0.5"
            value={form.guidance}
            onChange={set("guidance")}
            className="w-full accent-[var(--signal)]"
          />
        </div>

        <div className="grid grid-cols-3 gap-3">
          <Field label="ODE steps">
            <input type="number" className="field-input" value={form.num_steps} onChange={set("num_steps")} />
          </Field>
          <Field label="frames">
            <input type="number" className="field-input" value={form.num_frames} onChange={set("num_frames")} />
          </Field>
          <Field label="seed">
            <input type="number" className="field-input" value={form.seed} onChange={set("seed")} />
          </Field>
        </div>

        <Field label="checkpoint">
          <select className="field-input" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            <option value="">server default</option>
            {checkpoints.map((c) => (
              <option key={c.path} value={c.path}>
                {c.run} / {c.name}
              </option>
            ))}
          </select>
        </Field>

        <button type="submit" className="btn-signal w-full" disabled={loading}>
          {loading ? "GENERATING…" : "▶  GENERATE MOTION"}
        </button>
      </form>

      {/* ── viewport ──────────────────────────────────────────────── */}
      <div className="space-y-4">
        <MediaViewer
          url={result?.media_url}
          caption={result?.caption}
          loading={loading}
          error={error}
          status={result?.step != null ? `step ${result.step}${result.cached ? " · cached" : ""}` : null}
        />

        {history.length > 0 && (
          <div className="surface p-4">
            <div className="label mb-3">session history · {history.length}</div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {history.map((h, i) => (
                <button
                  key={i}
                  type="button"
                  title={h.caption}
                  onClick={() => setResult(h)}
                  className={`shrink-0 overflow-hidden rounded-lg border transition ${
                    result === h ? "border-[var(--signal)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]"
                  }`}
                >
                  {h.media_url.toLowerCase().endsWith(".mp4") ? (
                    <video src={mediaUrl(h.media_url)} className="h-16 w-16 object-cover" muted />
                  ) : (
                    <img src={mediaUrl(h.media_url)} alt={h.caption} className="h-16 w-16 object-cover" />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="label mb-1.5 block">{label}</span>
      {children}
    </label>
  );
}
