"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import RemoteCheckpointPicker from "./RemoteCheckpointPicker";
import VizJobResult from "./VizJobResult";
import { useVizJob } from "@/lib/useVizJob";

const MODES = [
  { id: "clip", label: "GT clips", hint: "render stored ground-truth motions" },
  { id: "compare", label: "GT vs prediction", hint: "GT + the model's prediction on each clip's caption" },
  { id: "samples", label: "Training samples", hint: "the trainer's fixed-prompt dumps across saved steps" },
];

// Selection is a set of clip ids serialised as the comma field the viz job wants.
const parseClips = (s) => (s || "").split(",").map((c) => c.trim()).filter(Boolean);
const joinClips = (arr) => arr.join(",");

export default function VisualizeTab() {
  const [mode, setMode] = useState("clip");
  const [clips, setClips] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  // Selected run's config (presets + training subset) — drives seen/unseen split.
  const [cfg, setCfg] = useState(null);
  // GT-clip browse (id + caption, plus `seen` in compare mode) — hints what exists.
  const [gtClips, setGtClips] = useState([]);
  const [gtErr, setGtErr] = useState(null);
  const [gtLoading, setGtLoading] = useState(false);
  // Caption search box (debounced → server-side, spans the whole dataset).
  const [query, setQuery] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  // Training-sample step selection (avoid rendering an entire dir of dumps).
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [steps, setSteps] = useState([]);
  const [stepsErr, setStepsErr] = useState(null);
  const [selSteps, setSelSteps] = useState([]);
  const { job, error, submitting, run: launch } = useVizJob();

  const selected = parseClips(clips);
  const isSel = (cid) => selected.includes(cid);
  const toggleClip = (cid) =>
    setClips(joinClips(isSel(cid) ? selected.filter((c) => c !== cid) : [...selected, cid]));

  useEffect(() => {
    api.clusterRuns().then((r) => setRuns(r.filter((x) => x.has_samples))).catch(() => {});
  }, []);

  // Debounce the search box so each keystroke doesn't fire an SSH-backed query.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(query.trim()), 350);
    return () => clearTimeout(t);
  }, [query]);

  // Load the GT clip list. Plain browse for "clip"; for "compare" wait until a run
  // is picked, then tag each clip seen/unseen against that run's training subset.
  // `debouncedQ` searches captions across the whole dataset server-side.
  useEffect(() => {
    if (mode !== "clip" && mode !== "compare") return;
    if (mode === "compare" && !cfg) { setGtClips([]); return; }
    setGtErr(null);
    setGtLoading(true);
    const params = mode === "compare"
      ? {
          subset_fraction: cfg.subset_fraction, subset_seed: cfg.subset_seed,
          subset_n: cfg.subset_n, tag_seen: true, limit: 120, q: debouncedQ,
        }
      : { limit: 120, q: debouncedQ };
    api.clusterGtClips(params)
      .then(setGtClips)
      .catch((e) => { setGtClips([]); setGtErr(e.message); })
      .finally(() => setGtLoading(false));
  }, [mode, cfg, debouncedQ]);

  // Load saved sample steps when a run is picked.
  useEffect(() => {
    if (mode !== "samples" || !run) { setSteps([]); setSelSteps([]); return; }
    setStepsErr(null);
    api
      .runSampleSteps(run)
      .then((s) => { setSteps(s); setSelSteps(s.map((x) => x.step)); })
      .catch((e) => { setSteps([]); setStepsErr(e.message); });
  }, [mode, run]);

  function go(e) {
    e.preventDefault();
    if (mode === "clip") launch({ mode: "clip", clips });
    else if (mode === "compare") launch({
      mode: "compare", clips, checkpoint,
      model_preset: cfg?.model_preset, train_preset: cfg?.train_preset,
      subset_fraction: cfg?.subset_fraction, subset_seed: cfg?.subset_seed,
    });
    else launch({ mode: "samples", run, steps: selSteps });
  }

  const cur = MODES.find((m) => m.id === mode);

  // One clickable clip row. `min-w-0` + `truncate` clamps long captions to a
  // single ellipsised line so the list never overflows the panel horizontally.
  const clipRow = (c) => (
    <button key={c.cid} type="button" onClick={() => toggleClip(c.cid)}
      className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left text-[11px] transition ${isSel(c.cid) ? "bg-[var(--signal-dim)] text-[var(--signal)]" : "text-slate-400 hover:bg-white/5"}`}>
      <span className="shrink-0 font-mono">{isSel(c.cid) ? "▣" : "▢"}</span>
      <span className="shrink-0 font-mono text-slate-300">{c.cid}</span>
      <span className="min-w-0 flex-1 truncate">{c.caption || "—"}</span>
    </button>
  );

  const clipList = (items) =>
    items.length === 0 ? (
      <p className="px-2 py-1 text-[11px] text-[var(--muted)]">none</p>
    ) : (
      <div className="max-h-44 space-y-1 overflow-y-auto rounded-lg border border-[var(--hairline)] bg-ink p-1.5">
        {items.map(clipRow)}
      </div>
    );

  const clipGroup = (title, dot, items) => (
    <div>
      <div className="mb-1 flex items-center gap-2">
        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        <span className="label">{title}</span>
        <span className="font-mono text-[10px] text-[var(--muted)]">{items.length}</span>
      </div>
      {clipList(items)}
    </div>
  );

  const note = "rounded-lg border border-dashed border-[var(--hairline)] px-3 py-3 text-[11px] text-[var(--muted)]";

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_1.1fr]">
      <form className="surface min-w-0 space-y-4 p-6" onSubmit={go}>
        <div className="grid grid-cols-3 gap-1.5">
          {MODES.map((m) => (
            <button key={m.id} type="button" onClick={() => setMode(m.id)}
              className={`rounded-md px-2 py-2 text-[12px] font-medium transition ${mode === m.id ? "border border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
              {m.label}
            </button>
          ))}
        </div>
        <p className="label">{cur.hint}</p>

        {mode === "compare" && (
          <RemoteCheckpointPicker value={checkpoint} onChange={setCheckpoint} onConfig={setCfg} />
        )}

        {(mode === "clip" || mode === "compare") && (
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="label">available GT clips · click to select</span>
              <span className="font-mono text-[11px] text-[var(--muted)]">{selected.length} selected</span>
            </div>
            {mode === "compare" && !cfg ? (
              <p className={note}>select a run above to list the clips it trained on, separated from unseen ones.</p>
            ) : (
              <>
                <input className="field-input mb-2" placeholder="search captions…"
                  value={query} onChange={(e) => setQuery(e.target.value)} />
                {gtErr ? (
                  <p className={note}>GT clip list unavailable ({gtErr}).</p>
                ) : gtClips.length === 0 ? (
                  <p className={note}>{gtLoading ? "loading clips…" : debouncedQ ? `no captions match “${debouncedQ}”.` : "no clips found."}</p>
                ) : mode === "compare" ? (
                  <div className="space-y-3">
                    {clipGroup("seen in training", "bg-[var(--signal)]", gtClips.filter((c) => c.seen))}
                    {clipGroup("unseen", "bg-slate-500", gtClips.filter((c) => !c.seen))}
                  </div>
                ) : (
                  clipList(gtClips)
                )}
              </>
            )}
          </div>
        )}

        {mode === "samples" && (
          <>
            <label className="block">
              <span className="label mb-1.5 block">run (remote, has samples)</span>
              <select className="field-input" value={run} onChange={(e) => setRun(e.target.value)}>
                <option value="">select run…</option>
                {runs.map((r) => <option key={r.run} value={r.run}>{r.run}</option>)}
              </select>
            </label>

            {run && (
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="label">steps to render</span>
                  <span className="flex gap-2 text-[11px]">
                    <button type="button" className="text-slate-400 hover:text-[var(--signal)]" onClick={() => setSelSteps(steps.map((s) => s.step))}>all</button>
                    <button type="button" className="text-slate-400 hover:text-[var(--signal)]" onClick={() => setSelSteps([])}>none</button>
                  </span>
                </div>
                {stepsErr ? (
                  <p className={note}>no steps found ({stepsErr})</p>
                ) : steps.length === 0 ? (
                  <p className={note}>loading steps…</p>
                ) : (
                  <>
                    <div className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto rounded-lg border border-[var(--hairline)] bg-ink p-2">
                      {steps.map((s) => {
                        const on = selSteps.includes(s.step);
                        return (
                          <button key={s.step} type="button"
                            onClick={() => setSelSteps(on ? selSteps.filter((x) => x !== s.step) : [...selSteps, s.step])}
                            className={`rounded-full border px-2.5 py-1 font-mono text-[11px] transition ${on ? "border-[var(--signal)] bg-[var(--signal-dim)] text-[var(--signal)]" : "border-[var(--hairline)] text-slate-400 hover:text-slate-200"}`}>
                            {s.step.toLocaleString()}
                          </button>
                        );
                      })}
                    </div>
                    <p className="mt-1.5 label">
                      {selSteps.length} step{selSteps.length === 1 ? "" : "s"} → ~{selSteps.length * 3} GIFs (3 prompts/step)
                    </p>
                  </>
                )}
              </div>
            )}
          </>
        )}

        <button type="submit" className="btn-signal w-full"
          disabled={submitting
            || ((mode === "clip" || mode === "compare") && selected.length === 0)
            || (mode === "compare" && !checkpoint)
            || (mode === "samples" && (!run || selSteps.length === 0))}>
          {submitting ? "SUBMITTING…" : `▶  RENDER ${cur.label.toUpperCase()}`}
        </button>
      </form>

      <div className="surface min-w-0 p-6">
        <VizJobResult job={job} error={error} submitting={submitting} emptyHint="Rendered clips will appear here." />
      </div>
    </div>
  );
}
