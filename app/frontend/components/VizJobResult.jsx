"use client";

import { mediaUrl } from "@/lib/api";
import { ACTIVE } from "@/lib/useVizJob";
import { useLightbox } from "./Lightbox";

const STATE_COLOR = {
  queued: "var(--amber)", submitting: "var(--muted)", pending: "var(--amber)",
  running: "var(--signal)", pulling: "var(--accent2)", done: "#34d399",
  failed: "#fb7185", cancelled: "#fb7185",
};

// Shows a viz job's live state chip + the pulled clips (or spinner / error).
export default function VizJobResult({ job, error, submitting, emptyHint }) {
  const { open } = useLightbox();   // hook before any early return
  if (error) {
    return <Box className="text-rose-300">⚠ {error}</Box>;
  }
  if (!job && !submitting) {
    return <Box className="text-[var(--muted)]">{emptyHint || "Launch a job to see results here."}</Box>;
  }

  const state = submitting ? "submitting" : job?.state;
  const color = STATE_COLOR[state] || "var(--muted)";
  const busy = submitting || (job && ACTIVE.has(job.state));
  const outputs = job?.outputs || [];
  // Any GT tile means this job's results are GT/PRED pairs — lay them out two-up.
  const paired = outputs.some((o) => o.kind === "gt");
  const nCached = outputs.filter((o) => o.from_registry).length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span
          className={`rounded px-2 py-0.5 font-mono text-[11px] ${busy ? "animate-pulse-soft" : ""}`}
          style={{ color, border: `1px solid ${color}` }}
        >
          {state}{state === "queued" && job?.queue_pos ? ` #${job.queue_pos}` : ""}
        </span>
        {job?.slurm_id && <span className="font-mono text-[var(--muted)]">#{job.slurm_id}</span>}
        {job?.run_name && <span className="truncate font-mono text-[var(--muted)]">{job.run_name}</span>}
        {nCached > 0 && (
          <span className="text-[11px] text-[var(--muted)]" title="GT already in the registry — paired in without re-rendering">
            {nCached} GT reused
          </span>
        )}
        {job?.error && <span className="text-rose-300">{job.error}</span>}
      </div>
      {job?.params?.checkpoint && (
        <div className="text-[11px] text-[var(--muted)]">
          checkpoint: <span className="font-mono text-slate-300">{ckptLabel(job.params.checkpoint)}</span>
        </div>
      )}

      {busy && (
        <Box>
          <div className="flex flex-col items-center gap-3 text-[var(--muted)]">
            <span className="h-8 w-8 animate-spin rounded-full border-2 border-[var(--hairline-strong)] border-t-[var(--signal)]" />
            <span className="label">
              {state === "queued"
                ? `waiting in local queue${job?.queue_pos ? ` · #${job.queue_pos}` : ""}…`
                : state === "pending"
                ? "queued on the cluster…"
                : state === "pulling"
                ? "pulling media…"
                : job?.kind && job.kind !== "viz"
                ? `${job.kind} running on GPU…`
                : "rendering on GPU…"}
            </span>
            <span className="text-[10px] text-[var(--muted)]">
              live progress bar + controls → header ▸ <span className="text-[var(--signal)]">system</span>
            </span>
          </div>
        </Box>
      )}

      {!busy && outputs.length > 0 && (
        // GT+PRED of the SAME clip arrive back-to-back (the backend sorts them
        // that way and splices registry GT in at the same position), so a 2-col
        // grid pairs each clip's GT and PRED on one row. Modes without GT
        // pairing use the wider responsive grid.
        <div className={`grid gap-3 ${paired ? "sm:grid-cols-2" : "sm:grid-cols-2 xl:grid-cols-3"}`}>
          {outputs.map((o, i) => <Tile key={i} o={o} onOpen={() => open(outputs, i)} />)}
        </div>
      )}
    </div>
  );
}

const KIND_META = {
  gt: { label: "GT", color: "var(--muted)" },
  pred: { label: "PRED", color: "var(--signal)" },
  sample: { label: "SAMPLE", color: "var(--accent2)" },
};

// One rendered clip: the media + a caption block that surfaces the TRUE text
// (and GT/PRED badge + clip id) the backend read from the job manifest, so the
// label always matches what the clip was conditioned on. Clicking the media
// opens it fullscreen (arrow keys then walk the whole result set).
function Tile({ o, onOpen }) {
  const k = KIND_META[o.kind];
  return (
    <figure className="min-w-0 overflow-hidden rounded-lg border border-[var(--hairline)] bg-black">
      <button type="button" onClick={onOpen} title="open fullscreen" className="group relative block w-full">
        {o.media_url.toLowerCase().endsWith(".mp4") ? (
          // `pointer-events-none` so the click lands on the button, not the
          // video's own controls — fullscreen is where you scrub.
          <video src={mediaUrl(o.media_url)} className="pointer-events-none w-full" autoPlay loop muted />
        ) : (
          <img src={mediaUrl(o.media_url)} alt={o.caption} className="w-full" />
        )}
        <span className="absolute inset-0 grid place-items-center bg-black/40 opacity-0 transition group-hover:opacity-100">
          <span className="rounded border border-[var(--signal)] bg-black/70 px-2 py-1 text-[10px] text-[var(--signal)]">⤢ fullscreen</span>
        </span>
      </button>
      <figcaption className="space-y-1 px-2 py-1.5">
        <div className="flex items-center gap-1.5 text-[10px]">
          {k && (
            <span className="rounded px-1.5 py-0.5 font-mono font-medium" style={{ color: k.color, border: `1px solid ${k.color}` }}>
              {k.label}
            </span>
          )}
          {o.clip_id && <span className="font-mono text-slate-300">{o.clip_id}</span>}
          {o.step != null && <span className="font-mono text-[var(--muted)]">step {o.step}</span>}
          {o.from_registry && (
            <span className="rounded border border-[var(--hairline)] px-1 py-0.5 font-mono text-[9px] text-[var(--muted)]"
              title="reused from the GT registry — this job didn't re-render it">
              cached
            </span>
          )}
        </div>
        <div className="max-h-16 overflow-y-auto break-words text-[11px] leading-snug text-slate-400">{o.caption}</div>
        <div className="flex gap-2 pt-0.5 text-[10px]">
          <a href={mediaUrl(o.media_url)} download className="text-[var(--muted)] hover:text-[var(--signal)]" title="download clip">⤓ {o.media_url.toLowerCase().endsWith(".mp4") ? "mp4" : "gif"}</a>
          {o.npy_url && <a href={mediaUrl(o.npy_url)} download className="text-[var(--muted)] hover:text-[var(--signal)]" title="download joints .npy">⤓ npy</a>}
        </div>
      </figcaption>
    </figure>
  );
}

// /…/runs/<kind>/<run>/checkpoints/<file>.pt → "<run>/<file>.pt"
function ckptLabel(path) {
  const parts = path.split("/");
  const run = parts[parts.length - 3];
  const file = parts[parts.length - 1];
  return run ? `${run}/${file}` : file;
}

function Box({ children, className = "" }) {
  return (
    <div className={`grid min-h-[180px] place-items-center rounded-lg border border-dashed border-[var(--hairline)] bg-ink p-6 text-center text-[12px] ${className}`}>
      {children}
    </div>
  );
}
