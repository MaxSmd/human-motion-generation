"use client";

import { mediaUrl } from "@/lib/api";
import { useLightbox } from "./Lightbox";

// Renders an MP4 (<video>) or GIF (<img>) clip inside the instrument viewport,
// picking by file extension. Clicking the clip blows it up fullscreen.
export default function MediaViewer({ url, caption, loading, error, status }) {
  const { open } = useLightbox();
  const full = url ? mediaUrl(url) : null;
  const isVideo = full?.toLowerCase().endsWith(".mp4");

  return (
    <div className="space-y-2">
      <div className="viewport aspect-square w-full">
        {/* top status strip */}
        <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between px-4 py-2.5 text-[10px]">
          <span className="label flex items-center gap-1.5">
            <span
              className="dot"
              style={{
                color: loading
                  ? "var(--amber)"
                  : error
                  ? "#fb7185"
                  : full
                  ? "var(--signal)"
                  : "var(--muted)",
              }}
            />
            {loading ? "rendering" : error ? "error" : full ? "ready" : "idle"}
          </span>
          {status && <span className="label">{status}</span>}
        </div>

        <div className="flex h-full w-full items-center justify-center p-3">
          {loading ? (
            <div className="flex flex-col items-center gap-3 text-[var(--muted)]">
              <span className="h-9 w-9 animate-spin rounded-full border-2 border-[var(--hairline-strong)] border-t-[var(--signal)]" />
              <span className="label">solving ODE · forward-kinematics</span>
            </div>
          ) : error ? (
            <p className="max-w-sm px-4 text-center text-[12px] leading-relaxed text-rose-300">
              {error}
            </p>
          ) : full ? (
            <button
              type="button"
              onClick={() => open([{ media_url: url, caption }])}
              title="open fullscreen"
              className="group relative flex h-full w-full items-center justify-center"
            >
              {isVideo ? (
                // `pointer-events-none`: the click belongs to the button —
                // scrubbing happens in the fullscreen view.
                <video
                  key={full}
                  src={full}
                  className="pointer-events-none max-h-full max-w-full rounded"
                  autoPlay
                  loop
                  muted
                />
              ) : (
                <img
                  key={full}
                  src={full}
                  alt={caption || "motion clip"}
                  className="max-h-full max-w-full rounded"
                />
              )}
              <span className="absolute bottom-2 right-2 rounded border border-[var(--signal)] bg-black/70 px-1.5 py-0.5 text-[9px] text-[var(--signal)] opacity-0 transition group-hover:opacity-100">
                ⤢ fullscreen
              </span>
            </button>
          ) : (
            <div className="text-center">
              <Crosshair />
              <p className="label mt-3">no signal</p>
            </div>
          )}
        </div>
      </div>
      {caption && (
        <p className="px-1 text-[12px] leading-snug text-slate-400">{caption}</p>
      )}
    </div>
  );
}

function Crosshair() {
  return (
    <svg width="60" height="60" viewBox="0 0 60 60" className="mx-auto opacity-30">
      <circle cx="30" cy="30" r="22" stroke="var(--muted)" strokeWidth="1" fill="none" />
      <circle cx="30" cy="30" r="3" fill="var(--muted)" />
      <line x1="30" y1="2" x2="30" y2="14" stroke="var(--muted)" strokeWidth="1" />
      <line x1="30" y1="46" x2="30" y2="58" stroke="var(--muted)" strokeWidth="1" />
      <line x1="2" y1="30" x2="14" y2="30" stroke="var(--muted)" strokeWidth="1" />
      <line x1="46" y1="30" x2="58" y2="30" stroke="var(--muted)" strokeWidth="1" />
    </svg>
  );
}
