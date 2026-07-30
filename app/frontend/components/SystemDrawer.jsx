"use client";

// SYSTEM drawer — the old Cluster tab, demoted to an ambient panel you glance at
// rather than a primary destination. Holds the link readout, the remote squeue,
// the current-training pause/resume controls and the raw global job list (the
// per-workspace history rails are filtered views of that same list).

import ClusterTab from "./ClusterTab";

export default function SystemDrawer({ status, open, onClose }) {
  if (!open) return null;
  return (
    <div className="mb-8 animate-fade-up rounded-xl border border-[var(--hairline-strong)] bg-panel/40 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="label flex items-center gap-2">
          <span className="dot" style={{ color: "var(--signal)" }} />
          system · cluster monitor
        </div>
        <button onClick={onClose}
          className="rounded-md border border-[var(--hairline)] px-2.5 py-1 text-[11px] text-slate-300 hover:border-[var(--signal)] hover:text-[var(--signal)]">
          close ✕
        </button>
      </div>
      <ClusterTab status={status} />
    </div>
  );
}
