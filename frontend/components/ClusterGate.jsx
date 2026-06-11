"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";

const POLL_MS = 4000;

const COPY = {
  vpn_down: {
    title: "Cluster unreachable",
    body: "No route to the head node. Connect to the VPN, then retry.",
    hint: "Bring up your VPN connection, then this unlocks automatically.",
  },
  ssh_error: {
    title: "SSH failing",
    body: "The head node is reachable but SSH did not authenticate.",
    hint: "Check your key / ~/.ssh/config and known_hosts (one-time `ssh head true`).",
  },
  disabled: {
    title: "Cluster mode off",
    body: "RMG_CLUSTER_MODE is disabled on the backend.",
    hint: "Set RMG_CLUSTER_MODE=1 and restart the backend to enable it.",
  },
  connecting: { title: "Connecting…", body: "Probing the head node.", hint: "" },
};

export default function ClusterGate({ children }) {
  const [status, setStatus] = useState(null);
  const [checking, setChecking] = useState(false);

  const poll = useCallback(async () => {
    setChecking(true);
    try {
      setStatus(await api.clusterStatus());
    } catch (e) {
      setStatus({ state: "vpn_down", detail: e.message });
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    poll();
    const t = setInterval(poll, POLL_MS);
    return () => clearInterval(t);
  }, [poll]);

  if (status?.state === "online") return children;

  const key = status?.state || "connecting";
  const copy = COPY[key] || COPY.connecting;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
      {/* scanline backdrop */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0 1px, transparent 1px 4px)",
        }}
      />
      <div className="surface relative w-full max-w-md p-8 text-center animate-fade-up">
        <div className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-full border border-[var(--amber)]/40">
          <span
            className="dot animate-pulse-soft"
            style={{ color: "var(--amber)", width: 12, height: 12 }}
          />
        </div>
        <div className="label mb-2 text-[var(--amber)]">no signal · {key}</div>
        <h2 className="display text-2xl font-bold text-white">{copy.title}</h2>
        <p className="mt-3 text-[13px] leading-relaxed text-slate-400">{copy.body}</p>
        {status?.detail && (
          <p className="mt-2 break-words font-mono text-[11px] text-[var(--muted)]">
            {status.detail}
          </p>
        )}
        {copy.hint && (
          <p className="mt-4 rounded-lg border border-[var(--hairline)] bg-ink px-3 py-2 text-left font-mono text-[11px] text-slate-400">
            {copy.hint}
          </p>
        )}
        <button onClick={poll} disabled={checking} className="btn-signal mt-6 w-full">
          {checking ? "RETRYING…" : "RETRY NOW"}
        </button>
        <p className="label mt-3">auto-retrying every {POLL_MS / 1000}s</p>
      </div>
    </div>
  );
}
