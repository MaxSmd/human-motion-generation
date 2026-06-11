// Thin fetch wrapper around the FastAPI backend.
//
// API base is configurable via NEXT_PUBLIC_API_BASE so the same build works
// whether the backend is local or tunnelled from the cluster.
export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"
).replace(/\/$/, "");

// Backend returns relative media URLs ("/media/xyz.mp4"); resolve against API base.
export function mediaUrl(path) {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

async function req(path, opts) {
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => req("/health"),
  checkpoints: () => req("/checkpoints"),

  generate: (body) =>
    req("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  gtList: ({ subset_n = 0, subset_fraction = 0.01, subset_seed = 0, limit = 60 }) =>
    req(
      `/gt?subset_n=${subset_n}&subset_fraction=${subset_fraction}` +
        `&subset_seed=${subset_seed}&limit=${limit}`
    ),
  gtClip: (cid) => req(`/gt/${encodeURIComponent(cid)}`),

  runs: () => req("/runs"),
  runSteps: (run) => req(`/runs/${encodeURIComponent(run)}/steps`),
  runSample: (run, step) => req(`/runs/${encodeURIComponent(run)}/sample/${step}`),

  // ── cluster control plane ────────────────────────────────────────────────
  clusterStatus: () => req("/cluster/status"),
  clusterSqueue: () => req("/cluster/squeue"),
  clusterRuns: () => req("/cluster/runs"),
  clusterCheckpoints: (run) => req(`/cluster/checkpoints?run=${encodeURIComponent(run)}`),
  clusterCancel: (slurmId) => req(`/cluster/cancel/${slurmId}`, { method: "POST" }),

  jobs: () => req("/cluster/jobs"),
  job: (id) => req(`/cluster/jobs/${id}`),
  jobLog: (id, lines = 200) => req(`/cluster/jobs/${id}/log?lines=${lines}`),
  submitViz: (body) => post("/cluster/jobs/viz", body),
  submitTrain: (body) => post("/cluster/jobs/train", body),
  previewTrain: (body) => post("/cluster/jobs/train/preview", body),
  submitEval: (body) => post("/cluster/jobs/eval", body),
};

function post(path, body) {
  return req(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
