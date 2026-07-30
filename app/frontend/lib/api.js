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
  metaJoints: () => req("/meta/joints"),

  generate: (body) =>
    req("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  // Caption-corpus support for a candidate phrase — "has the text encoder ever
  // seen this language?". Drives the constraint→text ablation's go/no-go badge.
  corpusPhrase: (q, examples = 4) =>
    req(`/corpus/phrase?q=${encodeURIComponent(q)}&examples=${examples}`),
  // Rank candidate phrasings of one motion by caption support — lets the
  // archetype map propose variants and the corpus pick the winner.
  corpusRank: (candidates, examples = 1) => post("/corpus/rank", { candidates, examples }),
  corpusStatus: () => req("/corpus/status"),

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
  getModel: () => req("/cluster/model"),
  setModel: (model) => post("/cluster/model", { model }),
  clusterStatus: () => req("/cluster/status"),
  clusterSqueue: () => req("/cluster/squeue"),
  clusterRuns: () => req("/cluster/runs"),
  clusterCheckpoints: (run) => req(`/cluster/checkpoints?run=${encodeURIComponent(run)}`),
  clusterCancel: (slurmId) => req(`/cluster/cancel/${slurmId}`, { method: "POST" }),
  clusterGtClips: ({ split = "train", subset_fraction = 1.0, subset_seed = 0, subset_n = 0, limit = 60, tag_seen = false, q = "" } = {}) =>
    req(
      `/cluster/gt-clips?split=${encodeURIComponent(split)}&subset_fraction=${subset_fraction}` +
        `&subset_seed=${subset_seed}&subset_n=${subset_n}&limit=${limit}&tag_seen=${tag_seen}` +
        `&q=${encodeURIComponent(q)}`
    ),
  // Clip ids whose GT render is already stored → those pair up without GPU work.
  clusterGtRegistry: () => req("/cluster/gt-registry"),
  runSampleSteps: (run) => req(`/cluster/run-sample-steps?run=${encodeURIComponent(run)}`),

  jobs: () => req("/cluster/jobs"),
  queue: () => req("/cluster/queue"),
  job: (id) => req(`/cluster/jobs/${id}`),
  jobLog: (id, lines = 200) => req(`/cluster/jobs/${id}/log?lines=${lines}`),
  cancelJob: (id) => req(`/cluster/jobs/${id}/cancel`, { method: "POST" }),
  deleteJob: (id) => req(`/cluster/jobs/${id}`, { method: "DELETE" }),
  pauseJob: (id) => req(`/cluster/jobs/${id}/pause`, { method: "POST" }),
  resumeJob: (id) => req(`/cluster/jobs/${id}/resume`, { method: "POST" }),
  jobProgress: (id) => req(`/cluster/jobs/${id}/progress`),
  adoptable: () => req("/cluster/jobs/adoptable"),
  adoptJob: (body) => post("/cluster/jobs/adopt", body),
  submitViz: (body) => post("/cluster/jobs/viz", body),
  submitTrain: (body) => post("/cluster/jobs/train", body),
  previewTrain: (body) => post("/cluster/jobs/train/preview", body),
  submitEval: (body) => post("/cluster/jobs/eval", body),

  evalRuns: () => req("/cluster/eval-runs"),
  evalResults: (run) => req(`/cluster/eval/${encodeURIComponent(run)}`),
  // Every matching run's metrics in ONE ssh round trip. Use this instead of
  // looping evalResults() over a run list: that costs two SSH calls per run and
  // spins forever whenever the login node is slow.
  evalResultsBulk: (match = "", opts) =>
    req(`/cluster/eval-results?match=${encodeURIComponent(match)}`, opts),
  analysisTable: (runs) =>
    req(`/cluster/analysis/table?runs=${encodeURIComponent(runs.join(","))}`),
  runMetrics: (run) => req(`/cluster/metrics?run=${encodeURIComponent(run)}`),
  runInfo: (run) => req(`/cluster/run-info?run=${encodeURIComponent(run)}`),
  analysisNpy: (job, name) =>
    req(`/cluster/analysis/npy?job=${encodeURIComponent(job)}&name=${encodeURIComponent(name)}`),
  compareNpy: (job, real, gen) =>
    req(`/cluster/analysis/compare-npy?job=${encodeURIComponent(job)}` +
      `&real=${encodeURIComponent(real)}&gen=${encodeURIComponent(gen)}`),
  // Extract contact constraints from a draft clip's foot plants → drop into
  // scene.contacts for a constrained resample (Pass B).
  autoContacts: (job, name, scene, fps = 20) =>
    post("/cluster/analysis/auto-contacts", { job, name, scene, fps }),
  forensics: (run) => req(`/cluster/analysis/forensics?run=${encodeURIComponent(run)}`),

  // ── obstacle-course study ────────────────────────────────────────────────
  // The catalogue is the SINGLE source of the course geometry: the viewport
  // draws it, the backend samples it, and the metrics score against it. The
  // frontend deliberately declares no geometry of its own.
  courses: () => req("/cluster/analysis/courses"),
  submitCourseStudy: (body) => post("/cluster/jobs/course-study", body),
  courseStudy: () => req("/cluster/analysis/course-study"),
};

function post(path, body) {
  return req(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
