// Maps a persisted backend job (.media/jobs.json, served by /cluster/jobs) to a
// single workspace category. The history rails are just filtered views of the
// global job list, so this classifier is what makes "constraint generations live
// with constraint generations" work — with zero backend change, since every job
// already carries the params we sniff here.
//
//   prompt      plain text → motion
//   constrained text → motion with fixed angles / hinge ranges
//   scene       text → motion inside a 3D room (placement + room guidance)
//   clip        rendered ground-truth clips
//   compare     GT vs the model's prediction
//   samples     a training run's fixed-prompt dumps
//   train       a training run
//   eval        an evaluation run
export function classifyJob(job) {
  if (!job) return "job";
  const { kind, mode, params = {} } = job;
  if (kind === "train") return "train";
  if (kind === "eval") return "eval";
  // viz kinds, split by mode + payload
  if (mode === "compare") return "compare";
  if (mode === "samples") return "samples";
  if (mode === "clip") return "clip";
  if (mode === "prompt") {
    if (params.scene) return "scene";
    if (params.constraints?.length || params.ranges?.length) return "constrained";
    return "prompt";
  }
  return kind || "job";
}

// Per-category display metadata (chip label + accent colour).
export const CATEGORY_META = {
  prompt:      { label: "Prompt",      color: "var(--signal)" },
  constrained: { label: "Constrained", color: "var(--amber)" },
  scene:       { label: "Scene",       color: "var(--accent2)" },
  clip:        { label: "GT clips",    color: "var(--muted)" },
  compare:     { label: "Compare",     color: "var(--signal)" },
  samples:     { label: "Samples",     color: "var(--accent2)" },
  train:       { label: "Train",       color: "var(--signal)" },
  eval:        { label: "Eval",        color: "var(--amber)" },
  job:         { label: "Job",         color: "var(--muted)" },
};

export function categoryMeta(cat) {
  return CATEGORY_META[cat] || CATEGORY_META.job;
}

// The categories surfaced by each workspace's history rail.
export const WORKSPACE_CATEGORIES = {
  create: ["prompt", "constrained", "scene"],
  library: ["clip", "compare", "samples"],
  lab: ["train", "eval"],
};
