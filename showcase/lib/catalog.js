// The Studio catalog.
//
// Each scenario is a base prompt with up to two optional axes — a hard joint
// constraint and an added phrase in the text — and one pre-rendered clip for
// every reachable combination. A scenario may also name a ground-truth capture,
// which unlocks the side-by-side comparison. Every clip is a tiny .npy under
// /data; clicking through plays a short loading sequence (GenerationProgress)
// and then loads the matching array. Nothing is sampled in the browser, and the
// panel says so while it loads.
//
// Combination keys: "" none · "c" constraint · "t" text · "ct" both.
// `held` / `jerk` are measured here from the rendered array.

export const SAMPLER = {
  model: "RMG-mid · 111.7 M · ckpt 300 k",
  omega: 6.5,
  steps: 800,
  seed: 0,
};

export const SCENARIOS = [
  {
    id: "walkturn",
    title: "Walk and turn",
    prompt: "a person walks forward then turns around and walks back",
    joint: "left knee",
    limb: "left_leg",
    constraint: {
      label: "Clamp the left knee to 0 – 10°",
      detail: "bend angle at the knee clamped into its range, every ODE step",
    },
    text: { label: "Add “while limping the left leg”", phrase: "while limping the left leg" },
    gt: "gt_walkturn",
    variants: {
      "": { file: "c_free.npy", held: "13 – 62°", jerk: 0.0141 },
      c: { file: "c_conflict.npy", held: "≤ 10°", jerk: 0.0134 },
      t: { file: "c_limp_free.npy", held: "10 – 52°", jerk: 0.2468 },
      ct: { file: "c_coherent.npy", held: "≤ 10°", jerk: 0.0194 },
    },
    note: "In this sample the clamp leaves smoothness essentially unchanged with the neutral prompt. The unconstrained limping draw is an unusually rough one, which is why the constrained version of it looks much smoother. Over 36 seed-paired renders no effect is resolved; see the results page.",
  },
  {
    id: "box",
    title: "Carry a box",
    prompt: "a person walks forward holding a box against their chest",
    joint: "left elbow",
    limb: "left_arm",
    constraint: { label: "Pin the left elbow at 90°", detail: "bend angle at the elbow held at 90°, every ODE step" },
    variants: {
      "": { file: "c_box_free.npy", held: "85 – 141°", jerk: 0.042 },
      c: { file: "c_elbow90.npy", held: "90.0°", jerk: 0.32 },
    },
  },
  {
    id: "run",
    title: "Run",
    prompt: "a person runs forward",
    joint: "left knee",
    limb: "left_leg",
    constraint: { label: "Clamp the left knee to 0 – 10°", detail: "bend angle at the knee clamped into its range, every ODE step" },
    variants: {
      "": { file: "n_run.npy", held: "18 – 97°", jerk: 0.0287 },
      c: { file: "c_run_knee.npy", held: "≤ 10°", jerk: 0.0218 },
    },
  },
  {
    id: "wave",
    title: "Wave",
    prompt: "a person waves their right hand",
    joint: "right elbow",
    limb: "right_arm",
    constraint: { label: "Pin the right elbow straight (0°)", detail: "bend angle at the elbow held at 0°, every ODE step" },
    variants: {
      "": { file: "n_wave.npy", held: "13 – 92°", jerk: 0.0848 },
      c: { file: "c_wave_elbow.npy", held: "0.0°", jerk: 0.0157 },
    },
    note: "In this sample the pinned elbow gives smoother motion than the unconstrained wave. This is a single seed.",
  },
  {
    id: "stiffwalk",
    title: "Stiff-legged walk",
    prompt: "a person walks forward",
    joint: "both knees",
    limb: "left_leg",
    constraint: { label: "Clamp both knees to 0 – 10°", detail: "bend angle at each knee clamped into its range, every ODE step" },
    variants: {
      "": { file: "n_walk.npy", held: "9 – 68°", jerk: 0.0112 },
      c: { file: "c_walk_stiff.npy", held: "≤ 10°", jerk: 0.0111 },
    },
    note: "Clamping both knees gives a stiff, marching gait. Jerk is unchanged in this sample, 0.0112 against 0.0111.",
  },
  {
    id: "kick",
    title: "Kick",
    prompt: "a person kicks with their right leg",
    joint: "right knee",
    limb: "right_leg",
    constraint: { label: "Clamp the right knee to 0 – 10°", detail: "bend angle at the knee clamped into its range, every ODE step" },
    variants: {
      "": { file: "n_kick.npy", held: "7 – 113°", jerk: 0.0239 },
      c: { file: "c_kick_knee.npy", held: "≤ 10°", jerk: 0.0273 },
    },
  },
  {
    id: "boxing",
    title: "Boxing",
    prompt: "a person is boxing, throwing jabs",
    joint: "left elbow",
    limb: "left_arm",
    constraint: { label: "Pin the left elbow at 90° (guard up)", detail: "bend angle at the elbow held at 90°, every ODE step" },
    variants: {
      "": { file: "n_boxjabs.npy", held: "27 – 156°", jerk: 0.0362 },
      c: { file: "c_box_guard.npy", held: "90.0°", jerk: 0.0644 },
    },
  },
  // Free motions — no constraint axis, just generation from the prompt.
  { id: "dance", title: "Dance", prompt: "a person is doing a dance", variants: { "": { file: "gen_dance.npy", jerk: 0.1768 } }, gt: "gt_dance" },
  { id: "spin", title: "Spin around", prompt: "a person walks forward, spins around, and walks back", variants: { "": { file: "gen_spin.npy", jerk: 0.0197 } }, gt: "gt_spin" },
  { id: "cartwheel", title: "Cartwheel", prompt: "a person does a cartwheel", variants: { "": { file: "n_cartwheel.npy", jerk: 0.0676 } } },
];

// Key for the active toggle set.
export const comboKey = (constraint, text) => `${constraint ? "c" : ""}${text ? "t" : ""}`;

export const scenarioById = (id) => SCENARIOS.find((s) => s.id === id);

// The variant for a scenario under the given toggles, or null if not rendered.
export function resolveVariant(scenario, constraint, text) {
  return scenario.variants[comboKey(constraint, text)] || null;
}

export const dataUrl = (file) => `/data/${file}`;
