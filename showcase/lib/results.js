// Every number rendered on this site, transcribed from the cluster run
// directories on 2026-07-23. Each block names the run(s) it came from so a
// value can be traced back to a results.json.
//
// Protocol for all eval numbers unless stated otherwise: HumanML3D test split,
// all 4096 clips, EMA weights, inference length mask on, Guo evaluator with its
// own Comp_v6 normalisation. FID is meaningless without its guidance scale and
// ODE step count, so both are always carried alongside.

// CVD-safe palette shared by every chart.
export const C = {
  mid: "#22d3ee",
  base: "#f59e0b",
  paper: "#a78bfa",
  real: "#34d399",
  warn: "#fb7185",
  muted: "#8e9cb3",
};

// ── Works this one builds on. Every external number, dataset and method on the
//    site is attributed here; the site cites by key.
export const REPO = "https://github.com/julsmzr/riemann-motion-generation";

export const REFERENCES = [
  {
    key: "rmg",
    text: "Miao, Huang and Li, Riemannian Motion Generation: A Unified Framework for Human Motion Representation and Generation via Riemannian Flow Matching (March 2026). The method and the RMG-main result this work reproduces at smaller scale.",
    cite: "Miao et al. 2026, arXiv:2603.15016",
    href: "https://arxiv.org/abs/2603.15016",
  },
  {
    key: "humanml3d",
    text: "Guo et al., Generating Diverse and Natural 3D Human Motions from Text (CVPR 2022). The HumanML3D dataset, and the movement/text encoders every metric on this site is computed with.",
    cite: "Guo et al. 2022",
    href: "https://github.com/EricGuo5513/HumanML3D",
  },
  {
    key: "amass",
    text: "Mahmood et al., AMASS: Archive of Motion Capture as Surface Shapes (ICCV 2019). The capture corpus underlying HumanML3D.",
    cite: "Mahmood et al. 2019",
    href: "https://amass.is.tue.mpg.de/",
  },
  {
    key: "repaint",
    text: "Lugmayr et al., RePaint: Inpainting using Denoising Diffusion Probabilistic Models (CVPR 2022). The reference point for constraining a sample by overwriting part of it at every reverse step.",
    cite: "Lugmayr et al. 2022",
    href: "https://arxiv.org/abs/2201.09865",
  },
  {
    key: "cfm",
    text: "Chen and Lipman, Flow Matching on General Geometries (ICLR 2024). Riemannian conditional flow matching, the training objective used here.",
    cite: "Chen and Lipman 2024",
    href: "https://arxiv.org/abs/2302.03660",
  },
  {
    key: "qwen",
    text: "Qwen3-Embedding-0.6B, the frozen text encoder used for conditioning.",
    cite: "Qwen3-Embedding",
    href: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
  },
];

// The dataset both models are trained and evaluated on. Split sizes are our
// packed coverage (splits.json on the cluster); the corpus size is the caption
// index the studio scores against; the rest are the published HumanML3D figures.
export const DATASET = {
  name: "HumanML3D",
  blurb:
    "Motion capture from AMASS and HumanAct12, retargeted to a shared 22-joint skeleton and paired with natural-language descriptions. The standard benchmark for text-to-motion.",
  stats: [
    ["motions", "14,616"],
    ["captions", "≈ 45,000"],
    ["motion capture", "28.6 h"],
    ["frame rate", "20 fps"],
    ["joints", "22"],
    ["clip length", "2 – 10 s"],
  ],
  splits: [
    ["train", "21,777"],
    ["val", "1,362"],
    ["test", "4,096"],
  ],
  augment:
    "Every capture also appears left/right mirrored, with its caption mirrored too. This is part of the standard splits, so the test split contains 1,986 mirrored clips alongside 2,110 originals. Split sizes are the clips we packed: 14,616 captures give 29,232 after mirroring, of which 27,235 are at least 40 frames long and are kept.",
};

export const MODELS = {
  base: {
    key: "base",
    name: "RMG-base",
    color: C.base,
    params: "24.7 M",
    arch: "384 wide · 6 layers · 8 heads · FFN ×8",
    steps: "150 k",
    batch: "256 (32 × 8)",
    precision: "bf16",
    warmup: "8 %",
    antipodal: false,
    time: "4.4 d",
    throughput: "0.39 steps/s",
    run: "rmg-base-v1",
  },
  mid: {
    key: "mid",
    name: "RMG-mid",
    color: C.mid,
    params: "111.7 M",
    paramsExact: 111_696_731,
    arch: "768 wide · 10 layers · 12 heads · FFN ×4",
    steps: "300 k",
    batch: "256 (64 × 4)",
    precision: "bf16 → tf32 @ 160 k",
    warmup: "5 %",
    antipodal: true,
    time: "5.7 d",
    throughput: "0.63 steps/s",
    run: "rmgui-train-rmg_mid-f8f7cf5e",
  },
  paper: {
    key: "paper",
    name: "RMG-main (published)",
    color: C.paper,
    params: "≈ 460 M",
    arch: "1024 wide · 24 layers · 8 heads · FFN ×4",
    steps: "600 k",
    batch: "256 (16 × 8 × 2)",
    precision: "not stated",
    warmup: "8 %",
    antipodal: null,
    time: "—",
    throughput: "—",
    run: "—",
  },
};

// Shared across both runs (paper Table 7 matches on every line here).
export const RECIPE = [
  ["objective", "Riemannian conditional flow matching"],
  ["manifold", "ℝ³ × (S³)²², 91-D ambient"],
  ["optimizer", "AdamW · β (0.9, 0.999) · wd 0"],
  ["peak LR", "1e-4, cosine to 0"],
  ["grad clip", "0.5"],
  ["CFG dropout", "0.1"],
  ["EMA decay", "0.9999"],
  ["text encoder", "Qwen3-Embedding-0.6B, 1024-d pooled, frozen"],
  ["conditioning", "MLP fuse → AdaLN-Zero, no cross-attention"],
];

// ── Guidance sweep, 200 ODE steps (runs: eval-mid, eval-base) ───────────────
export const OMEGA = {
  w: [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
  mid: {
    fid: [1.7929, 0.9643, 0.7982, 0.6829, 0.607, 0.6376, 0.7499, 0.8123],
    r1: [0.439, 0.4855, 0.4824, 0.4927, 0.4983, 0.4907, 0.49, 0.4941],
  },
  base: {
    fid: [8.8573, 8.2915, 8.3899, 8.0493, 8.4633, 8.587, 8.7898, 8.6708],
    r1: [0.2126, 0.2344, 0.2251, 0.2241, 0.2209, 0.2185, 0.2234, 0.2236],
  },
  best: { mid: { w: 6.5, fid: 0.607 }, base: { w: 5.5, fid: 8.0493 } },
};

// ── ODE step convergence at ω 6.5 (runs: eval-mid-w55to75-ode100/200,
//    eval-mid-ode400, …gridfull-ode800-w65, …ode1600, …ode3200) ──────────────
export const STEPS = {
  s: [100, 200, 400, 800, 1600, 3200],
  fid: [0.8659, 0.607, 0.4766, 0.4288, 0.4096, 0.4236],
  r1: [0.488, 0.4983, 0.5022, 0.5071, 0.5044, 0.4973],
  mm: [3.294, 3.225, 3.132, 3.126, 3.12, 3.126],
  // The smallest value in the sweep, NOT the operating point: 0.4288 / 0.4096 /
  // 0.4236 across 800–3200 steps are within the replication spread of each other.
  min: { s: 1600, fid: 0.4096 },
  quoted: 800,
};

// ── Full metric set across the sampling-step sweep, each model at its own
//    guidance optimum, with the real-motion row as the attainable bound. The sweep itself
//    is a figure above; this table carries the metrics the figures do not plot.
//    Base was swept over all eight ω at both 200 and 1600 steps; its best over
//    the whole 1600-step sweep is 8.0816 at ω 4.5.
export const STEP_METRICS = [
  { model: "Reference (real motion)", w: null, steps: null, fid: 0.0019, r1: 0.513, r2: null, r3: 0.797, mm: 3.1, div: 9.79, mmod: null, ref: true },
  { model: "RMG-mid", w: 6.5, steps: 100, fid: 0.8659, r1: 0.488, r2: 0.6638, r3: 0.7566, mm: 3.294, div: 8.986, mmod: null },
  { model: "RMG-mid", w: 6.5, steps: 200, fid: 0.607, r1: 0.4983, r2: 0.676, r3: 0.7666, mm: 3.225, div: 9.278, mmod: 1.794 },
  { model: "RMG-mid", w: 6.5, steps: 400, fid: 0.4766, r1: 0.5022, r2: 0.6838, r3: 0.7778, mm: 3.132, div: 9.185, mmod: 2.085 },
  { model: "RMG-mid", w: 6.5, steps: 800, fid: 0.4288, r1: 0.5071, r2: 0.6902, r3: 0.7864, mm: 3.126, div: 9.056, mmod: 1.813, hi: true },
  { model: "RMG-mid", w: 6.5, steps: 1600, fid: 0.4096, r1: 0.5044, r2: 0.6909, r3: 0.7898, mm: 3.12, div: 9.007, mmod: 1.829 },
  { model: "RMG-mid", w: 6.5, steps: 3200, fid: 0.4236, r1: 0.4973, r2: 0.6826, r3: 0.7815, mm: 3.126, div: 9.084, mmod: 1.942 },
  { model: "RMG-base", w: 5.5, steps: 200, fid: 8.0493, r1: 0.2241, r2: 0.3625, r3: 0.4792, mm: 5.396, div: 8.058, mmod: 2.034 },
  { model: "RMG-base", w: 5.5, steps: 1600, fid: 8.3267, r1: 0.2119, r2: 0.3558, r3: 0.4596, mm: 5.432, div: 7.761, mmod: 2.267 },
];

// ── Operating-point selection on the validation split. The 1,362-clip val split
//    was swept over guidance at 200 ODE steps and again at the 800-step setting
//    we report at, with a second seed at ω 6.5 to measure how much of the
//    difference between cells is noise.
//
//    Runs: rmgui-eval-mid300k-valsweep-ode200 (eight ω) and
//    rmgui-eval-mid300k-val-ode800-{w55, w65b, w65-seed1b, w75c}.
//
//    The reading is that this split does not resolve the guidance scale: two
//    seeds at ω 6.5 differ by more than the whole spread across 5.5, 6.5 and 7.5.
export const VAL_SELECTION = {
  split: "HumanML3D validation, 1,362 clips",
  sweep200: {
    w: [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    fid: [2.0701, 0.9119, 0.6866, 0.5359, 0.6183, 0.6633, 0.7497, 0.8086],
    r1: [0.4397, 0.4933, 0.4598, 0.5082, 0.4918, 0.5067, 0.4963, 0.4851],
  },
  ode800: [
    { w: 5.5, seed: 0, fid: 0.523, r1: 0.5 },
    { w: 6.5, seed: 0, fid: 0.5997, r1: 0.4918 },
    { w: 6.5, seed: 1, fid: 0.4586, r1: 0.5074 },
    { w: 7.5, seed: 0, fid: 0.5605, r1: 0.5112 },
  ],
  // Two seeds at one setting, so a coarse estimate; it is quoted as a range.
  seedRange: 0.1412,
  omegaSpread: 0.0375,
  testSd: 0.024,
  kept: 6.5,
};

// ── Replication noise: four independent evals, identical settings. The seed
//    changes the noise draw and the shuffle; the fourth is a single-ω rerun at
//    seed 0, whose RNG trajectory differs from the eight-ω sweep that produced
//    the first. (runs: eval-mid ω6.5 · …w65-seed1 · …w65-seed2 ·
//    …w65-ode200-bs64-control) ──────────────────────────────────────────────
export const REPLICATION = {
  setting: "ω 6.5 · 200 ODE steps · full split",
  fid: [0.607, 0.6146, 0.6471, 0.6574],
  mean: 0.632,
  sd: 0.024,
  n: 4,
  r1sd: 0.004,
  // sd from four replications is itself a coarse estimate; we use ±2 sd (0.05)
  // as the resolution of this harness rather than quoting it as a CI.
  resolution: 0.05,
};

// ── FID vs training step, held at ω 6.5 / 800 ODE steps
//    (runs: rmgui-eval-mid{50,100,150,200,250}k-w65-ode800 + the 300 k cell) ─
export const CKPT_SWEEP = {
  step: [50000, 100000, 150000, 200000, 250000, 300000],
  fid: [24.0662, 2.2772, 12.0416, 1.3065, 0.6781, 0.4288],
  r1: [0.2402, 0.4563, 0.3523, 0.4811, 0.4966, 0.5071],
  diversity: [5.545, 8.523, 6.57, 8.898, 8.981, 9.056],
  // The 150 k checkpoint is the only non-monotone point, and it sits inside the
  // bf16 window that ended at 160 k.
  precisionSwitch: 160000,
};

// ── Operating point, both models. `steps` is the ODE step count at evaluation,
//    which is a separate lever from training steps and is carried on every row.
//
//    The published row is the paper's MODEL row (arXiv:2603.15016). Its R@1 and
//    diversity are 0.525 / 9.56; it reports no multimodal distance. Do not fill
//    that gap with the paper's ground-truth row (0.511 / 2.97 / 9.5) — those are
//    reference-motion numbers and belong in CALIBRATION, where they are used.
export const HEADLINE = {
  mid: { fid: 0.607, r1: 0.4983, r3: 0.7666, mm: 3.225, div: 9.278, mmod: 1.794, w: 6.5, steps: 200 },
  midBest: { fid: 0.4288, r1: 0.5071, r3: 0.7898, mm: 3.126, div: 9.056, mmod: 1.813, w: 6.5, steps: 800 },
  base: { fid: 8.0493, r1: 0.2241, r3: 0.4792, mm: 5.396, div: 8.058, mmod: 2.034, w: 5.5, steps: 200 },
  paper: { fid: 0.043, r1: 0.525, mm: null, div: 9.56, mmod: null, w: 6.5, steps: null },
  gt: { fid: 0.0019, r1: 0.513, r3: 0.797, mm: 3.1, div: 9.79, mmod: null },
};

// The quoted headline. 800 steps is the operating point we report: from 800 to
// 3200 every value sits inside the replication spread, so the extra integration
// is not a real gain and quoting the 1600-step minimum (0.4096) would be
// quoting the bottom of the noise.
export const QUOTED = {
  fid: 0.4288,
  sd: 0.024,
  w: 6.5,
  steps: 800,
  plateauFrom: 800,
  note: "single-pass full split; the published number is a 20-replication mean",
};

// ── Jerk distribution over every clip we measured (mean ‖Δ³x‖ per clip). Real
//    = 17 ground-truth captures; generated = 9 paired samples + 10 free prompts,
//    all mid @ ω6.5 / 800 steps. Measured here from the rendered arrays.
//    `realPaired` / `genPaired` are the nine (caption, capture, sample) triples
//    only. Those are the sole matched comparison we have: same caption, same
//    motion, one real and one generated. The free prompts have no capture to
//    compare against, so pooling them in would compare generations of one set of
//    motions against captures of another.
export const JERK_DIST = {
  realPaired: [0.0062, 0.0057, 0.0035, 0.0031, 0.0086, 0.0142, 0.0036, 0.021, 0.0019],
  genPaired: [0.013, 0.0152, 0.0197, 0.0239, 0.0372, 0.0513, 0.0825, 0.0907, 0.1768],
};

// ── Generation consistency: one prompt, 20 independent seeds, mid @ ω6.5 /
//    800 steps. Filled once the seeded renders land; the section is hidden
//    until then. The band is the IQR of real walking captures.
export const CONSISTENCY = {
  prompt: "a person walks forward",
  jerk: [
    0.0112, 0.0114, 0.0105, 0.0113, 0.0133, 0.0102, 0.0126, 0.0089, 0.01, 0.0126, 0.0101, 0.011,
    0.011, 0.0113, 0.0411, 0.0108, 0.0116, 0.0108, 0.0542, 0.0096,
  ],
  gtBand: { lo: 0.0029, med: 0.0032, hi: 0.0046 },
};

// ── Harness calibration: our measurement of quantities with a published value.
//    Run: eval_sanity, 2026-07-09/10, real motions only, no model involved. ──
export const CALIBRATION = [
  { metric: "FID, ground truth vs ground truth", ours: "0.0019", published: "0.002", note: "two-pass unit-length crop protocol" },
  { metric: "R@1 on ground truth", ours: "0.513", published: "0.511", note: "the retrieval ceiling of this harness" },
  { metric: "Multimodal distance, ground truth", ours: "3.10", published: "≈ 2.97", note: "" },
  { metric: "Diversity, ground truth", ours: "9.79", published: "9.5", note: "" },
  { metric: "263-D features vs official new_joint_vecs", ours: "0.0 % rel. diff", published: "bit-exact", note: "(T,R) → forward kinematics → features" },
];



// ── Scaling. Measured at matched settings (each model at its own ω optimum,
//    200 ODE steps, full split). TWO POINTS. The ratio is reported because it
//    is what we measured; it is not fitted, not extrapolated, and no claim is
//    made about a third configuration. base and mid also differ in training
//    length, precision and the antipodal quotient, so even the 13.3× is not
//    attributable to parameter count alone.
export const SCALING = {
  measured: { from: 8.0493, to: 0.607, params: 4.52, steps: 2 },
  ratio: 13.3,
  gapNow: 10, // quoted mid (0.4288) against published 0.043
  confounds: "training length (2×), precision (bf16 vs bf16→tf32), antipodal quotient (absent vs present)",
};


// ── Constraint satisfaction. Every projected sample in the factorial below,
//    measured from the rendered joint positions: the bend angle at the
//    constrained joint over all frames, against the limit it was given. Both
//    bounds are reported, since for a fixed angle a shortfall matters as much
//    as an overshoot. The full-split figure in CONSTRAINED_EVAL is the stronger
//    statement; this table also shows how much of an allowed range is used.
export const SATISFACTION = [
  { spec: "clamp to 0 – 10°", limit: "0 – 10°", n: 12, lo: 6.63, hi: 10.0, dev: 0.0 },
  { spec: "pin at 10°", limit: "10°", n: 12, lo: 10.0, hi: 10.0, dev: 0.0 },
  { spec: "pin at 0°", limit: "0°", n: 12, lo: 0.0, hi: 0.01, dev: 0.012 },
];

// ── Constrained generation scored on the full split (4096 clips, ω 6.5, 200 ODE
//    steps). Runs: rmgui-eval-mid300k-w65-ode200-{cnull,cclamp,cpin}-fix, against
//    the matched unconstrained single-ω control …-bs64-control.
//
//    The null row is the control: clamping to [0°, 180°] leaves every angle
//    already inside the range, so the projector runs at every step and can never
//    change anything. It reproduces the unconstrained FID to 0.001, which is what
//    licenses reading the other two rows.
//
//    Constraining every clip is deliberately off-distribution — HumanML3D has
//    almost no locked-knee walking — so FID must rise and its rise is not
//    evidence of worse generation. R-precision and multimodal distance are the
//    axis that carries meaning here: they say whether the sample still matches
//    its caption once a joint is held.
export const CONSTRAINED_EVAL = {
  setting: "mid · ω 6.5 · 200 ODE steps · full test split · 4096 clips",
  rows: [
    { label: "Unconstrained", spec: "—", fid: 0.6471, r1: 0.498, r3: 0.7677, mm: 3.222, div: 9.179, mmod: 1.905, worst: null, ref: true },
    { label: "Null projection", spec: "knee 0 – 180°", fid: 0.6462, r1: 0.4866, r3: 0.7737, mm: 3.221, div: 9.094, mmod: 1.905, worst: 0 },
    { label: "Knee clamped", spec: "knee 0 – 10°", fid: 1.1492, r1: 0.4814, r3: 0.7517, mm: 3.401, div: 8.564, mmod: 2.009, worst: 8.6e-6 },
    { label: "Knee pinned", spec: "knee at 10°", fid: 1.5444, r1: 0.46, r3: 0.7395, mm: 3.499, div: 8.485, mmod: 2.065, worst: 8.7e-5 },
  ],
};

// ── Constraint → text factorial. Three constraint specs on the same joint of
//    the same base prompt ("a person is walking forward"), four text conditions,
//    with and without the projection, three seeds each. The design is balanced:
//    joint, pin and prompt are held fixed inside each row, so the text condition
//    is the only thing that varies.
//
//    pairs = [seed, jerk free, jerk projected, max realised bend °].
export const FACTORIAL = {
  design: "3 constraint specs × 4 text conditions × {free, projected} × 3 seeds = 72 renders",
  sampler: "mid · ω 6.5 · 800 ODE steps · 120 frames",
  base: "a person is walking forward",
  joint: "left knee",
  n: 36,
  medianRatio: 1.09,
  iqr: [0.73, 1.24],
  range: [0.03, 11.27],
  smootherPairs: 14,
  // Paired test on the log ratio, which is symmetric under the null. The last
  // number is the one worth quoting: it is the smallest effect this design could
  // have detected, so "no effect" means "none larger than this".
  signP: 0.24,
  wilcoxonP: 0.82,
  sdLogRatio: 1.35,
  detectable: 1.9,
  cells: [
    { kind: "clamp", target: "0 – 10°", text: "none", label: "no added text", clips: null, pairs: [[0, 0.01081, 0.01097, 10.0], [1, 0.00976, 0.01059, 10.0], [2, 0.21306, 0.01436, 10.0]] },
    { kind: "clamp", target: "0 – 10°", text: "archetype", label: "limping", clips: 77, pairs: [[0, 0.01552, 0.01724, 10.0], [1, 0.23968, 0.01831, 10.0], [2, 0.01642, 0.01898, 10.0]] },
    { kind: "clamp", target: "0 – 10°", text: "semantic", label: "limping, dragging their left leg", clips: 12, pairs: [[0, 0.04233, 0.06342, 10.0], [1, 0.02644, 0.02923, 10.0], [2, 0.0268, 0.13789, 10.0]] },
    { kind: "clamp", target: "0 – 10°", text: "literal", label: "with the left knee kept between 0 and 10 degrees", clips: 0, pairs: [[0, 0.01126, 0.01451, 10.0], [1, 0.01211, 0.01169, 10.0], [2, 0.01183, 0.01144, 10.0]] },
    { kind: "pin", target: "10°", text: "none", label: "no added text", clips: null, pairs: [[0, 0.01053, 0.01199, 10.0], [1, 0.01645, 0.01117, 10.0], [2, 0.21306, 0.01476, 10.0]] },
    { kind: "pin", target: "10°", text: "archetype", label: "limping", clips: 77, pairs: [[0, 0.0338, 0.01702, 10.0], [1, 0.01593, 0.01896, 10.0], [2, 0.01642, 0.02001, 10.0]] },
    { kind: "pin", target: "10°", text: "semantic", label: "limping, dragging their left leg", clips: 12, pairs: [[0, 0.0244, 0.03371, 10.0], [1, 0.03861, 0.02902, 10.0], [2, 0.0268, 0.14113, 10.0]] },
    { kind: "pin", target: "10°", text: "literal", label: "with the left knee held at 10 degrees", clips: 0, pairs: [[0, 0.01274, 0.01686, 10.0], [1, 0.34221, 0.01148, 10.0], [2, 0.0424, 0.01402, 10.0]] },
    { kind: "pin", target: "0°", text: "none", label: "no added text", clips: null, pairs: [[0, 0.01081, 0.01113, 0.01], [1, 0.00976, 0.01286, 0.0], [2, 0.21308, 0.01265, 0.0]] },
    { kind: "pin", target: "0°", text: "archetype", label: "limping", clips: 77, pairs: [[0, 0.01552, 0.17492, 0.01], [1, 0.23968, 0.01988, 0.01], [2, 0.01642, 0.01842, 0.01]] },
    { kind: "pin", target: "0°", text: "semantic", label: "limping, dragging their left leg", clips: 12, pairs: [[0, 0.04233, 0.04618, 0.01], [1, 0.02644, 0.0282, 0.0], [2, 0.0268, 0.2659, 0.01]] },
    { kind: "pin", target: "0°", text: "literal", label: "with the left knee held at 0 degrees", clips: 0, pairs: [[0, 0.01089, 0.00995, 0.01], [1, 0.01159, 0.01296, 0.0], [2, 0.01217, 0.01126, 0.0]] },
  ],
};

// Spread of the UNCONSTRAINED arm alone, across the three seeds of each cell.
// This is the yardstick any claimed constraint cost has to beat.
export const FREE_SPREAD = { worst: 26.9, median: 8.8, unit: "× between the roughest and smoothest of three seeds" };


// ── Caption-corpus attestation. The index covers the 14,616 unmirrored
//    HumanML3D CLIPS (each carries several captions); a term's count is the
//    number of clips with that lemma in any of its captions.
//
//    `all` is the conjunction: clips containing EVERY content word. It falls off
//    a cliff with phrase length — "holding a box against their chest" scores 0
//    although each of its words is attested — so the conjunction alone is not a
//    measure of whether the model knows a phrase. The weakest single term is the
//    more honest summary, and both are shown.
export const ATTESTATION = {
  total: 14616,
  phrases: [
    {
      phrase: "limping",
      all: 77,
      terms: [["limping", 77]],
      weakest: ["limping", 77],
    },
    {
      phrase: "limping, dragging their left leg",
      all: 12,
      terms: [["limping", 77], ["dragging", 63], ["left", 5540], ["leg", 1713]],
      weakest: ["dragging", 63],
    },
    {
      phrase: "with the left knee held at 10 degrees",
      all: 0,
      terms: [["left", 5540], ["knee", 818], ["held", 1464], ["degrees", 245]],
      weakest: ["degrees", 245],
    },
    {
      phrase: "holding a box against their chest",
      all: 0,
      terms: [["holding", 1466], ["box", 154], ["against", 48], ["chest", 488]],
      weakest: ["against", 48],
    },
  ],
};

// Summary of the nine paired clips in lib/clips.js, which hold the per-clip
// measurements themselves. `normMedian` is the same statistic after dividing
// each clip's jerk by its mean per-frame joint displacement, which removes the
// trivial "faster motion has larger third differences" confound. The ratio is
// unchanged, so the gap is not a speed artefact.
export const PAIRED_JERK_SUMMARY = { median: 4.3, spread: "2.1× – 91×", normMedian: 4.3 };


// ── Generation forensics: generated against real, on quantities that do not go
//    through the Guo evaluator. Measured at the operating point we report at
//    (mid @ 300 k, ω 6.5, 800 ODE steps) over 256 test captions, with the real
//    clip for the same caption as the reference, by
//    `rmg.scripts.forensics` (run: forensics-mid300k-w65-ode800).
//
//    An earlier ad-hoc pass reported these on a superseded sampler path and did
//    not state its settings; its figures do not reproduce here and are not used.
export const FORENSICS = [
  { q: "quaternion norm", gen: "1.0000", real: "1.0000", note: "unit at every step, never renormalised" },
  { q: "angular velocity", gen: "0.0644", real: "0.0539", unit: "rad / frame", note: "generations turn joints about 20 % faster than captures" },
  { q: "translation velocity", gen: "0.0200", real: "0.0180", unit: "m / frame", note: "" },
  { q: "root displacement", gen: "0.936", real: "0.996", unit: "m", note: "mean over clips, start to end" },
  { q: "largest root displacement", gen: "11.20", real: "8.92", unit: "m", note: "single furthest-travelling clip of the 256" },
];

// ── Jerk against the two sampling levers, from the same runs. Each row is 256
//    test captions; `q1`/`q3` are the quartiles ACROSS CLIPS, which is the
//    spread worth drawing here — it is not a seed replicate. `real` is the
//    median over the same captions' captures, so the comparison is controlled.
//
//    The reading: refining the integration 16-fold moves jerk by about a tenth
//    while it halves FID, so the smoothness gap is not integration error.
export const JERK_SWEEP = {
  n: 256,
  realMedian: 0.0034,
  steps: {
    s: [100, 200, 400, 800, 1600],
    med: [0.0275, 0.0267, 0.0239, 0.0246, 0.0236],
    q1: [0.0185, 0.0171, 0.0158, 0.016, 0.0161],
    q3: [0.0605, 0.0498, 0.0481, 0.045, 0.0429],
  },
  omega: {
    w: [2.5, 4.5, 6.5, 8.5],
    med: [0.0342, 0.0271, 0.0267, 0.0287],
    q1: [0.0172, 0.0155, 0.0171, 0.0196],
    q3: [0.0899, 0.0548, 0.0498, 0.0543],
    steps: 200,
  },
};

// ── How the clips shown on this site were produced. Only what the render jobs
//    themselves record: prompt, seed and sampler settings. Whether a draw was
//    ever rejected and re-rolled is NOT recorded, so this states the protocol
//    rather than claiming the selection was blind.
//    TODO(team): if any clip here was picked from several draws, say so here.
export const SELECTION = {
  sampler: "All clips shown are RMG-mid at checkpoint 300 k, ω 6.5, 800 ODE steps, EMA weights.",
  paired:
    "Nine held-out HumanML3D test captions covering locomotion, turning, manipulation, striking and dance. The reference is the capture for that caption and the sample is the seed-0 draw from the same caption.",
  prompted: "Ten prompts written for this report, each the seed-0 draw. Corpus support ranges from 45 to 1,045 clips.",
  constraint:
    "Each pair is one prompt sampled with and without the projection from the same seed, so the two clips of a pair differ only by the constraint. Seeds are 5, 5, 0 and 0 in the order shown, and the clips are 100 to 120 frames. These four are examples. Their jerk ratios run from 0.08× to 7.6×, which is the range a single draw can take, and the second pair starts from an unusually rough unconstrained sample. The measurement is the replicated factorial below.",
};


// ── What this work does NOT show. Stated on the site rather than left for the
//    reader to infer from what is missing.
export const LIMITS = [
  {
    head: "No flat-representation baseline",
    body: "We did not train a matched model on the 263-D feature vector, so the contribution of the manifold representation is not measured here. The only comparison available to us is against a published model four times larger.",
  },
  {
    head: "Effect of the constraint on smoothness",
    body: "Over 36 seed-paired renders the median jerk ratio is 1.09×, with a sign test at p = 0.24 and a Wilcoxon signed-rank test at p = 0.82. The seed-to-seed spread is wide enough that this design resolves only effects of about 1.9× or larger, so the result places an upper bound on the cost. More seeds would narrow it.",
  },
  {
    head: "No constraint-matched reference",
    body: "Constrained generation is scored on the full split, but FID compares against unconstrained real motion. Since holding a joint moves the samples off the data distribution deliberately, the rise from 0.65 to 1.54 is a measure of that displacement. It says nothing about sample quality. Retrieval and multimodal distance are the axes we read here, and reading FID would need a constraint-matched reference set.",
  },
  {
    head: "Single-pass evaluation",
    body: "Each FID here is one pass over the full split, while the published figure is a mean over 20 replications. Our replication spread is ±0.024 over four runs at ω 6.5 and 200 ODE steps, so differences below about 0.05 are not resolved.",
  },
  {
    head: "Few physical measures",
    body: "Smoothness, joint angular velocity and root travel are the physical properties we measure. Foot skating, ground penetration and contact consistency are not measured, and a valid quaternion does not imply any of them.",
  },
  {
    head: "Multimodality",
    body: "Multimodality is 1.81 for the 112 M model at ω 6.5 and 800 ODE steps, and 2.03 for the 25 M model at ω 5.5 and 200 steps, so the two are not measured at matched settings. Across the 112 M guidance sweep at 200 steps it varies between 1.79 and 2.71 with no clear trend, and the published work reports no value for its model, so we do not read anything into the difference.",
  },
];
