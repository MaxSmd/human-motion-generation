// The motion library shipped with the site.
//
// Every entry is a (T, 22, 3) float32 joint array under /data, a few tens of
// kilobytes each, animated on the visitor's GPU. `jerk` is mean ‖Δ³x‖ over
// joints and frames, measured here from the same file the browser loads.
// `clips` on generated entries is the number of HumanML3D captions containing
// every content word of the prompt, from the 14 616-caption corpus.

export const SAMPLER = {
  model: "RMG-mid · 111.7 M · ckpt 300 k",
  omega: 6.5,
  steps: 800,
  fps: 20,
};

// Generated against a held-out reference: the same test caption drives the
// model, and the ground-truth capture for that caption sits beside it.
export const PAIRS = [
  { id: "walkturn", label: "walk and turn", caption: "a person walks forward then turns around and walks back.", gt: 0.0062, gen: 0.013 },
  { id: "fwdback", label: "walk forward and back", caption: "walking forward and then back.", gt: 0.0057, gen: 0.0152 },
  { id: "spin", label: "spin around", caption: "a person walks forward, spins around, and walks back", gt: 0.0035, gen: 0.0197 },
  { id: "walkleft", label: "walk to the left", caption: "a person walks slowly forward then toward the left hand side and stands facing that direction", gt: 0.0031, gen: 0.0239 },
  { id: "pickup", label: "pick up and place", caption: "a person walks forward, picks something up, sets it down on something a little higher, turns around", gt: 0.0086, gen: 0.0372 },
  { id: "punch", label: "punch combination", caption: "a person punches left, right, and then left again.", gt: 0.0142, gen: 0.0513 },
  { id: "waltz", label: "waltz", caption: "a person is dancing the waltz, going in a counter-clockwise direction with the left arm out", gt: 0.0036, gen: 0.0825 },
  { id: "boxjab", label: "boxing jab", caption: "a person is boxing, jabbing mostly with his right hand.", gt: 0.021, gen: 0.0907 },
  { id: "dance", label: "freestyle dance", caption: "a person is doing a dance.", gt: 0.0019, gen: 0.1768 },
];

// Free generation from a written prompt, no reference motion. `clips` is the
// prompt's corpus attestation.
export const PROMPTED = [
  { id: "run", prompt: "a person runs forward", jerk: 0.0287, clips: 835, travel: 4.22 },
  { id: "jump", prompt: "a person jumps forward", jerk: 0.0252, clips: 1045, travel: 1.43 },
  { id: "kick", prompt: "a person kicks with their right leg", jerk: 0.0239, clips: 483, travel: 0.06 },
  { id: "boxjabs", prompt: "a person is boxing, throwing jabs", jerk: 0.0362, clips: 191, travel: 0.02 },
  { id: "jacks", prompt: "a person does jumping jacks", jerk: 0.0379, clips: 182, travel: 0.05 },
  { id: "sit", prompt: "a person sits down on a chair", jerk: 0.0271, clips: 164, travel: 0.05 },
  { id: "crawl", prompt: "a person crawls forward on the ground", jerk: 0.0405, clips: 157, travel: 2.33 },
  { id: "kneel", prompt: "a person kneels down", jerk: 0.021, clips: 155, travel: 0.4 },
  { id: "throw", prompt: "a person throws a ball", jerk: 0.0936, clips: 153, travel: 0.26 },
  { id: "cartwheel", prompt: "a person does a cartwheel", jerk: 0.0676, clips: 45, travel: 3.27 },
];

// Constraint demonstrations, rendered as matched pairs: the same prompt and the
// same seed, once free and once with one joint projected at every ODE step.
// `freeRange` is what the unconstrained sample does at that joint, `held` is
// what the projected one measures.
//
// One seed each, re-rendered 2026-07-27 with the sign-preserving projector.
// The jerk figures describe these particular samples and nothing more: three of
// the four cost nothing to constrain and one costs 7.6x, which is the span a
// single seed can take. The replicated measurement is FACTORIAL in lib/results.js.
export const CONSTRAINT_PAIRS = [
  {
    id: "walkturn",
    label: "walk, neutral prompt",
    sub: "knee → 0 – 10°",
    prompt: "a person walks forward then turns around and walks back",
    joint: "left knee",
    limb: "left_leg",
    pin: "0 – 10°",
    freeFile: "c_free.npy",
    pinFile: "c_conflict.npy",
    freeRange: "13.2 – 62.4°",
    held: "9.02 – 10.00°",
    free: 0.0141,
    pinned: 0.0134,
  },
  {
    id: "limp",
    label: "walk, prompt says limping",
    sub: "knee → 0 – 10°",
    prompt: "a person walks forward then turns around and walks back while limping the left leg",
    joint: "left knee",
    limb: "left_leg",
    pin: "0 – 10°",
    freeFile: "c_limp_free.npy",
    pinFile: "c_coherent.npy",
    freeRange: "9.6 – 52.0°",
    held: "7.83 – 10.00°",
    free: 0.2468,
    pinned: 0.0194,
  },
  {
    id: "run",
    label: "run, knee clamped",
    sub: "knee → 0 – 10°",
    prompt: "a person runs forward",
    joint: "left knee",
    limb: "left_leg",
    pin: "0 – 10°",
    freeFile: "n_run.npy",
    pinFile: "c_run_knee.npy",
    freeRange: "18.1 – 97.2°",
    held: "8.81 – 10.00°",
    free: 0.0287,
    pinned: 0.0218,
  },
  {
    id: "box",
    label: "carry a box, elbow pinned",
    sub: "elbow → 90°",
    prompt: "a person walks forward holding a box against their chest",
    joint: "left elbow",
    limb: "left_arm",
    pin: "90°",
    freeFile: "c_box_free.npy",
    pinFile: "c_elbow90.npy",
    freeRange: "85.2 – 141.2°",
    held: "90.00°",
    free: 0.042,
    pinned: 0.32,
  },
];

export const gtUrl = (id) => `/data/gt_${id}.npy`;
export const genUrl = (id) => `/data/gen_${id}.npy`;
export const promptUrl = (id) => `/data/n_${id}.npy`;
export const constrainedUrl = (file) => `/data/${file}`;
