// Client-side kinematic-quality metrics for a generated clip, computed straight
// from the joint-position .npy (T, J, 3). These are the "did the constraint
// wreck the motion?" readouts that live as *fields* next to the Studio viewport
// (not a whole eval tab): jitter (jerk), foot-skate, and acceleration, plus a
// per-joint jitter vector so the cost of pinning one specific joint is visible.
//
// Everything here is a finite-difference over positions — cheap enough to rerun
// on every resample. FID / R@k stay server-side in shared/eval; this is the fast
// local loop. Units assume fps frames/second (HumanML3D default 20).

// HumanML3D foot joints (ankles + toes) — the ones foot-skate cares about.
const FOOT_JOINTS = [7, 8, 10, 11];
const FLOOR_CONTACT_H = 0.05; // m; a foot below this counts as planted

function jointAt(data, J, f, j) {
  const o = (f * J + j) * 3;
  return [data[o], data[o + 1], data[o + 2]];
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mag = (a) => Math.hypot(a[0], a[1], a[2]);

// Full-clip analysis. Returns scalar summaries + per-frame series (for the
// sparkline) + a per-joint jerk vector (for "this pinned joint got jittery").
export function analyzeMotion(joints, fps = 20) {
  const { shape, data } = joints;
  const [T, J] = shape;
  const dt = 1 / fps;

  const jerkSeries = new Float32Array(T); // mean joint jerk magnitude per frame
  const accelSeries = new Float32Array(T); // mean joint accel magnitude per frame
  const perJointJerk = new Float64Array(J);
  let jerkCount = 0;
  let accelCount = 0;

  for (let f = 0; f < T; f++) {
    // central 2nd difference → acceleration; needs neighbours on both sides
    if (f >= 1 && f <= T - 2) {
      let sum = 0;
      for (let j = 0; j < J; j++) {
        const pPrev = jointAt(data, J, f - 1, j);
        const pCur = jointAt(data, J, f, j);
        const pNext = jointAt(data, J, f + 1, j);
        const ax = (pNext[0] - 2 * pCur[0] + pPrev[0]) / (dt * dt);
        const ay = (pNext[1] - 2 * pCur[1] + pPrev[1]) / (dt * dt);
        const az = (pNext[2] - 2 * pCur[2] + pPrev[2]) / (dt * dt);
        sum += Math.hypot(ax, ay, az);
      }
      accelSeries[f] = sum / J;
      accelCount++;
    }
    // central 3rd difference → jerk; needs two neighbours each side
    if (f >= 2 && f <= T - 2) {
      let sum = 0;
      for (let j = 0; j < J; j++) {
        const p2 = jointAt(data, J, f - 2, j);
        const p1 = jointAt(data, J, f - 1, j);
        const n1 = jointAt(data, J, f + 1, j);
        const n2 = jointAt(data, J, f + 2, j);
        // (-p[-2] + 2p[-1] - 2p[+1] + p[+2]) / (2 dt^3)
        const jx = (-p2[0] + 2 * p1[0] - 2 * n1[0] + n2[0]) / (2 * dt ** 3);
        const jy = (-p2[1] + 2 * p1[1] - 2 * n1[1] + n2[1]) / (2 * dt ** 3);
        const jz = (-p2[2] + 2 * p1[2] - 2 * n1[2] + n2[2]) / (2 * dt ** 3);
        const m = Math.hypot(jx, jy, jz);
        sum += m;
        perJointJerk[j] += m;
      }
      jerkSeries[f] = sum / J;
      jerkCount++;
    }
  }

  const denom = Math.max(1, jerkCount);
  for (let j = 0; j < J; j++) perJointJerk[j] /= denom;

  const meanOf = (series, n) => {
    let s = 0;
    for (let i = 0; i < series.length; i++) s += series[i];
    return n ? s / n : 0;
  };

  return {
    T,
    fps,
    jitter: meanOf(jerkSeries, jerkCount), // m/s³, mean joint jerk
    accel: meanOf(accelSeries, accelCount), // m/s², mean joint accel
    footSkate: footSkateMetric(joints, fps), // cm/s, planted-foot horizontal drift
    jerkSeries,
    perJointJerk: Array.from(perJointJerk),
  };
}

// Mean horizontal speed of a foot while it is planted (height < contact thresh),
// in cm/s. A clean clip plants the foot and it stays put → ~0. Pins and hinge
// clamps often reintroduce skate, so this is a sharp "we broke it" signal.
export function footSkateMetric(joints, fps = 20) {
  const { shape, data } = joints;
  const [T, J] = shape;
  const dt = 1 / fps;
  let total = 0;
  let contacts = 0;
  for (let f = 0; f < T - 1; f++) {
    for (const j of FOOT_JOINTS) {
      if (j >= J) continue;
      const p = jointAt(data, J, f, j);
      const q = jointAt(data, J, f + 1, j);
      const planted = p[1] < FLOOR_CONTACT_H && q[1] < FLOOR_CONTACT_H;
      if (!planted) continue;
      const horiz = Math.hypot(q[0] - p[0], q[2] - p[2]); // ignore vertical
      total += horiz / dt; // m/s
      contacts++;
    }
  }
  return contacts ? (total / contacts) * 100 : 0; // → cm/s
}

// Constraint-satisfaction proxy from positions alone. We can't read the joint
// quaternion back from a position .npy, but a held joint's *outgoing bone* keeps
// a constant direction relative to its parent bone over the pinned window. So we
// report the angular std-dev (deg) of that relative direction inside the window:
// near 0 ⇒ the pin/hinge is holding; large ⇒ the model is fighting it / drifting.
// Returns null for joints with no child bone (end-effectors like wrist/head).
export function jointHoldStability(joints, jointIdx, parents, children, win) {
  const { shape, data } = joints;
  const [T, J] = shape;
  const child = children[jointIdx];
  const parent = parents[jointIdx];
  if (child == null || parent == null || parent < 0) return null;

  const lo = Math.max(0, win?.start ?? 0);
  const hi = Math.min(T, win?.end ?? T);
  if (hi - lo < 2) return null;

  // angle (deg) between parent→joint bone and joint→child bone, per frame
  const angles = [];
  for (let f = lo; f < hi; f++) {
    const pj = jointAt(data, J, f, parent);
    const jj = jointAt(data, J, f, jointIdx);
    const cj = jointAt(data, J, f, child);
    const a = sub(jj, pj);
    const b = sub(cj, jj);
    const na = mag(a);
    const nb = mag(b);
    if (na < 1e-6 || nb < 1e-6) continue;
    const cos = Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)));
    angles.push((Math.acos(cos) * 180) / Math.PI);
  }
  if (angles.length < 2) return null;
  const mean = angles.reduce((s, x) => s + x, 0) / angles.length;
  const variance = angles.reduce((s, x) => s + (x - mean) ** 2, 0) / angles.length;
  return { meanAngle: mean, stdDeg: Math.sqrt(variance), n: angles.length };
}

// First child of each joint, derived from the parent table (for the hold proxy).
export function childrenFromParents(parents) {
  const children = new Array(parents.length).fill(null);
  for (let j = 0; j < parents.length; j++) {
    const p = parents[j];
    if (p != null && p >= 0 && children[p] == null) children[p] = j;
  }
  return children;
}
