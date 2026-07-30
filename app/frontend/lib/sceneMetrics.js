// Client-side scene-violation metrics — a JS port of src/rmg/flow/scene.py's
// signed-distance fields, so the Room-tab player can show, live per frame, how
// well a clip respects the room/obstacle constraints it was sampled under.
// Joints arrive already placed in the room's world frame (backend applies spawn
// placement before dumping the .npy), so we compare directly against the scene.

const VIOL_EPS = 1e-3; // metres; below this we treat a joint as compliant

function rotYInv(px, py, pz, yawRad) {
  const c = Math.cos(-yawRad), s = Math.sin(-yawRad);
  return [c * px + s * pz, py, -s * px + c * pz];
}

// Signed distance to an (optionally yaw-rotated) axis box. <0 inside.
export function sdfBox(p, center, half, yawRad = 0) {
  let qx = p[0] - center[0], qy = p[1] - center[1], qz = p[2] - center[2];
  if (yawRad) [qx, qy, qz] = rotYInv(qx, qy, qz, yawRad);
  const ax = Math.abs(qx) - half[0], ay = Math.abs(qy) - half[1], az = Math.abs(qz) - half[2];
  const outside = Math.hypot(Math.max(ax, 0), Math.max(ay, 0), Math.max(az, 0));
  const inside = Math.min(Math.max(ax, ay, az), 0);
  return outside + inside;
}
export function sdfSphere(p, c, r) {
  return Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - r;
}
export function sdfCylinder(p, cxz, r, cy, halfH) {
  const dxz = Math.hypot(p[0] - cxz[0], p[2] - cxz[1]) - r;
  const dy = Math.abs(p[1] - cy) - halfH;
  const outside = Math.hypot(Math.max(dxz, 0), Math.max(dy, 0));
  const inside = Math.min(Math.max(dxz, dy), 0);
  return outside + inside;
}

function obstacleSdf(p, o) {
  if (o.kind === "sphere") return sdfSphere(p, [o.x, o.y, o.z], o.radius);
  if (o.kind === "cylinder") return sdfCylinder(p, [o.x, o.z], o.radius, o.y, o.height / 2);
  return sdfBox(p, [o.x, o.y, o.z], [o.w / 2, o.h / 2, o.d / 2], ((o.rotation || 0) * Math.PI) / 180);
}

// Per-frame metrics. `frameJoints` is an array of [x,y,z] (length J).
export function frameViolations(frameJoints, scene) {
  const { width: w, depth: d, height: h } = scene.room;
  const objects = scene.objects || [];
  let roomOut = 0;        // max distance any joint is outside the room
  let pen = 0;            // max penetration depth into any obstacle
  let penLabel = null;    // which obstacle is most penetrated
  const violating = new Set();

  for (let j = 0; j < frameJoints.length; j++) {
    const p = frameJoints[j];
    const sr = sdfBox(p, [0, h / 2, 0], [w / 2, h / 2, d / 2]);
    if (sr > VIOL_EPS) { roomOut = Math.max(roomOut, sr); violating.add(j); }
    for (const o of objects) {
      const depth = -obstacleSdf(p, o);
      if (depth > VIOL_EPS) {
        violating.add(j);
        if (depth > pen) { pen = depth; penLabel = o.label || o.kind; }
      }
    }
  }
  return { roomOut, pen, penLabel, violating, nViol: violating.size, worst: Math.max(roomOut, pen) };
}

// Read joint f of the flat npy into [x,y,z] triples.
export function frameJointsOf(joints, frame) {
  const { shape, data } = joints;
  const J = shape[1];
  const out = new Array(J);
  for (let j = 0; j < J; j++) {
    const o = (frame * J + j) * 3;
    out[j] = [data[o], data[o + 1], data[o + 2]];
  }
  return out;
}

// SMPL 22-joint name → index (matches shared.geometry.skeleton.JOINT_NAMES).
const JOINT_INDEX = {
  pelvis: 0, L_Hip: 1, R_Hip: 2, Spine1: 3, L_Knee: 4, R_Knee: 5, Spine2: 6,
  L_Ankle: 7, R_Ankle: 8, Spine3: 9, L_Foot: 10, R_Foot: 11, Neck: 12,
  L_Collar: 13, R_Collar: 14, Head: 15, L_Shoulder: 16, R_Shoulder: 17,
  L_Elbow: 18, R_Elbow: 19, L_Wrist: 20, R_Wrist: 21,
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// JS mirror of flow.scene._contact_target — the point a contact pulls its joint
// toward, given that joint's current position `jp` ([x,y,z], already placed).
function contactTarget(jp, contact, scene) {
  if (contact.target === "floor") return [jp[0], 0, jp[2]];
  if (contact.target === "point") return [contact.x, contact.y, contact.z];
  const o = (scene.objects || []).find((x) => x.id === contact.object_id);
  if (!o) return jp;
  if (contact.target === "obstacle_top") {
    if (o.kind === "box") {
      const yaw = ((o.rotation || 0) * Math.PI) / 180;
      const [lx, , lz] = rotYInv(jp[0] - o.x, jp[1] - o.y, jp[2] - o.z, yaw);
      const cx = clamp(lx, -o.w / 2, o.w / 2), cz = clamp(lz, -o.d / 2, o.d / 2);
      const c = Math.cos(yaw), s = Math.sin(yaw); // rotate (cx,cz) back by +yaw
      return [o.x + c * cx + s * cz, o.y + o.h / 2, o.z - s * cx + c * cz];
    }
    if (o.kind === "cylinder") {
      const dx = jp[0] - o.x, dz = jp[2] - o.z;
      const dist = Math.hypot(dx, dz) || 1e-9;
      const sc = Math.min(dist, o.radius) / dist;
      return [o.x + dx * sc, o.y + o.height / 2, o.z + dz * sc];
    }
    if (o.kind === "sphere") return [o.x, o.y + o.radius, o.z];
  }
  return [o.x, o.y, o.z];
}

// Per-contact realized satisfaction: fraction of in-window frames where the
// joint lands within `tol` of its target, plus the mean / worst distance.
export function contactReport(joints, scene) {
  const contacts = scene.contacts || [];
  if (!contacts.length) return [];
  const T = joints.shape[0];
  return contacts.map((c) => {
    const ji = JOINT_INDEX[c.joint] ?? 0;
    const fs = Math.max(0, Number(c.frame_start) || 0);
    const fe = c.frame_end === "" || c.frame_end == null ? T : Math.min(T, Number(c.frame_end));
    const tol = Number(c.tol) || 0.05;
    let held = 0, sum = 0, worst = 0, n = 0;
    for (let f = fs; f < fe; f++) {
      const jp = frameJointsOf(joints, f)[ji];
      const tgt = contactTarget(jp, c, scene);
      const dist = Math.hypot(jp[0] - tgt[0], jp[1] - tgt[1], jp[2] - tgt[2]);
      n += 1; sum += dist; worst = Math.max(worst, dist);
      if (dist <= tol + VIOL_EPS) held += 1;
    }
    return {
      id: c.id, joint: c.joint, target: c.target,
      heldPct: n ? held / n : 0, meanDist: n ? sum / n : 0, worst,
    };
  });
}

// Whole-clip summary + per-frame worst-violation series (for the timeline).
export function clipMetrics(joints, scene) {
  const T = joints.shape[0];
  const series = new Float32Array(T);
  let framesViolating = 0;
  let peak = 0;
  for (let f = 0; f < T; f++) {
    const m = frameViolations(frameJointsOf(joints, f), scene);
    series[f] = m.worst;
    if (m.worst > VIOL_EPS) framesViolating += 1;
    if (m.worst > peak) peak = m.worst;
  }
  return { series, peak, framesViolating, T, pctViolating: T ? framesViolating / T : 0 };
}
