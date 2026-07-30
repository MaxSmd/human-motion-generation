// Constraint → text. The formalization layer behind the Lab's ablation study.
//
// A pin/hinge is a geometric fact ("L_Knee bend ∈ [0°,10°]"). The sampler can
// enforce it, but the TEXT PRIOR never hears about it — so for a prompt like "a
// person is walking forward" the model keeps proposing a normal gait, the
// projection keeps overwriting it, and the clip shakes. Measured, that fight is
// worth ~7× in jerk (0.019 coherent vs 0.136 conflicting, same pin, same seed).
//
// The hypothesis this module exists to test: if we SAY the constraint in the
// prompt — in language the encoder actually knows — the prior proposes the
// constrained motion itself and the projection has nothing left to fight.
//
// Three renderings per constraint — the text arms of the study:
//
//   literal   — mechanical restatement: "with the left knee kept between 0 and
//               10 degrees". Faithful, but degree-talk is not caption language.
//   semantic  — per-joint lexicon, one clause each: "limping, dragging their left
//               leg". Idiomatic, but composed mechanically, so it degrades on
//               multi-joint constraints ("crouching on their left leg AND
//               crouching on their right leg").
//   archetype — the whole constraint set matched to a named motion, phrased as a
//               caption would ("crouching"), with the variant chosen by corpus
//               support rather than taste. See ARCHETYPES below.
//
// The lexicon PROPOSES; the corpus DISPOSES — every phrase gets scored against
// the real HumanML3D captions (/corpus/phrase) before you spend GPU time, because
// agreeing text only helps if the prior has support there. That caveat is not
// theoretical: "holding a box against their chest" is coherent with an elbow pin
// and still shook (0.154), and it has ~0 corpus support.
//
// Not theoretical in the other direction either: the corpus prefers "squatting
// down" (551 clips) to "crouching down" (328), and "with their arms straight"
// (369) to "with both arms stretched out straight" (18). Hand-picked phrasings
// were wrong by up to 20×, which is why the archetype map proposes variants
// instead of committing.

// ── joint → family + side ───────────────────────────────────────────────────

const FAMILIES = [
  [/knee/i, "knee"],
  [/elbow/i, "elbow"],
  [/shoulder/i, "shoulder"],
  [/hip/i, "hip"],
  [/ankle/i, "ankle"],
  [/foot/i, "foot"],
  [/wrist/i, "wrist"],
  [/collar/i, "collar"],
  [/neck/i, "neck"],
  [/head/i, "head"],
  [/spine/i, "spine"],
  [/pelvis/i, "pelvis"],
];

export function jointFamily(name = "") {
  for (const [re, fam] of FAMILIES) if (re.test(name)) return fam;
  return "other";
}

export function jointSide(name = "") {
  if (/^L[_-]/i.test(name)) return "left";
  if (/^R[_-]/i.test(name)) return "right";
  return null;
}

// Plain-English joint noun, e.g. "L_Knee" → "left knee".
export function jointPhrase(name = "") {
  const side = jointSide(name);
  const fam = jointFamily(name);
  const noun = fam === "other" ? name.replace(/^[LR][_-]/, "").toLowerCase() : fam;
  return side ? `${side} ${noun}` : noun;
}

// ── bend buckets ────────────────────────────────────────────────────────────
// Bend = angle between a joint's incoming and outgoing bone. 0° = straight.

export const BUCKETS = [
  { id: "straight", max: 15, label: "straight" },
  { id: "slight", max: 45, label: "slightly bent" },
  { id: "bent", max: 100, label: "bent" },
  { id: "deep", max: 150, label: "deeply bent" },
  { id: "folded", max: 181, label: "folded" },
];

export function bucketOf(deg) {
  return BUCKETS.find((b) => deg < b.max) || BUCKETS[BUCKETS.length - 1];
}

// ── the semantic lexicon ────────────────────────────────────────────────────
//
// family → bucket → { any, locomotion }. `locomotion` is used when the base
// prompt describes gait, where the same geometry reads very differently: a knee
// locked straight is "a straight leg" when standing but "a limp" when walking —
// and only the latter is language HumanML3D actually contains.
//
// `reliable: false` marks families whose bend angle has no clean anatomical
// reading (the bone chain through a hip or collar isn't a hinge), so the phrase
// is a guess. Those get flagged in the UI rather than quietly trusted.

const LEXICON = {
  knee: {
    reliable: true,
    straight: {
      locomotion: "limping, dragging their {side} leg",
      any: "keeping their {side} leg straight",
    },
    slight: { any: "with their {side} knee slightly bent" },
    bent: {
      locomotion: "lifting their {side} knee high with each step",
      any: "with their {side} knee bent",
    },
    deep: { any: "crouching down on their {side} leg" },
    folded: { any: "kneeling on their {side} knee" },
  },
  elbow: {
    reliable: true,
    straight: { any: "keeping their {side} arm straight" },
    slight: { any: "with their {side} arm almost straight" },
    bent: { any: "with their {side} arm bent at the elbow" },
    deep: { any: "with their {side} hand raised up near their shoulder" },
    folded: { any: "with their {side} arm folded in against their body" },
  },
  shoulder: {
    reliable: false,
    straight: { any: "with their {side} arm held out from their body" },
    slight: { any: "with their {side} arm raised slightly" },
    bent: { any: "with their {side} arm raised" },
    deep: { any: "with their {side} arm raised high" },
    folded: { any: "with their {side} arm pulled in tight" },
  },
  hip: {
    reliable: false,
    straight: { any: "keeping their {side} leg in line with their body" },
    slight: { any: "with their {side} leg slightly forward" },
    bent: { any: "raising their {side} leg" },
    deep: { any: "lifting their {side} knee up toward their chest" },
    folded: { any: "with their {side} leg tucked up" },
  },
  ankle: {
    reliable: false,
    straight: { any: "with their {side} foot pointed" },
    slight: { any: "with their {side} foot slightly flexed" },
    bent: { any: "with their {side} foot flat" },
    deep: { any: "with their {side} toes pulled up" },
    folded: { any: "with their {side} foot turned up" },
  },
  neck: {
    reliable: false,
    straight: { any: "holding their head up" },
    slight: { any: "with their head tilted slightly" },
    bent: { any: "with their head tilted" },
    deep: { any: "looking down" },
    folded: { any: "with their chin tucked to their chest" },
  },
  spine: {
    reliable: false,
    straight: { any: "keeping their back straight" },
    slight: { any: "leaning slightly" },
    bent: { any: "bending at the waist" },
    deep: { any: "bent far over" },
    folded: { any: "doubled over" },
  },
};

const LOCOMOTION_RE = /\b(walk|walks|walking|run|runs|running|jog|jogs|jogging|stride|strides|march|marches|marching|step|steps|stepping|pace|paces|pacing|limp|limps|limping)\b/i;

export function isLocomotion(prompt = "") {
  return LOCOMOTION_RE.test(prompt);
}

// ── constraint normalization ────────────────────────────────────────────────
//
// A pin and a hinge are the same thing to the verbalizer: a target bend plus a
// tolerance. Collapsing them here keeps the renderers from branching twice.

export function normalizeSpec(spec, numFrames = 100) {
  const isPin = spec.bend_deg != null;
  const lo = isPin ? Number(spec.bend_deg) : Number(spec.bend_min);
  const hi = isPin ? Number(spec.bend_deg) : Number(spec.bend_max);
  const start = Number(spec.frame_start) || 0;
  const end =
    spec.frame_end === "" || spec.frame_end == null ? numFrames : Number(spec.frame_end);
  return {
    kind: isPin ? "pin" : "hinge",
    joint: spec.joint,
    lo, hi,
    mid: (lo + hi) / 2,
    width: hi - lo,
    start, end,
    partial: start > 0 || end < numFrames,
  };
}

// ── renderers ───────────────────────────────────────────────────────────────

// NO TEMPORAL CLAUSE. An earlier version appended the constraint's active window
// ("for the first 2.5 seconds") to every phrase, and it measurably hurt: caption
// language almost never timestamps a modifier, so the clause is off-distribution
// on its own AND it dilutes the part that has to land. The window still applies —
// the projection enforces it — we just don't say it.

// Mechanical restatement — faithful to the geometry, alien to the caption
// distribution. This is the arm that tests whether faithfulness is enough.
export function literalPhrase(spec, { numFrames = 100 } = {}) {
  const n = normalizeSpec(spec, numFrames);
  const j = jointPhrase(n.joint);
  return n.kind === "pin"
    ? `with the ${j} held at ${round(n.lo)} degrees`
    : `with the ${j} kept between ${round(n.lo)} and ${round(n.hi)} degrees`;
}

// Idiomatic restatement — lossy about the exact angle, but reaches for language
// the text encoder has actually seen. Returns null when the constraint is too
// loose to be worth saying (a 120°-wide hinge barely restricts anything, and
// inventing a phrase for it would just add noise to the prompt).
export function semanticPhrase(spec, { numFrames = 100, basePrompt = "" } = {}) {
  const n = normalizeSpec(spec, numFrames);
  const fam = jointFamily(n.joint);
  const entry = LEXICON[fam];
  if (!entry) return null;
  if (n.kind === "hinge" && n.width > 90) return null; // barely a constraint

  const bucket = bucketOf(n.mid);
  const slot = entry[bucket.id];
  if (!slot) return null;

  const loco = isLocomotion(basePrompt);
  const template = (loco && slot.locomotion) || slot.any;
  if (!template) return null;

  const side = jointSide(n.joint) || "";
  const text = template.replace(/\{side\}\s?/g, side ? `${side} ` : "");
  return {
    text: text.replace(/\s+/g, " ").trim(),
    family: fam,
    bucket: bucket.id,
    usedLocomotion: !!(loco && slot.locomotion),
    reliable: entry.reliable !== false,
  };
}

function round(v) {
  return Math.round(Number(v));
}

// ── the archetype map ───────────────────────────────────────────────────────
//
// The per-joint lexicon above is COMPOSITIONAL: one clause per constraint, glued
// together. That's mechanical, and it degrades exactly where it matters — two
// deeply-bent knees become "crouching on their left leg and crouching on their
// right leg" when a person would just say "crouching". The prior knows the whole
// motion, not the conjunction of its joint angles.
//
// So this map matches the constraint SET (plus locomotion context) against a
// small library of recognizable motions, and emits the idiom for the whole thing.
// It's the "better mapping" arm of the study — run head-to-head against the
// compositional one, since which is actually better is an empirical question.
//
// Each archetype offers several VARIANTS rather than one phrase. We don't pick;
// the corpus does (see /corpus/rank) — the variant with the most caption support
// wins, because support is what predicts whether the prior moves at all. That
// turns "which phrasing is best?" from my taste into a measurement.

const groupSpecs = (specs, numFrames) => {
  const g = {};
  for (const raw of specs) {
    const n = normalizeSpec(raw, numFrames);
    // a hinge wider than 90° barely restricts anything — naming it would put a
    // motion in the prompt that the constraint isn't actually asking for
    if (n.kind === "hinge" && n.width > 90) continue;
    const fam = jointFamily(n.joint);
    (g[fam] = g[fam] || []).push({ ...n, fam, side: jointSide(n.joint), bucket: bucketOf(n.mid).id });
  }
  return g;
};

const has = (arr, ...buckets) => (arr || []).filter((x) => buckets.includes(x.bucket));
const sideWord = (x) => (x.side ? `${x.side} ` : "");

export const ARCHETYPES = [
  {
    id: "limp",
    label: "limp",
    // One knee locked straight while walking. The canonical conflict case: the
    // prior wants a normal gait, the pin forbids it, and the sampler tears.
    match: (g, ctx) => {
      const k = has(g.knee, "straight");
      return ctx.locomotion && k.length === 1 ? k : null;
    },
    variants: ([k]) => [
      `limping`,
      `walking with a limp`,
      `limping on their ${k.side} leg`,
      `limping, dragging their ${k.side} leg`,
      `dragging their ${k.side} leg`,
    ],
  },
  {
    id: "both_legs_stiff",
    label: "stiff legs",
    match: (g) => {
      const k = has(g.knee, "straight");
      return k.length >= 2 ? k : null;
    },
    variants: () => [`keeping both legs straight`, `with stiff legs`, `without bending their knees`],
  },
  {
    id: "stiff_leg",
    label: "straight leg",
    match: (g) => {
      const k = has(g.knee, "straight");
      return k.length === 1 ? k : null;
    },
    variants: ([k]) => [
      `keeping their ${k.side} leg straight`,
      `with their ${k.side} leg held straight`,
      `without bending their ${k.side} knee`,
    ],
  },
  {
    id: "crouch",
    label: "crouch",
    match: (g) => {
      const k = has(g.knee, "deep", "folded");
      return k.length >= 2 ? k : null;
    },
    variants: () => [`crouching down`, `squatting down`, `crouching low`, `in a deep squat`],
  },
  {
    id: "kneel",
    label: "kneel",
    match: (g) => {
      const k = has(g.knee, "folded");
      return k.length === 1 ? k : null;
    },
    variants: ([k]) => [`kneeling down`, `kneeling on their ${k.side} knee`, `dropping to one knee`],
  },
  {
    id: "high_knee",
    label: "high knees",
    match: (g, ctx) => {
      const k = has(g.knee, "bent", "deep");
      return ctx.locomotion && k.length >= 1 ? k : null;
    },
    variants: (k) =>
      k.length >= 2
        ? [`marching with high knees`, `lifting their knees high`, `marching in place`]
        : [`lifting their ${k[0].side} knee high`, `marching, raising their ${k[0].side} knee`],
  },
  {
    id: "knee_bent",
    label: "bent knee",
    match: (g) => {
      const k = has(g.knee, "bent", "deep");
      return k.length === 1 ? k : null;
    },
    variants: ([k]) => [
      `with their ${k.side} knee bent`,
      `bending their ${k.side} knee`,
      `with their ${k.side} leg bent`,
    ],
  },
  {
    id: "both_arms_straight",
    label: "arms stretched",
    match: (g) => {
      const e = has(g.elbow, "straight");
      return e.length >= 2 ? e : null;
    },
    variants: () => [
      `with both arms stretched out straight`,
      `with both arms extended`,
      `holding both arms out straight`,
      `with their arms straight`,
    ],
  },
  {
    id: "arm_straight",
    label: "arm stretched",
    match: (g) => {
      const e = has(g.elbow, "straight");
      return e.length === 1 ? e : null;
    },
    variants: ([e]) => [
      `with their ${e.side} arm stretched out straight`,
      `with their ${e.side} arm extended`,
      `holding their ${e.side} arm out straight`,
      `keeping their ${e.side} arm straight`,
    ],
  },
  {
    id: "both_arms_folded",
    label: "arms folded",
    match: (g) => {
      const e = has(g.elbow, "folded", "deep");
      return e.length >= 2 ? e : null;
    },
    variants: () => [`with both arms folded`, `with their arms crossed`, `with both hands up by their shoulders`],
  },
  {
    id: "arm_folded",
    label: "arm pinched",
    match: (g) => {
      const e = has(g.elbow, "folded", "deep");
      return e.length === 1 ? e : null;
    },
    variants: ([e]) => [
      `with their ${e.side} arm folded up`,
      `with their ${e.side} hand up by their shoulder`,
      `bending their ${e.side} elbow all the way`,
    ],
  },
  {
    id: "arm_bent",
    label: "bent arm",
    match: (g) => {
      const e = has(g.elbow, "bent");
      return e.length >= 1 ? e : null;
    },
    variants: (e) =>
      e.length >= 2
        ? [`with both arms bent at the elbows`, `with their elbows bent`]
        : [
            `with their ${e[0].side} arm bent at the elbow`,
            `bending their ${e[0].side} elbow`,
            `with their ${e[0].side} elbow bent`,
          ],
  },
  {
    id: "bent_over",
    label: "bent over",
    match: (g) => {
      const s = has(g.spine, "bent", "deep", "folded");
      return s.length >= 1 ? s : null;
    },
    variants: (s) =>
      s[0].bucket === "folded"
        ? [`doubled over`, `bent all the way over`, `bending over at the waist`]
        : [`bending over`, `bent over at the waist`, `leaning forward`, `hunched over`],
  },
  {
    id: "head_down",
    label: "head down",
    match: (g) => {
      const n = has(g.neck, "deep", "folded");
      return n.length >= 1 ? n : null;
    },
    variants: () => [`looking down`, `looking down at the ground`, `with their head lowered`],
  },
  {
    id: "arms_raised",
    label: "arms raised",
    match: (g) => {
      const s = has(g.shoulder, "bent", "deep");
      return s.length >= 1 ? s : null;
    },
    variants: (s) =>
      s.length >= 2
        ? [`with both arms raised`, `raising their arms`, `with their arms up`]
        : [`with their ${s[0].side} arm raised`, `raising their ${s[0].side} arm`],
  },
];

// Match the constraint set against the archetype library. Greedy and ordered:
// the most specific archetype wins and CONSUMES its specs, so a limp doesn't also
// report "bent knee". Anything left over falls back to the compositional lexicon
// rather than being silently dropped — a constraint the prompt never mentions is
// exactly the failure mode this whole tab exists to measure.
export function matchArchetypes(specs, { numFrames = 100, basePrompt = "" } = {}) {
  const ctx = { locomotion: isLocomotion(basePrompt) };
  const groups = groupSpecs(specs, numFrames);
  const used = new Set();
  const hits = [];

  for (const arch of ARCHETYPES) {
    // hide already-consumed specs from later archetypes
    const avail = {};
    for (const [fam, arr] of Object.entries(groups)) {
      const left = arr.filter((x) => !used.has(x));
      if (left.length) avail[fam] = left;
    }
    const consumed = arch.match(avail, ctx);
    if (!consumed || consumed.length === 0) continue;
    consumed.forEach((x) => used.add(x));
    hits.push({ id: arch.id, label: arch.label, variants: arch.variants(consumed), consumed });
  }

  const leftover = Object.values(groups).flat().filter((x) => !used.has(x));
  return { hits, leftover };
}

// ── composition ─────────────────────────────────────────────────────────────

// Attach modifier clauses to a base prompt: "a person walks forward" +
// ["limping, dragging their left leg"] → "a person walks forward, limping,
// dragging their left leg". Trailing punctuation on the base is absorbed.
export function composePrompt(base, phrases) {
  const clean = String(base || "").trim().replace(/[.\s]+$/, "");
  const parts = (phrases || []).map((p) => String(p).trim()).filter(Boolean);
  if (parts.length === 0) return clean;
  const joined =
    parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
  return `${clean}, ${joined}`;
}

// ── the study: an N×2 factorial ─────────────────────────────────────────────
//
// text ∈ TEXT_MODES  ×  projection ∈ {off, on}. The two rows
// answer different halves of the question, and you need both to claim a
// mechanism rather than a correlation:
//
//   projection OFF → does the text move the PRIOR toward the constraint at all?
//                    (measured as bend agreement with a constraint we did not
//                    enforce — nothing is projecting, so this is the model's own
//                    opinion.) This is where "limping" should beat degree-talk.
//   projection ON  → what does enforcement then COST? (jerk, as a multiple of
//                    the free cell.) If the prior already wants the motion, the
//                    projection has nothing to fight and the cost collapses.
//
// none/off is the reference cell: what this model does with nothing opposing it.

export const TEXT_MODES = [
  {
    id: "none",
    label: "no text",
    describe: "Prompt untouched — the constraint is never spoken. The naive path.",
  },
  {
    id: "literal",
    label: "literal",
    describe: "Mechanical restatement (degree-talk). Faithful to the geometry, alien to caption language.",
  },
  {
    id: "semantic",
    label: "semantic",
    describe: "Per-joint lexicon, one clause per constraint. Idiomatic, but composed mechanically.",
  },
  {
    id: "archetype",
    label: "archetype",
    describe:
      "Whole constraint set matched to a named motion (limp · crouch · bent over), phrased the way a caption would. Variant chosen by corpus support, not taste.",
  },
];

export const cellId = (text, projected) => `${text}-${projected ? "on" : "off"}`;

export const CELLS = TEXT_MODES.flatMap((t) =>
  [false, true].map((projected) => ({
    id: cellId(t.id, projected),
    text: t.id,
    projected,
    reference: t.id === "none" && !projected,
    label: `${t.label} · projection ${projected ? "on" : "off"}`,
  }))
);

// Build each text mode's prompt for a given base + constraint set.
//
// `overrides` lets the user hand-edit a prompt before running — the lexicon is a
// starting point, not an authority, and a study you can't hand-tune isn't much use.
// `picks.archetype` selects among an archetype's variants; the caller passes the
// corpus-ranked winner once /corpus/rank resolves, and until then variant 0 stands
// in so the UI has something to show.
export function buildTextModes(basePrompt, specs, { numFrames = 100, overrides = {}, picks = {} } = {}) {
  const literal = specs.map((s) => literalPhrase(s, { numFrames })).filter(Boolean);
  const semantic = specs.map((s) => semanticPhrase(s, { numFrames, basePrompt })).filter(Boolean);
  const semanticText = semantic.map((s) => s.text);

  const { hits, leftover } = matchArchetypes(specs, { numFrames, basePrompt });
  // Uncovered constraints fall back to the compositional clause so the prompt
  // still mentions every constraint being enforced.
  const leftoverText = leftover
    .map((x) => semanticPhrase(x, { numFrames, basePrompt }))
    .filter(Boolean)
    .map((s) => s.text);
  const archText = [...hits.map((h) => picks[h.id] ?? h.variants[0]), ...leftoverText];

  const added = {
    none: "",
    literal: literal.join(" and "),
    semantic: semanticText.join(" and "),
    archetype: archText.join(" and "),
  };
  const auto = {
    none: String(basePrompt || "").trim(),
    literal: composePrompt(basePrompt, literal),
    semantic: composePrompt(basePrompt, semanticText),
    archetype: composePrompt(basePrompt, archText),
  };

  return TEXT_MODES.map((t) => ({
    ...t,
    auto: auto[t.id],
    prompt: overrides[t.id] ?? auto[t.id],
    edited: overrides[t.id] != null && overrides[t.id] !== auto[t.id],
    // the bolted-on clause alone — what we ask the corpus about. The base
    // prompt's own coverage isn't under test; the added language is.
    added: added[t.id],
    meta: t.id === "semantic" ? semantic : [],
    // which archetypes fired, + what else they could have said (for the UI's
    // variant picker and the ranking request)
    archetypes: t.id === "archetype" ? hits : [],
    uncovered: t.id === "archetype" ? leftover : [],
    // nothing rendered → no lexicon/archetype entry for this family+bucket, so
    // the arm would silently duplicate `none` and the cell would be a wasted run
    empty: t.id !== "none" && added[t.id] === "",
  }));
}
