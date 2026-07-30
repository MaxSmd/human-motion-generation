// Axis numerics shared by every chart. Two jobs:
//
//   niceTicks() — pick tick VALUES that are round (1/2/2.5/5 × 10^k) instead of
//   slicing the data range into n equal parts, which is what produced ticks like
//   0.13333 and 1.0e+3 in the first place.
//
//   fmtNum()    — render a number the way a reader expects: 1000 is "1000", not
//   "1.0e+3"; 300000 is "300k"; 0.0125 keeps the digits that matter. Scientific
//   notation is a last resort for genuinely tiny/huge magnitudes only.

// Round `x` to the nearest "nice" number (1, 2, 2.5, 5, 10 × a power of ten).
function niceStep(x) {
  if (!(x > 0)) return 1;
  const e = Math.floor(Math.log10(x));
  const f = x / 10 ** e; // 1 ≤ f < 10
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nf * 10 ** e;
}

// Tick values on round numbers inside [lo, hi]. Returns { ticks, step } — the
// step is handed back so the formatter knows how many decimals actually carry
// information (a 0.25 step needs 2 decimals; a 50 step needs none).
export function niceTicks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {
    return { ticks: [lo], step: 1 };
  }
  const step = niceStep((hi - lo) / Math.max(1, count));
  const ticks = [];
  // guard against fp drift pushing an edge tick just outside the domain
  const eps = step * 1e-9;
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + eps; v += step) {
    ticks.push(Math.abs(v) < eps ? 0 : v);
  }
  // Degenerate domains (all ticks fell outside) still need something to draw.
  if (ticks.length < 2) return { ticks: [lo, (lo + hi) / 2, hi], step: (hi - lo) / 2 };
  return { ticks, step };
}

// Decimals worth showing for a value spaced `step` apart.
function decimalsFor(step) {
  if (!(step > 0)) return 2;
  const d = Math.ceil(-Math.log10(step) + 1e-9);
  return Math.min(6, Math.max(0, d));
}

// [threshold to switch on, divisor, suffix]. Below 10k the number is written out
// in full — 1000 stays "1000", 1500 stays "1500".
const SUFFIX = [
  [1e9, 1e9, "B"],
  [1e6, 1e6, "M"],
  [1e4, 1e3, "k"],
];

// Human-readable number for an axis tick or an inline stat.
//   fmtNum(1000)            -> "1000"
//   fmtNum(300000)          -> "300k"
//   fmtNum(0.125, 0.025)    -> "0.125"
//   fmtNum(1.5e-6)          -> "1.5e-6"
export function fmtNum(v, step = null) {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  // Genuinely tiny values have no readable fixed form — exponential is the
  // honest rendering there (and only there).
  if (a < 1e-4) return trim(v.toExponential(1).replace(/\.0e/, "e"));
  for (const [at, div, suf] of SUFFIX) {
    if (a >= at) {
      const s = v / div;
      // ≤1 decimal on the scaled value: 1.2M, 300k, 12k
      return `${trim(s.toFixed(Math.abs(s) >= 100 ? 0 : 1))}${suf}`;
    }
  }
  const d = step != null ? decimalsFor(step) : a >= 100 ? 0 : a >= 1 ? 2 : a >= 0.01 ? 3 : 4;
  return trim(v.toFixed(d));
}

// "1.50" -> "1.5", "3.0" -> "3"
function trim(s) {
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}
