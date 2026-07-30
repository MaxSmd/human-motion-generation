// Numeric helpers with no client dependencies, so server components can call
// them directly (a plain function exported from a "use client" module becomes a
// non-callable client reference).

/**
 * Fritsch-Carlson monotone cubic through (xs, ys), sampled at `n` points. Used
 * to draw a smooth guide through a set of points without overshooting into
 * non-monotone wiggles between them.
 */
export function monotoneTrend(xs, ys, n = 48) {
  const k = xs.length;
  const h = [], d = [];
  for (let i = 0; i < k - 1; i++) {
    h[i] = xs[i + 1] - xs[i];
    d[i] = (ys[i + 1] - ys[i]) / h[i];
  }
  const m = new Array(k);
  m[0] = d[0];
  m[k - 1] = d[k - 2];
  for (let i = 1; i < k - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < k - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) {
      const t = 3 / Math.sqrt(s);
      m[i] = t * a * d[i];
      m[i + 1] = t * b * d[i];
    }
  }
  const f = (x) => {
    let i = k - 2;
    for (let j = 0; j < k - 1; j++)
      if (x <= xs[j + 1]) {
        i = j;
        break;
      }
    const t = (x - xs[i]) / h[i];
    const h00 = 2 * t ** 3 - 3 * t ** 2 + 1;
    const h10 = t ** 3 - 2 * t ** 2 + t;
    const h01 = -2 * t ** 3 + 3 * t ** 2;
    const h11 = t ** 3 - t ** 2;
    return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1];
  };
  const out = [];
  for (let j = 0; j <= n; j++) {
    const x = xs[0] + ((xs[k - 1] - xs[0]) * j) / n;
    out.push([x, f(x)]);
  }
  return out;
}
