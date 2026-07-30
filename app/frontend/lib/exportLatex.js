// Turn the app's on-screen figures and tables into paper-ready source: booktabs
// tabulars for tables and pgfplots/TikZ for the line & bar charts. Everything a
// figure needs is already in the component (the same props that drew the SVG),
// so these functions take that data — or, for tables, the live <table> node —
// and emit standalone LaTeX you can paste straight into a manuscript.

// ---- download / clipboard --------------------------------------------------

function triggerDownload(href, filename) {
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function downloadText(text, filename, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }));
  triggerDownload(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

// ---- shared helpers --------------------------------------------------------

// Escape the handful of characters that are special in LaTeX text mode. Cells
// that are pure math (a `data-tex` source, see below) skip this.
function esc(s) {
  return String(s)
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([&%$#_{}])/g, "\\$1")
    .replace(/~/g, "\\textasciitilde{}")
    .replace(/\^/g, "\\textasciicircum{}");
}

// Any CSS colour the charts use → a 6-digit HTML hex for \definecolor, or null
// when it can't be resolved statically (e.g. a var()/rgba with alpha) so the
// caller can fall back to a cycled palette colour.
function toHex(color) {
  if (!color) return null;
  const c = String(color).trim();
  let m = c.match(/^#([0-9a-f]{6})$/i);
  if (m) return m[1].toUpperCase();
  m = c.match(/^#([0-9a-f]{3})$/i);
  if (m) return m[1].split("").map((h) => h + h).join("").toUpperCase();
  m = c.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
  if (m) {
    return [m[1], m[2], m[3]]
      .map((n) => Math.max(0, Math.min(255, +n)).toString(16).padStart(2, "0"))
      .join("")
      .toUpperCase();
  }
  return null;
}

const MARK = { circle: "*", square: "square*", triangle: "triangle*", diamond: "diamond*", cross: "x" };

const num = (v) => (Number.isFinite(v) ? +v.toPrecision(6) : v);

// pgfplots defaults to a scaled axis ("·10^3") and scientific tick labels once
// the range leaves [0.001, 1000]. Exported figures should read like the on-screen
// ones: literal numbers, no thousands separator noise.
const FIXED_TICKS = [
  "scaled ticks=false",
  "/pgf/number format/.cd, fixed, fixed zerofill=false, precision=4, 1000 sep={}",
];

function wrapFigure(body, { caption, label } = {}) {
  if (!caption && !label) return body;
  return [
    "\\begin{figure}[t]",
    "  \\centering",
    body.split("\n").map((l) => (l ? "  " + l : l)).join("\n"),
    caption ? `  \\caption{${esc(caption)}}` : null,
    label ? `  \\label{${label}}` : null,
    "\\end{figure}",
  ].filter((l) => l != null).join("\n");
}

// ---- tables → booktabs -----------------------------------------------------

// Pull text out of a cell, preferring the TeX source that <Katex> stamps as
// `data-tex` (so an ω header comes back as `$\omega$`, not a pile of spans).
function cellText(td) {
  const parts = [];
  let hadTex = false;
  td.childNodes.forEach((n) => {
    if (n.nodeType === 1 && n.dataset && n.dataset.tex != null) {
      parts.push(`$${n.dataset.tex}$`);
      hadTex = true;
    } else if (n.nodeType === 1 && n.querySelector && n.querySelector("[data-tex]")) {
      const k = n.querySelector("[data-tex]");
      parts.push(`$${k.dataset.tex}$`);
      hadTex = true;
    } else {
      parts.push(n.textContent || "");
    }
  });
  const raw = parts.join("").replace(/\s+/g, " ").trim();
  return hadTex ? raw : esc(raw);
}

// A live <table> DOM node → a booktabs tabular. Column count/alignment is taken
// from the header row; the first column is left-aligned, the rest right-aligned
// (the usual "labels on the left, numbers on the right" table). Multi-row
// <thead> is collapsed to its last row of <th> for the header labels.
export function tableToLatex(tableEl, { caption, label, align } = {}) {
  if (!tableEl) return "";
  const headRows = [...tableEl.querySelectorAll("thead tr")];
  const headCells = headRows.length
    ? [...headRows[headRows.length - 1].querySelectorAll("th,td")]
    : [];
  const bodyRows = [...tableEl.querySelectorAll("tbody tr")];
  const ncol = Math.max(
    headCells.length,
    ...bodyRows.map((r) => r.querySelectorAll("td,th").length),
    1
  );
  const colspec = align || "l" + "r".repeat(ncol - 1);

  const header = headCells.length ? headCells.map(cellText).join(" & ") + " \\\\" : null;
  const body = bodyRows.map(
    (r) => [...r.querySelectorAll("td,th")].map(cellText).join(" & ") + " \\\\"
  );

  const lines = [
    "\\begin{tabular}{" + colspec + "}",
    "\\toprule",
    ...(header ? [header, "\\midrule"] : []),
    ...body,
    "\\bottomrule",
    "\\end{tabular}",
  ];
  const tab = lines.join("\n");
  if (!caption && !label) return tab;
  return [
    "\\begin{table}[t]",
    "  \\centering",
    tab.split("\n").map((l) => "  " + l).join("\n"),
    caption ? `  \\caption{${esc(caption)}}` : null,
    label ? `  \\label{${label}}` : null,
    "\\end{table}",
  ].filter((l) => l != null).join("\n");
}

// A live <table> → CSV (numbers stay raw; math cells fall back to their TeX).
export function tableToCsv(tableEl) {
  if (!tableEl) return "";
  const rows = [...tableEl.querySelectorAll("thead tr, tbody tr")];
  const q = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  return rows
    .map((r) =>
      [...r.querySelectorAll("th,td")]
        .map((c) => {
          const k = c.querySelector?.("[data-tex]") || (c.dataset?.tex != null ? c : null);
          const t = k ? (k.dataset.tex ?? k.textContent) : c.textContent;
          return q((t || "").replace(/\s+/g, " ").trim());
        })
        .join(",")
    )
    .join("\n");
}

// ---- line chart → pgfplots -------------------------------------------------

const PALETTE = ["3B82F6", "F59E0B", "10B981", "EF4444", "A855F7", "14B8A6", "EC4899", "84CC16"];

// `series` is exactly LineChart's prop: [{label,color,points:[[x,y]],marker,dash}].
export function lineChartToTikz(series = [], opts = {}) {
  const { xLabel = "", yLabel = "", yScale = "linear", width, height, caption, label } = opts;
  const defs = [];
  const plots = [];
  const legends = [];
  const seen = {};
  series.forEach((s, i) => {
    const hex = toHex(s.color) || PALETTE[i % PALETTE.length];
    const name = `c${i}`;
    if (!seen[hex]) seen[hex] = true;
    defs.push(`\\definecolor{${name}}{HTML}{${hex}}`);
    const mark = MARK[s.marker] || "none";
    const dash = s.dash ? ", dashed" : "";
    const pts = (Array.isArray(s.points) ? s.points : [])
      .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && (yScale !== "log" || y > 0))
      .map(([x, y]) => `(${num(x)},${num(y)})`)
      .join(" ");
    plots.push(`\\addplot[color=${name}, mark=${mark}, thick${dash}] coordinates {${pts}};`);
    if (s.label) legends.push(s.label);
  });

  const axisOpts = [
    width ? `width=${(width / 72).toFixed(2)}in` : "width=3.4in",
    height ? `height=${(height / 72).toFixed(2)}in` : "height=2.2in",
    yScale === "log" ? "ymode=log" : null,
    xLabel ? `xlabel={${esc(xLabel)}}` : null,
    yLabel ? `ylabel={${esc(yLabel)}}` : null,
    "grid=both",
    "grid style={gray!20}",
    legends.length > 1 ? "legend pos=outer north east" : null,
    // plain numbers on the ticks — no "·10^3" axis multiplier, no 1.0e+3.
    // (a log axis keeps pgfplots' 10^k labels: there that IS the right form)
    ...(yScale === "log" ? [] : FIXED_TICKS),
    "tick label style={font=\\footnotesize}",
  ].filter(Boolean);

  const body = [
    "% requires \\usepackage{pgfplots}",
    ...defs,
    "\\begin{tikzpicture}",
    "\\begin{axis}[",
    axisOpts.map((o) => "  " + o + ",").join("\n"),
    "]",
    ...plots,
    legends.length > 1 ? `\\legend{${legends.map(esc).join(", ")}}` : null,
    "\\end{axis}",
    "\\end{tikzpicture}",
  ].filter((l) => l != null).join("\n");

  return wrapFigure(body, { caption, label });
}

// ---- bar chart → pgfplots --------------------------------------------------

// `bars` is exactly BarChart's prop: [{label, values:[{v,color}]}]; `legend` its
// per-series legend [{label,color}].
export function barChartToTikz(bars = [], opts = {}) {
  const { unit = "", horizontal = true, legend = [], width, height, caption, label } = opts;
  bars = bars.filter((b) => Array.isArray(b?.values));
  const nseries = Math.max(0, ...bars.map((b) => b.values.length));
  const labels = bars.map((b) => b.label);

  const defs = [];
  const plots = [];
  for (let k = 0; k < nseries; k++) {
    const color = legend[k]?.color || bars.find((b) => b.values[k])?.values[k]?.color;
    const hex = toHex(color) || PALETTE[k % PALETTE.length];
    defs.push(`\\definecolor{b${k}}{HTML}{${hex}}`);
    const coords = bars
      .map((b) => `(${symbolic(b.label)},${num(b.values[k]?.v ?? 0)})`)
      .join(" ");
    plots.push(`\\addplot[fill=b${k}, draw=none] coordinates {${coords}};`);
  }

  const barType = horizontal ? "xbar" : "ybar";
  const valLabel = unit ? `{value (${esc(unit)})}` : "{value}";
  const axisOpts = [
    barType,
    "bar width=6pt",
    width ? `width=${(width / 72).toFixed(2)}in` : "width=3.4in",
    height ? `height=${(height / 72).toFixed(2)}in` : "height=3.0in",
    horizontal
      ? `symbolic y coords={${labels.map(symbolic).join(",")}}`
      : `symbolic x coords={${labels.map(symbolic).join(",")}}`,
    horizontal ? "ytick=data" : "xtick=data",
    horizontal ? `xlabel=${valLabel}` : `ylabel=${valLabel}`,
    !horizontal ? "x tick label style={rotate=45,anchor=east,font=\\footnotesize}" : "tick label style={font=\\footnotesize}",
    "grid=both",
    "grid style={gray!20}",
    legend.length > 1 ? "legend pos=outer north east" : null,
    ...FIXED_TICKS,
  ].filter(Boolean);

  const body = [
    "% requires \\usepackage{pgfplots}",
    ...defs,
    "\\begin{tikzpicture}",
    "\\begin{axis}[",
    axisOpts.map((o) => "  " + o + ",").join("\n"),
    "]",
    ...plots,
    legend.length > 1 ? `\\legend{${legend.map((l) => esc(l.label)).join(", ")}}` : null,
    "\\end{axis}",
    "\\end{tikzpicture}",
  ].filter((l) => l != null).join("\n");

  return wrapFigure(body, { caption, label });
}

// pgfplots symbolic coords can't contain commas/braces — sanitise labels.
function symbolic(s) {
  return String(s).replace(/[,{}]/g, " ").trim();
}
