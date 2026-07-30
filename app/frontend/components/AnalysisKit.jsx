"use client";

// Shared furniture for the Lab's analysis tabs (Analysis · Clip comparison ·
// Constraint analysis). Extracted when clip comparison and constraint analysis
// moved out of AnalysisTab into tabs of their own — these are the bits all three
// still agree on, so a colour or a section header stays consistent across them.

export const SIGNAL = "#22d3ee";
export const AMBER = "#fbbf24";
export const GREEN = "#34d399";
export const ROSE = "#fb7185";
export const SLATE = "#64748b";

export const ROLE_COLOR = { real: SIGNAL, gen: AMBER, clip: SIGNAL };
export const ROLE_LABEL = { real: "GT", gen: "GEN", clip: "CLIP" };

// Numbered section header, e.g. "03 · Clip comparison · real vs gen".
export function Section({ n, title, sub, children }) {
  return (
    <section>
      <div className="mb-3 flex items-baseline gap-3">
        <span className="font-mono text-[11px] tracking-widest text-[var(--signal)]">{n}</span>
        <h2 className="display text-lg font-bold text-white">{title}</h2>
        <span className="label">{sub}</span>
      </div>
      {children}
    </section>
  );
}

export function Chart({ title, children }) {
  return (
    <figure className="rounded-lg border border-[var(--hairline)] bg-ink p-3">
      <figcaption className="label mb-2">{title}</figcaption>
      {children}
    </figure>
  );
}

// Dashed-border placeholder for "you haven't generated the inputs yet".
export function EmptyHint({ children }) {
  return (
    <p className="rounded-lg border border-dashed border-[var(--hairline)] px-3 py-10 text-center text-[12px] text-[var(--muted)]">
      {children}
    </p>
  );
}

// /…/runs/<kind>/<run>/checkpoints/<file>.pt → "<run>/<file>.pt"
export function ckptLabel(path) {
  if (!path) return null;
  const p = path.split("/");
  return p.length >= 3 ? `${p[p.length - 3]}/${p[p.length - 1]}` : p[p.length - 1];
}
