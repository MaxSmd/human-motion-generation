// Typeset display equations without a math library.
//
// The site has no KaTeX dependency and does not need one: the handful of
// equations here are built from a serif italic for variables, an upright face
// for operators and numerals, and real <sub>/<sup>, which is what makes the
// difference between typeset maths and a line of Unicode approximations.

/** Italic variable. */
export function V({ children }) {
  return <span style={{ fontFamily: "var(--font-math)", fontStyle: "italic" }}>{children}</span>;
}

/** Upright text inside an equation: operator names, function names, units. */
export function T({ children }) {
  return <span style={{ fontFamily: "var(--font-math)", fontStyle: "normal" }}>{children}</span>;
}

/** Binary operator with the right amount of air around it. */
export function O({ children }) {
  return <span style={{ fontFamily: "var(--font-math)", padding: "0 0.34em" }}>{children}</span>;
}

export function Sub({ children }) {
  return (
    <sub style={{ fontSize: "0.68em", fontFamily: "var(--font-math)", fontStyle: "italic", lineHeight: 0 }}>
      {children}
    </sub>
  );
}

export function Sup({ children }) {
  return (
    <sup style={{ fontSize: "0.68em", fontFamily: "var(--font-math)", lineHeight: 0 }}>{children}</sup>
  );
}

/** Stacked fraction with a hairline rule. */
export function Frac({ num, den }) {
  return (
    <span
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        verticalAlign: "middle",
        margin: "0 0.22em",
        lineHeight: 1.18,
      }}
    >
      <span style={{ padding: "0 0.25em 0.1em" }}>{num}</span>
      <span style={{ width: "100%", height: 1, background: "currentColor", opacity: 0.55 }} />
      <span style={{ padding: "0.1em 0.25em 0" }}>{den}</span>
    </span>
  );
}

/**
 * A numbered display equation. `note` is the one-line reading of it, which is
 * what a visitor who skips the symbols will actually take away.
 */
export function Formula({ children, n, note }) {
  return (
    <figure className="my-6">
      <div className="flex items-center gap-4">
        <div
          className="min-w-0 flex-1 overflow-x-auto px-2 py-1 text-center text-[var(--text)]"
          style={{ fontFamily: "var(--font-math)", fontSize: 19, letterSpacing: "0.005em" }}
        >
          <span style={{ whiteSpace: "nowrap" }}>{children}</span>
        </div>
        {n && <span className="shrink-0 font-mono text-[11px] text-[var(--muted)]">({n})</span>}
      </div>
      {note && (
        <figcaption className="mt-2.5 text-center text-[12.5px] leading-relaxed text-[var(--muted)]">
          {note}
        </figcaption>
      )}
    </figure>
  );
}
