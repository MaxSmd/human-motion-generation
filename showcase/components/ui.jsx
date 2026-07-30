// Presentational primitives. Deliberately spare: the site carries results, so
// the furniture is a frame, a caption and a number format, nothing more.

import { Reveal } from "./motion";

export function Page({ children }) {
  return <main className="mx-auto max-w-[1180px] px-5 py-10 md:py-16">{children}</main>;
}

export function PageHead({ kicker, title, lede }) {
  return (
    <header className="mb-14">
      {kicker && <div className="label mb-3">{kicker}</div>}
      <h1 className="display text-[34px] leading-[1.06] text-[var(--text)] md:text-[46px]">{title}</h1>
      {lede && <p className="mt-4 max-w-2xl text-[14px] leading-relaxed text-[var(--muted)]">{lede}</p>}
    </header>
  );
}

export function Section({ n, title, sub, children, id, center = false }) {
  return (
    <section className="mb-16 scroll-mt-24" id={id}>
      {title && (
        <Reveal className={`mb-5 ${center ? "text-center" : ""}`}>
          {center ? (
            <>
              {n && <div className="mb-1.5 font-mono text-[11px] text-[var(--signal)]">{n}</div>}
              <h2 className="display text-[21px] text-[var(--text)] md:text-[25px]">{title}</h2>
              {sub && <p className="mx-auto mt-2 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">{sub}</p>}
            </>
          ) : (
            <>
              <div className="flex items-baseline gap-3">
                {n && <span className="font-mono text-[11px] text-[var(--signal)]">{n}</span>}
                <h2 className="display text-[21px] text-[var(--text)] md:text-[25px]">{title}</h2>
              </div>
              {sub && <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">{sub}</p>}
            </>
          )}
        </Reveal>
      )}
      {children}
    </section>
  );
}

/** Caption under a figure. Carries the reading, not decoration. */
export function Caption({ children }) {
  return <p className="mt-3 text-[12.5px] leading-relaxed text-[var(--muted)]">{children}</p>;
}

export function Stat({ k, v, sub, tone = "signal" }) {
  const color = {
    signal: "var(--signal)",
    warn: "var(--warn)",
    real: "var(--real)",
    text: "var(--text)",
  }[tone];
  return (
    <div className="surface p-4">
      <div className="label mb-2">{k}</div>
      <div className="display text-[27px] leading-none" style={{ color }}>
        {v}
      </div>
      {sub && <div className="mt-2 text-[11px] leading-snug text-[var(--muted)]">{sub}</div>}
    </div>
  );
}

/** Math line. Unicode + mono, no typesetting dependency. */
export function Eq({ children, note }) {
  return (
    <div className="my-3 overflow-x-auto rounded-lg border border-[var(--hairline)] bg-black/30 px-4 py-3">
      <span className="font-mono text-[13.5px] text-[var(--text)]" style={{ letterSpacing: "0.01em" }}>
        {children}
      </span>
      {note && <div className="mt-1.5 font-mono text-[10.5px] text-[var(--muted)]">{note}</div>}
    </div>
  );
}

export function Table({ head, rows, align = [] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--hairline)]">
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr className="bg-black/30">
            {head.map((h, i) => (
              <th
                key={i}
                className="whitespace-nowrap px-3 py-2.5 font-mono text-[9.5px] uppercase tracking-[0.16em] text-[var(--muted)]"
                style={{ textAlign: align[i] || "left" }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className="border-t border-[var(--hairline)] transition-colors hover:bg-white/[0.025]"
              style={r._hi ? { background: "var(--signal-dim)" } : undefined}
            >
              {r.cells.map((c, j) => (
                <td
                  key={j}
                  className={`px-3 py-2 ${j === 0 ? "text-[var(--text)]" : "font-mono text-[var(--text)]"}`}
                  style={{
                    textAlign: align[j] || "left",
                    ...(r._hi && j === 0 ? { color: "var(--signal)", fontWeight: 600 } : {}),
                    ...(r._dim ? { color: "var(--muted)" } : {}),
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Key/value chip grid — model cards, recipes, configurations. */
export function Specs({ items, cols = "sm:grid-cols-3 lg:grid-cols-5" }) {
  return (
    <div className={`grid grid-cols-2 gap-2.5 ${cols}`}>
      {items.map(([k, v]) => (
        <div key={k} className="surface p-3">
          <div className="label mb-1.5">{k}</div>
          <div className="font-mono text-[12.5px] leading-snug text-[var(--text)]">{v}</div>
        </div>
      ))}
    </div>
  );
}
