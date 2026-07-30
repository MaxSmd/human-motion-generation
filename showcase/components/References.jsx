// Citations. `Cite` is the inline marker; `References` is the list at the foot
// of a page. Keys index REFERENCES in lib/results.js.

import { REFERENCES, REPO } from "@/lib/results";

const BY_KEY = Object.fromEntries(REFERENCES.map((r, i) => [r.key, { ...r, n: i + 1 }]));

/** Inline citation: <Cite k="repaint" /> → [4]. */
export function Cite({ k }) {
  const r = BY_KEY[k];
  if (!r) return null;
  return (
    <a
      href={`#ref-${r.key}`}
      className="whitespace-nowrap font-mono text-[0.85em] text-[var(--signal)] hover:underline"
      title={r.cite}
    >
      [{r.n}]
    </a>
  );
}

export default function References() {
  return (
    <section className="mb-16 mt-4 border-t border-[var(--hairline)] pt-8">
      <div className="label mb-4">References</div>
      <ol className="space-y-2.5">
        {REFERENCES.map((r, i) => (
          <li key={r.key} id={`ref-${r.key}`} className="flex gap-3 text-[12.5px] leading-relaxed">
            <span className="shrink-0 font-mono text-[var(--signal)]">[{i + 1}]</span>
            <span className="text-[var(--muted)]">
              {r.text}{" "}
              <a
                href={r.href}
                target="_blank"
                rel="noreferrer noopener"
                className="font-mono text-[11.5px] text-[var(--signal)] hover:underline"
              >
                {r.cite} ↗
              </a>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** Who made this and where the code is. */
export function Colophon() {
  return (
    <div className="mb-14 text-[12px] leading-relaxed text-[var(--muted)]">
      A reproduction of RMG <Cite k="rmg" /> at 25 M and 112 M parameters, carried out for the practical
      course on deep representation learning. All figures were measured by us on the full HumanML3D test
      split, except where a row is marked “published”, which is quoted from the cited work. The code,
      configurations and run directories behind each figure will be released at{" "}
      <a
        href={REPO}
        target="_blank"
        rel="noreferrer noopener"
        className="font-mono text-[11.5px] text-[var(--signal)] hover:underline"
      >
        github.com/julsmzr/riemann-motion-generation ↗
      </a>
      .
    </div>
  );
}
