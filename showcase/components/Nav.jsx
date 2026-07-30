"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/training", label: "Training" },
  { href: "/results", label: "Results" },
  { href: "/studio", label: "Studio" },
];

export default function Nav() {
  const path = usePathname();
  const active = (href) => (href === "/" ? path === "/" : path.startsWith(href));
  return (
    <nav className="sticky top-0 z-30 border-b border-[var(--hairline)] bg-[var(--ink)]/80 backdrop-blur">
      <div className="mx-auto flex max-w-[1200px] items-center gap-2 px-4 py-3 sm:gap-4 sm:px-5">
        <Link href="/" className="flex shrink-0 items-center gap-2">
          <span className="dot" style={{ color: "var(--signal)", background: "var(--signal)" }} />
          <span className="display text-[15px] text-[var(--text)]">RMG</span>
        </Link>
        <div className="swipe-rail ml-auto flex gap-0.5 overflow-x-auto sm:ml-2 sm:gap-1">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href}
              className={`whitespace-nowrap rounded-md px-2.5 py-1.5 font-mono text-[12px] transition sm:px-3 ${
                active(l.href)
                  ? "bg-[var(--signal-dim)] text-[var(--signal)]"
                  : "text-[var(--muted)] hover:text-[var(--text)]"
              }`}>
              {l.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
