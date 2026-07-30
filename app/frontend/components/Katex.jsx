"use client";

import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

// Renders a TeX string to HTML (inline). Throw-safe: falls back to raw text.
export default function Katex({ tex, className = "" }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { throwOnError: false, output: "html" });
    } catch {
      return null;
    }
  }, [tex]);

  // `data-tex` carries the raw TeX source so exporters (tables → LaTeX) can
  // recover `$\omega$` from a rendered cell instead of scraping KaTeX spans.
  if (html == null) return <span className={className} data-tex={tex}>{tex}</span>;
  return <span className={className} data-tex={tex} dangerouslySetInnerHTML={{ __html: html }} />;
}
