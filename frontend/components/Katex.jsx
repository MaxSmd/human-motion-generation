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

  if (html == null) return <span className={className}>{tex}</span>;
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
