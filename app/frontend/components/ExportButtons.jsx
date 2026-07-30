"use client";

import { useState } from "react";
import { tableToLatex, tableToCsv, downloadText, copyText } from "@/lib/exportLatex";

// A tiny export toolbar for a hand-written <table>: copies booktabs LaTeX to the
// clipboard and downloads a .tex / .csv. Pass a getter for the live table node
// (`getTable={() => ref.current}`) so extraction reads exactly what's on screen
// (KaTeX cells come back as their `$...$` source via the <Katex> data-tex hook).
export function TableTools({ getTable, name = "table", caption, label, className = "" }) {
  const [copied, setCopied] = useState(false);

  const withTable = (fn) => {
    const el = getTable?.();
    if (el) fn(el);
  };
  const copy = () =>
    withTable(async (el) => {
      const ok = await copyText(tableToLatex(el, { caption, label }));
      if (ok) {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }
    });
  const tex = () => withTable((el) => downloadText(tableToLatex(el, { caption, label }), `${name}.tex`));
  const csv = () => withTable((el) => downloadText(tableToCsv(el), `${name}.csv`, "text/csv"));

  return (
    <span className={`flex gap-1 ${className}`}>
      <ExportBtn onClick={copy} title="copy booktabs LaTeX">{copied ? "✓ latex" : "⧉ latex"}</ExportBtn>
      <ExportBtn onClick={tex} title="download .tex">⤓ tex</ExportBtn>
      <ExportBtn onClick={csv} title="download .csv">⤓ csv</ExportBtn>
    </span>
  );
}

// Small shared button used by both TableTools and the chart export controls.
export function ExportBtn({ onClick, title, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="rounded border border-[var(--hairline-strong)] bg-ink px-1.5 py-0.5 text-[9px] text-slate-400 transition hover:border-[var(--signal)] hover:text-[var(--signal)]"
    >
      {children}
    </button>
  );
}
