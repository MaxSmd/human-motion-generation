"use client";
// Corpus attestation for a freeform prompt — "has the text encoder ever seen
// this language?" Scored against the real HumanML3D captions the encoder trained
// on (/corpus/phrase). The predictive one: a phrase with no support won't move
// the prior no matter how faithfully it describes the motion you want, so this
// sits under every prompt box where we enter text — Studio, the scene/room
// editor, and the constraint lab's formalized prompts.
//
// Extracted from ConstraintAnalysisTab's PhraseCoverage so all three entry
// points score identically and read the same badge.

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { GREEN, AMBER, ROSE } from "./AnalysisKit";

const TONE = {
  strong: { c: GREEN, t: "well attested — the prior knows this language" },
  medium: { c: GREEN, t: "attested — should move the prior" },
  weak: { c: AMBER, t: "thin support — may not move the prior" },
  none: { c: ROSE, t: "no support — expect the prompt to fight the prior" },
};

export default function PhraseAttestation({ phrase, defaultOpen = false }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(defaultOpen);

  useEffect(() => {
    let cancelled = false;
    setStats(null); setError(null);
    const q = (phrase || "").trim();
    if (!q) return undefined;
    const t = setTimeout(() => {
      api.corpusPhrase(q)
        .then((s) => !cancelled && setStats(s))
        .catch((e) => !cancelled && setError(e.message));
    }, 400); // debounce: the phrase changes on every keystroke of the editor
    return () => { cancelled = true; clearTimeout(t); };
  }, [phrase]);

  if (!(phrase || "").trim()) return null;
  if (error) return <p className="mt-2 text-[11px] text-[var(--muted)]">corpus check unavailable — {error}</p>;
  if (!stats) return <p className="mt-2 text-[11px] text-[var(--muted)]">checking corpus support…</p>;

  const tone = TONE[stats.coverage] || TONE.none;

  return (
    <div className="mt-2">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 text-left">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: tone.c }} />
        <span className="font-mono text-[11px]" style={{ color: tone.c }}>
          {stats.all_clips} / {stats.total_clips} clips
        </span>
        <span className="text-[11px] text-[var(--muted)]">{tone.t}</span>
        <span className="ml-auto text-[10px] text-[var(--muted)]">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2 border-t border-[var(--hairline)] pt-2">
          <div className="flex flex-wrap gap-1.5">
            {stats.terms.map((t) => (
              <span key={t.word}
                className="rounded border px-1.5 py-0.5 font-mono text-[10px]"
                style={{
                  borderColor: t.clips === 0 ? "rgba(251,113,133,0.5)" : "var(--hairline)",
                  color: t.clips === 0 ? ROSE : "var(--muted)",
                }}>
                {t.word}{t.lemma && t.lemma !== t.word ? `→${t.lemma}` : ""} · {t.clips}
              </span>
            ))}
          </div>
          {stats.weakest?.clips === 0 && (
            <p className="text-[11px] text-rose-300">
              “{stats.weakest.word}” never appears in a caption — that term is what breaks the phrase.
            </p>
          )}
          {stats.examples?.length > 0 && (
            <div>
              <div className="label mb-1">captions that match</div>
              {stats.examples.map((e, i) => (
                <p key={i} className="text-[11px] leading-snug text-slate-400">“{e}”</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
