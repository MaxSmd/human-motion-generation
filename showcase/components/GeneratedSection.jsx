"use client";

// The gen-vs-ground-truth gallery and the smoothness plot, sharing one
// selection: picking a clip above highlights its row below. Clips are aligned
// so both figures face the same way, and the small analysis panels update with
// the selection.

import { useState } from "react";
import MotionGallery from "./MotionGallery";
import { Dumbbell } from "./charts";
import { Stage } from "./motion";
import { Caption } from "./ui";
import { PAIRS, gtUrl, genUrl } from "@/lib/clips";
import { PAIRED_JERK_SUMMARY, C } from "@/lib/results";

const items = PAIRS.map((p) => ({
  id: p.id,
  label: p.label,
  sub: `${(p.gen / p.gt).toFixed(1)}× jerk`,
  caption: p.caption,
  clips: [
    { url: gtUrl(p.id), color: C.real, label: "ground truth" },
    { url: genUrl(p.id), color: C.mid, label: "generated" },
  ],
  meta: [
    ["jerk, real", p.gt.toFixed(4)],
    ["jerk, generated", p.gen.toFixed(4), p.gen / p.gt > 8 ? "warn" : undefined],
    ["ratio", `${(p.gen / p.gt).toFixed(1)}×`],
  ],
}));

export default function GeneratedSection() {
  const [sel, setSel] = useState(0);
  return (
    <>
      <MotionGallery
        items={items}
        stageLabel="ground truth vs generated"
        note="mid · ω 6.5 · 800 ODE steps"
        height={440}
        align
        stats
        onSelect={setSel}
      />
      <div className="mt-4">
        <Stage label="jerk, real vs generated" note="mean ‖Δ³x‖ · log scale">
          <Dumbbell
            rows={PAIRS.map((p) => ({ label: p.label, a: p.gt, b: p.gen }))}
            aLabel="● real"
            bLabel="● generated"
            aColor={C.real}
            bColor={C.mid}
            highlight={sel}
            width={700}
          />
          <Caption>
            Jerk is the mean third difference of joint positions and measures how abrupt the motion is.
            The clip currently shown is highlighted. Across the nine pairs the generated motion has a
            median jerk {PAIRED_JERK_SUMMARY.median}× that of its reference, ranging over{" "}
            {PAIRED_JERK_SUMMARY.spread}. The walking clips are closest and the two dance clips are
            furthest apart. The freestyle-dance capture is the smoothest reference in the set, at
            0.0019, which is part of why its ratio is the largest.
          </Caption>
        </Stage>
      </div>
    </>
  );
}
