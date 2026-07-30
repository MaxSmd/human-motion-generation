"use client";

import { LinePlot } from "./charts";
import { Stage } from "./motion";
import { Caption } from "./ui";
import { CURVES } from "@/lib/curves";
import { C } from "@/lib/results";

const zip = (a, b) => a.map((x, i) => [x, b[i]]);
const SWITCH = CURVES.switch;

/**
 * The two loss traces, raw, on a log axis. The axis absorbs the ~10x offset
 * between the objectives (base predates the antipodal quotient, so its target
 * still carries the near-antipodal blow-ups), leaving both shapes readable
 * without an arbitrary normalisation.
 */
export function LossCurves() {
  const series = [
    { label: "base", color: C.base, points: zip(CURVES.base.step, CURVES.base.loss), width: 1.7 },
    { label: "mid", color: C.mid, points: zip(CURVES.mid.step, CURVES.mid.loss), width: 1.7 },
  ];
  return (
    <Stage label="training loss" note="raw · log scale">
      <LinePlot
        series={series}
        width={720}
        height={330}
        yLog
        xLabel="training step"
        yLabel="loss"
        xTicks={[0, 50000, 100000, 150000, 200000, 250000, 300000]}
        vlines={[{ x: SWITCH, label: "bf16 → tf32", color: C.warn }]}
        right={52}
      />
      <Caption>
        The mid loss steps up at 160 k, where the run switched from bf16 to tf32 by resuming from that
        checkpoint, and then descends steadily again, still falling at 300 k. The 150 k checkpoint sits
        just before the switch, at a loss of 3.28 against 4.80 for the lowest checkpoint after it, and
        its FID is higher than that of the checkpoints on either side (see below). Under bf16 the loss
        was therefore no longer tracking sample quality. The values before and after the switch are not
        directly comparable.
      </Caption>
    </Stage>
  );
}

/** Learning rate and gradient norm, both marked at the resume. */
export function ScheduleCurves() {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Stage label="learning rate" note="cosine to zero">
        <LinePlot
          series={[
            { label: "base", color: C.base, points: zip(CURVES.base.step, CURVES.base.lr), width: 1.7 },
            { label: "mid", color: C.mid, points: zip(CURVES.mid.step, CURVES.mid.lr), width: 1.7 },
          ]}
          width={560}
          height={250}
          xLabel="training step"
          yLabel="learning rate"
          xTicks={[0, 100000, 200000, 300000]}
          vlines={[{ x: SWITCH, label: "resume", color: C.warn }]}
          right={48}
        />
        <Caption>
Both runs use a 1e-4 peak and cosine decay. The resume continues the existing schedule, so
          the switch is not visible here.
        </Caption>
      </Stage>
      <Stage label="gradient norm" note="pre-clip · log scale">
        <LinePlot
          series={[
            { label: "base", color: C.base, points: zip(CURVES.base.step, CURVES.base.gradNorm), width: 1.7 },
            { label: "mid", color: C.mid, points: zip(CURVES.mid.step, CURVES.mid.gradNorm), width: 1.7 },
          ]}
          width={560}
          height={250}
          yLog
          xLabel="training step"
          yLabel="pre-clip gradient norm"
          xTicks={[0, 100000, 200000, 300000]}
          vlines={[{ x: SWITCH, label: "resume", color: C.warn }]}
          right={48}
        />
        <Caption>
Base runs three to five times higher throughout. This is the antipodal instability as it
          appears in the gradient. Both runs are clipped to 0.5 before the step.
        </Caption>
      </Stage>
    </div>
  );
}
