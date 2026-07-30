"use client";

// Constraints: the method, then the measurements. The limit is held exactly, on
// the full split as well as in the factorial. For smoothness the figure gives an
// upper bound on the cost, and its caption states what that bound is.

import MotionGallery from "./MotionGallery";
import ConstraintMethod from "./ConstraintMethod";
import { Table, Caption } from "./ui";
import { Stage, Reveal } from "./motion";
import { PairSpread } from "./charts";
import { CONSTRAINT_PAIRS, constrainedUrl } from "@/lib/clips";
import { FACTORIAL, SATISFACTION, CONSTRAINED_EVAL, FREE_SPREAD, ATTESTATION, SELECTION, C } from "@/lib/results";

const items = CONSTRAINT_PAIRS.map((c) => ({
  id: c.id,
  label: c.label,
  sub: c.sub,
  caption: c.prompt,
  clips: [
    { url: constrainedUrl(c.freeFile), color: C.muted, label: "unconstrained" },
    { url: constrainedUrl(c.pinFile), color: C.mid, label: `${c.joint} pinned` },
  ],
  highlight: c.limb,
  accent: "#fbbf24",
  meta: [
    ["unconstrained range", c.freeRange],
    ["pinned to", c.pin],
    ["measured", c.held, "good"],
    ["jerk", `${c.free.toFixed(3)} → ${c.pinned.toFixed(3)}`],
  ],
}));

// One row per (constraint spec × text condition): the three seed-paired
// free→projected ratios, plotted as points on a log axis.
const spreadRows = FACTORIAL.cells.map((c) => ({
  label: `${c.kind} ${c.target}`,
  sub: c.label,
  values: c.pairs.map(([, f, p]) => p / f),
}));

export default function ConstraintsSection() {
  return (
    <div>
      <div>
        <Reveal>
          <h3 className="display mb-4 text-[17px] text-[var(--text)]">Method</h3>
          <ConstraintMethod />
        </Reveal>
      </div>

      <div className="mt-8">
        <Reveal>
          <MotionGallery
            items={items}
            stageLabel="unconstrained and constrained"
            note="mid · ω 6.5 · 800 ODE steps · same seed"
            height={440}
            align
            stats
          />
          <Caption>{SELECTION.constraint}</Caption>
        </Reveal>
      </div>

      <div className="mt-8">
        <Reveal>
          <h3 className="display mb-1 text-[17px] text-[var(--text)]">Constraint satisfaction</h3>
          <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
The bend angle at the constrained joint, measured over every frame of every constrained
            sample in the study below.
          </p>
          <Table
            head={["Constraint", "Limit", "Samples", "Measured range", "Largest deviation"]}
            align={["left", "right", "right", "right", "right"]}
            rows={SATISFACTION.map((r) => ({
              cells: [
                r.spec,
                r.limit,
                r.n,
                r.lo === r.hi ? `${r.hi.toFixed(2)}°` : `${r.lo.toFixed(2)}° – ${r.hi.toFixed(2)}°`,
                `${r.dev.toFixed(2)}°`,
              ],
            }))}
          />
          <Caption>
            No violations across 36 constrained samples and every frame of each. The clamp leaves the
            knee free to move inside its range and it uses 6.6° to 10.0° of it. The two fixed angles
            hold at their target in every frame, with the 0.01° at the pin to zero coming from the
            numerical precision of the projection and of the angle readout. The same measurement over
            all 4096 clips of the test split is in the table below.
          </Caption>
        </Reveal>
      </div>

      <div className="mt-8">
        <Reveal>
          <h3 className="display mb-1 text-[17px] text-[var(--text)]">Constrained generation, full split</h3>
          <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
            The same evaluation as the rest of this page, with the projection active for every one of the
            4096 clips. {CONSTRAINED_EVAL.setting}.
          </p>
          <Table
            head={["Sampling", "Constraint", "FID", "R@1 ↑", "R@3 ↑", "MM-dist ↓", "Diversity →", "MModality", "Worst violation"]}
            align={["left", "left", "right", "right", "right", "right", "right", "right", "right"]}
            rows={CONSTRAINED_EVAL.rows.map((r) => ({
              _dim: r.ref,
              cells: [
                r.label, r.spec, r.fid.toFixed(4), r.r1.toFixed(4), r.r3.toFixed(4),
                r.mm.toFixed(3), r.div.toFixed(3), r.mmod.toFixed(3),
                r.worst == null ? "—" : r.worst === 0 ? "0°" : `${r.worst.toExponential(1)}°`,
              ],
            }))}
          />
          <Caption>
            The null projection clamps the knee to 0 – 180°, a range every angle already satisfies, so the
            projector runs at every step and can never change anything. It returns FID 0.6462 against
            0.6471 unconstrained, so the two rows below it can be read against it. Holding the joint
            raises FID to 1.15 and 1.54. That rise is expected: the training set contains almost no
            locked-knee walking, so constraining every clip moves the samples off the data distribution
            by construction, and there is no constraint-matched reference set to score against.
            Retrieval and multimodal distance are the informative axis here. Retrieval falls from
            0.498 to 0.481 under the clamp and to 0.460 under the pin, so the samples still match their
            captions with the joint held. Diversity falls as the pinned joint stops varying. Across all
            4096 clips the largest violation of the limit is {"8.7 × 10⁻⁵"} degrees.
          </Caption>
        </Reveal>
      </div>

      <div className="mt-8">
        <Reveal>
          <h3 className="display mb-1 text-[17px] text-[var(--text)]">Effect on smoothness</h3>
          <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
If a constraint conflicts with the motion the model would otherwise produce, this should
            appear as rougher motion. To test this, the joint, the limit and the base prompt are held
            fixed and only the text condition varies. Each cell is sampled with and without the
            projection from the same three seeds.
          </p>
          <Stage
            label="jerk ratio, constrained over unconstrained"
            note={`${FACTORIAL.design} · log scale`}
          >
            <PairSpread
              rows={spreadRows}
              width={760}
              refLine={1}
              refLabel="no change"
              color={C.mid}
              warnAbove={FACTORIAL.detectable}
            />
            <Caption>
Each point is one seed, giving the jerk of the constrained sample divided by the jerk of the
              unconstrained sample from the same seed and prompt. The two arms share the noise draw, so
              each point is a matched pair. Over all {FACTORIAL.n} pairs the median is{" "}
              {FACTORIAL.medianRatio}×, with an interquartile range of {FACTORIAL.iqr[0]}–
              {FACTORIAL.iqr[1]}×, and the constrained sample is smoother in {FACTORIAL.smootherPairs} of
              them. On the log ratio the sign test gives p = {FACTORIAL.signP} and the Wilcoxon signed-rank
              test p = {FACTORIAL.wilcoxonP}. The spread between seeds is large enough that at{" "}
              {FACTORIAL.n} pairs this design would only resolve an effect of about{" "}
              {FACTORIAL.detectable}× or larger. The result is therefore an upper bound on the cost.
              Points above {FACTORIAL.detectable}× are marked, since that is
              where a single pair exceeds what the design can resolve. For comparison, sampling the unconstrained
              prompt at three seeds gives
              clips whose jerk differs by a median of {FREE_SPREAD.median}× across the twelve cells and,
              in the worst cell, {FREE_SPREAD.worst}×.
            </Caption>
          </Stage>
        </Reveal>
      </div>

      <div className="mt-8">
        <Reveal>
          <h3 className="display mb-1 text-[17px] text-[var(--text)]">Effect of the text condition</h3>
          <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
We expected that describing the constrained motion in the prompt would reduce the conflict,
            and that this would depend on whether the phrasing occurs in the training captions. The
            factorial does not support this.
          </p>
          <Table
            head={["Added text", "Clips with every word", "Median cost, clamp 0–10°", "Median cost, pin 10°", "Median cost, pin 0°"]}
            align={["left", "right", "right", "right", "right"]}
            rows={textRows()}
          />
          <Caption>
The ordering of the four text conditions does not follow their corpus support, and all
            differences between them are smaller than the seed spread within a single cell. At this
            sample size we measure no effect of the added text.
          </Caption>
        </Reveal>
      </div>

      <div className="mt-8">
        <Reveal>
          <h3 className="display mb-1 text-[17px] text-[var(--text)]">Corpus support</h3>
          <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
The conjunction over every content word is a limited measure of corpus support, since it
            reaches zero for most longer phrases, including phrases made up of well-attested words.
          </p>
          <Table
            head={["Phrase", "Clips with every word", "Per-word support", "Weakest word"]}
            align={["left", "right", "left", "right"]}
            rows={ATTESTATION.phrases.map((p) => ({
              _hi: p.all === 0,
              cells: [
                `“${p.phrase}”`,
                p.all,
                p.terms.map(([w, n]) => `${w} ${n}`).join(" · "),
                `${p.weakest[0]} ${p.weakest[1]}`,
              ],
            }))}
          />
          <Caption>
Counts are clips, out of the {ATTESTATION.total.toLocaleString("en-GB")} unmirrored HumanML3D
            clips, whose captions contain the lemma. “holding a box against their chest” scores zero on
            the conjunction although each of its four words appears in dozens to thousands of clips, the
            limiting term being “against” at 48. We therefore report the weakest single term alongside
            the conjunction.
          </Caption>
        </Reveal>
      </div>

    </div>
  );
}

// Median cost per (text condition × constraint spec), from the factorial.
function textRows() {
  const median = (a) => {
    const s = [...a].sort((x, y) => x - y);
    return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
  };
  const texts = ["none", "archetype", "semantic", "literal"];
  const specs = [
    ["clamp", "0 – 10°"],
    ["pin", "10°"],
    ["pin", "0°"],
  ];
  return texts.map((t) => {
    const any = FACTORIAL.cells.find((c) => c.text === t);
    const cost = specs.map(([kind, target]) => {
      const cell = FACTORIAL.cells.find((c) => c.text === t && c.kind === kind && c.target === target);
      return cell ? `${median(cell.pairs.map(([, f, p]) => p / f)).toFixed(2)}×` : "—";
    });
    return {
      cells: [t === "none" ? "no added text" : `“${any.label}”`, any.clips == null ? "—" : any.clips, ...cost],
    };
  });
}
