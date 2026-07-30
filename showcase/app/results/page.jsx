import { Page, PageHead, Section, Caption, Table } from "@/components/ui";
import { Reveal, Stage } from "@/components/motion";
import { LinePlot, LogBars, Violin } from "@/components/charts";
import MotionGallery from "@/components/MotionGallery";
import GeneratedSection from "@/components/GeneratedSection";
import ConstraintsSection from "@/components/ConstraintsSection";
import References, { Cite } from "@/components/References";
import {
  OMEGA, STEPS, STEP_METRICS, HEADLINE, QUOTED, CALIBRATION, CONSISTENCY, REPLICATION, JERK_DIST,
  FORENSICS, SELECTION, SCALING, VAL_SELECTION, JERK_SWEEP, C,
} from "@/lib/results";
import { PROMPTED, promptUrl } from "@/lib/clips";

export const metadata = { title: "RMG · Results" };

const promptItems = PROMPTED.map((p) => ({
  id: p.id,
  label: p.prompt.replace("a person ", ""),
  sub: `${p.clips} clips`,
  caption: p.prompt,
  clips: [{ url: promptUrl(p.id), color: C.mid, label: "generated" }],
  meta: [
    ["corpus support", `${p.clips} clips`, p.clips < 100 ? "warn" : "good"],
    ["jerk", p.jerk.toFixed(4)],
    ["travel", `${p.travel.toFixed(2)} m`],
  ],
}));

const SCALE = [
  { params: 24.7, fid: HEADLINE.base.fid, label: "base", color: C.base },
  { params: 111.7, fid: HEADLINE.mid.fid, label: "mid", color: C.mid },
  { params: 460, fid: HEADLINE.paper.fid, label: "published", color: C.paper },
];

// The measured replication spread, carried onto every mid FID point as an error
// bar. It was measured at 200 steps; we do not have a per-setting estimate, so
// the same absolute spread is drawn throughout and the caption says so.
const err = (pts) => pts.map(([x, y]) => [x, Math.max(y - REPLICATION.sd, 1e-4), y + REPLICATION.sd]);

export default function ResultsPage() {
  return (
    <Page>
      <PageHead
        kicker="Results"
        title="Results"
        lede="Figures are measured on the full HumanML3D test split unless another split or clip count is named. Each is reported with its guidance scale ω and ODE step count, since both affect the result."
      />

      <Section n="01" title="Evaluation setup" sub="Quantities with a published value, measured with our harness.">
        <Reveal>
          <Table
            head={["Quantity", "Measured here", "Published"]}
            align={["left", "right", "right"]}
            rows={CALIBRATION.map((c) => ({ cells: [c.metric, c.ours, c.published] }))}
          />
        </Reveal>
        <Caption>
          Real motion scored against real motion gives 0.0019, against a published 0.002{" "}
          <Cite k="rmg" />, so absolute FID values here are comparable to those in that work. Repeating
          a full-split evaluation at a fixed setting gives {REPLICATION.mean} ± {REPLICATION.sd} over{" "}
          {REPLICATION.n} runs, so differences below about {REPLICATION.resolution} are not resolved.
          Each FID on this page is a single pass, while the published figure is a mean over 20
          replications.
        </Caption>

      </Section>

      <Section n="02" title="Guidance" sub="Classifier-free guidance ω scales how strongly the caption pulls the sample. Swept on the full split at 200 steps.">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Reveal>
            <Stage label="FID vs guidance" note="log scale · lower is better">
              <LinePlot
                series={[
                  { label: "base", color: C.base, points: OMEGA.w.map((w, i) => [w, OMEGA.base.fid[i]]) },
                  {
                    label: "mid",
                    color: C.mid,
                    points: OMEGA.w.map((w, i) => [w, OMEGA.mid.fid[i]]),
                    err: err(OMEGA.w.map((w, i) => [w, OMEGA.mid.fid[i]])),
                  },
                ]}
                width={560}
                height={300}
                yLog
                xLabel="guidance scale ω"
                yLabel="FID ↓"
                xTicks={OMEGA.w}
                right={46}
              />
              <Caption>
                With the ± {REPLICATION.sd} replication spread drawn on, mid is flat between ω 5.5 and
                7.5. The minimum at 6.5 is not separable from its neighbours, and the section below
                checks that choice against the validation split. Base remains near 8 at every ω, so
                guidance does not compensate for the weaker model.
              </Caption>
            </Stage>
          </Reveal>
          <Reveal delay={90}>
            <Stage label="retrieval vs guidance" note="R@1 · higher is better">
              <LinePlot
                series={[
                  { label: "base", color: C.base, points: OMEGA.w.map((w, i) => [w, OMEGA.base.r1[i]]) },
                  { label: "mid", color: C.mid, points: OMEGA.w.map((w, i) => [w, OMEGA.mid.r1[i]]) },
                ]}
                width={560}
                height={300}
                xLabel="guidance scale ω"
                yLabel="R@1 ↑"
                xTicks={OMEGA.w}
                hlines={[{ y: 0.513, color: C.real, label: "ground-truth ceiling", anchor: "left" }]}
                right={46}
              />
              <Caption>
                R-precision is the share of clips whose caption is the nearest of 32{" "}
                <Cite k="humanml3d" />. From ω 3.5 upwards mid stays within 6 % of the ceiling, and is closest at ω 6.5 at 2.9 % below it. Retrieval
                replicates to ± {REPLICATION.r1sd}, so the variation across this sweep is small.
              </Caption>
            </Stage>
          </Reveal>
        </div>
      </Section>

      <Section
        n="03"
        title="Operating point"
        sub="The guidance scale swept again on the held-out validation split, at 800 ODE steps."
      >
        <Reveal>
          <Table
            head={["Split", "ω", "Seed", "ODE steps", "FID", "R@1 ↑"]}
            align={["left", "right", "right", "right", "right", "right"]}
            rows={[
              ...VAL_SELECTION.ode800.map((r) => ({
                _hi: r.w === VAL_SELECTION.kept && r.seed === 0,
                cells: ["validation, 1,362 clips", r.w, r.seed, 800, r.fid.toFixed(4), r.r1.toFixed(4)],
              })),
              { _dim: true, cells: ["test, 4,096 clips", 5.5, 0, 800, "0.4354", "0.5063"] },
              { _dim: true, cells: ["test, 4,096 clips", 6.5, 0, 800, "0.4288", "0.5071"] },
            ]}
          />
          <Caption>
            The validation split does not resolve the guidance scale. Two seeds at ω 6.5 give 0.5997 and
            0.4586, a range of {VAL_SELECTION.seedRange.toFixed(3)}, while the whole spread across ω 5.5,
            6.5 and 7.5 is {VAL_SELECTION.omegaSpread.toFixed(3)}. At 1,362 clips the repeat spread is
            about five times the ±{VAL_SELECTION.testSd} measured on the full test split, which is
            steeper than the square root of the sample-size ratio. On test the two candidates differ by
            0.007, also inside the repeat spread. We keep ω {VAL_SELECTION.kept} and report the test
            value at it; ω 5.5 would give 0.4354. The eight-point validation sweep at 200 steps agrees,
            placing the minimum at 5.5 with the three central cells within 0.08 of each other.
          </Caption>
        </Reveal>
      </Section>

      <Section n="04" title="Sampling steps" sub="The number of ODE steps controls how accurately the flow is integrated. It requires no retraining, only compute at inference.">
        <Reveal>
          <Stage label="FID vs ODE steps" note="mid · ω 6.5 · log–log">
            <LinePlot
              series={[
                {
                  label: "mid",
                  color: C.mid,
                  points: STEPS.s.map((s, i) => [s, STEPS.fid[i]]),
                  err: err(STEPS.s.map((s, i) => [s, STEPS.fid[i]])),
                  values: true,
                  area: true,
                },
              ]}
              width={720}
              height={320}
              xLog
              yLog
              xLabel="ODE integration steps"
              yLabel="FID ↓"
              xTicks={STEPS.s}
              valueDigits={3}
              right={40}
            />
            <Caption>
              FID falls by about half from 100 to 800 steps and then flattens. The values from 800
              onwards, 0.429, 0.410 and 0.424, lie within the ± {REPLICATION.sd} replication spread of
              each other, so we report the value at {QUOTED.steps} steps. Since the curve has converged,
              the remaining distance to the published model does not come from the integration.
            </Caption>
          </Stage>
        </Reveal>

        <div className="mt-6">
          <Reveal delay={70}>
            <h3 className="display mb-1 text-[17px] text-[var(--text)]">Full metrics</h3>
            <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
              The complete metric set across the step sweep, each model at the guidance scale it is reported at.
              The first row is real motion scored against real motion, which is the best value each
              column can take.
            </p>
            <Table
              head={["System", "ω", "ODE steps", "FID ↓", "R@1 ↑", "R@2 ↑", "R@3 ↑", "MM-dist ↓", "Diversity →", "MModality"]}
              align={["left", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}
              rows={STEP_METRICS.map((r) => ({
                _hi: r.hi,
                _dim: r.ref,
                cells: [
                  r.model,
                  r.w ?? "—",
                  r.steps ?? "—",
                  ...METRICS.map((m) => cell(r, m)),
                ],
              }))}
            />
            <Caption>
              Best value per column is marked within each model. Retrieval and multimodal distance
              improve alongside FID up to 800 steps and then stop. Base was swept over all eight
              guidance values at both step counts, and its best result is 8.05 at 200 steps against
              8.08 at 1600, so the extra integration that is worth a factor of two for mid does nothing
              for it. Four separate evaluations at ω 6.5 and 200 steps gave {REPLICATION.fid[0]},{" "}
              {REPLICATION.fid[1]}, {REPLICATION.fid[2]} and {REPLICATION.fid[3]}, which is the spread
              quoted throughout this page.
            </Caption>
          </Reveal>
        </div>

      </Section>

      <Section n="05" title="Scale" sub="The two models we trained and the published one, by parameter count.">
        <Reveal>
          <Stage label="FID vs parameters" note="each system at its reported ω · log–log">
            <LinePlot
              series={[{ label: "", color: C.muted, points: SCALE.map((s) => [s.params, s.fid]), width: 1.4, dot: 0, dashed: true }]}
              width={720}
              height={330}
              xLog
              yLog
              xLabel="parameters (millions)"
              yLabel="FID ↓"
              xTicks={[25, 112, 460]}
              labelSeries={false}
              markers={SCALE.map((s) => ({ x: s.params, y: s.fid, label: s.label, color: s.color }))}
              hlines={[{ y: HEADLINE.gt.fid, color: C.real, label: "reference floor", anchor: "left" }]}
              right={30}
            />
            <Caption>
              Three points, two of which are ours. From base to mid, a 4.5× increase in parameters
              corresponds to a 13× lower FID. These two runs also differ in {SCALING.confounds}, so the
              ratio cannot be attributed to parameter count alone. The published work does not state the
              ODE step count its FID was measured at. Across our own sweep FID ranges from 0.87 at 100
              steps to 0.41 at 1600, so the published point's height on this axis is uncertain by about
              that factor. The dashed line joins the three points and is not a fit.
            </Caption>
          </Stage>
        </Reveal>
      </Section>

      <Section n="06" title="Generated motion" sub="Each entry is one held-out caption, showing the capture and the model sample on a shared clock and orientation.">
        <Reveal>
          <p className="mb-4 max-w-3xl text-[12.5px] leading-relaxed text-[var(--muted)]">{SELECTION.paired}</p>
        </Reveal>
        <GeneratedSection />

        <div className="mt-5">
          <Reveal>
            <Stage label="jerk distribution" note="paired clips only · log scale">
              <Violin
                groups={[
                  { label: `real  (n ${JERK_DIST.realPaired.length})`, color: C.real, values: JERK_DIST.realPaired },
                  { label: `generated  (n ${JERK_DIST.genPaired.length})`, color: C.mid, values: JERK_DIST.genPaired },
                ]}
                width={640}
                height={330}
                yLog
                yLabel="jerk"
              />
              <Caption>
Jerk is the mean over joints and frames of{" "}
                {"‖x_t − 3x_{t−1} + 3x_{t−2} − x_{t−3}‖"}, in metres per frame³ at 20 fps. The nine captions above as two distributions. Only the paired clips are included, since
                the free prompts elsewhere on this page have no reference capture. The medians of
                the two distributions are 0.0057 and 0.0372. Taking the ratio within each pair first, as
                in the figure above, gives a median of 4.3×; dividing each clip's jerk by its mean
                per-frame joint displacement, which accounts for faster motion producing larger third
                differences, leaves that median unchanged at 4.3×. Smoothness is the clearest difference between generated
                and real motion in these measurements, and it is not reflected in FID.
              </Caption>
            </Stage>
          </Reveal>
        </div>

        <div className="mt-4">
          <Reveal delay={80}>
            <h3 className="display mb-1 text-[17px] text-[var(--text)]">Rotation and translation statistics</h3>
            <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
Quantities measured on the generated state directly, without the evaluator.
            </p>
            <Table
              head={["Quantity", "Generated", "Real", "Unit"]}
              align={["left", "right", "right", "left"]}
              rows={FORENSICS.map((f) => ({ cells: [f.q, f.gen, f.real, f.unit || "—"] }))}
            />
            <Caption>
              Measured at the operating point over 256 test captions against the captures for the same
              captions. Joint rotations in the samples are about 20 % faster than in the captures, which
              is consistent with the difference in smoothness above. Mean root travel is within 6 % of
              real, and the single furthest-travelling sample goes further than any capture in the set.
              Quaternion norm is included as a check that no sample leaves the manifold.
            </Caption>
          </Reveal>
        </div>

        <div className="mt-6">
          <Reveal delay={90}>
            <h3 className="display mb-1 text-[17px] text-[var(--text)]">Smoothness across sampling settings</h3>
            <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
              Whether the roughness above is integration error or a property of the learned field. Both
              panels are {JERK_SWEEP.n} test captions per point, with the band showing the quartiles
              across clips and the line the median of the captures for the same captions.
            </p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <Stage label="jerk vs ODE steps" note="mid · ω 6.5 · log–log">
                <LinePlot
                  series={[{
                    label: "generated",
                    color: C.mid,
                    points: JERK_SWEEP.steps.s.map((x, i) => [x, JERK_SWEEP.steps.med[i]]),
                    err: JERK_SWEEP.steps.s.map((x, i) => [x, JERK_SWEEP.steps.q1[i], JERK_SWEEP.steps.q3[i]]),
                    area: true,
                  }]}
                  width={560}
                  height={290}
                  xLog
                  yLog
                  xLabel="ODE integration steps"
                  yLabel="jerk"
                  xTicks={JERK_SWEEP.steps.s}
                  hlines={[{ y: JERK_SWEEP.realMedian, color: C.real, label: "real motion", anchor: "left" }]}
                  right={46}
                />
                <Caption>
                  Sixteen times more integration moves the median from 0.0275 to 0.0236, about a tenth,
                  while FID halves over the same range. The samples remain near seven times rougher than
                  the captures at every step count. The smoothness gap therefore reflects the learned
                  velocity field; integration accuracy does not control it.
                </Caption>
              </Stage>
              <Stage label="jerk vs guidance" note="mid · 200 ODE steps · log y">
                <LinePlot
                  series={[{
                    label: "generated",
                    color: C.mid,
                    points: JERK_SWEEP.omega.w.map((x, i) => [x, JERK_SWEEP.omega.med[i]]),
                    err: JERK_SWEEP.omega.w.map((x, i) => [x, JERK_SWEEP.omega.q1[i], JERK_SWEEP.omega.q3[i]]),
                    area: true,
                  }]}
                  width={560}
                  height={290}
                  yLog
                  xLabel="guidance scale ω"
                  yLabel="jerk"
                  xTicks={JERK_SWEEP.omega.w}
                  hlines={[{ y: JERK_SWEEP.realMedian, color: C.real, label: "real motion", anchor: "left" }]}
                  right={46}
                />
                <Caption>
                  Weak guidance is the roughest setting, at 0.0342 for ω 2.5. From ω 4.5 upwards the
                  median is flat to within 0.002, so ω 6.5 lies in a flat region on this measure as well
                  as on FID.
                </Caption>
              </Stage>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section n="07" title="Free generation" sub="Ten prompts written for this report, each the seed-0 draw. The count beside each is the number of the 14,616 training clips whose captions contain every content word of the prompt.">
        <Reveal>
          <MotionGallery items={promptItems} stageLabel="prompted generation" note="mid · ω 6.5 · 800 ODE steps" height={430} align stats />
        </Reveal>
      </Section>

      <Section n="08" title="Seed variation" sub="How much the model varies between draws of the same prompt, which sets the scale for reading the constraint results below.">
            <Reveal>
              <Stage label="generation consistency" note="one prompt · 20 seeds · jerk · log scale">
                <Violin
                  groups={[{ label: `“${CONSISTENCY.prompt}”`, color: C.mid, values: CONSISTENCY.jerk }]}
                  width={620}
                  height={300}
                  yLog
                  yLabel="jerk"
                  refBand={{ ...CONSISTENCY.gtBand, line: CONSISTENCY.gtBand.med, label: "real walking (IQR)" }}
                />
                <Caption>
                  The same prompt sampled from 20 seeds. Eighteen fall in a narrow band around 0.011,
                  roughly three times the median of real walking. Two, at 0.041 and 0.054, are four to
                  five times the others, so about one draw in ten is noticeably rougher. This variation
                  also affects the constraint measurements below.
                </Caption>
              </Stage>
            </Reveal>
      </Section>

      <Section n="09" title="Constraints" sub="Applying a joint limit during sampling.">
        <ConstraintsSection />
      </Section>

      <References />
    </Page>
  );
}

// Best value per metric, computed within each model so the two are not ranked
// against each other. Diversity and multimodality have no direction and are
// left unmarked.
const METRICS = [
  { key: "fid", dir: "min", d: 4 },
  { key: "r1", dir: "max", d: 4 },
  { key: "r2", dir: "max", d: 4 },
  { key: "r3", dir: "max", d: 4 },
  { key: "mm", dir: "min", d: 3 },
  { key: "div", dir: null, d: 3 },
  { key: "mmod", dir: null, d: 3 },
];

const BEST = (() => {
  const out = {};
  for (const r of STEP_METRICS) {
    if (r.ref) continue;
    out[r.model] ??= {};
    for (const m of METRICS) {
      if (!m.dir || r[m.key] == null) continue;
      const cur = out[r.model][m.key];
      const better = cur == null || (m.dir === "min" ? r[m.key] < cur : r[m.key] > cur);
      if (better) out[r.model][m.key] = r[m.key];
    }
  }
  return out;
})();

function cell(r, m) {
  const v = r[m.key];
  if (v == null) return "—";
  const text = v.toFixed(m.d);
  const best = !r.ref && m.dir && v === BEST[r.model]?.[m.key];
  return best ? <span style={{ color: "var(--signal)", fontWeight: 700 }}>{text}</span> : text;
}
