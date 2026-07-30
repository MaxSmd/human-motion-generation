"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useRef } from "react";
import { Page, Section, Table, Caption } from "@/components/ui";
import { Reveal, Stage, useScrollProgress, smooth, lerp } from "@/components/motion";
import { Formula, V, T, O, Sub, Sup, Frac } from "@/components/Formula";
import { LogBars } from "@/components/charts";
import Viewport from "@/components/Viewport";
import References, { Cite, Colophon } from "@/components/References";
import { HEADLINE, QUOTED, REPLICATION, LIMITS, C } from "@/lib/results";

const SkeletonPlayer = dynamic(() => import("@/components/SkeletonPlayer"), {
  ssr: false,
  loading: () => <div className="viewport" style={{ height: 410 }} />,
});
const Architecture = dynamic(() => import("@/components/Architecture"), {
  ssr: false,
  loading: () => <div className="viewport" style={{ height: 460 }} />,
});

// Each row carries the inference budget it was measured at, because FID moves
// with it: quoting a 800-step number beside a 200-step one without saying so
// would make the gap look smaller than it is.
const GAP_ROWS = [
  { label: "Reference floor", sub: "real motion against real motion", v: 0.0019, color: C.real },
  { label: "RMG-main, published", sub: "460 M · 600 k train · ODE n/d", v: 0.043, color: C.paper },
  { label: "RMG-mid, this work", sub: "112 M · 300 k train · 800 ODE", v: 0.4288, color: C.mid, hi: true },
  { label: "RMG-base, this work", sub: "24.7 M · 150 k train · 200 ODE", v: 8.0493, color: C.base },
];

const num = (v) => (v == null ? "—" : v);

export default function Overview() {
  return (
    <Page>
      <Hero />

      <Section title="Abstract" center>
        <Reveal>
          <p className="mx-auto max-w-[52rem] text-center text-[15.5px] leading-[1.8] text-[var(--text)]">
            A pose consists of a root translation and 22 joint rotations, so it lives on the product
            manifold <span className="text-[var(--signal)]">ℝ³ × (S³)²²</span>. We reproduce Riemannian
            conditional flow matching for text-to-motion <Cite k="rmg" /> at two scales, 25 M and 112 M
            parameters, and evaluate on the HumanML3D test split <Cite k="humanml3d" /> using a harness
            calibrated against published reference values. The 112 M model reaches{" "}
            <span className="text-[var(--signal)]">FID 0.43 ± 0.03</span> with retrieval at 98 % of the
            ground-truth ceiling, about ten times higher than the published 460 M result. Since the
            joints are independent factors of the manifold, a joint limit can be imposed by projecting
            one factor after each integration step. Applied to every clip of the test split, this holds
            the limit to within 10⁻⁴ degrees without retraining, and retrieval falls from 0.498 to 0.481
            with the knee clamped. Over 36 seed-paired renders we resolve no effect on smoothness;
            the design is sensitive to effects of about 1.9× or larger.
          </p>
        </Reveal>
      </Section>

      <Section n="01" title="Introduction" center>
        <Reveal>
          <div className="mx-auto max-w-3xl space-y-4 text-[14px] leading-[1.75] text-[var(--muted)]">
            <p>
              Text-to-motion models are commonly trained on a flat 263-dimensional feature vector that
              combines positions, velocities, foot contacts and rotations. This representation does not
              encode the constraint that a rotation must remain a rotation, so validity is restored
              afterwards by renormalisation, and any quantity that should be held fixed has to be learned
              from data.
            </p>
            <p>
              A unit quaternion is a point on the three-sphere, and a pose is 22 such points together
              with a translation. Generating on this manifold keeps every intermediate state of the
              solver a valid pose. It also makes each joint an independent factor, which allows a limit
              on a single joint to be imposed by projection during sampling.
            </p>
            <p>
              This report describes our reproduction at two scales, the evaluation setup, and a study of
              sampling-time joint constraints.
            </p>
          </div>
        </Reveal>
      </Section>

      <Section n="02" title="Method" sub="Conditional flow matching on the manifold." center>
        <Reveal>
          <div className="mx-auto max-w-3xl text-[14px] leading-[1.75] text-[var(--muted)]">
            A motion is a sequence of points on <T>M</T>. Training draws a noise pose and a data pose,
            connects them by a geodesic, and regresses the network onto the velocity of that path.
            Sampling integrates the learned velocity from noise to data in geodesic steps.
          </div>
        </Reveal>

        <div className="mx-auto mt-6 max-w-3xl">
          <Reveal>
            <Formula
              n="1"
              note={
                <>
The state space: a translation and 22 unit quaternions, 91 ambient dimensions with
                  69 intrinsic degrees of freedom.
                </>
              }
            >
              <V>M</V>
              <O>=</O>
              <T>ℝ</T>
              <Sup>3</Sup>
              <O>×</O>
              <T>(</T>
              <V>S</V>
              <Sup>3</Sup>
              <T>)</T>
              <Sup>22</Sup>
              <O>,</O>
              <V>x</V>
              <O>=</O>
              <T>(</T>
              <V>p</V>
              <O>,</O>
              <V>q</V>
              <Sub>1</Sub>
              <O>,…,</O>
              <V>q</V>
              <Sub>22</Sub>
              <T>)</T>
            </Formula>
          </Reveal>

          <Reveal>
            <Formula
              n="2"
              note={
                <>
                  The conditional path between a prior sample <V>x</V>
                  <Sub>0</Sub> and a data pose <V>x</V>
                  <Sub>1</Sub>, in slerp form, with <V>θ</V> the angle between them.
                </>
              }
            >
              <V>γ</V>
              <T>(</T>
              <V>t</V>
              <T>)</T>
              <O>=</O>
              <Frac
                num={
                  <>
                    <T>sin</T>
                    <T>(</T>
                    <T>1−</T>
                    <V>t</V>
                    <T>)</T>
                    <V>θ</V>
                  </>
                }
                den={
                  <>
                    <T>sin</T> <V>θ</V>
                  </>
                }
              />
              <V>x</V>
              <Sub>0</Sub>
              <O>+</O>
              <Frac
                num={
                  <>
                    <T>sin</T> <V>tθ</V>
                  </>
                }
                den={
                  <>
                    <T>sin</T> <V>θ</V>
                  </>
                }
              />
              <V>x</V>
              <Sub>1</Sub>
            </Formula>
          </Reveal>

          <Reveal>
            <Formula
              n="3"
              note={
                <>
                  The regression target, the velocity of that path. It equals Log
                  <Sub>γ(t)</Sub>(<V>x</V>
                  <Sub>1</Sub>) ⁄ (1−<V>t</V>). Appendix A of the published work <Cite k="rmg" /> gives
                  this with a sine in place of the cosine. That form is not tangent to the sphere at{" "}
                  <V>t</V> = 0, and our implementation uses the cosine.
                </>
              }
            >
              <V>γ̇</V>
              <T>(</T>
              <V>t</V>
              <T>)</T>
              <O>=</O>
              <Frac
                num={<V>θ</V>}
                den={
                  <>
                    <T>sin</T> <V>θ</V>
                  </>
                }
              />
              <T>(</T>
              <O>−</O>
              <T>cos</T>
              <T>(</T>
              <T>1−</T>
              <V>t</V>
              <T>)</T>
              <V>θ</V>
              <O>·</O>
              <V>x</V>
              <Sub>0</Sub>
              <O>+</O>
              <T>cos</T>
              <T> </T>
              <V>tθ</V>
              <O>·</O>
              <V>x</V>
              <Sub>1</Sub>
              <T>)</T>
            </Formula>
          </Reveal>

          <Reveal>
            <Formula
              n="4"
              note={
                <>
                  The training objective, with <V>t</V> uniform on [<V>ε</V>, 1−<V>ε</V>] and <V>c</V> the
                  pooled caption. Before each path is drawn, <V>x</V>
                  <Sub>1</Sub> is replaced by whichever of ±<V>x</V>
                  <Sub>1</Sub> lies in the same hemisphere as <V>x</V>
                  <Sub>0</Sub>. Both represent the same rotation, and the choice keeps <V>θ</V> ≤ π⁄2, so
                  the <V>θ</V> ⁄ sin <V>θ</V> factor in (3) never diverges. This applies to the 112 M run
                  only; it changes the training target, so it can only be used from the start of a run.
                </>
              }
            >
              <V>ℒ</V>
              <O>=</O>
              <V>𝔼</V>
              <Sub>
                <V>t</V>,<V>x</V>
                <Sub>0</Sub>,<V>x</V>
                <Sub>1</Sub>
              </Sub>
              <O> </O>
              <T>‖</T>
              <V>v</V>
              <Sub>θ</Sub>
              <T>(</T>
              <V>γ</V>
              <T>(</T>
              <V>t</V>
              <T>),</T>
              <V>t</V>
              <T>,</T>
              <V>c</V>
              <T>)</T>
              <O>−</O>
              <V>γ̇</V>
              <T>(</T>
              <V>t</V>
              <T>)</T>
              <T>‖</T>
              <Sup>2</Sup>
            </Formula>
          </Reveal>

          <Reveal>
            <Formula
              n="5"
              note={
                <>
                  Classifier-free guidance is combined in the ambient space and then projected onto the
                  tangent space at <V>x</V>. The projection is linear, so projecting each branch first
                  gives the same result.
                </>
              }
            >
              <V>v</V>
              <O>=</O>
              <V>v</V>
              <Sub>θ</Sub>
              <T>(</T>
              <V>x</V>
              <T>,</T>
              <V>t</V>
              <T>,∅)</T>
              <O>+</O>
              <V>ω</V>
              <T>(</T>
              <V>v</V>
              <Sub>θ</Sub>
              <T>(</T>
              <V>x</V>
              <T>,</T>
              <V>t</V>
              <T>,</T>
              <V>c</V>
              <T>)</T>
              <O>−</O>
              <V>v</V>
              <Sub>θ</Sub>
              <T>(</T>
              <V>x</V>
              <T>,</T>
              <V>t</V>
              <T>,∅))</T>
              <O>,</O>
              <V>Π</V>
              <Sub>x</Sub>
              <V>v</V>
              <O>=</O>
              <V>v</V>
              <O>−</O>
              <T>⟨</T>
              <V>x</V>
              <T>,</T>
              <V>v</V>
              <T>⟩</T>
              <V>x</V>
            </Formula>
          </Reveal>

          <Reveal>
            <Formula
              n="6"
              note={
                <>
                  One Riemannian Euler step. The exponential map on the sphere is a rotation, so the
                  result stays on the manifold and no quaternion is renormalised.
                </>
              }
            >
              <V>x</V>
              <Sub>
                <V>t</V>+<V>h</V>
              </Sub>
              <O>=</O>
              <T>Exp</T>
              <Sub>
                <V>x</V>
              </Sub>
              <T>(</T>
              <V>h</V>
              <O>·</O>
              <V>Π</V>
              <Sub>x</Sub>
              <V>v</V>
              <T>)</T>
              <O>,</O>
              <T>Exp</T>
              <Sub>
                <V>x</V>
              </Sub>
              <T>(</T>
              <V>u</V>
              <T>)</T>
              <O>=</O>
              <T>cos‖</T>
              <V>u</V>
              <T>‖ </T>
              <V>x</V>
              <O>+</O>
              <T>sin‖</T>
              <V>u</V>
              <T>‖ </T>
              <Frac
                num={<V>u</V>}
                den={
                  <>
                    <T>‖</T>
                    <V>u</V>
                    <T>‖</T>
                  </>
                }
              />
            </Formula>
          </Reveal>

          <Reveal>
            <Formula
              n="7"
              note={
                <>
                  A joint limit is a projection onto the feasible set <V>C</V>
                  <Sub>j</Sub> of one factor, applied after each step of (6). The remaining factors and
                  the weights are unchanged.
                </>
              }
            >
              <V>x</V>
              <Sup>(j)</Sup>
              <O>←</O>
              <V>P</V>
              <Sub>
                <V>C</V>
                <Sub>j</Sub>
              </Sub>
              <T>(</T>
              <V>x</V>
              <Sup>(j)</Sup>
              <T>)</T>
            </Formula>
          </Reveal>
        </div>
      </Section>

      <Section
        n="03"
        title="Architecture"
        sub="The caption embedding and the flow time are combined into one modulation signal for ten transformer blocks. The final stages map the output back onto the manifold."
        center
      >
        <Reveal>
          <Viewport height={460} label="pipeline diagram">
            <Architecture />
          </Viewport>
        </Reveal>
      </Section>

      <Section n="04" title="Results" sub="Full HumanML3D test split, 4096 clips. Each model is evaluated at the guidance scale it is reported at." center>
        <Reveal>
          <Stage label="Fréchet distance, Guo motion features" note="log scale · lower is better">
            <LogBars rows={GAP_ROWS} digits={4} />
          </Stage>
        </Reveal>

        <div className="mt-4">
          <Reveal delay={80}>
            <Table
              head={["System", "Params", "Train steps", "ω", "ODE steps", "FID ↓", "R@1 ↑", "R@3 ↑", "MM-dist ↓", "Diversity →", "MModality"]}
              align={["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}
              rows={[
                { _dim: true, cells: ["Reference (real motion)", "—", "—", "—", "—", HEADLINE.gt.fid, HEADLINE.gt.r1, HEADLINE.gt.r3, HEADLINE.gt.mm, HEADLINE.gt.div, num(HEADLINE.gt.mmod)] },
                { cells: ["RMG-main, published", "460 M", "600 k", HEADLINE.paper.w, "n/d", HEADLINE.paper.fid, HEADLINE.paper.r1, "—", num(HEADLINE.paper.mm), HEADLINE.paper.div, num(HEADLINE.paper.mmod)] },
                { _hi: true, cells: ["RMG-mid, this work", "111.7 M", "300 k", HEADLINE.midBest.w, HEADLINE.midBest.steps, HEADLINE.midBest.fid, HEADLINE.midBest.r1, HEADLINE.midBest.r3, HEADLINE.midBest.mm, HEADLINE.midBest.div, num(HEADLINE.midBest.mmod)] },
                { cells: ["RMG-base, this work", "24.7 M", "150 k", HEADLINE.base.w, HEADLINE.base.steps, HEADLINE.base.fid, HEADLINE.base.r1, HEADLINE.base.r3, HEADLINE.base.mm, HEADLINE.base.div, num(HEADLINE.base.mmod)] },
              ]}
            />
          </Reveal>
          <Caption>
            Train steps and ODE steps are separate settings: the first is the length of training, the
            second the number of integration steps at sampling time. The published work does not state
            its ODE step count, so that entry is left as n/d. Our own FID varies by a factor of two
            across the step counts we swept, so the published FID and ours are not measured at a matched
            inference budget. For diversity, values closer to the reference 9.79 are better. Multimodality
            measures the spread of samples drawn for one caption and is undefined for the reference,
            which has one capture per caption. The published row contains only the metrics that work
            reports for its model. Retrieval on this data saturates at 0.513, so 0.507 corresponds to
            98 % of the attainable value. Our figures are single passes over the full split and repeat
            to ± {REPLICATION.sd}; the published figure is a mean over 20 replications. The full sweeps
            are on the{" "}
            <Link href="/results" className="text-[var(--signal)] hover:underline">results page</Link>.
          </Caption>
        </div>
      </Section>

      <Section n="05" title="Limitations" center>
        <Reveal>
          <div className="mx-auto grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-2">
            {LIMITS.map((l) => (
              <div key={l.head} className="surface p-4 text-left">
                <div className="label mb-2">{l.head}</div>
                <p className="text-[12.5px] leading-relaxed text-[var(--muted)]">{l.body}</p>
              </div>
            ))}
          </div>
        </Reveal>
      </Section>

      <Section n="06" title="Conclusion" center>
        <Reveal>
          <div className="mx-auto max-w-3xl space-y-4 text-[14px] leading-[1.75] text-[var(--muted)]">
            <p>
              Sampling on the manifold keeps every quaternion on the sphere, so no renormalisation is
              required at any point. Joint limits can be applied during sampling by projection. Over
              the full test split they were met to within 10⁻⁴ degrees on a model trained without
              constraints, at a cost of 0.017 in R@1 for a clamped knee.
            </p>
            <p>
              At 112 M parameters and half the published training budget the model reaches an FID about
              ten times higher than the published 460 M result. Our two configurations differ by 4.5× in
              parameters and 13× in FID, but they also differ in training length, precision and the
              antipodal treatment, so we treat this as an observation about two runs and do not
              extrapolate from it.
            </p>
            <p>
              Two experiments follow from this. The first is the same recipe at full scale and matched
              training length. The second is a matched model trained on the flat representation, which
              would allow the contribution of the geometry to be measured directly. For the constraints,
              a larger number of seeds would tighten the bound on the projection's effect on smoothness,
              which at 36 pairs stands at about 1.9×.
            </p>
          </div>
        </Reveal>
      </Section>

      <Colophon />
      <References />
    </Page>
  );
}

function Hero() {
  const ref = useRef(null);
  const p = useScrollProgress(ref);
  const t = smooth(Math.min(1, p / 0.55));

  return (
    <section ref={ref} className="mb-24 grid grid-cols-1 items-center gap-10 md:grid-cols-[1fr_470px]">
      <div>
        <Reveal dir="none" blur>
          <div className="label mb-4">Project report</div>
          <h1 className="display text-[40px] leading-[1.04] text-[var(--text)] md:text-[54px]">
            Generating motions on
            <span className="text-[var(--signal)]"> the Riemannian manifold</span>
          </h1>
        </Reveal>
        <Reveal delay={140}>
          <p className="mt-6 max-w-lg text-[15px] leading-[1.7] text-[var(--muted)]">
            A reproduction of text-conditioned motion generation on the product manifold ℝ³ × (S³)²²,
            at 25 M and 112 M parameters, with a study of joint constraints applied during sampling.
          </p>
        </Reveal>
      </div>

      <div
        style={{
          transform: `perspective(1200px) rotateX(${lerp(9, 0, t)}deg) scale(${lerp(0.93, 1, t)})`,
          opacity: lerp(0.55, 1, Math.min(1, t * 1.8)),
          transition: "transform .12s linear",
        }}
      >
        <div className="surface !p-3">
          <Viewport height={410}>
            <SkeletonPlayer url="/data/gt_waltz.npy" height={410} autoRotate controls={false} />
          </Viewport>
          <div className="mt-2.5 px-2 pb-1 text-center font-mono text-[10.5px] leading-relaxed text-[var(--muted)]">
            “a person is dancing the waltz, going in a counter-clockwise direction with the left arm out”
            <span className="mt-1 block text-[var(--real)]">ground-truth capture</span>
          </div>
        </div>
      </div>
    </section>
  );
}
