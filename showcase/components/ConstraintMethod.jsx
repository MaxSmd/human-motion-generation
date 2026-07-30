// The constraint method, as implemented in src/rmg/flow/constraints.py.
//
// One projector does both jobs on the site: a pin is the degenerate range
// min == max. It acts on the bend angle at a joint, which is a function of a
// single quaternion, so the projection is closed-form and exact.

import { Formula, V, T, O, Sub, Sup, Frac } from "./Formula";
import { Cite } from "./References";

export default function ConstraintMethod() {
  return (
    <div>
      <div className="mx-auto max-w-3xl text-[14px] leading-[1.75] text-[var(--muted)]">
        <p>
          The constrained quantity is the <span className="text-[var(--text)]">bend angle</span> at a
          joint, defined as the angle between the bone arriving at the joint and the bone leaving it,
          and equal to zero when the limb is straight. On this manifold it depends on a single
          quaternion, so it can be corrected in isolation.
        </p>
      </div>

      <div className="mx-auto max-w-3xl">
        <Formula
          n="8"
          note={
            <>
              The bend at joint <V>j</V>, with <V>u</V> the incoming rest bone, <V>v</V> the outgoing
              rest bone and <V>q</V> the quaternion orienting the outgoing bone. Forward kinematics here
              rotates each bone by the quaternion of the joint it leads to, so the bend at <V>j</V> is
              governed by the quaternion of its child.
            </>
          }
        >
          <V>α</V>
          <O>=</O>
          <O>∠</O>
          <T>(</T>
          <V>u</V>
          <O>,</O>
          <V>d</V>
          <T>)</T>
          <O>,</O>
          <V>d</V>
          <O>=</O>
          <V>q</V>
          <O>·</O>
          <V>v</V>
        </Formula>

        <Formula
          n="9"
          note={
            <>
              The projection. The bend is clamped into its feasible range and the outgoing bone is
              rotated by the difference, within the plane the bend already lies in. A fixed angle is the
              case <V>α</V>
              <Sub>min</Sub> = <V>α</V>
              <Sub>max</Sub>, so the same projector covers both.
            </>
          }
        >
          <V>β</V>
          <O>=</O>
          <T>clamp(</T>
          <V>α</V>
          <O>,</O>
          <V>α</V>
          <Sub>min</Sub>
          <O>,</O>
          <V>α</V>
          <Sub>max</Sub>
          <T>)</T>
          <O>,</O>
          <V>q</V>
          <O>←</O>
          <V>Δ</V>
          <O>⊗</O>
          <V>q</V>
          <O>,</O>
          <V>Δ</V>
          <O>=</O>
          <T>exp</T>
          <T>(</T>
          <Frac num={<T>1</T>} den={<T>2</T>} />
          <T>(</T>
          <V>β</V>
          <O>−</O>
          <V>α</V>
          <T>)</T>
          <V>n̂</V>
          <T>)</T>
          <O>,</O>
          <V>n̂</V>
          <O>=</O>
          <Frac
            num={
              <>
                <V>u</V>
                <O>×</O>
                <V>d</V>
              </>
            }
            den={
              <>
                <T>‖</T>
                <V>u</V>
                <O>×</O>
                <V>d</V>
                <T>‖</T>
              </>
            }
          />
        </Formula>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card head="Exactness">
The correction is available in closed form, so after each step the bend lies inside the range
          to the precision of the arithmetic. Over all 4096 clips of the test split the largest
          deviation from the limit is 8.7 × 10⁻⁵ degrees.
        </Card>
        <Card head="Locality">
<V>Δ</V> rotates the outgoing bone within the plane the bend already occupies, so the twist
          about the bone and the direction of the bend are preserved. Only the magnitude of the bend
          changes.
        </Card>
        <Card head="Applied during sampling">
The projector runs after every Euler step and acts on one factor of the product. No
          term is added to the loss and no weight is trained, so the same checkpoint is used for
          constrained and unconstrained generation.
        </Card>
      </div>

      <div className="surface mt-4 p-5">
        <div className="label mb-3">relation to RePaint</div>
        <div className="space-y-3 text-[13px] leading-relaxed text-[var(--muted)]">
          <p>
            RePaint <Cite k="repaint" /> imposes known image content during sampling by replacing the
            known region with a noised copy of it after every reverse step. The replaced region differs
            from what the model would have generated there, so the two regions can disagree at the
            boundary. RePaint addresses this by renoising and resampling the same interval several
            times, which lets the model adjust the generated region to the imposed one.
          </p>
          <p>
            Our setting differs in two respects. The constrained quantity depends on a single
            independent factor of{" "}
            <span className="text-[var(--text)]">
              ℝ<sup>3</sup> × (S<sup>3</sup>)<sup>22</sup>
            </span>
            , so the projection stays on the manifold and no renoising is required for validity. The
            correction is a rotation of the constrained joint, so no external content is inserted and
            there is no boundary to blend.
          </p>
          <p>
            The remaining similarity is that the rest of the body continues to be integrated from a
            state it did not produce, with no opportunity to adjust to the correction. This is where a
            conflict between the constraint and the model would be expected to appear. On the full split
            it appears as a drop in retrieval from 0.498 to 0.481 with the knee clamped. In smoothness no
            effect is resolved, with the design sensitive only to effects of about 1.9× or larger. A
            RePaint-style resampling step is the natural method to try against the retrieval cost.
          </p>
        </div>
      </div>
    </div>
  );
}

function Card({ head, children }) {
  return (
    <div className="surface flex h-full flex-col p-4">
      <div className="label mb-2">{head}</div>
      <p className="text-[12.5px] leading-relaxed text-[var(--muted)]">{children}</p>
    </div>
  );
}
