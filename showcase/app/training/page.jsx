import { Page, PageHead, Section, Caption } from "@/components/ui";
import { Reveal, Stage } from "@/components/motion";
import { LossCurves, ScheduleCurves } from "@/components/TrainingCurves";
import { LinePlot } from "@/components/charts";
import { monotoneTrend } from "@/lib/interp";
import References, { Cite } from "@/components/References";
import { MODELS, RECIPE, CKPT_SWEEP, DATASET, REPLICATION, C } from "@/lib/results";

export const metadata = { title: "RMG · Training" };

// Rows that differ per model; each pulls straight from MODELS.
const PER_MODEL = [
  ["parameters", "params"],
  ["width / depth", "arch"],
  ["training steps", "steps"],
  ["effective batch", "batch"],
  ["warmup", "warmup"],
  ["precision", "precision"],
  ["antipodal quotient", (m) => (m.antipodal == null ? "not stated" : m.antipodal ? "yes" : "no")],
  ["wall-clock", "time"],
  ["throughput", "throughput"],
];

export default function TrainingPage() {
  return (
    <Page>
      <PageHead
        kicker="Training"
        title="Training"
        lede="Two models were trained with the same recipe at different scales. The published configuration is included for comparison and was not trained by us."
      />

      <Section n="01" title="Data" sub={DATASET.blurb}>
        <Reveal>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.6fr_1fr]">
            <div className="surface p-5">
              <div className="label mb-3">{DATASET.name}</div>
              <div className="grid grid-cols-3 gap-x-4 gap-y-4">
                {DATASET.stats.map(([k, v]) => (
                  <div key={k}>
                    <div className="display text-[22px] leading-none text-[var(--text)]">{v}</div>
                    <div className="mt-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">{k}</div>
                  </div>
                ))}
              </div>
            </div>
            <div className="surface flex flex-col p-5">
              <div className="label mb-3">split</div>
              <div className="flex-1 space-y-2.5">
                {DATASET.splits.map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between border-b border-[var(--hairline)] pb-2">
                    <span className="text-[12.5px] text-[var(--muted)]">{k}</span>
                    <span className="font-mono text-[14px] text-[var(--text)]">{v}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[11px] leading-snug text-[var(--muted)]">
                {DATASET.augment}. HumanML3D <Cite k="humanml3d" />, over AMASS captures <Cite k="amass" />.
              </p>
            </div>
          </div>
        </Reveal>
      </Section>

      <Section n="02" title="Configurations">
        <Reveal>
          <ConfigTable />
        </Reveal>
        <Caption>
The rows below the rule are identical across all three and match the published configuration{" "}
          <Cite k="rmg" />. The two models we trained differ in width, depth and training length, and
          also in precision and the antipodal treatment, so the comparison between them reflects more
          than parameter count.
        </Caption>
      </Section>

      <Section n="03" title="Loss">
        <Reveal>
          <LossCurves />
        </Reveal>
        <div className="mt-4">
          <Reveal>
            <ScheduleCurves />
          </Reveal>
        </div>
      </Section>

      <Section
        n="04"
        title="Sample quality during training"
        sub="Six checkpoints of the mid run, evaluated at ω 6.5 and 800 ODE steps on the full split."
      >
        <Reveal>
          <FidVsStep />
        </Reveal>
        <div className="mt-4">
          <Reveal delay={70}>
            <Retrieval />
          </Reveal>
        </div>
      </Section>

      <Section n="05" title="Stability">
        <Reveal>
          <Stability />
        </Reveal>
      </Section>

      <References />
    </Page>
  );
}

function ConfigTable() {
  const cols = [MODELS.base, MODELS.mid, MODELS.paper];
  const cell = (m, spec) => (typeof spec === "function" ? spec(m) : m[spec]);

  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--hairline)]">
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr className="bg-black/30">
            <th className="px-3 py-2.5 text-left font-mono text-[9.5px] uppercase tracking-[0.16em] text-[var(--muted)]" />
            {cols.map((m) => (
              <th key={m.key} className="px-3 py-2.5 text-left align-bottom">
                <div className="font-mono text-[9.5px] uppercase tracking-[0.16em]" style={{ color: m.color }}>
                  {m.name}
                </div>
                <div className="mt-1 font-mono text-[9px] normal-case tracking-normal text-[var(--muted)]">
                  {m.key === "paper" ? "published configuration" : m.run}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PER_MODEL.map(([label, spec]) => (
            <tr key={label} className="border-t border-[var(--hairline)]">
              <td className="px-3 py-2 text-[var(--muted)]">{label}</td>
              {cols.map((m) => (
                <td
                  key={m.key}
                  className="px-3 py-2 font-mono"
                  style={{ color: m.key === "paper" ? "var(--muted)" : "var(--text)" }}
                >
                  {cell(m, spec)}
                </td>
              ))}
            </tr>
          ))}

          <tr className="border-t border-[var(--hairline-strong)]">
            <td
              colSpan={4}
              className="bg-black/20 px-3 py-1.5 font-mono text-[9.5px] uppercase tracking-[0.16em] text-[var(--muted)]"
            >
              shared across all three
            </td>
          </tr>
          {RECIPE.map(([label, value]) => (
            <tr key={label} className="border-t border-[var(--hairline)]">
              <td className="px-3 py-2 text-[var(--muted)]">{label}</td>
              <td colSpan={3} className="px-3 py-2 font-mono text-[var(--text)]">
                {value}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FidVsStep() {
  // The trend the FID would follow if the 150 k checkpoint (inside the bf16
  // window) were dropped: a monotone guide through the other five, in log-FID.
  const kept = CKPT_SWEEP.step
    .map((s, i) => [s, CKPT_SWEEP.fid[i]])
    .filter(([s]) => s !== 150000);
  const trendLog = monotoneTrend(kept.map((p) => p[0]), kept.map((p) => Math.log10(p[1])), 56);
  const trend = trendLog.map(([x, y]) => [x, 10 ** y]);

  const series = [
    { label: "", color: C.muted, points: trend, dashed: true, width: 1.4, dot: 0 },
    {
      label: "mid",
      color: C.mid,
      points: CKPT_SWEEP.step.map((s, i) => [s, CKPT_SWEEP.fid[i]]),
      values: true,
    },
  ];

  return (
    <Stage label="FID vs training step" note="full split · log scale">
      <LinePlot
        series={series}
        width={720}
        height={330}
        yLog
        xLabel="training step"
        yLabel="FID ↓"
        xTicks={CKPT_SWEEP.step}
        vlines={[{ x: CKPT_SWEEP.precisionSwitch, label: "bf16 → tf32", color: C.warn }]}
        annotations={[{ x: 150000, y: 12.0416, text: "150 k", dy: -16, color: C.warn }]}
        valueDigits={2}
        right={44}
      />
      <Caption>
FID decreases with training except at 150 k, which lies above the trend through the other five
        checkpoints (dashed). That checkpoint falls inside the bf16 window that ended at 160 k, and its
        training loss is lower than that of any checkpoint after the switch. After training continues in
        tf32 the curve returns to the trend and reaches 0.43 at 300 k. Each point is a single evaluation
        pass, repeating to ± {REPLICATION.sd}.
      </Caption>
    </Stage>
  );
}

function Retrieval() {
  return (
    <Stage label="retrieval over the same checkpoints" note="R@1 ↑">
      <LinePlot
        series={[{ label: "mid", color: C.mid, points: CKPT_SWEEP.step.map((s, i) => [s, CKPT_SWEEP.r1[i]]), values: true }]}
        width={720}
        height={250}
        xLabel="training step"
        yLabel="R@1 ↑"
        xTicks={CKPT_SWEEP.step}
        hlines={[{ y: 0.513, color: C.real, label: "ground-truth ceiling 0.513", anchor: "left" }]}
        vlines={[{ x: CKPT_SWEEP.precisionSwitch, color: C.warn }]}
        valueDigits={3}
        yPad={0.14}
        right={44}
      />
      <Caption>
Retrieval also drops at 150 k and recovers to within 1 % of the ceiling by 300 k. It reaches
        its final level before FID stops improving.
      </Caption>
    </Stage>
  );
}

const STABILITY = [
  {
    n: "1",
    head: "θ / sin θ near antipodal",
    body: "The geodesic target velocity contains a θ ⁄ sin θ factor that diverges as two quaternions approach opposite poles. In bf16 such a batch produces Inf or NaN, which then propagates to the weights, the EMA and the optimizer moments within one step.",
  },
  {
    n: "2",
    head: "align into one hemisphere",
    body: "Before each path is drawn, the data quaternion is replaced by whichever of its two representatives shares a hemisphere with the prior sample, which gives θ ≤ π⁄2 and makes the singularity unreachable. This changes the training target, so it can only be applied from the start of a run, and base does not have it.",
  },
  {
    n: "3",
    head: "reject non-finite steps",
    body: "Any step whose loss or gradient is non-finite, or exceeds 5× and 25× the running scale, is skipped, with no optimizer step and no EMA update. A single bad batch then costs one step.",
  },
  {
    n: "4",
    head: "switch to tf32",
    body: "bf16 remained unstable through 160 k, so the run was continued in tf32. The final checkpoint lies entirely after this switch, and the FID curve above shows the effect.",
  },
];

function Stability() {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      {STABILITY.map((s, i) => (
        <Reveal key={s.n} delay={i * 60}>
          <div className="surface flex h-full flex-col p-4">
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <span className="text-[13.5px] font-semibold text-[var(--text)]">{s.head}</span>
              <span className="label shrink-0">{s.n}</span>
            </div>
            <p className="text-[12px] leading-relaxed text-[var(--muted)]">{s.body}</p>
          </div>
        </Reveal>
      ))}
    </div>
  );
}
