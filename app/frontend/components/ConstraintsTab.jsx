"use client";

// Frontend-only PLACEHOLDER for the V2 constraint editor. None of these
// controls are wired to the backend — they preview the constrained-sampler UI
// (joint pins / hinge limits / obstacles) described as out-of-scope in plan.md.

const JOINTS = [
  "pelvis", "spine", "neck", "head",
  "L_shoulder", "L_elbow", "L_wrist",
  "R_shoulder", "R_elbow", "R_wrist",
  "L_hip", "L_knee", "L_ankle",
  "R_hip", "R_knee", "R_ankle",
];

export default function ConstraintsTab() {
  return (
    <div className="relative">
      {/* preview banner */}
      <div className="surface mb-5 flex flex-wrap items-center justify-between gap-4 border-[var(--amber)]/25 p-5">
        <div className="flex items-center gap-4">
          <span className="grid h-10 w-10 place-items-center rounded-lg border border-[var(--amber)]/40 text-lg">
            🔒
          </span>
          <div>
            <div className="display text-lg font-bold text-white">
              Constraint editor — preview
            </div>
            <p className="mt-0.5 text-[12px] text-[var(--muted)]">
              A look at the planned V2 surface. Pin joints, clamp hinge ranges, and
              drop obstacles, then resample on the constrained manifold.
              <span className="text-[var(--amber)]"> Not yet connected to the backend.</span>
            </p>
          </div>
        </div>
        <span className="rounded-full border border-[var(--amber)]/40 bg-[var(--amber)]/10 px-3 py-1 text-[11px] font-semibold tracking-wider text-[var(--amber)]">
          V2 · ROADMAP
        </span>
      </div>

      {/* locked controls (visually dimmed) */}
      <div className="pointer-events-none select-none opacity-60">
        <div className="grid gap-5 lg:grid-cols-3">
          {/* joint pins */}
          <section className="surface p-5">
            <Header n="A" title="Joint pins" desc="lock joints to a target position" />
            <div className="mt-4 grid grid-cols-2 gap-1.5">
              {JOINTS.map((j, i) => (
                <span
                  key={j}
                  className={`flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-[11px] ${
                    i === 6 || i === 9
                      ? "border-[var(--signal)]/50 bg-[var(--signal-dim)] text-[var(--signal)]"
                      : "border-[var(--hairline)] text-slate-400"
                  }`}
                >
                  <span className="dot" style={{ color: i === 6 || i === 9 ? "var(--signal)" : "var(--muted)" }} />
                  {j}
                </span>
              ))}
            </div>
            <p className="label mt-4">2 joints pinned · drag in 3D to set targets</p>
          </section>

          {/* hinge limits */}
          <section className="surface p-5">
            <Header n="B" title="Hinge limits" desc="clamp per-joint rotation range" />
            <div className="mt-5 space-y-5">
              {[
                ["L_elbow", 12, 145],
                ["R_knee", 5, 160],
                ["neck", -40, 40],
              ].map(([name, lo, hi]) => (
                <div key={name}>
                  <div className="mb-1.5 flex justify-between text-[11px]">
                    <span className="text-slate-300">{name}</span>
                    <span className="font-mono text-[var(--signal)]">
                      {lo}° → {hi}°
                    </span>
                  </div>
                  <div className="relative h-1.5 rounded-full bg-[var(--hairline)]">
                    <span
                      className="absolute h-1.5 rounded-full bg-[var(--signal)]"
                      style={{ left: `${((lo + 90) / 270) * 100}%`, right: `${100 - ((hi + 90) / 270) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* obstacles */}
          <section className="surface p-5">
            <Header n="C" title="Obstacles" desc="forbid joints from entering regions" />
            <div className="viewport mt-4 grid aspect-square place-items-center">
              <div className="relative h-32 w-32">
                <span className="absolute left-4 top-6 h-12 w-12 rounded-full border border-rose-400/50 bg-rose-400/10" />
                <span className="absolute bottom-3 right-3 h-16 w-10 rounded-md border border-[var(--amber)]/50 bg-[var(--amber)]/10" />
                <span className="label absolute inset-x-0 -bottom-6 text-center">drag to place volumes</span>
              </div>
            </div>
          </section>
        </div>

        <div className="mt-5 flex items-center justify-between gap-4 surface p-5">
          <div className="text-[12px] text-[var(--muted)]">
            Resamples via the constrained Riemannian sampler — projecting each ODE
            step back onto the feasible set.
          </div>
          <button disabled className="btn-signal whitespace-nowrap">
            APPLY &amp; RESAMPLE
          </button>
        </div>
      </div>

      {/* roadmap footnote */}
      <p className="mt-6 text-center text-[11px] text-[var(--muted)]">
        Tracked in <span className="text-slate-400">plan.md</span> · out of scope for V1 ·
        backend endpoint <span className="font-mono text-slate-400">POST /constrain</span> not implemented.
      </p>
    </div>
  );
}

function Header({ n, title, desc }) {
  return (
    <div className="flex items-start gap-3">
      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-[var(--hairline-strong)] font-mono text-[12px] text-[var(--signal)]">
        {n}
      </span>
      <div>
        <div className="display text-base font-bold text-white">{title}</div>
        <div className="label mt-0.5">{desc}</div>
      </div>
    </div>
  );
}
