"use client";

import dynamic from "next/dynamic";
import { Page, PageHead } from "@/components/ui";

// three.js cannot be statically prerendered, so the studio is client-only.
const ConstraintStudio = dynamic(() => import("@/components/ConstraintStudio"), {
  ssr: false,
  loading: () => (
    <div className="grid place-items-center py-32 text-[13px] text-[var(--muted)]">loading studio…</div>
  ),
});

export default function StudioPage() {
  return (
    <Page>
      <PageHead
        kicker="Studio"
        title="Constraint study"
        lede="Select a motion, choose whether a joint is constrained and whether the text mentions it, and compare the resulting samples. Where a ground-truth capture exists it can be shown alongside."
      />
      <div className="mt-5">
        <span
          className="rounded border border-[var(--hairline-strong)] px-2 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--warn)]"
          title="clips were sampled on the cluster in advance and are replayed here"
        >
          pre-computed
        </span>
      </div>
      <div className="mt-8">
        <ConstraintStudio />
      </div>
    </Page>
  );
}
