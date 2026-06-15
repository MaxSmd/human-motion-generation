"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

export const ACTIVE = new Set(["queued", "submitting", "pending", "running", "pulling"]);

// Submit a job (default: viz) and poll it to completion. Shared by Generate /
// Visualize / Model. Extra submits queue locally on the backend (one cluster job
// at a time), so this just tracks whatever it gets back.
export function useVizJob(submitFn = api.submitViz) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const run = useCallback(async (body) => {
    setSubmitting(true);
    setError(null);
    setJob(null);
    try {
      setJob(await submitFn(body));
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }, [submitFn]);

  useEffect(() => {
    if (!job || !ACTIVE.has(job.state)) return;
    const t = setInterval(async () => {
      try {
        setJob(await api.job(job.id));
      } catch {
        /* transient — keep last state */
      }
    }, 3000);
    return () => clearInterval(t);
  }, [job?.id, job?.state]);

  return { job, error, submitting, active: !!job && ACTIVE.has(job.state), run };
}
