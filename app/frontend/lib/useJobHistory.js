"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { classifyJob } from "./classifyJob";

// Read-only view of the backend's persisted job list (.media/jobs.json), filtered
// to a workspace's categories. This is the heart of "history per type": every
// workspace polls the SAME global list and shows only its own kinds, so a clip a
// constraint generation never leaks into the wrong place — yet nothing is lost.
//
// It deliberately does NOT submit jobs. The mode editors keep their existing
// submit paths (useVizJob / api.generate); their results land in jobs.json and
// this hook surfaces them on the next poll. Decoupling submit from history keeps
// the working sampling pipelines untouched.
export function useJobHistory(categories, { pollMs = 4000 } = {}) {
  const [all, setAll] = useState([]);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setAll(await api.jobs());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, pollMs);
    return () => clearInterval(t);
  }, [refresh, pollMs]);

  const key = categories ? categories.join(",") : "";
  const jobs = useMemo(() => {
    if (!categories) return all;
    const set = new Set(categories);
    return all.filter((j) => set.has(classifyJob(j)));
  }, [all, key]); // eslint-disable-line react-hooks/exhaustive-deps

  return { jobs, error, refresh };
}
