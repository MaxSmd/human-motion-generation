"""Run and score a whole obstacle-course study.

The unit of work here is not a clip but a **cell**: one (course, arm, seed) whose
three prompts are sampled together. `submit` lays the full matrix on the queue
and `collect` walks it back, scoring every pulled `.npy` against the course it
was sampled for and folding the per-clip reports into one row per (course, arm).

Cells are tagged in `job.params["ablation"]` — the same channel the constraint
ablation uses — so a study can be reassembled from the job registry alone, days
later, with no state kept here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import courses as C
from . import scene_eval as SE

STUDY = "course"


# --------------------------------------------------------------------------- submit


def build_jobs(course_keys: list[str] | None, seeds: list[int], checkpoint: str,
               model_preset: str, train_preset: str, num_steps: int = 800,
               guidance: float = 6.5, room_guidance: float = 0.75,
               arms: list[str] | None = None) -> list[dict]:
    """Every `/cluster/jobs/viz` body of the study, in submission order.

    Order is (seed, arm) within a course rather than (arm, seed): the arms of one
    seed then land adjacently in the queue, which keeps a matched pair together
    if the study is ever cut short.
    """
    keys = course_keys or list(C.COURSES)
    arms = arms or list(C.ARMS)
    out = []
    for key in keys:
        course = C.COURSES[key]
        for seed in seeds:
            for arm in arms:
                out.append(C.job_params(course, arm, int(seed), checkpoint,
                                        model_preset, train_preset,
                                        num_steps=num_steps, guidance=guidance,
                                        room_guidance=room_guidance))
    return out


# --------------------------------------------------------------------------- collect


def resolve_media(media_dir: Path, url: str) -> Path | None:
    """Local path for a `/media/...` URL, or None if it escapes the media root.

    Resolving from the URL rather than from the job id is load-bearing: viz jobs
    are **fused**, and a follower's outputs keep the *lead's* directory, so
    `media/jobs/<follower_id>/…` does not exist. Rebuilding the path from the
    job's own id silently loses every arm but the lead — which is exactly what
    the pilot did before this.
    """
    if not url:
        return None
    rel = url.split("/media/", 1)[-1].lstrip("/")
    path = (media_dir / rel).resolve()
    root = media_dir.resolve()
    if root not in path.parents:
        return None
    return path


def collect(jobs: list[dict], media_dir: Path) -> dict:
    """Score every finished cell of the study found in `jobs`.

    `jobs` is the raw job registry (list of `ClusterJob.to_dict()`); anything not
    tagged as this study, not finished, or with no pulled `.npy` is skipped and
    counted, so a partially-complete study still returns usable numbers instead
    of an error.
    """
    rows: list[dict] = []
    skipped: list[dict] = []
    for job in jobs:
        ab = (job.get("params") or {}).get("ablation") or {}
        if ab.get("study") != STUDY:
            continue
        key, arm = ab.get("course"), ab.get("arm")
        if key not in C.COURSES or arm not in C.ARMS:
            continue
        if job.get("state") != "done":
            skipped.append({"job": job.get("id"), "course": key, "arm": arm,
                            "reason": job.get("state")})
            continue
        course = C.COURSES[key]
        for out in job.get("outputs") or []:
            npy = out.get("npy_url")
            if not npy:
                continue
            path = resolve_media(media_dir, npy)
            if path is None or not path.exists():
                skipped.append({"job": job["id"], "course": key, "arm": arm,
                                "reason": "npy missing"})
                continue
            try:
                res = SE.evaluate(np.load(path), course, arm)
            except ValueError as e:
                skipped.append({"job": job["id"], "course": key, "arm": arm,
                                "reason": str(e)})
                continue
            res["job"] = job["id"]
            res["seed"] = ab.get("seed")
            # Read from the job's own params, not the ablation tag: the sampler
            # settings are what actually differ, and a pilot re-run of the same
            # (course, arm, seed) at a different step count must NOT pool into
            # the same cell — it silently mixes two experiments.
            res["num_steps"] = int((job.get("params") or {}).get("num_steps", 0))
            res["guidance"] = float((job.get("params") or {}).get("guidance", 0.0))
            res["clip"] = Path(npy).name
            res["media_url"] = out.get("media_url")
            res["npy_url"] = npy
            res["caption"] = out.get("caption") or ""
            rows.append(res)

    # Runs at different sampler settings are different experiments. Default the
    # report to the most thoroughly sampled configuration present (a 200-step
    # pipeline pilot must not dilute an 800-step study) and list the rest so
    # nothing is hidden.
    configs = sorted({r["num_steps"] for r in rows}, reverse=True)
    primary = configs[0] if configs else 0
    main = [r for r in rows if r["num_steps"] == primary]
    table = SE.aggregate(main)
    return {
        "clips": main,
        "table": table,
        "deltas": deltas(table),
        "num_steps": primary,
        "other_configs": [{"num_steps": s, "n_clips": sum(r["num_steps"] == s for r in rows)}
                          for s in configs[1:]],
        "skipped": skipped,
        "n_clips": len(main),
        "n_clips_all": len(rows),
    }


# Metrics where the comparison of interest is "did the trajectory arm change
# this, and in which direction". `lower` says which way is better, so the report
# can mark a regression without the reader having to remember each convention.
HEADLINE = [
    ("penetration_max", True, "m"),
    ("penetration_frame_frac", True, ""),
    ("path_dev_mean", True, "m"),
    ("foot_skate_mean", True, "m/s"),
    ("slide_per_root_step", True, ""),
    ("support_clearance_mean", False, "m"),
    ("jerk_mean", True, ""),
    ("root_speed_mean", None, "m/s"),
    ("pelvis_rise", None, "m"),
]


def deltas(table: list[dict]) -> list[dict]:
    """Per-course `room+traj` minus `room`, and `room` minus `free`.

    Two separate questions that a single "constrained vs unconstrained" number
    would blur: what the euclidean room energy alone buys, and what the
    trajectory channel adds on top of it.
    """
    by: dict[tuple[str, str], dict] = {(r["course"], r["arm"]): r for r in table}
    out = []
    for key in C.COURSES:
        for base, arm in (("free", "room"), ("room", "room+traj")):
            a, b = by.get((key, base)), by.get((key, arm))
            if not a or not b:
                continue
            row = {"course": key, "arm": arm, "vs": base, "metrics": {}}
            for name, lower, unit in HEADLINE:
                ma, mb = a["metrics"].get(name), b["metrics"].get(name)
                if not ma or not mb:
                    continue
                d = mb["mean"] - ma["mean"]
                row["metrics"][name] = {
                    "base": ma["mean"], "arm": mb["mean"], "delta": round(d, 5),
                    "pct": round(100.0 * d / ma["mean"], 1) if abs(ma["mean"]) > 1e-9 else None,
                    "better": None if lower is None else bool((d < 0) == lower),
                    "unit": unit,
                }
            out.append(row)
    return out
