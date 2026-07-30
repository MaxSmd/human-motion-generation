"""Build + submit SLURM jobs from the three unified `slurm/rmg_{train,eval,viz}.sbatch`
scripts. Everything is an env var + an `OVERRIDES` passthrough for arbitrary extra
Hydra args, so any config is reachable without per-job scripts.

The scripts live in the cluster's git checkout (the single source of truth) — we
do **not** rsync anything up; the user keeps it current via `git pull`.

Each job is `cd <project> && KEY=val … sbatch [--partition=… …] --parsable
--job-name=rmgui-… slurm/<script>`. Values are `shlex.quote`-d (user prompts /
clip-ids / overrides flow through here). `--parsable` makes sbatch print just the
job id; `--job-name` tags our jobs so the user can audit them.

`render_command()` returns the exact remote command WITHOUT running it — the
frontend shows it as a dry-run preview before any launch.
"""

from __future__ import annotations

import json
import secrets
import shlex

from .. import config as cfgmod, gtreg
from . import ssh
from .squeue import resolve_run_dir

JOB_PREFIX = "rmgui"

SCRIPTS = {"viz": "rmg/viz.sbatch", "train": "rmg/train.sbatch", "eval": "rmg/eval.sbatch"}


def _run_name(kind: str, tag: str = "") -> str:
    suffix = f"{kind}-{tag}" if tag else kind
    return f"{JOB_PREFIX}-{suffix}-{secrets.token_hex(4)}"


def _runs_root(kind: str) -> str:
    """Per-model, per-task runs dir on the cluster, e.g. `<project>/runs/rmg/train`."""
    return f"{ssh.abs_remote(cfgmod.cluster_runs_dir())}/{cfgmod.cluster_model()}/{kind}"


def train_run_dir(run_name: str) -> str:
    """Abs path of a train run's output dir — where `train.py` writes its
    `.progress` / `.complete` markers (output_dir = MGEN_RUNS_DIR/run_name and the
    sbatch sets MGEN_RUNS_DIR = the train runs root)."""
    return f"{_runs_root('train')}/{run_name}"


def eval_progress_dir(run_name: str) -> str:
    """Abs path of an eval run's `eval/` subdir — where `evaluate.py` writes its
    `results.json`, `.progress.json` and `.complete` markers (output_dir =
    RUNS_ROOT/run_name, results under output_dir/eval/)."""
    return f"{_runs_root('eval')}/{run_name}/eval"


def viz_progress_dir(run_name: str) -> str:
    """Abs path of a viz run's `viz/` subdir — where `visualize.py` writes its
    rendered media, manifest and `.progress.json` marker."""
    return f"{_runs_root('viz')}/{run_name}/viz"


def _sbatch_flags(params: dict, partition: str | None) -> list[str]:
    """SBATCH CLI overrides shared by every kind (these override the directives
    baked into the sbatch scripts). `partition` is the caller's resolved default;
    a `partition` param always wins, and `walltime` overrides the script's --time.
    `sbatch_extra` is free-form (`--constraint=…`, `--gres=…`, `--exclude=…`,
    `--nodelist=…`) — e.g. to pin a 12g job onto a Turing+ node."""
    partition = params.get("partition") or partition
    flags: list[str] = []
    if partition:
        flags.append(f"--partition={partition}")
    if params.get("walltime"):
        flags.append(f"--time={params['walltime']}")
    # The 12g partition is mixed-GPU and only its RTX 2080 Ti nodes (sm_75) can run
    # the CUDA-13 container — TITAN Xp/X (sm_61/52) crash with "no kernel image".
    # Auto-pin 12g jobs to the compatible cards unless the user set their own gres.
    extra = str(params.get("sbatch_extra") or "")
    if partition == "12g" and "gres" not in extra:
        flags.append("--gres=gpu:RTX2080Ti:1")
    if extra:
        flags.extend(shlex.split(extra))
    return flags


def sbatch_flags_for_train(params: dict) -> list[str]:
    """Bigger presets default to the 24g partition for GPU memory headroom; the
    small ones take whatever train.sbatch bakes in (no --partition flag)."""
    big = params.get("model_preset") in ("dit_small", "dit_mid", "dit_large")
    return _sbatch_flags(params, "24g" if big else None)


def sbatch_flags_for_eval(params: dict) -> list[str]:
    """Evals fit in <12GB even for dit_mid (measured <50% of a 24g card), and the
    cluster auto-cancels jobs under 50% GPU-mem utilization after 2h — so default
    to the 12g partition and leave the 24g cards for training."""
    return _sbatch_flags(params, "12g")


def sbatch_flags_for_viz(params: dict) -> list[str]:
    """Viz renders a handful of clips/prompts — strictly lighter than an eval
    sweep, which already fits in 12GB for dit_mid — so it defaults to 12g too
    (viz.sbatch bakes in 24g, which is overkill and queues behind training)."""
    return _sbatch_flags(params, "12g")


def render_command(
    script: str, env: dict[str, str], job_name: str, sbatch_flags: list[str] | None = None
) -> str:
    """The exact remote command (for dry-run preview + actual submit)."""
    proj = shlex.quote(ssh.abs_remote(cfgmod.cluster_project_dir()))
    assignments = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v != "")
    flags = " ".join(shlex.quote(f) for f in (sbatch_flags or []))
    flags = f"{flags} " if flags else ""
    return (
        f"cd {proj} && {assignments} "
        f"sbatch {flags}--parsable --job-name={shlex.quote(job_name)} slurm/{script}"
    )


def _submit(script: str, env: dict[str, str], job_name: str, sbatch_flags=None) -> str:
    cmd = render_command(script, env, job_name, sbatch_flags)
    res = ssh.run(cmd, timeout=30)
    jobid = res.stdout.strip().splitlines()[-1].split(";")[0].strip()  # --parsable → "12345"
    if not jobid.isdigit():
        raise ssh.SSHError(1, cmd, f"unexpected sbatch output: {res.stdout!r} {res.stderr!r}")
    return jobid


def _overrides(pairs: list[tuple[str, object]], extra: str | None) -> str:
    """Join `key=value` Hydra tokens (skipping None/'') + any free-form extra."""
    toks = [f"{k}={v}" for k, v in pairs if v not in (None, "")]
    if extra:
        toks.append(str(extra))
    return " ".join(toks)


# --------------------------------------------------------------------------- viz


def build_viz(params: dict) -> tuple[str, dict[str, str], str, str]:
    mode = params.get("mode", "clip")
    run_name = _run_name("viz", mode)
    env: dict[str, str] = {"MODE": mode, "RUN_NAME": run_name, "RUNS_ROOT": _runs_root("viz")}
    if mode == "clip":
        env["CLIPS"] = params["clips"]
    elif mode == "prompt":
        env.update({
            "CKPT": params["checkpoint"], "MODEL_PRESET": params.get("model_preset", "dit_base"),
            "PRESET": params.get("train_preset", "rmg_base"), "PROMPTS": params["prompts"],
            "NUM_FRAMES": str(params.get("num_frames", 100)),
            "NUM_STEPS": str(params.get("num_steps", 50)),
            "GUIDANCE": str(params.get("guidance", 6.5)),
            "SEED": str(params.get("seed", 0)),
            "USE_EMA": "true" if params.get("use_ema", True) else "false",
        })
        # Sampling-time constraints: JSON forwarded to visualize.py as env vars
        # (avoids quoting a structured list through the Hydra CLI). CONSTRAINTS =
        # fixed angles, RANGES = hinge limits.
        if params.get("constraints"):
            env["CONSTRAINTS"] = json.dumps(params["constraints"])
        if params.get("ranges"):
            env["RANGES"] = json.dumps(params["ranges"])
        # Euclidean room/scene → RMG_SCENE (+ guidance weight) in visualize.py.
        if params.get("scene"):
            env["SCENE"] = json.dumps(params["scene"])
            env["ROOM_GUIDANCE"] = str(params.get("room_guidance", 0.0))
        # Spatial (mask-control) targets → RMG_TRAJECTORY in visualize.py.
        if params.get("trajectory"):
            env["TRAJECTORY"] = json.dumps(params["trajectory"])
    elif mode == "compare":
        clips_raw = str(params.get("clips", "auto"))
        env.update({
            "CKPT": params["checkpoint"], "MODEL_PRESET": params.get("model_preset", "dit_base"),
            "PRESET": params.get("train_preset", "rmg_base"), "CLIPS": clips_raw,
            "SUBSET_FRACTION": str(params.get("subset_fraction", 0.01)),
            "SUBSET_SEED": str(params.get("subset_seed", 0)),
            "NUM_STEPS": str(params.get("num_steps", 50)),
            "GUIDANCE": str(params.get("guidance", 6.5)),
        })
        # GT is identical every time it's rendered, so skip the clips the registry
        # already holds — the app splices those renders back in on pull. Only
        # possible for explicit ids ('auto' is picked cluster-side, after submit).
        if clips_raw.strip().lower() != "auto":
            cached = gtreg.known([c.strip() for c in clips_raw.split(",") if c.strip()])
            if cached:
                env["SKIP_GT"] = ",".join(cached)
    elif mode == "samples":
        run = params["run"]
        steps = params.get("steps")
        if params.get("samples_file"):
            env["SAMPLES_FILE"] = params["samples_file"]
        elif steps:
            # Render only the chosen steps (comma-separated explicit .pt paths) so
            # a run with dozens of dumps doesn't balloon into hundreds of GIFs.
            rundir = resolve_run_dir(run)
            env["SAMPLES_FILE"] = ",".join(
                f"{rundir}/samples/step-{int(s):09d}.pt" for s in steps
            )
        else:
            env["SAMPLES_FILE"] = f"{resolve_run_dir(run)}/samples"
    else:
        raise ValueError(f"unknown viz mode {mode!r}")
    if params.get("overrides"):
        env["OVERRIDES"] = str(params["overrides"])
    return SCRIPTS["viz"], env, f"{JOB_PREFIX}-viz-{mode}", run_name


# ---------------------------------------------------------------------- viz fusion
#
# The cluster runs ONE job at a time, so N queued viz jobs mean N separate waits at
# the back of the SLURM queue — painful when a tab queues a whole study at once
# (the constraint analysis submits 24). Compatible jobs are therefore fused into a
# SINGLE sbatch: one queue wait, one model load, N renders.
#
# What must match (the fusion key) is everything that configures the model and
# sampler, since a job builds those once. Everything that varies per render —
# prompt text, constraints, scene, frame count, seed — travels per item inside the
# job instead.

_FUSION_KEYS = {
    "clip": ("overrides",),
    "prompt": ("checkpoint", "model_preset", "train_preset", "num_steps", "guidance",
               "use_ema", "overrides"),
    "compare": ("checkpoint", "model_preset", "train_preset", "num_steps", "guidance",
                "subset_fraction", "subset_seed", "overrides"),
}

# Placement is a property of the sbatch itself, so it has to match as well.
_FUSION_PLACEMENT = ("partition", "sbatch_extra", "walltime")


def viz_fusion_key(params: dict) -> str | None:
    """Signature for `build_viz_fused`: two viz jobs may fuse iff their keys are
    equal. None means unfusable — mode=samples renders whole step-dump files, and
    mode=compare with clips='auto' picks its clips cluster-side (after submit), so
    there is no list to merge here."""
    mode = params.get("mode", "clip")
    keys = _FUSION_KEYS.get(mode)
    if keys is None:
        return None
    if mode == "compare" and str(params.get("clips", "auto")).strip().lower() == "auto":
        return None
    return json.dumps(
        [mode, [params.get(k) for k in (*keys, *_FUSION_PLACEMENT)]],
        sort_keys=True, default=str,
    )


def _merged_clips(params_list: list[dict]) -> str:
    """Union of every job's clip ids, order preserved (two jobs may ask for the
    same clip — it only needs rendering once; the puller hands it to both)."""
    seen: list[str] = []
    for p in params_list:
        for c in str(p.get("clips", "")).split(","):
            c = c.strip()
            if c and c not in seen:
                seen.append(c)
    return ",".join(seen)


def viz_batch_items(job_id: str, params: dict) -> list[dict]:
    """One `RMG_BATCH` entry per prompt of a mode=prompt job. `job` tags each entry
    with the app job that asked for it, so the manifest can route the render back."""
    items = []
    for prompt in str(params.get("prompts", "")).split("|"):
        prompt = prompt.strip()
        if not prompt:
            continue
        items.append({
            "job": job_id,
            "prompt": prompt,
            "num_frames": int(params.get("num_frames", 100)),
            "seed": int(params.get("seed", 0)),
            "constraints": params.get("constraints") or [],
            "ranges": params.get("ranges") or [],
            "scene": params.get("scene") or None,
            "room_guidance": float(params.get("room_guidance", 0.0)),
            "trajectory": params.get("trajectory") or None,
        })
    return items


def fused_walltime(n_items: int) -> str:
    """viz.sbatch bakes in 30 min, which is right for a couple of renders and far
    too short once a study's worth is fused into one job. Scale with the item count
    (model load + a batched ODE per constraint group + a render each), capped at 6h."""
    mins = min(6 * 60, 20 + 6 * n_items)
    return f"{mins // 60:02d}:{mins % 60:02d}:00"


def build_viz_fused(
    items: list[tuple[str, dict]],
) -> tuple[str, dict[str, str], str, str, list[str]]:
    """Build ONE viz submission covering several app jobs. `items` is
    [(job_id, params), …], all sharing a `viz_fusion_key`; the first is the lead
    and donates the shared model/sampler config plus the run_name. Returns the
    usual builder tuple plus the sbatch flags (the walltime scales with the fused
    item count, so it can't be derived from the params alone).

    clip/compare fuse by unioning their clip lists — a viz job already renders many
    clips, and the puller re-derives which clip belongs to which job from each job's
    own params. prompt fuses into an `RMG_BATCH` spec: one entry per prompt, each
    carrying its OWN constraints/ranges/scene/frames/seed (the per-batch env vars
    can't express that), which visualize.py groups by constraint signature.
    """
    params_list = [p for _, p in items]
    lead = dict(params_list[0])
    mode = lead.get("mode", "clip")
    script, env, job_name, run_name = build_viz(lead)
    if mode in ("clip", "compare"):
        env["CLIPS"] = _merged_clips(params_list)
        if mode == "compare":
            cached = gtreg.known([c for c in env["CLIPS"].split(",") if c])
            env["SKIP_GT"] = ",".join(cached)
        n_items = len(env["CLIPS"].split(","))
    elif mode == "prompt":
        batch = [it for jid, p in items for it in viz_batch_items(jid, p)]
        env["BATCH"] = json.dumps(batch)
        # RMG_BATCH supersedes these; drop them so the previewed command shows only
        # what actually drives the job (the lead's prompts are inside BATCH already).
        for k in ("PROMPTS", "CONSTRAINTS", "RANGES", "SCENE", "ROOM_GUIDANCE", "TRAJECTORY"):
            env.pop(k, None)
        n_items = len(batch)
    else:
        raise ValueError(f"viz mode {mode!r} is not fusable")
    flags = sbatch_flags_for_viz(
        {**lead, "walltime": lead.get("walltime") or fused_walltime(n_items)}
    )
    return script, env, f"{job_name}-x{len(items)}", run_name, flags


# --------------------------------------------------------------------------- train


def build_train(params: dict) -> tuple[str, dict[str, str], str, str]:
    """All training config is reachable: structured knobs become Hydra OVERRIDES,
    plus a free-form `overrides` string for anything not surfaced in the form."""
    run_name = params.get("run_name") or _run_name("train", params.get("train_preset", "rmg_mid"))
    env: dict[str, str] = {
        "RUN_NAME": run_name,
        "RUNS_ROOT": _runs_root("train"),
        "MODEL_PRESET": params.get("model_preset", "dit_mid"),
        "PRESET": params.get("train_preset", "rmg_mid"),
    }
    ov = _overrides([
        ("data.subset_n", params.get("subset_n")),
        ("data.subset_fraction", params.get("subset_fraction")),
        ("data.subset_seed", params.get("subset_seed")),
        ("train.max_steps", params.get("max_steps")),
        ("train.micro_batch_size", params.get("micro_batch_size")),
        ("train.grad_accum", params.get("grad_accum")),
        ("train.sample_every", params.get("sample_every")),
        ("train.ckpt_every", params.get("ckpt_every")),
        ("train.optimizer.lr", params.get("lr")),
        ("train.guidance_scale", params.get("guidance")),
        ("train.precision", params.get("precision")),
        # RAM-cache the encoded features (kills the per-step encode pipeline that
        # otherwise starves the GPU). Hydra wants a lowercase bool; only sent when on.
        ("data.preload", "true" if params.get("preload") else None),
        ("representation", params.get("representation")),
    ], params.get("overrides"))
    if ov:
        env["OVERRIDES"] = ov
    return SCRIPTS["train"], env, f"{JOB_PREFIX}-train", run_name


# --------------------------------------------------------------------------- eval


def build_eval(params: dict) -> tuple[str, dict[str, str], str, str]:
    run_name = _run_name("eval", params.get("run", ""))
    env: dict[str, str] = {
        "CKPT": params["checkpoint"],
        "RUN_NAME": run_name,
        "RUNS_ROOT": _runs_root("eval"),
        "MODEL_PRESET": params.get("model_preset", "dit_mid"),
        "PRESET": params.get("train_preset", "rmg_mid"),
    }
    for key, env_name in [
        ("eval_split", "EVAL_SPLIT"), ("max_clips", "MAX_CLIPS"),
        ("guidance_scales", "GUIDANCE_SCALES"), ("subset_fraction", "SUBSET_FRACTION"),
        ("subset_seed", "SUBSET_SEED"), ("num_sample_steps", "NUM_SAMPLE_STEPS"),
    ]:
        if params.get(key) not in (None, ""):
            env[env_name] = str(params[key])
    if params.get("use_ema") is not None:
        env["USE_EMA"] = "true" if params["use_ema"] else "false"
    if params.get("overrides"):
        env["OVERRIDES"] = str(params["overrides"])
    return SCRIPTS["eval"], env, f"{JOB_PREFIX}-eval", run_name


# --------------------------------------------------------------------------- mardm
#
# MARDM diverges from rmg: training is two-stage (AE → gen) and eval needs both
# checkpoints. These builders map the app's params onto the `slurm/mardm/*`
# env-var contracts (which accept an `OVERRIDES` passthrough, mirroring rmg).

MARDM_SCRIPTS = {
    "train_ae": "mardm/train_mardm_ae.sbatch",
    "train_gen": "mardm/train_mardm.sbatch",
    "eval": "mardm/evaluate_mardm.sbatch",
}


def build_mardm_train(params: dict) -> tuple[str, dict[str, str], str, str]:
    """Two-stage MARDM training. `stage` ∈ {ae, gen}; gen requires an AE ckpt.

    rmg's structured knobs map onto MARDM's hydra keys (top-level `subset_frac`/
    `limit_clips`, `train.*`); anything else flows through the free-form
    `overrides` string → the sbatch `OVERRIDES` passthrough."""
    stage = params.get("stage", "gen")
    if stage not in ("ae", "gen"):
        raise ValueError(f"unknown MARDM train stage {stage!r} (expected ae|gen)")
    run_name = params.get("run_name") or _run_name("train", stage)
    env: dict[str, str] = {"RUN_NAME": run_name, "RUNS_ROOT": _runs_root("train")}
    ov = _overrides([
        ("subset_frac", params.get("subset_fraction")),
        ("limit_clips", params.get("subset_n")),
        ("train.max_steps", params.get("max_steps")),
        ("train.ckpt_every", params.get("ckpt_every")),
        ("train.optimizer.lr", params.get("lr")),
        ("train.precision", params.get("precision")),
    ], params.get("overrides"))
    if stage == "ae":
        script, job = MARDM_SCRIPTS["train_ae"], f"{JOB_PREFIX}-train-ae"
    else:
        ae = params.get("ae_checkpoint")
        if not ae:
            raise ValueError("MARDM gen training needs an AE checkpoint (ae_checkpoint)")
        env["AE_CKPT"] = ae
        env["CONFIG_NAME"] = params.get("config_name") or "gen"
        script, job = MARDM_SCRIPTS["train_gen"], f"{JOB_PREFIX}-train-gen"
    if ov:
        env["OVERRIDES"] = ov
    return script, env, job, run_name


def build_mardm_eval(params: dict) -> tuple[str, dict[str, str], str, str]:
    """MARDM eval needs the stage-1 AE checkpoint AND the stage-2 gen checkpoint.
    The gen checkpoint reuses the shared `checkpoint` param (the eval picker)."""
    gen = params.get("checkpoint")
    ae = params.get("ae_checkpoint")
    if not gen:
        raise ValueError("MARDM eval needs a gen checkpoint (checkpoint)")
    if not ae:
        raise ValueError("MARDM eval needs an AE checkpoint (ae_checkpoint)")
    run_name = _run_name("eval", params.get("run", ""))
    env: dict[str, str] = {
        "AE_CKPT": ae,
        "GEN_CKPT": gen,
        "CONFIG_NAME": params.get("config_name") or "gen",
        "RUN_NAME": run_name,
        "RUNS_ROOT": _runs_root("eval"),
    }
    # The eval picker may send a single value or a bracketed list; either is a
    # valid MARDM GUIDANCE (forwarded to +eval.guidance_scales).
    guidance = params.get("guidance_scales")
    if guidance in (None, ""):
        guidance = params.get("guidance")
    if guidance not in (None, ""):
        env["GUIDANCE"] = str(guidance)
    if params.get("overrides"):
        env["OVERRIDES"] = str(params["overrides"])
    return MARDM_SCRIPTS["eval"], env, f"{JOB_PREFIX}-eval", run_name


# --------------------------------------------------------------- model dispatch

# Per-model task → builder. Each builder embeds its own sbatch script. MARDM has
# no `viz` (no standalone generate/render pipeline yet), so Generate/Visualize
# are gated off in the UI when MARDM is active.
_BUILDERS: dict[str, dict] = {
    "rmg": {"viz": build_viz, "train": build_train, "eval": build_eval},
    "mardm": {"train": build_mardm_train, "eval": build_mardm_eval},
}


def has_task(model: str, kind: str) -> bool:
    return kind in _BUILDERS.get(model, {})


def builder_for(model: str, kind: str):
    """Resolve the (model, kind) builder. Raises ValueError on an unsupported task."""
    try:
        return _BUILDERS[model][kind]
    except KeyError as e:
        raise ValueError(f"task {kind!r} is not available for model {model!r}") from e


def submit(builder, params: dict) -> dict:
    """Run a `build_*` and submit. Returns {slurm_id, run_name, job_name, command}."""
    script, env, job_name, run_name = builder(params)
    sbatch_flags = params.get("sbatch_flags")  # optional [--partition=…, --time=…]
    command = render_command(script, env, job_name, sbatch_flags)
    slurm_id = _submit(script, env, job_name, sbatch_flags)
    return {"slurm_id": slurm_id, "run_name": run_name, "job_name": job_name, "command": command}
