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

from .. import config as cfgmod
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
    elif mode == "compare":
        env.update({
            "CKPT": params["checkpoint"], "MODEL_PRESET": params.get("model_preset", "dit_base"),
            "PRESET": params.get("train_preset", "rmg_base"), "CLIPS": params.get("clips", "auto"),
            "SUBSET_FRACTION": str(params.get("subset_fraction", 0.01)),
            "SUBSET_SEED": str(params.get("subset_seed", 0)),
            "NUM_STEPS": str(params.get("num_steps", 50)),
            "GUIDANCE": str(params.get("guidance", 6.5)),
        })
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


# --------------------------------------------------------------------------- train


def build_train(params: dict) -> tuple[str, dict[str, str], str, str]:
    """All training config is reachable: structured knobs become Hydra OVERRIDES,
    plus a free-form `overrides` string for anything not surfaced in the form."""
    run_name = params.get("run_name") or _run_name("train", params.get("train_preset", "rmg_base"))
    env: dict[str, str] = {
        "RUN_NAME": run_name,
        "RUNS_ROOT": _runs_root("train"),
        "MODEL_PRESET": params.get("model_preset", "dit_base"),
        "PRESET": params.get("train_preset", "rmg_base"),
    }
    ov = _overrides([
        ("data.subset_n", params.get("subset_n")),
        ("data.subset_fraction", params.get("subset_fraction")),
        ("data.subset_seed", params.get("subset_seed")),
        ("train.max_steps", params.get("max_steps")),
        ("train.sample_every", params.get("sample_every")),
        ("train.ckpt_every", params.get("ckpt_every")),
        ("train.optimizer.lr", params.get("lr")),
        ("train.guidance_scale", params.get("guidance")),
        ("train.precision", params.get("precision")),
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
        "MODEL_PRESET": params.get("model_preset", "dit_base"),
        "PRESET": params.get("train_preset", "rmg_base"),
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
