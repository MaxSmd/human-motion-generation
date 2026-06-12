"""Config loading for the backend, decoupled from Hydra's CLI.

Two ways a config is obtained:

1. `compose_default()` — replays Hydra's `defaults:` composition of
   `configs/train.yaml` (data + model + train + representation + `_self_`)
   using plain OmegaConf, so we don't need a Hydra runtime / `@hydra.main`.

2. `load_run_config(run_dir)` — the trainer writes a full snapshot of the
   resolved config to `runs/<run>/config.json`. When generating from a real
   checkpoint we prefer this: it carries the exact model dims, representation,
   and text-encoder used for that run, which may differ from the repo defaults.

Path defaults come from env vars so the same image runs locally or on the
cluster:
    RMG_DATA_ROOT   — packed HumanML3D dir (humanml3d.zip, splits.json, offsets)
    RMG_RUNS_DIR    — where training runs (checkpoints + samples) live
    RMG_OFFSETS     — optional direct path to target_offsets.pt (overrides data root)
    RMG_CHECKPOINT  — optional explicit checkpoint .pt to warm-load for /generate
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

def _find_repo_root() -> Path:
    """Repo root = the dir holding `src/rmg/configs`. Located by marker, not a
    fixed parent depth, so it works both in-repo (app/backend/) and in the Docker
    image (backend flattened to /app/backend, root /app)."""
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "rmg" / "configs").is_dir():
            return d
    return here.parents[2]


REPO = _find_repo_root()
CONFIGS = REPO / "src" / "rmg" / "configs"


def _register_resolvers() -> None:
    """`configs/*.yaml` interpolate `${hydra:runtime.cwd}` as a fallback. Outside
    a Hydra runtime that resolver is missing, so register a stand-in that points
    at the repo root. Env-var branches (`${oc.env:RMG_DATA_ROOT,...}`) short-circuit
    before this is ever evaluated when the env var is set."""
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver(
            "hydra", lambda key: str(REPO) if key == "runtime.cwd" else "", replace=True
        )


def _load_leaf(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        del cfg["defaults"]  # sub-configs carry an empty `defaults: []`
    return cfg  # type: ignore[return-value]


def compose_default(
    *,
    data: str = "local_submodule",
    model: str = "dit_base",
    train: str = "rmg_base",
    representation: str = "t_plus_r",
) -> DictConfig:
    """Compose the same config Hydra would for `rmg.scripts.train` defaults."""
    _register_resolvers()
    cfg = OmegaConf.create({})
    cfg.data = _load_leaf(CONFIGS / "data" / f"{data}.yaml")
    cfg.model = _load_leaf(CONFIGS / "model" / f"{model}.yaml")
    cfg.train = _load_leaf(CONFIGS / "train" / f"{train}.yaml")
    cfg.representation = _load_leaf(CONFIGS / "representation" / f"{representation}.yaml")
    self_cfg = _load_leaf(CONFIGS / "train.yaml")
    cfg = OmegaConf.merge(cfg, self_cfg)
    return cfg  # type: ignore[return-value]


def load_run_config(run_dir: str | Path) -> DictConfig | None:
    """Load `runs/<run>/config.json` (the trainer's resolved-config snapshot)."""
    p = Path(run_dir) / "config.json"
    if not p.exists():
        return None
    _register_resolvers()
    with open(p) as f:
        return OmegaConf.create(json.load(f))  # type: ignore[return-value]


# --------------------------------------------------------------------------- paths


def data_root(cfg: DictConfig | None = None) -> Path:
    env = os.environ.get("RMG_DATA_ROOT")
    if env:
        return Path(env)
    cfg = cfg or compose_default()
    return Path(str(cfg.data.root))


def runs_dir() -> Path:
    env = os.environ.get("RMG_RUNS_DIR")
    if env:
        return Path(env)
    return REPO / "runs"


def offsets_path(cfg: DictConfig | None = None) -> Path:
    """Path to `target_offsets.pt` — RMG_OFFSETS wins, else data_root/offsets_name."""
    env = os.environ.get("RMG_OFFSETS")
    if env:
        return Path(env)
    cfg = cfg or compose_default()
    return data_root(cfg) / str(cfg.data.offsets_name)


def default_checkpoint() -> Path | None:
    """Checkpoint to warm-load for /generate. RMG_CHECKPOINT wins; otherwise the
    most-recently-modified `latest.pt`/`ckpt-*.pt` under any run in RMG_RUNS_DIR."""
    env = os.environ.get("RMG_CHECKPOINT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    rd = runs_dir()
    if not rd.exists():
        return None
    ckpts = list(rd.glob("*/checkpoints/*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------- cluster
#
# Cluster mode turns the backend into a SLURM control plane (see cluster-plan.md).
# Everything is driven over the user's own `~/.ssh/config` host alias, multiplexed
# through one ControlMaster opened for the app session. Defaults match the user's
# layout: alias `head`, project ~/riemann-motion-generation, runs <project>/runs
# (split into per-task subdirs train/ eval/ viz/).


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def cluster_mode() -> bool:
    return _env_bool("RMG_CLUSTER_MODE", True)


def cluster_host() -> str:
    return os.environ.get("RMG_CLUSTER_HOST", "head")


def cluster_model() -> str:
    """Active model namespace under the runs root: `runs/<model>/<kind>/<run>`.
    Default `rmg`; the momask/mardm merge will make this per-submission."""
    return os.environ.get("RMG_CLUSTER_MODEL", "rmg")


RUN_KINDS = ("train", "eval", "viz")


def cluster_runs_dir() -> str:
    """Remote runs root, holding per-task subdirs `train/`, `eval/`, `viz/`
    (shell-expanded on the cluster). Defaults to `<project>/runs` so all run data
    lives inside the repo checkout rather than scattered in `$HOME`."""
    env = os.environ.get("RMG_CLUSTER_RUNS")
    if env:
        return env
    return f"{cluster_project_dir()}/runs"


def cluster_project_dir() -> str:
    """Remote git project dir (holds slurm/*.sbatch). Default ~/riemann-motion-generation."""
    return os.environ.get("RMG_CLUSTER_PROJECT", "~/riemann-motion-generation")


def ssh_control_path() -> str:
    """ControlMaster socket for the app session. Kept short (macOS sun_path limit)."""
    return os.environ.get("RMG_SSH_CONTROL_PATH", str(Path.home() / ".rmg-cm.sock"))


def ssh_extra_opts() -> list[str]:
    """Extra `-o Key=Val` style tokens from RMG_SSH_OPTS (space-separated)."""
    raw = os.environ.get("RMG_SSH_OPTS", "").strip()
    return raw.split() if raw else []


def jobs_state_path() -> Path:
    """Where the job registry is persisted so jobs survive a backend restart."""
    env = os.environ.get("RMG_JOBS_STATE")
    return Path(env) if env else (REPO / ".media" / "jobs.json")
