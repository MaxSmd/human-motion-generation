"""Run the unmodified official MoMask evaluator on a modern runtime."""

from __future__ import annotations

import argparse
import ast
import functools
import inspect
import os
import runpy
import sys
import types
from pathlib import Path


def _parse_args() -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return args.entrypoint.resolve(), forwarded


def _clip_is_used(source_path: Path) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "clip"
        for node in ast.walk(tree)
    )


def _install_clip_compat(official_repo: Path) -> str:
    eval_source = official_repo / "utils" / "eval_t2m.py"
    if not eval_source.is_file():
        raise FileNotFoundError(f"official eval module not found: {eval_source}")
    if _clip_is_used(eval_source):
        raise RuntimeError(
            "official utils/eval_t2m.py uses CLIP at runtime; install the official "
            "OpenAI CLIP package instead of using the import-only compatibility stub"
        )
    sys.modules["clip"] = types.ModuleType("clip")
    return "unused-import-stub"


def _install_torch_load_compat(torch_module: types.ModuleType) -> str:
    """Preserve pre-PyTorch-2.6 loading semantics for trusted official assets."""
    original_load = torch_module.load
    try:
        parameters = inspect.signature(original_load).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "weights_only" not in parameters:
        return "legacy-default"

    @functools.wraps(original_load)
    def legacy_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch_module.load = legacy_load
    return "explicit-weights-only-false"


def main() -> None:
    entrypoint, forwarded = _parse_args()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"official evaluator entrypoint not found: {entrypoint}")

    # PyTorch 2.6 changed torch.load's default to weights_only=True. The
    # released MoMask checkpoints predate that behavior.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import numpy as np

    aliases: list[str] = []
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
        aliases.append("np.float")

    official_repo = entrypoint.parent
    clip_mode = _install_clip_compat(official_repo)

    import scipy
    import torch

    torch_load_mode = _install_torch_load_compat(torch)

    print(
        "[momask-vq-official-compat] "
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"numpy={np.__version__} scipy={scipy.__version__} "
        f"clip={clip_mode} aliases={','.join(aliases) or 'none'} "
        f"torch_load={torch_load_mode}",
        flush=True,
    )

    sys.path.insert(0, str(official_repo))
    sys.argv = [str(entrypoint), *forwarded]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
