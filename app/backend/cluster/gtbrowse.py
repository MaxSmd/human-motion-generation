"""Browse available ground-truth clips (id + caption) on the cluster.

The Visualize tab needs to *hint which GT clips exist* and show each clip's
assigned text, without launching a GPU job. The packed `humanml3d.zip` may only
be mounted during jobs, so instead we read the always-present HumanML3D
submodule checkout (`cluster_humanml3d_dir()`): split lists for the clip ids and
`texts.zip` for the captions — both plain files we can read over SSH (`cat` +
`unzip -p`), no torch, no GPU.

Results are cached in-process (the single SSH connection is precious) keyed by
the query, so the dropdown can poll cheaply.
"""

from __future__ import annotations

import random
import shlex
import time

from .. import config as cfgmod
from . import ssh

# query-key → (ts, rows). Small TTL; the split lists never change mid-session.
_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_TTL = 300.0


def _subset_ids(train_ids: list[str], fraction: float, seed: int) -> list[str]:
    """Replay the dataset's train-subset selection on the regular (non-mirror)
    ids → sorted ids the model trained on at this fraction/seed. Mirrors the
    logic in shared.data.select_clip_ids so the browse list matches training."""
    regular = [c for c in train_ids if not c.startswith("M")]
    if fraction >= 1.0:
        return sorted(regular)
    n_keep = max(1, int(round(len(regular) * fraction)))
    rng = random.Random(seed)
    return sorted(rng.sample(regular, k=n_keep))


def gt_clips(
    split: str = "train",
    *,
    subset_fraction: float = 0.01,
    subset_seed: int = 0,
    limit: int = 60,
) -> list[dict]:
    """[{cid, caption}] for clips in `split`. For the train split the list is
    narrowed to the model's subset (fraction/seed); val/test are returned whole.
    Captions are the first line of each clip's texts.zip entry."""
    split = split if split in ("train", "val", "test") else "train"
    key = (split, round(subset_fraction, 6), subset_seed, limit)
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _TTL:
        return hit[1]

    base = shlex.quote(ssh.abs_remote(cfgmod.cluster_humanml3d_dir()))
    listing = ssh.run(f"cat {base}/{shlex.quote(split)}.txt", timeout=15, check=False)
    ids = [l.strip() for l in listing.stdout.splitlines() if l.strip()]
    if not ids:
        return []

    if split == "train":
        ids = _subset_ids(ids, subset_fraction, subset_seed)
    ids = ids[: max(1, limit)]

    # Batch-read the first caption per clip from texts.zip in ONE remote command
    # (the single SSH slot is serialized — never fan out). `<cid>\t<line>` rows;
    # caption is the text before the first '#'.
    quoted = " ".join(shlex.quote(c) for c in ids)
    script = (
        f"cd {base} 2>/dev/null || exit 0; "
        f"for c in {quoted}; do "
        "printf '%s\\t' \"$c\"; "
        "unzip -p texts.zip \"$c.txt\" 2>/dev/null | head -n1; "
        "done"
    )
    res = ssh.run(script, timeout=30, check=False)
    caps: dict[str, str] = {}
    for line in res.stdout.splitlines():
        cid, _, rest = line.partition("\t")
        cid = cid.strip()
        if cid:
            caps[cid] = rest.split("#", 1)[0].strip()

    rows = [{"cid": c, "caption": caps.get(c, "")} for c in ids]
    _CACHE[key] = (time.time(), rows)
    return rows
