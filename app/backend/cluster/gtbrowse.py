"""Browse available ground-truth clips (id + caption) on the cluster.

The Visualize tab needs to *hint which GT clips exist* and show each clip's
assigned text, without launching a GPU job. The packed `humanml3d.zip` may only
be mounted during jobs, so instead we read the always-present HumanML3D
submodule checkout (`cluster_humanml3d_dir()`): split lists for the clip ids and
`texts.zip` for the captions — both plain files we can read over SSH (`cat` +
`unzip -p`), no torch, no GPU.

For "GT vs prediction" the tab also wants to mark which clips a given run trained
on. `tag_seen=True` replays that run's subset selection (fraction/seed/n) and
returns each clip tagged `seen` — clips in the training set vs unseen clips
(held-out train clips, or val/test when the run trained on the whole split).

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


def _select_seen(regular: list[str], fraction: float, seed: int, n: int) -> list[str]:
    """Replay the dataset's train-subset selection on the regular (non-mirror)
    ids → sorted ids the model trained on. Mirrors shared.data.select_clip_ids
    (regular-only; mirrors are dupes and uninteresting to browse)."""
    if n > 0:
        rng = random.Random(seed)
        return sorted(rng.sample(regular, k=min(n, len(regular))))
    if fraction < 1.0:
        rng = random.Random(seed)
        n_keep = max(1, int(round(len(regular) * fraction)))
        return sorted(rng.sample(regular, k=n_keep))
    return sorted(regular)  # whole split — the run saw everything


def _read_split(base: str, split: str) -> list[str]:
    """Clip ids in `<split>.txt` (empty list if the file is missing)."""
    res = ssh.run(f"cat {base}/{shlex.quote(split)}.txt", timeout=15, check=False)
    return [l.strip() for l in res.stdout.splitlines() if l.strip()]


def _captions(base: str, ids: list[str]) -> dict[str, str]:
    """{cid: caption} — the first caption per clip from texts.zip, read in ONE
    remote command (the single SSH slot is serialized — never fan out). Entries
    inside texts.zip are stored UNDER a `texts/` prefix (`texts/<cid>.txt`) —
    request that path, not bare `<cid>.txt`, or unzip matches nothing and every
    caption comes back empty. Fall back to the unpacked `texts/` dir when the zip
    is absent. Capture into a var and printf with an explicit '\\n' so every clip
    emits exactly one terminated row even when the lookup is empty — otherwise a
    caption-less clip leaves no newline and the next clip's id is appended onto
    the same line. Caption is the text before the first '#'."""
    if not ids:
        return {}
    quoted = " ".join(shlex.quote(c) for c in ids)
    script = (
        f"cd {base} 2>/dev/null || exit 0; "
        f"for c in {quoted}; do "
        "cap=$(unzip -p texts.zip \"texts/$c.txt\" 2>/dev/null | head -n1); "
        "[ -z \"$cap\" ] && cap=$(head -n1 \"texts/$c.txt\" 2>/dev/null); "
        "printf '%s\\t%s\\n' \"$c\" \"$cap\"; "
        "done"
    )
    res = ssh.run(script, timeout=30, check=False)
    caps: dict[str, str] = {}
    for line in res.stdout.splitlines():
        cid, _, rest = line.partition("\t")
        cid = cid.strip()
        if cid:
            caps[cid] = rest.split("#", 1)[0].strip()
    return caps


def _search_caps(base: str, query: str, cap: int) -> dict[str, str]:
    """Caption search across the WHOLE dataset in one remote command: zipgrep the
    captions inside texts.zip for `query` → ordered {cid: caption} (regular clips
    only; mirrors are caption dupes). Fixed-string + case-insensitive so the box
    behaves like a plain substring filter, not a regex. `texts/<cid>.txt:<line>`
    rows; first caption per clip wins; output capped (many lines per clip)."""
    if not query:
        return {}
    # Prefer grepping the unpacked `texts/` dir (near-instant); fall back to
    # zipgrep over texts.zip (extracts every entry — slow, but always present).
    # Both emit `texts/<cid>.txt:<caption line>`.
    q = shlex.quote(query)
    res = ssh.run(
        f"cd {base} 2>/dev/null || exit 0; "
        f"if [ -d texts ]; then grep -riF -- {q} texts/ 2>/dev/null; "
        f"else zipgrep -iF -- {q} texts.zip 2>/dev/null; fi | head -n {int(cap)}",
        timeout=30, check=False,
    )
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        name, sep, rest = line.partition(":")  # texts/<cid>.txt : <caption line>
        if not sep:
            continue
        cid = name.rsplit("/", 1)[-1]
        if cid.endswith(".txt"):
            cid = cid[:-4]
        if not cid or cid.startswith("M") or cid in out:
            continue
        out[cid] = rest.split("#", 1)[0].strip()
    return out


def gt_clips(
    split: str = "train",
    *,
    subset_fraction: float = 1.0,
    subset_seed: int = 0,
    subset_n: int = 0,
    limit: int = 60,
    tag_seen: bool = False,
    q: str = "",
) -> list[dict]:
    """Available GT clips for the Visualize tab.

    Plain browse (`tag_seen=False`): `[{cid, caption}]` for the first `limit`
    clips of `split`.

    Seen/unseen (`tag_seen=True`): `[{cid, caption, seen}]` for a run's training
    subset (fraction/seed/n) → up to `limit/2` clips the run trained on (`seen`)
    plus up to `limit/2` it did not. Unseen clips are the held-out train clips
    when the run used a subset, else val+test clips when it trained on the whole
    train split.

    `q` (caption substring) searches the whole dataset (zipgrep), then keeps only
    matches that belong to the relevant universe (the split, or the run's
    seen/unseen sets) so search results stay consistent with the unfiltered list."""
    split = split if split in ("train", "val", "test") else "train"
    q = (q or "").strip()
    key = (
        split, round(subset_fraction, 6), subset_seed, subset_n, limit, tag_seen, q,
    )
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _TTL:
        return hit[1]

    base = shlex.quote(ssh.abs_remote(cfgmod.cluster_humanml3d_dir()))
    # On a search, pull captions straight from zipgrep (over-fetch to survive the
    # per-universe filter below); otherwise read them for the chosen ids only.
    matched = _search_caps(base, q, cap=max(limit * 6, 200)) if q else {}

    if not tag_seen:
        split_set = set(_read_split(base, split))
        if q:
            ids = [c for c in matched if c in split_set][: max(1, limit)]
            rows = [{"cid": c, "caption": matched[c]} for c in ids]
        else:
            ids = [c for c in _read_split(base, split)][: max(1, limit)]
            caps = _captions(base, ids)
            rows = [{"cid": c, "caption": caps.get(c, "")} for c in ids]
        _CACHE[key] = (time.time(), rows)
        return rows

    # Seen/unseen for a run's training subset.
    train = _read_split(base, "train")
    if not train:
        return []
    regular = [c for c in train if not c.startswith("M")]
    seen = _select_seen(regular, subset_fraction, subset_seed, subset_n)
    seen_set = set(seen)

    held_out = [c for c in regular if c not in seen_set]
    if held_out:
        unseen = held_out  # run used a subset → held-out train clips are unseen
    else:
        # Run trained on the whole train split → unseen lives in the other splits.
        unseen = sorted(c for c in (_read_split(base, "val") + _read_split(base, "test"))
                        if not c.startswith("M"))

    if q:
        seen = [c for c in seen if c in matched]
        unseen = [c for c in unseen if c in matched]

    half = max(1, limit // 2)
    seen_pick = seen[:half]
    unseen_pick = unseen[:half]
    caps = matched if q else _captions(base, seen_pick + unseen_pick)
    rows = (
        [{"cid": c, "caption": caps.get(c, ""), "seen": True} for c in seen_pick]
        + [{"cid": c, "caption": caps.get(c, ""), "seen": False} for c in unseen_pick]
    )
    _CACHE[key] = (time.time(), rows)
    return rows
