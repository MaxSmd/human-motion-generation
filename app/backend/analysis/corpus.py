"""HumanML3D caption corpus — "does the text prior actually know this phrase?"

Constraint→text ablations live or die on one question: when we bolt a phrase like
"while limping" onto a prompt, has the text encoder ever *seen* that language? The
offline A/B that motivated this module found the answer predicts the outcome:

    "limping"                  → 160 clips → constrained jerk 0.019 (clean)
    "holding a box to a chest" →   2 clips → constrained jerk 0.154 (shakes)

Agreeing text only moves the prior if the prior has support there. So we index the
raw HumanML3D captions once and score a candidate phrase against them *before*
spending GPU time on it.

The index is built from the dataset's `texts/` dir (one file per clip, lines of
`caption#lemma/POS lemma/POS …#start#end`). We key on the LEMMA column, so a query
for "limping" hits captions that say "limps" / "limped". Built lazily, cached in
memory, and persisted to the state dir so a restart is instant.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from ..config import REPO

# Words that carry no motion semantics — every caption has them, so matching on
# them tells us nothing about whether the prior knows the *motion*.
STOPWORDS = frozenset("""
a an the this that these those and or but so then while as if of to in on at by for with
without from into onto over under up down out off again back forward
is are was were be been being do does did done has have had
he she they them their his her its it one someone somebody person man woman guy human
figure character subject who which what very quite slightly bit little more most some
keep keeps keeping kept remain remains remaining stay stays staying continue continues
seem seems appear appears try tries trying begin begins cause causes allow allows
""".split())

# Irregular surface→lemma pairs the suffix stripper below can't reach. Only the
# ones that actually show up in motion captions — this is not a general stemmer.
IRREGULAR = {
    "held": "hold", "bent": "bend", "kept": "keep", "stood": "stand", "sat": "sit",
    "lay": "lie", "lain": "lie", "swung": "swing", "bore": "bear", "drew": "draw",
    "fell": "fall", "flew": "fly", "got": "get", "hung": "hang", "knelt": "kneel",
    "leant": "lean", "leapt": "leap", "left": "left", "rose": "rise", "ran": "run",
    "sprang": "spring", "struck": "strike", "swept": "sweep", "threw": "throw",
    "wound": "wind", "spun": "spin", "slid": "slide", "shook": "shake",
    "crouched": "crouch", "feet": "foot", "knees": "knee", "legs": "leg",
}

_INDEX: dict | None = None


def texts_dir() -> Path:
    """Raw HumanML3D `texts/` dir. MGEN_TEXTS_DIR wins; else the vendored checkout."""
    env = os.environ.get("MGEN_TEXTS_DIR") or os.environ.get("RMG_TEXTS_DIR")
    if env:
        return Path(env)
    return REPO / "external" / "HumanML3D" / "HumanML3D" / "texts"


def _cache_path() -> Path:
    state = os.environ.get("MGEN_JOBS_STATE")
    base = Path(state).parent if state else REPO / ".appstate"
    return base / "caption_index.json"


def _parse_line(line: str) -> tuple[str, list[str]] | None:
    """`caption#a/DET man/NOUN kick/VERB#0.0#0.0` → (caption, [lemmas])."""
    parts = line.split("#")
    if len(parts) < 2 or not parts[0].strip():
        return None
    caption = parts[0].strip()
    lemmas = []
    for tok in parts[1].split():
        lemma = tok.rsplit("/", 1)[0].lower()
        if lemma and lemma not in STOPWORDS and lemma.isalpha():
            lemmas.append(lemma)
    return caption, lemmas


def _build() -> dict:
    """Scan texts/ → {vocab: {lemma: [clip_idx]}, caps: [[caption]], clips: [cid]}."""
    d = texts_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"HumanML3D texts dir not found at {d} — set MGEN_TEXTS_DIR to the "
            f"dataset's texts/ directory (one .txt per clip)."
        )
    t0 = time.time()
    # An empty dir is the common failure here: docker happily creates the bind
    # mount's source if it doesn't exist, so a missing checkout looks like a
    # present-but-empty corpus. Left unchecked, every phrase would score 0 and be
    # labelled "no support" — a confident wrong answer, which is worse than none.
    vocab: dict[str, set[int]] = {}
    caps: list[list[str]] = []
    clips: list[str] = []
    for path in sorted(d.glob("*.txt")):
        cid = path.stem
        if cid.startswith("M"):
            continue  # mirrored duplicates — same captions, would double every count
        idx = len(clips)
        clip_caps: list[str] = []
        try:
            raw = path.read_text(errors="ignore")
        except OSError:
            continue
        for line in raw.splitlines():
            parsed = _parse_line(line)
            if not parsed:
                continue
            caption, lemmas = parsed
            clip_caps.append(caption)
            for lemma in lemmas:
                vocab.setdefault(lemma, set()).add(idx)
        if not clip_caps:
            continue
        clips.append(cid)
        caps.append(clip_caps)
    if not clips:
        raise FileNotFoundError(
            f"HumanML3D texts dir at {d} has no readable caption files — the "
            f"dataset checkout is missing or empty, so phrase support can't be "
            f"scored. Point MGEN_TEXTS_DIR at the dataset's texts/ directory."
        )
    return {
        "vocab": {w: sorted(s) for w, s in vocab.items()},
        "caps": caps,
        "clips": clips,
        "build_seconds": round(time.time() - t0, 2),
    }


def _load() -> dict:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    cache = _cache_path()
    if cache.is_file():
        try:
            with open(cache) as f:
                data = json.load(f)
            if not data.get("clips"):
                raise ValueError("empty cached index")  # written before the corpus mounted
            data["vocab"] = {w: set(v) for w, v in data["vocab"].items()}
            _INDEX = data
            return _INDEX
        except (OSError, ValueError, KeyError):
            pass  # corrupt / empty cache → rebuild
    data = _build()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump({**data, "vocab": {w: list(v) for w, v in data["vocab"].items()}}, f)
    except OSError:
        pass  # read-only state dir → in-memory only
    data["vocab"] = {w: set(v) if not isinstance(v, set) else v for w, v in data["vocab"].items()}
    _INDEX = data
    return _INDEX


def _stem_candidates(word: str) -> list[str]:
    """Surface form → plausible lemmas. Crude on purpose: we only need to bridge
    query "limping" → corpus lemma "limp" without pulling spacy into the image."""
    w = word.lower()
    out = [w]
    if w in IRREGULAR:
        out.append(IRREGULAR[w])
    if w.endswith("ing"):
        stem = w[:-3]
        out += [stem, stem + "e"]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            out.append(stem[:-1])  # running → runn → run
    if w.endswith("ed"):
        stem = w[:-2]
        out += [stem, stem + "e", w[:-1]]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            out.append(stem[:-1])
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s"):
        out.append(w[:-1])
    return out


def _resolve(word: str, vocab: dict) -> tuple[str | None, set[int]]:
    """Surface word → (canonical lemma, clips mentioning it in any inflection).

    Unions over *every* candidate stem rather than stopping at the first hit: the
    dataset's tagger is inconsistent (some captions lemmatise "limping" to "limp",
    others leave it), so first-match would report 6 clips for a word the corpus
    actually covers in ~160. The reported lemma is the best-supported form.
    """
    hits: set[int] = set()
    best: tuple[int, str] | None = None
    for cand in _stem_candidates(word):
        got = vocab.get(cand)
        if not got:
            continue
        hits |= got
        if best is None or len(got) > best[0]:
            best = (len(got), cand)
    return (best[1] if best else None), hits


def phrase_stats(query: str, *, examples: int = 4) -> dict:
    """Score a candidate phrase against the caption corpus.

    Returns per-content-word clip support plus the strict conjunction (clips whose
    captions mention *every* content word). The per-word breakdown is the useful
    part: it names which token is out-of-distribution, which is exactly the term
    that will make the constrained sample fight its prior.
    """
    index = _load()
    vocab, caps, clips = index["vocab"], index["caps"], index["clips"]
    total = len(clips)

    words = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in STOPWORDS]
    seen: set[str] = set()
    terms = []
    hit_sets: list[set[int]] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        lemma, hits = _resolve(w, vocab)
        terms.append({
            "word": w,
            "lemma": lemma,
            "clips": len(hits),
            "frac": round(len(hits) / total, 5) if total else 0.0,
        })
        hit_sets.append(hits)

    both = set.intersection(*hit_sets) if hit_sets else set()
    ex: list[str] = []
    if both:
        need = [t["lemma"] for t in terms if t["lemma"]]
        for idx in sorted(both):
            for caption in caps[idx]:
                low = caption.lower()
                # prefer a caption that visibly contains the terms, not just the clip
                if all(any(c in low for c in _surface_forms(lemma)) for lemma in need):
                    ex.append(caption)
                    break
            if len(ex) >= examples:
                break
        if not ex:  # conjunction holds across sibling captions, not within one
            ex = [caps[idx][0] for idx in sorted(both)[:examples]]

    weakest = min(terms, key=lambda t: t["clips"]) if terms else None
    return {
        "query": query,
        "total_clips": total,
        "terms": terms,
        "all_clips": len(both),
        "all_frac": round(len(both) / total, 5) if total else 0.0,
        "weakest": weakest,
        "coverage": _coverage_label(len(both), weakest["clips"] if weakest else 0),
        "examples": ex,
    }


def rank_phrases(candidates: list[str], *, examples: int = 1) -> list[dict]:
    """Score several candidate phrasings of the same motion, best-supported first.

    The archetype map offers variants ("limping" / "walking with a limp" /
    "dragging their left leg") instead of committing to one. Which phrasing the
    encoder actually knows is a fact about the corpus, not a matter of taste, so
    we measure all of them and let the winner be the one with real support.

    Ties break toward the SHORTER phrase: fewer content words means a looser
    conjunction, so equal support at greater length is support for less specific
    language, and the extra words are just prompt dilution.
    """
    scored = [phrase_stats(c, examples=examples) for c in candidates if c and c.strip()]
    return sorted(scored, key=lambda s: (-s["all_clips"], len(s["terms"]), len(s["query"])))


def _surface_forms(lemma: str) -> list[str]:
    # removesuffix, not rstrip: rstrip("e") strips EVERY trailing e, turning
    # "see" into "s" and matching the wrong captions in the example picker.
    return [lemma, lemma + "s", lemma + "ing", lemma + "ed", lemma.removesuffix("e") + "ing"]


def _coverage_label(all_clips: int, weakest_clips: int) -> str:
    """Bucket the support into a go/no-go signal for the ablation.

    Keyed on the CONJUNCTION, not the per-word counts, because that is the part
    that predicted the two outcomes we have actually measured:

        "limping, dragging their left leg" → 12 clips co-occur → worked (0.019)
        "holding a box against their chest" →  0 clips co-occur → failed (0.154)

    The second phrase's individual words are all common (hold 1466, chest 488) —
    it's the *combination* the corpus has never seen, and that's what the encoder
    has no representation for. So per-word support can't be the signal; a phrase
    can be built entirely from known words and still be out of distribution.

    Calibrated on n=2 measured cases. Treat it as a prior, not a verdict — which
    is exactly why the ablation runs rather than trusting this.
    """
    if all_clips == 0:
        return "none"
    if all_clips >= 25:
        return "strong"
    if all_clips >= 5:
        return "medium"
    return "weak"


def index_status() -> dict:
    """Cheap probe: is the corpus reachable / already indexed?"""
    d = texts_dir()
    return {
        "texts_dir": str(d),
        "available": d.is_dir(),
        "indexed": _INDEX is not None or _cache_path().is_file(),
        "clips": len(_INDEX["clips"]) if _INDEX else None,
    }
