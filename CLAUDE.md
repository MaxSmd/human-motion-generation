# Engineering notes

Non-obvious traps in this repo. `README.md` covers what the project is, the
results, setup and the dev loop; this file is the stuff that bites you.

## Layering

`shared/` is the only cross-model dependency. `rmg`, `momask` and `mardm` must
never import each other — if two models need the same thing, it belongs in
`shared`. `momask` in particular is verified rmg-free (`import momask` loads zero
`rmg` modules); keep it that way.

## Data and geometry

- **`weights_only=False`** is required when loading our packed `.pt` blobs
  (`dict[str, Tensor | list[str]]`, which torch ≥ 2.6 rejects under strict mode).
  Safe because we produce those files ourselves.
- **HumanML3D's L/R variable naming is swapped** upstream
  (`l_hip, r_hip, sdr_r, sdr_l = [2, 1, 17, 16]` — `l_hip` holds index 2, the
  *right* hip). When touching this, copy upstream's variable names verbatim and
  keep them confined to the smallest possible scope.
- **`mirror_augment=True` is train-only.** The dataset already ignores it when
  `split != "train"`, but pass `False` explicitly for eval splits.
- **No temporal sign-continuity on quaternions.** `HumanML3DDataset.__getitem__`
  restricts to the upper hemisphere (`q_w ≥ 0`) and deliberately does *not* call
  `make_continuous`. Sign propagation let near-180° joint frames settle in the
  lower hemisphere, making `(x0, x1)` near-antipodal and blowing the
  flow-matching target up to ~7000. `q` and `−q` are the same rotation, so FK and
  the 263-D features are unaffected.
- **`mardm.data.EssentialDataset`'s preload cache duplicates that preprocessing
  by hand.** If you change the shared loader's normalisation, change the cache
  too — otherwise `preload=True` and `preload=False` silently produce different
  features, and the cluster runs use `preload=True`.
- **Never call `normalize_quaternions` on state being integrated.** Inside the ODE
  loop it flips signs mid-trajectory; this was a real bug in the bend projector.

## Evaluation

- FID is only comparable across runs that share a guidance scale and feature
  normalisation — never quote it against another paper's number without both.
- Our FID figures are single-pass over the full test split; published RMG numbers
  are 20-replication means. The replication spread here is ±0.024, so differences
  smaller than that are noise.
- `shared.eval`'s Guo evaluator uses `get_co_embeddings` for index-paired metrics.
  `get_motion_embeddings` reorders by length and is only valid for set statistics
  (FID, diversity).

## Cluster

- **Don't `pip install` inside an sbatch** — rebuild the enroot image instead
  (`containers/requirements.txt` → `sbatch slurm/build_image.sbatch`).
- Every slurm script resolves the repo root from `SLURM_SUBMIT_DIR` when set, and
  otherwise walks up from its own location: `../..` for `slurm/<model>/*`, one
  `dirname` for scripts at `slurm/` root. Keep that right when moving files.
- **`WANDB_MODE=offline` by default**; cluster egress is not always available.
- Mixed GPU generations on the `12g` partition: request `--gres=gpu:1,ccc:75`.
  `--exclude` is rejected, and the older nodes have no kernels for this torch
  build.
- Don't stack the SLURM queue directly — batch through the backend's local queue.

## The app

- The images **bake `src/`** rather than mounting it, so any code change needs
  `docker compose build` (+ `up`) before it is live.
- Job history lives on a host bind mount (`.appstate/`), not in a named volume,
  so `docker compose down -v` cannot wipe it. The media cache *is* regenerable
  and does live in a volume.
- `nn.MultiheadAttention` doesn't reliably dispatch to flash/SDPA; on
  speed-critical paths call `F.scaled_dot_product_attention` directly.

## Testing

`pytest` is 256 tests and needs neither GPU nor data. `torchdiffeq` is a declared
dependency but easy to miss in a fresh venv — without it the entire `tests/mardm`
suite fails at *collection*, which reads like an environment quirk and silently
hides real regressions. Install it rather than ignoring those files.
