"""Warm in-memory state for the backend.

Loading a checkpoint + building the sampler is slow, so we do it once and keep
it resident. The representation/skeleton are cheap and shared; each checkpoint
gets its own lazily-built bundle (model + EMA-applied weights + sampler + text
encoder), reconstructed from that run's `config.json` so the dims always match
the weights.

This in-process inference path is rmg-specific (it builds rmg's sampler +
representation); other models are driven via the cluster control plane until a
model-pluggable inference path is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor

from rmg.flow import (
    CONSTRAINABLE_REPRESENTATIONS,
    RiemannianEulerSampler,
    SamplerCfg,
    WrappedGaussianPrior,
    build_bend_projector,
    build_room_energy_fn,
    parse_bends,
    parse_scene,
    place_motion,
)
from rmg.models import DiTConfig, Qwen3EmbeddingEncoder, RandomTextEncoder, RMGDiT
from rmg.models.text_encoder import TextEncoder
from shared.geometry import Skeleton
from rmg.representation import build_representation
from shared.utils import EMA, load_checkpoint

from . import config as cfgmod


@dataclass
class ModelBundle:
    model: RMGDiT
    sampler: RiemannianEulerSampler
    text_encoder: TextEncoder
    representation_name: str
    step: int
    cfg: DictConfig


class AppState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_cfg = cfgmod.compose_default()
        self._skeleton: Skeleton | None = None
        self._bundles: dict[str, ModelBundle] = {}

    # ------------------------------------------------------------------ skeleton

    def skeleton(self) -> Skeleton:
        """Reference skeleton (T-pose offsets) for forward kinematics. Cached."""
        if self._skeleton is None:
            offs = cfgmod.offsets_path(self.default_cfg)
            if not offs.exists():
                raise FileNotFoundError(
                    f"target_offsets.pt not found at {offs}. Set RMG_DATA_ROOT (or "
                    "RMG_OFFSETS) to a packed HumanML3D dir once it is copied down "
                    "from the cluster."
                )
            offsets = torch.load(offs, weights_only=True).float()
            self._skeleton = Skeleton(offsets=offsets)
        return self._skeleton

    # ------------------------------------------------------------------ model

    @staticmethod
    def _run_dir_of(ckpt: Path) -> Path:
        # runs/<run>/checkpoints/<ckpt>.pt  → runs/<run>
        return ckpt.parent.parent

    def _build_text_encoder(self, cfg: DictConfig) -> TextEncoder:
        te = cfg.text_encoder
        if str(te.type) == "qwen3":
            return Qwen3EmbeddingEncoder(
                model_name=str(te.model_name),
                cache_dir=str(te.cache_dir) if te.get("cache_dir") else None,
                max_length=int(te.max_length),
            )
        return RandomTextEncoder(text_dim=int(te.text_dim))

    def load_bundle(self, ckpt_path: str | Path) -> ModelBundle:
        key = str(Path(ckpt_path).resolve())
        if key in self._bundles:
            return self._bundles[key]

        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")

        # Prefer the run's own config snapshot so dims match the weights.
        run_cfg = cfgmod.load_run_config(self._run_dir_of(ckpt)) or self.default_cfg

        rep = build_representation(
            str(run_cfg.representation.name),
            num_joints=int(run_cfg.representation.get("num_joints", 22)),
        )

        dit_cfg = DiTConfig(
            input_dim=rep.ambient_dim,
            hidden_dim=int(run_cfg.model.hidden_dim),
            depth=int(run_cfg.model.depth),
            num_heads=int(run_cfg.model.num_heads),
            ffn_mult=int(run_cfg.model.ffn_mult),
            text_dim=int(run_cfg.model.text_dim),
            time_freq_dim=int(run_cfg.model.time_freq_dim),
            time_scale=float(run_cfg.model.get("time_scale", 1.0)),
            max_seq_len=int(run_cfg.model.max_seq_len),
        )
        model = RMGDiT(dit_cfg).to(self.device)
        state = load_checkpoint(ckpt, map_location=self.device)
        model.load_state_dict(state.model)
        if state.ema is not None:
            ema = EMA(model, decay=0.0)
            ema.load_state_dict(state.ema)
            ema.copy_to(model)
        model.eval()

        # Sampler shares the training prior; per-request guidance/num_steps are
        # passed at sample() time so they're not baked into the bundle.
        manifold = rep.build_manifold()
        if hasattr(rep, "prior_mu_from_skeleton"):
            try:
                mu = rep.prior_mu_from_skeleton(self.skeleton())
            except FileNotFoundError:
                mu = rep.prior_mu()  # offsets missing; T+R mu is skeleton-free
        else:
            mu = rep.prior_mu()
        prior = WrappedGaussianPrior(
            manifold, mu, sigma=float(run_cfg.train.get("prior_sigma", 1.0))
        )
        sampler = RiemannianEulerSampler(manifold=manifold, prior=prior, cfg=SamplerCfg())

        bundle = ModelBundle(
            model=model,
            sampler=sampler,
            text_encoder=self._build_text_encoder(run_cfg),
            representation_name=str(run_cfg.representation.name),
            step=int(state.step),
            cfg=run_cfg,
        )
        self._bundles[key] = bundle
        return bundle

    # ------------------------------------------------------------------ generate

    @torch.no_grad()
    def generate(
        self,
        bundle: ModelBundle,
        text: str,
        num_frames: int,
        guidance: float,
        num_steps: int,
        seed: int,
        constraints: list[dict] | None = None,
        ranges: list[dict] | None = None,
        scene: dict | None = None,
        room_guidance: float = 0.0,
    ) -> Tensor:
        """Text → (T, ambient_dim) sample on the manifold.

        `constraints` are exact bend-angle pins, `ranges` are bend min/max limits
        — both projected onto the joint's feasible bend each ODE step (see
        flow.constraints); `scene` adds euclidean room/obstacle guidance + exact
        spawn placement. All apply only to the quaternion-on-S^3 representations
        (tr/trp).
        """
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        cond = bundle.text_encoder.encode([text], device=self.device)

        fixed_values = fixed_mask = project_fn = energy_fn = None
        scene_obj = parse_scene(scene)
        if constraints or ranges or scene_obj:
            if bundle.representation_name not in CONSTRAINABLE_REPRESENTATIONS:
                raise NotImplementedError(
                    f"constraints need a quaternion representation "
                    f"({CONSTRAINABLE_REPRESENTATIONS}); this run uses "
                    f"{bundle.representation_name!r}."
                )
            num_joints = int(bundle.cfg.representation.get("num_joints", 22))
            # Both pins (exact bend) and ranges (bend min/max) are bend-angle
            # constraints projected each ODE step — see flow.constraints. Each
            # acts on the joint's controller quaternion (per-chain off-by-one).
            bend_specs = [*parse_bends(constraints), *parse_bends(ranges)]
            if bend_specs:
                project_fn = build_bend_projector(
                    bend_specs, self.skeleton(),
                    num_frames=int(num_frames), num_joints=num_joints, device=self.device,
                )
            if scene_obj and room_guidance:
                energy_fn = build_room_energy_fn(scene_obj, self.skeleton(), num_joints=num_joints)

        samples = bundle.sampler.sample(
            bundle.model,
            shape=(1, int(num_frames)),
            cond=cond,
            guidance_scale=float(guidance),
            num_steps=int(num_steps),
            device=self.device,
            generator=gen,
            fixed_values=fixed_values,
            fixed_mask=fixed_mask,
            project_fn=project_fn,
            energy_fn=energy_fn,
            guidance_weight=float(room_guidance) if energy_fn is not None else 0.0,
        )
        sample = samples[0]

        # Exact spawn placement: rigidly move the clip so it starts at the spawn
        # pose inside the room (translation + root orientation). tr/trp only.
        if scene_obj is not None:
            nj = int(bundle.cfg.representation.get("num_joints", 22))
            qd = 3 + 4 * nj
            trans = sample[:, :3]
            quats = sample[:, 3:qd].reshape(sample.shape[0], nj, 4)
            trans2, quats2 = place_motion(trans, quats, scene_obj.spawn)
            sample = sample.clone()
            sample[:, :3] = trans2
            sample[:, 3:qd] = quats2.reshape(sample.shape[0], -1)
        return sample


# Module-level singleton, initialised on FastAPI startup.
STATE: AppState | None = None


def get_state() -> AppState:
    global STATE
    if STATE is None:
        STATE = AppState()
    return STATE
