"""Kinematics-aware metric R and the metric projection of the clean endpoint.

ProjFlow (Watanabe et al., CVPR 2026, Eq. 7/11) projects the predicted clean
motion x̂₁ onto the observation set under

    R = w_kin (I₃ ⊗ I_N ⊗ L_kin) + λ I,

L_kin the graph Laplacian of the 22-joint HumanML3D skeleton. R never couples
different frames or spatial channels, so R⁻¹ = I ⊗ R_J⁻¹ with the 22 × 22 block
R_J = w_kin L_kin + λ I applied along the joint axis.

For inpainting the observation operator A is a selector mask, so the
observation-space system A R⁻¹ Aᵀ + Σ is block-diagonal over (sample, frame):
each block is R_J⁻¹ restricted to the joints selected in that frame plus the
pseudo-observation variances. Upstream (external/ProjFlow,
diffusions/transport/projflow_helpers.py) assembles one m × m matrix per sample
in a Python loop; here every (sample, frame) block is padded to 22 × 22 and
solved in one batched Cholesky call. The math is the same (including the 1e-8
jitter); tests/projflow checks the two against each other.

Tensors follow upstream's layout: (B, D, L, J) = (batch, xyz, frames, joints).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# HumanML3D 22-joint kinematic chains (pelvis-rooted).
T2M_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)
NUM_JOINTS = 22
_JITTER = 1e-8


def skeleton_laplacian(num_joints: int = NUM_JOINTS, *, device=None, dtype=torch.float32) -> Tensor:
    """Unnormalised graph Laplacian L = D − A of the skeleton, (J, J)."""
    if num_joints != NUM_JOINTS:
        raise ValueError(f"only the {NUM_JOINTS}-joint HumanML3D skeleton is supported, got {num_joints}")
    adj = torch.zeros(num_joints, num_joints, device=device, dtype=dtype)
    for chain in T2M_CHAINS:
        for u, v in zip(chain[:-1], chain[1:]):
            adj[u, v] = adj[v, u] = 1.0
    return torch.diag(adj.sum(dim=1)) - adj


@dataclass(frozen=True)
class KinematicMetric:
    """R_J = w_kin L_kin + ridge I and the quantities ProjFlow derives from it.

    `euclidean()` gives the R = I ablation (paper Table 3, "Euclid.").
    """

    rinv_joint: Tensor       # (J, J) R_J⁻¹
    diag_rinv_joint: Tensor  # (J,)   diag(R_J⁻¹): r_i in σ² = r_i (1/π − 1)
    joint_weights_q: Tensor  # (J,)   q_j = 1 / ||col_j(R_J⁻¹)||²  (supp. Eq. 51)
    energy_joint: Tensor     # (J, J) w_kin L_kin + ridge I, the curvature norm (Eq. 17)

    @classmethod
    def kinematic(cls, w_kin: float = 10.0, ridge: float = 1.0, *, device=None,
                  dtype=torch.float32) -> "KinematicMetric":
        lap = skeleton_laplacian(device=device, dtype=dtype)
        lam, u = torch.linalg.eigh(lap)
        denom = w_kin * lam + float(ridge)
        rinv = (u / denom) @ u.mT
        col_norm_sq = (u**2) @ (1.0 / denom**2)
        return cls(
            rinv_joint=rinv,
            diag_rinv_joint=(u**2) @ (1.0 / denom),
            joint_weights_q=(1.0 / (col_norm_sq + 1e-12)).clamp_min(1e-12),
            energy_joint=w_kin * lap + ridge * torch.eye(NUM_JOINTS, device=device, dtype=dtype),
        )

    @classmethod
    def euclidean(cls, w_kin: float = 10.0, ridge: float = 1.0, *, device=None,
                  dtype=torch.float32) -> "KinematicMetric":
        # R = I everywhere, including the curvature norm ||·||_R (Eq. 17).
        # Matches upstream's use_projflow=False metric (R⁻¹ = I, unit weights);
        # upstream never evaluates curvature in that mode.
        eye = torch.eye(NUM_JOINTS, device=device, dtype=dtype)
        ones = torch.ones(NUM_JOINTS, device=device, dtype=dtype)
        return cls(rinv_joint=eye, diag_rinv_joint=ones, joint_weights_q=ones, energy_joint=eye)

    def apply_rinv(self, b: Tensor) -> Tensor:
        """(I ⊗ R_J⁻¹) b along the joint axis; b is (B, D, L, J)."""
        return torch.einsum("bdlj,jk->bdlk", b, self.rinv_joint.to(b))


def project_clean_endpoint(
    x1_hat: Tensor,
    selector: Tensor,
    targets: Tensor,
    metric: KinematicMetric,
    sigma2: Tensor | None = None,
) -> Tensor:
    """x̂₁ + R⁻¹ Aᵀ (A R⁻¹ Aᵀ + Σ)⁻¹ (y − A x̂₁) for a selector-mask A.

    Args:
        x1_hat:   (B, D, L, J) predicted clean motion.
        selector: (B, D, L, J) 0/1 rows of A; a (frame, joint) is selected for
                  every channel or none (upstream reads channel 0).
        targets:  (B, D, L, J) y on selected rows.
        sigma2:   (B, D, L, J) diagonal Σ (0 = hard constraint), or None.
    """
    B, D, L, J = x1_hat.shape
    dtype = x1_hat.dtype
    sel = selector[:, 0] > 0.5                                    # (B, L, J)
    sel_f = sel.to(dtype)
    rinv = metric.rinv_joint.to(x1_hat)

    # Per-(b, l) block: S R_J⁻¹ S + diag(σ² + jitter) on selected joints, identity
    # on unselected ones (their λ comes out 0 because their residual is zeroed).
    s_outer = sel_f.unsqueeze(-1) * sel_f.unsqueeze(-2)          # (B, L, J, J)
    diag = torch.full_like(sel_f, _JITTER)
    if sigma2 is not None:
        diag = diag + sigma2[:, 0] * sel_f
    diag = torch.where(sel, diag, torch.ones_like(diag))
    K = rinv * s_outer + torch.diag_embed(diag)
    K = 0.5 * (K + K.mT)

    resid = (targets - x1_hat) * sel_f.unsqueeze(1)                 # (B, D, L, J)
    rhs = resid.permute(0, 2, 3, 1)                                 # (B, L, J, D)
    chol = torch.linalg.cholesky(K)
    lam = torch.cholesky_solve(rhs, chol)                           # (B, L, J, D)
    lam = lam.permute(0, 3, 1, 2) * sel_f.unsqueeze(1)              # (B, D, L, J)
    return x1_hat + metric.apply_rinv(lam)
