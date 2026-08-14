"""Text-conditioned token transformers for MoMask."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def cosine_mask_ratio(step: int, total_steps: int) -> float:
    """Muse/MoMask-style cosine schedule from mostly masked to unmasked."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    tau = min(max(step / total_steps, 0.0), 1.0)
    return math.cos(math.pi * tau / 2.0)


def top_k_logits(logits: Tensor, thres: float = 0.9) -> Tensor:
    """Keep the highest-probability tail used by the official MoMask sampler."""
    if thres >= 1.0:
        return logits
    if thres < 0.0:
        raise ValueError("top-k filter threshold must be non-negative")
    k = max(1, math.ceil((1.0 - thres) * logits.shape[-1]))
    values, indices = logits.topk(k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, indices, values)


def sample_logits(logits: Tensor, *, temperature: float = 1.0, topk_filter_thres: float = 1.0) -> Tensor:
    filtered = top_k_logits(logits, topk_filter_thres)
    probs = F.softmax(filtered / max(temperature, 1e-6), dim=-1)
    return torch.distributions.Categorical(probs=probs).sample()


@dataclass
class TokenTransformerConfig:
    vocab_size: int = 512
    text_dim: int = 1024
    code_dim: int = 512
    hidden_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    ffn_dim: int = 1024
    max_seq_len: int = 196
    dropout: float = 0.1
    architecture: str = "legacy"


class _TransformerBackbone(nn.Module):
    def __init__(self, cfg: TokenTransformerConfig) -> None:
        super().__init__()
        if cfg.architecture not in {"legacy", "paper"}:
            raise ValueError("transformer architecture must be 'legacy' or 'paper'")
        self.cfg = cfg
        if cfg.architecture == "paper":
            position = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, cfg.hidden_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / cfg.hidden_dim)
            )
            pe = torch.zeros(1, cfg.max_seq_len, cfg.hidden_dim)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_embed", pe)
            self.pos_dropout = nn.Dropout(cfg.dropout)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.hidden_dim))
            self.pos_dropout = nn.Identity()
        self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        if cfg.architecture == "paper":
            self.register_buffer("null_text", torch.zeros(cfg.text_dim))
        else:
            self.null_text = nn.Parameter(torch.zeros(cfg.text_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.architecture != "paper",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.depth)
        self.norm = nn.Identity() if cfg.architecture == "paper" else nn.LayerNorm(cfg.hidden_dim)
        if cfg.architecture != "paper":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _paper_init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _condition(self, cond: Tensor | None, B: int, device: torch.device, drop_cond_mask: Tensor | None) -> Tensor:
        if cond is None:
            cond = self.null_text.unsqueeze(0).expand(B, -1)
        if drop_cond_mask is not None:
            cond = cond.clone()
            cond[drop_cond_mask] = self.null_text.to(device=device, dtype=cond.dtype)
        return self.text_proj(cond)

    def _encode(self, x: Tensor, cond: Tensor | None, mask: Tensor | None, drop_cond_mask: Tensor | None) -> Tensor:
        B, T, _ = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        h = self.pos_dropout(x + self.pos_embed[:, :T])
        c = self._condition(cond, B, h.device, drop_cond_mask).unsqueeze(1)
        src = torch.cat([c, h], dim=1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask],
                dim=1,
            )
        out = self.encoder(src, src_key_padding_mask=key_padding_mask)
        return self.norm(out[:, 1:])


class MaskedMotionTransformer(_TransformerBackbone):
    """Bidirectional masked transformer for base-layer motion tokens."""

    def __init__(self, cfg: TokenTransformerConfig | None = None) -> None:
        cfg = cfg or TokenTransformerConfig()
        super().__init__(cfg)
        self.mask_token_id = cfg.vocab_size
        self.pad_token_id = cfg.vocab_size + 1
        if cfg.architecture == "paper":
            self.token_embed = nn.Embedding(cfg.vocab_size + 2, cfg.code_dim)
            self.input_proj = nn.Linear(cfg.code_dim, cfg.hidden_dim)
            self.to_logits = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim, eps=1e-12),
                nn.Linear(cfg.hidden_dim, cfg.vocab_size),
            )
            self.apply(self._paper_init)
        else:
            self.token_embed = nn.Embedding(cfg.vocab_size + 1, cfg.hidden_dim)
            self.input_proj = nn.Identity()
            self.to_logits = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(
        self,
        tokens: Tensor,
        *,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        drop_cond_mask: Tensor | None = None,
    ) -> Tensor:
        if self.cfg.architecture == "paper" and mask is not None:
            tokens = torch.where(mask, tokens, self.pad_token_id)
        max_token_id = self.pad_token_id if self.cfg.architecture == "paper" else self.mask_token_id
        h = self.input_proj(self.token_embed(tokens.clamp(0, max_token_id)))
        h = self._encode(h, cond=cond, mask=mask, drop_cond_mask=drop_cond_mask)
        return self.to_logits(h)

    def training_loss(
        self,
        tokens: Tensor,
        *,
        cond: Tensor | None = None,
        valid_mask: Tensor | None = None,
        cond_drop_prob: float = 0.1,
        force_full_mask: bool = False,
    ) -> Tensor:
        B, T = tokens.shape
        device = tokens.device
        if force_full_mask:
            predict_mask = (
                valid_mask.bool().clone()
                if valid_mask is not None
                else torch.ones(B, T, dtype=torch.bool, device=device)
            )
            corrupted = torch.where(predict_mask, self.mask_token_id, tokens)
        else:
            tau = torch.rand(B, device=device)
            ratio = torch.cos(math.pi * tau / 2.0).clamp_min(1.0 / max(T, 1))
            corrupted = tokens.clone()
            if self.cfg.architecture == "paper":
                candidates = (
                    valid_mask.bool()
                    if valid_mask is not None
                    else torch.ones(B, T, dtype=torch.bool, device=device)
                )
                n_masked = torch.round(T * ratio).long().clamp(min=1)
                ranks = torch.rand(B, T, device=device).argsort(dim=-1).argsort(dim=-1)
                predict_mask = (ranks < n_masked.unsqueeze(-1)) & candidates
                random_replace = predict_mask & (torch.rand(B, T, device=device) < 0.1)
                mask_replace = (
                    predict_mask
                    & ~random_replace
                    & (torch.rand(B, T, device=device) < 0.88)
                )
                random_tokens = torch.randint_like(corrupted, high=self.cfg.vocab_size)
                corrupted = torch.where(random_replace, random_tokens, corrupted)
                corrupted = torch.where(mask_replace, self.mask_token_id, corrupted)
            else:
                predict_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for i in range(B):
                    candidates = (
                        valid_mask[i]
                        if valid_mask is not None
                        else torch.ones(T, dtype=torch.bool, device=device)
                    )
                    idx = candidates.nonzero(as_tuple=False).flatten()
                    if idx.numel() == 0:
                        continue
                    n = max(1, int(math.ceil(idx.numel() * float(ratio[i]))))
                    chosen = idx[torch.randperm(idx.numel(), device=device)[:n]]
                    predict_mask[i, chosen] = True
                replace_prob = torch.rand(B, T, device=device)
                random_tokens = torch.randint_like(corrupted, high=self.cfg.vocab_size)
                corrupted = torch.where(predict_mask & (replace_prob < 0.8), self.mask_token_id, corrupted)
                corrupted = torch.where(
                    predict_mask & (replace_prob >= 0.8) & (replace_prob < 0.9),
                    random_tokens,
                    corrupted,
                )
        drop = torch.rand(B, device=device) < cond_drop_prob
        logits = self(corrupted, cond=cond, mask=valid_mask, drop_cond_mask=drop)
        loss_mask = predict_mask if valid_mask is None else predict_mask & valid_mask
        if not bool(loss_mask.any()):
            raise ValueError("masked-transformer loss requires at least one valid token")
        return F.cross_entropy(logits[loss_mask], tokens[loss_mask])

    @torch.no_grad()
    def generate(
        self,
        *,
        cond: Tensor | None,
        seq_len: int,
        steps: int = 10,
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 1.0,
        sample: bool = False,
        remask_kept_tokens: bool = True,
        mask: Tensor | None = None,
    ) -> Tensor:
        if cond is not None:
            B, device = cond.shape[0], cond.device
        elif mask is not None:
            B, device = mask.shape[0], mask.device
        else:
            B, device = 1, self.pos_embed.device
        tokens = torch.full((B, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        unknown = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        if mask is not None:
            unknown = unknown & mask
        for step in range(1, steps + 1):
            logits_c = self(tokens, cond=cond, mask=mask)
            if guidance_scale != 1.0 and cond is not None:
                logits_u = self(tokens, cond=None, mask=mask)
                logits = logits_u + guidance_scale * (logits_c - logits_u)
            else:
                logits = logits_c
            pred = (
                sample_logits(logits, temperature=temperature, topk_filter_thres=topk_filter_thres)
                if sample
                else logits.argmax(dim=-1)
            )
            tokens = torch.where(unknown, pred, tokens)
            conf = logits.softmax(dim=-1).gather(2, pred.unsqueeze(-1)).squeeze(-1)
            if not remask_kept_tokens:
                conf = conf.masked_fill(~unknown, 1e5)

            ratio = cosine_mask_ratio(step, steps)
            next_unknown = torch.zeros_like(unknown)
            for i in range(B):
                active = mask[i] if mask is not None else torch.ones(seq_len, dtype=torch.bool, device=device)
                n_remask = int(round(active.sum().item() * ratio))
                if n_remask > 0:
                    scores_i = conf[i].masked_fill(~active, float("inf"))
                    remask = scores_i.argsort()[:n_remask]
                    next_unknown[i, remask] = True
            tokens[next_unknown] = self.mask_token_id
            unknown = next_unknown
        if mask is not None:
            tokens = torch.where(mask, tokens, torch.zeros_like(tokens))
        return tokens


class ResidualTransformer(_TransformerBackbone):
    """Predicts residual-layer tokens from already generated lower layers."""

    def __init__(
        self,
        cfg: TokenTransformerConfig | None = None,
        num_quantizers: int = 6,
        separate_level_heads: bool = True,
    ) -> None:
        cfg = cfg or TokenTransformerConfig()
        super().__init__(cfg)
        self.num_quantizers = num_quantizers
        self.separate_level_heads = separate_level_heads
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.level_embed = nn.Embedding(num_quantizers, cfg.hidden_dim)
        if separate_level_heads:
            self.to_logits = nn.ModuleList(
                [nn.Linear(cfg.hidden_dim, cfg.vocab_size) for _ in range(num_quantizers)]
            )
        else:
            self.to_logits = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(
        self,
        prev_tokens: Tensor,
        target_level: int,
        *,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        drop_cond_mask: Tensor | None = None,
    ) -> Tensor:
        if prev_tokens.dim() != 3:
            raise ValueError(f"prev_tokens must be (B, L, T), got {tuple(prev_tokens.shape)}")
        if not 1 <= target_level < self.num_quantizers:
            raise ValueError(f"target_level must be in [1, {self.num_quantizers - 1}]")
        B, L, T = prev_tokens.shape
        if L != target_level:
            raise ValueError(f"expected {target_level} previous levels, got {L}")
        h = self.token_embed(prev_tokens).sum(dim=1)
        h = h + self.level_embed(torch.full((B, T), target_level, device=prev_tokens.device, dtype=torch.long))
        h = self._encode(h, cond=cond, mask=mask, drop_cond_mask=drop_cond_mask)
        if self.separate_level_heads:
            return self.to_logits[target_level](h)
        return self.to_logits(h)

    def training_loss(
        self,
        tokens: Tensor,
        target_level: int,
        *,
        cond: Tensor | None = None,
        valid_mask: Tensor | None = None,
        cond_drop_prob: float = 0.2,
    ) -> Tensor:
        prev = tokens[:, :target_level]
        target = tokens[:, target_level]
        drop = torch.rand(tokens.shape[0], device=tokens.device) < cond_drop_prob
        logits = self(prev, target_level, cond=cond, mask=valid_mask, drop_cond_mask=drop)
        if valid_mask is None:
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        if not bool(valid_mask.any()):
            raise ValueError("residual-transformer loss requires at least one valid token")
        return F.cross_entropy(logits[valid_mask], target[valid_mask])

    @torch.no_grad()
    def generate_residuals(
        self,
        base_tokens: Tensor,
        *,
        cond: Tensor | None,
        num_quantizers: int | None = None,
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 1.0,
        sample: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        tokens = base_tokens.unsqueeze(1)
        Q = num_quantizers or self.num_quantizers
        for level in range(1, Q):
            logits_c = self(tokens, level, cond=cond, mask=mask)
            if guidance_scale != 1.0 and cond is not None:
                logits_u = self(tokens, level, cond=None, mask=mask)
                logits = logits_u + guidance_scale * (logits_c - logits_u)
            else:
                logits = logits_c
            next_tokens = (
                sample_logits(logits, temperature=temperature, topk_filter_thres=topk_filter_thres)
                if sample
                else logits.argmax(dim=-1)
            )
            tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        return tokens


class CodebookResidualTransformer(_TransformerBackbone):
    """Official-style residual transformer using per-quantizer code embeddings.

    Unlike `ResidualTransformer`, this mirrors MoMask's residual stage more
    closely: previous residual levels are embedded with level-specific code
    embeddings and summed in code space before the transformer predicts the
    next quantizer's token distribution.
    """

    def __init__(
        self,
        cfg: TokenTransformerConfig | None = None,
        num_quantizers: int = 6,
        code_dim: int = 256,
        share_weight: bool = False,
    ) -> None:
        cfg = cfg or TokenTransformerConfig()
        super().__init__(cfg)
        if num_quantizers < 2:
            raise ValueError("num_quantizers must be >= 2")
        self.num_quantizers = num_quantizers
        self.code_dim = code_dim
        self.share_weight = share_weight
        self.pad_id = cfg.vocab_size
        n_residual = num_quantizers - 1
        if share_weight:
            if n_residual < 2:
                raise ValueError("share_weight requires at least 3 quantizers")
            self.embed_proj_shared_weight = nn.Parameter(torch.empty(n_residual - 1, cfg.vocab_size + 1, code_dim))
            self.token_embed_weight_ = nn.Parameter(torch.empty(1, cfg.vocab_size + 1, code_dim))
            self.output_proj_weight_ = nn.Parameter(torch.empty(1, cfg.vocab_size + 1, code_dim))
            nn.init.normal_(self.embed_proj_shared_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.token_embed_weight_, mean=0.0, std=0.02)
            nn.init.normal_(self.output_proj_weight_, mean=0.0, std=0.02)
            self.output_proj_bias = None
        else:
            self.token_embed_weight = nn.Parameter(torch.empty(n_residual, cfg.vocab_size + 1, code_dim))
            self.output_proj_weight = nn.Parameter(torch.empty(n_residual, cfg.vocab_size + 1, code_dim))
            self.output_proj_bias = nn.Parameter(torch.zeros(n_residual, cfg.vocab_size + 1))
            nn.init.normal_(self.token_embed_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.output_proj_weight, mean=0.0, std=0.02)
        self.input_proj = nn.Linear(code_dim, cfg.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim, eps=1e-12 if cfg.architecture == "paper" else 1e-5),
            nn.Linear(cfg.hidden_dim, code_dim),
        )
        self.quant_embed = (
            nn.Linear(num_quantizers, cfg.hidden_dim)
            if cfg.architecture == "paper"
            else nn.Embedding(num_quantizers, cfg.hidden_dim)
        )
        if cfg.architecture == "paper":
            self.apply(self._paper_init)

    def _token_embed_weight(self) -> Tensor:
        if self.share_weight:
            return torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)
        return self.token_embed_weight

    def _output_proj_weight(self) -> Tensor:
        if self.share_weight:
            return torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
        return self.output_proj_weight

    def _history_codes(self, prev_tokens: Tensor, target_level: int) -> Tensor:
        B, L, T = prev_tokens.shape
        if L != target_level:
            raise ValueError(f"expected {target_level} previous levels, got {L}")
        safe_tokens = prev_tokens.clamp(0, self.pad_id)
        token_embed_weight = self._token_embed_weight()
        out = torch.zeros(B, T, self.code_dim, device=prev_tokens.device, dtype=token_embed_weight.dtype)
        for level in range(target_level):
            out = out + F.embedding(safe_tokens[:, level], token_embed_weight[level])
        return out

    def _encode_codes(
        self,
        history_codes: Tensor,
        target_level: int | Tensor,
        *,
        cond: Tensor | None,
        mask: Tensor | None,
        drop_cond_mask: Tensor | None,
    ) -> Tensor:
        B, T, _ = history_codes.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        h = self.pos_dropout(self.input_proj(history_codes) + self.pos_embed[:, :T])
        c = self._condition(cond, B, h.device, drop_cond_mask).unsqueeze(1)
        levels = (
            torch.full((B,), target_level, device=h.device, dtype=torch.long)
            if isinstance(target_level, int)
            else target_level.to(device=h.device, dtype=torch.long)
        )
        if levels.shape != (B,):
            raise ValueError(f"target levels must have shape ({B},), got {tuple(levels.shape)}")
        if self.cfg.architecture == "paper":
            q = self.quant_embed(F.one_hot(levels, num_classes=self.num_quantizers).to(h.dtype)).unsqueeze(1)
        else:
            q = self.quant_embed(levels).unsqueeze(1)
        src = torch.cat([c, q, h], dim=1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = torch.cat(
                [torch.zeros(B, 2, dtype=torch.bool, device=mask.device), ~mask],
                dim=1,
            )
        out = self.encoder(src, src_key_padding_mask=key_padding_mask)
        return self.norm(out[:, 2:])

    def _project_logits(self, h: Tensor, target_level: int | Tensor) -> Tensor:
        code = self.output_proj(h)
        if isinstance(target_level, int):
            idx = target_level - 1
            weight = self._output_proj_weight()[idx, : self.cfg.vocab_size]
            logits = code @ weight.t()
        else:
            idx = target_level.to(device=h.device, dtype=torch.long) - 1
            weight = self._output_proj_weight()[idx, : self.cfg.vocab_size]
            logits = torch.einsum("btc,bvc->btv", code, weight)
        if self.output_proj_bias is not None:
            bias = self.output_proj_bias[idx, : self.cfg.vocab_size]
            logits = logits + (bias if isinstance(target_level, int) else bias.unsqueeze(1))
        return logits

    def forward(
        self,
        prev_tokens: Tensor,
        target_level: int,
        *,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        drop_cond_mask: Tensor | None = None,
    ) -> Tensor:
        if prev_tokens.dim() != 3:
            raise ValueError(f"prev_tokens must be (B, L, T), got {tuple(prev_tokens.shape)}")
        if not 1 <= target_level < self.num_quantizers:
            raise ValueError(f"target_level must be in [1, {self.num_quantizers - 1}]")
        history_codes = self._history_codes(prev_tokens, target_level)
        h = self._encode_codes(
            history_codes,
            target_level,
            cond=cond,
            mask=mask,
            drop_cond_mask=drop_cond_mask,
        )
        return self._project_logits(h, target_level)

    def training_loss(
        self,
        tokens: Tensor,
        target_level: int,
        *,
        cond: Tensor | None = None,
        valid_mask: Tensor | None = None,
        cond_drop_prob: float = 0.2,
    ) -> Tensor:
        prev = tokens[:, :target_level]
        target = tokens[:, target_level]
        drop = torch.rand(tokens.shape[0], device=tokens.device) < cond_drop_prob
        logits = self(prev, target_level, cond=cond, mask=valid_mask, drop_cond_mask=drop)
        if valid_mask is None:
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        if not bool(valid_mask.any()):
            raise ValueError("residual-transformer loss requires at least one valid token")
        return F.cross_entropy(logits[valid_mask], target[valid_mask])

    def sampled_training_loss(
        self,
        tokens: Tensor,
        *,
        cond: Tensor | None = None,
        valid_mask: Tensor | None = None,
        cond_drop_prob: float = 0.2,
    ) -> tuple[Tensor, Tensor]:
        """Official residual objective with one quantizer level per sample."""
        if tokens.dim() != 3 or tokens.shape[1] != self.num_quantizers:
            raise ValueError(
                f"tokens must be (B, {self.num_quantizers}, T), got {tuple(tokens.shape)}"
            )
        B, Q, T = tokens.shape
        device = tokens.device
        candidates = (
            valid_mask.bool()
            if valid_mask is not None
            else torch.ones(B, T, dtype=torch.bool, device=device)
        )
        if not bool(candidates.any()):
            raise ValueError("residual-transformer loss requires at least one valid token")

        noise = torch.rand(B, device=device)
        schedule = 1.0 - torch.cos(noise * math.pi * 0.5)
        target_levels = torch.round(schedule * (Q - 2)).long() + 1

        safe_tokens = torch.where(
            candidates.unsqueeze(1),
            tokens,
            torch.full_like(tokens, self.pad_id),
        )
        embed_weight = self._token_embed_weight()
        level_codes = [
            F.embedding(safe_tokens[:, level], embed_weight[level])
            for level in range(Q - 1)
        ]
        history_by_level = torch.stack(level_codes, dim=1).cumsum(dim=1)
        batch_idx = torch.arange(B, device=device)
        history = history_by_level[batch_idx, target_levels - 1]
        target = tokens[batch_idx, target_levels]
        drop = torch.rand(B, device=device) < cond_drop_prob
        h = self._encode_codes(
            history,
            target_levels,
            cond=cond,
            mask=candidates,
            drop_cond_mask=drop,
        )
        logits = self._project_logits(h, target_levels)
        return F.cross_entropy(logits[candidates], target[candidates]), target_levels

    @torch.no_grad()
    def generate_residuals(
        self,
        base_tokens: Tensor,
        *,
        cond: Tensor | None,
        num_quantizers: int | None = None,
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 1.0,
        sample: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        tokens = base_tokens.unsqueeze(1)
        Q = num_quantizers or self.num_quantizers
        for level in range(1, Q):
            logits_c = self(tokens, level, cond=cond, mask=mask)
            if guidance_scale != 1.0 and cond is not None:
                logits_u = self(tokens, level, cond=None, mask=mask)
                logits = logits_u + guidance_scale * (logits_c - logits_u)
            else:
                logits = logits_c
            next_tokens = (
                sample_logits(logits, temperature=temperature, topk_filter_thres=topk_filter_thres)
                if sample
                else logits.argmax(dim=-1)
            )
            if mask is not None:
                next_tokens = torch.where(mask, next_tokens, torch.zeros_like(next_tokens))
            tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        return tokens
