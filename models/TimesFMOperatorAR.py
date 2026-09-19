"""Frozen TimesFM 2.5 operator with Generated-K native-block exits.

TimesFM's deployed point forecast recursively writes back its median 128-point
output block through a cached decoder.  This adapter preserves that exact K1
path and trains only lightweight far exits against the blocks produced by the
same cached recursion.  The public 200M checkpoint remains immutable and is
not duplicated in a compiler checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AutoTimesSpectrumTangent import (
    DEFAULT_BAND_EDGES,
    DEFAULT_BANDS,
    DEFAULT_BETA,
    build_phase_template,
    correct_far_commits,
    from_channel_batch,
)


_TOLERANCE = 1e-6


@dataclass
class _DecodeCache:
    next_index: torch.Tensor
    num_masked: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class _RolloutState:
    caches: list
    n: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    first: torch.Tensor
    hidden: torch.Tensor


class _FarExit(nn.Module):
    def __init__(self, state_dim, hidden_dim, block_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, block_len),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)


def _update_running_stats(n, mu, sigma, values, mask):
    """Byte-for-byte formula used by TimesFM 2.5 torch.util."""
    legitimate = torch.logical_not(mask)
    increment_n = torch.sum(legitimate.to(values.dtype), dim=-1)
    safe_increment_n = torch.where(increment_n == 0, 1.0, increment_n)
    increment_mu = torch.sum(values * legitimate, dim=-1) / safe_increment_n
    increment_mu = torch.where(increment_n == 0, 0.0, increment_mu)
    increment_variance = torch.sum(
        ((values - increment_mu.unsqueeze(-1)) ** 2) * legitimate,
        dim=-1,
    ) / safe_increment_n
    increment_variance = torch.where(
        increment_n == 0, 0.0, increment_variance
    )
    increment_sigma = torch.sqrt(increment_variance)

    new_n = n + increment_n
    safe_new_n = torch.where(new_n == 0, 1.0, new_n)
    new_mu = (n * mu + increment_mu * increment_n) / safe_new_n
    new_mu = torch.where(new_n == 0, 0.0, new_mu)
    new_variance = (
        n * sigma.pow(2)
        + increment_n * increment_sigma.pow(2)
        + n * (mu - new_mu).pow(2)
        + increment_n * (increment_mu - new_mu).pow(2)
    ) / safe_new_n
    new_variance = torch.where(new_n == 0, 0.0, new_variance)
    new_sigma = torch.sqrt(torch.clamp(new_variance, min=0.0))
    return new_n, new_mu, new_sigma


def _revin(values, mu, sigma, reverse=False):
    if len(mu.shape) == len(values.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(values.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]
    if reverse:
        return values * sigma + mu
    return (values - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)


class Model(nn.Module):
    """Generated-trajectory compiler over frozen TimesFM 2.5 200M."""

    MODES = {"q1", "generated"}

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.mode = str(getattr(configs, "external_timesfm_mode", "q1"))
        if self.mode not in self.MODES:
            raise ValueError(
                f"external TimesFM mode must be one of {sorted(self.MODES)}"
            )

        factory = getattr(configs, "external_timesfm_backbone_factory", None)
        if factory is not None:
            self.timesfm = factory()
        else:
            source = Path(str(getattr(configs, "external_timesfm_source", "")))
            checkpoint = Path(str(
                getattr(configs, "external_timesfm_checkpoint", "")
            ))
            if not source.is_dir() or not checkpoint.exists():
                raise ValueError(
                    "local TimesFM source and checkpoint are both required"
                )
            source_path = str(source.resolve())
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
            from safetensors.torch import load_file
            from timesfm.timesfm_2p5.timesfm_2p5_torch import (
                TimesFM_2p5_200M_torch_module,
            )

            self.timesfm = TimesFM_2p5_200M_torch_module()
            weights = (
                checkpoint / "model.safetensors"
                if checkpoint.is_dir() else checkpoint
            )
            self.timesfm.load_state_dict(load_file(str(weights)), strict=True)

        self.input_patch_len = int(self.timesfm.p)
        self.patch_len = int(self.timesfm.o)
        self.input_patches_per_block = int(self.timesfm.m)
        self.quantiles = int(self.timesfm.q)
        self.point_index = int(self.timesfm.aridx)
        state_dim = int(self.timesfm.md)
        if self.seq_len % self.input_patch_len:
            raise ValueError("TimesFM input patch must divide the lookback")
        if self.patch_len % self.input_patch_len:
            raise ValueError("TimesFM output block must contain whole input patches")

        self.commit_patches = (
            1
            if self.mode == "q1"
            else int(getattr(configs, "distilled_multi_commit_patches", 8))
        )
        requested = int(getattr(
            configs,
            "distilled_multi_commit_eval_patches",
            self.commit_patches,
        ))
        self.eval_commit_patches = self.commit_patches if requested == 0 else requested
        if not 1 <= self.eval_commit_patches <= self.commit_patches:
            raise ValueError("TimesFM evaluation width must lie in [1, K]")

        hidden_dim = int(getattr(
            configs, "distilled_multi_commit_hidden", 512
        ))
        self.far_heads = nn.ModuleList([
            _FarExit(state_dim, hidden_dim, self.patch_len)
            for _ in range(self.commit_patches - 1)
        ])
        for parameter in self.timesfm.parameters():
            parameter.requires_grad = False
        if self.mode == "q1":
            for parameter in self.far_heads.parameters():
                parameter.requires_grad = False
        self.train_pred_len = self.commit_patches * self.patch_len

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        frozen_prefix = f"{prefix}timesfm."
        for name in [name for name in state if name.startswith(frozen_prefix)]:
            del state[name]
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        missing = [
            name for name in incompatible.missing_keys
            if not name.startswith("timesfm.")
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "incompatible TimesFM compiler checkpoint: "
                f"missing={missing} unexpected={unexpected}"
            )
        return type(incompatible)(missing, unexpected)

    @staticmethod
    def _instance_stats(history):
        mean = history.mean(dim=1, keepdim=True).detach()
        variance = torch.mean(
            torch.square(history - mean), dim=1, keepdim=True
        )
        std = torch.sqrt(variance + 1e-5).detach()
        return mean, std

    @staticmethod
    def _to_channel_batch(values):
        batch, length, channels = values.shape
        return (
            values.permute(0, 2, 1).reshape(batch * channels, length, 1),
            batch,
            channels,
        )

    def _make_caches(self, batch_size, capacity, device):
        return [
            _DecodeCache(
                next_index=torch.zeros(
                    batch_size, dtype=torch.int32, device=device
                ),
                num_masked=torch.zeros(
                    batch_size, dtype=torch.int32, device=device
                ),
                key=torch.zeros(
                    batch_size,
                    capacity,
                    int(self.timesfm.h),
                    int(self.timesfm.hd),
                    device=device,
                ),
                value=torch.zeros(
                    batch_size,
                    capacity,
                    int(self.timesfm.h),
                    int(self.timesfm.hd),
                    device=device,
                ),
            )
            for _ in range(int(self.timesfm.x))
        ]

    def _initial_rollout(self, history, total_blocks):
        if history.ndim != 2 or history.size(1) != self.seq_len:
            raise ValueError("TimesFM rollout expects [B*C, seq_len]")
        batch_size = history.size(0)
        patched = history.reshape(batch_size, -1, self.input_patch_len)
        masks = torch.zeros_like(patched, dtype=torch.bool)
        n = torch.zeros(batch_size, device=history.device)
        mu = torch.zeros(batch_size, device=history.device)
        sigma = torch.zeros(batch_size, device=history.device)
        patch_mu = []
        patch_sigma = []
        for index in range(patched.size(1)):
            n, mu, sigma = _update_running_stats(
                n, mu, sigma, patched[:, index], masks[:, index]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        future_input_patches = max(int(total_blocks) - 1, 0) * (
            self.input_patches_per_block
        )
        caches = self._make_caches(
            batch_size,
            patched.size(1) + future_input_patches,
            history.device,
        )
        normalized = _revin(patched, context_mu, context_sigma)
        (_, hidden, output, _), caches = self.timesfm(
            normalized, masks, caches
        )
        points = _revin(output, context_mu, context_sigma, reverse=True)
        points = points.reshape(
            batch_size, -1, self.patch_len, self.quantiles
        )[:, -1, :, self.point_index]
        return _RolloutState(
            caches=caches,
            n=n,
            mu=mu,
            sigma=sigma,
            first=points,
            hidden=hidden[:, -1],
        )

    def _consume_blocks(self, state, blocks):
        batch_size, width, block_len = blocks.shape
        if block_len != self.patch_len:
            raise ValueError("TimesFM writeback block length changed")
        patched = blocks.reshape(
            batch_size, width * self.input_patches_per_block,
            self.input_patch_len,
        )
        masks = torch.zeros_like(patched, dtype=torch.bool)
        patch_mu = []
        patch_sigma = []
        n, mu, sigma = state.n, state.mu, state.sigma
        for index in range(patched.size(1)):
            n, mu, sigma = _update_running_stats(
                n, mu, sigma, patched[:, index], masks[:, index]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        current_mu = torch.stack(patch_mu, dim=1)
        current_sigma = torch.stack(patch_sigma, dim=1)
        normalized = _revin(patched, current_mu, current_sigma)
        (_, hidden, output, _), caches = self.timesfm(
            normalized, masks, state.caches
        )
        points = _revin(output, current_mu, current_sigma, reverse=True)
        points = points.reshape(
            batch_size, -1, self.patch_len, self.quantiles
        )[:, -1, :, self.point_index]
        return _RolloutState(
            caches=caches,
            n=n,
            mu=mu,
            sigma=sigma,
            first=points,
            hidden=hidden[:, -1],
        )

    def _propose_blocks(self, state, width):
        width = int(width)
        if not 1 <= width <= self.commit_patches:
            raise ValueError("requested TimesFM commit width is unavailable")
        blocks = [state.first]
        scale = torch.where(
            state.sigma < _TOLERANCE,
            torch.ones_like(state.sigma),
            state.sigma,
        )[:, None]
        blocks.extend(
            state.first + self.far_heads[index](state.hidden) * scale
            for index in range(width - 1)
        )
        return torch.stack(blocks, dim=1)

    @torch.no_grad()
    def _recursive_q1_targets(self, history, width):
        state = self._initial_rollout(history, int(width))
        targets = [state.first]
        for _ in range(1, int(width)):
            state = self._consume_blocks(state, state.first[:, None, :])
            targets.append(state.first)
        return torch.stack(targets, dim=1)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.timesfm.eval()
        for head in self.far_heads:
            head.train(mode and self.mode == "generated")
        return self

    def optimizer_parameter_groups(self, learning_rate):
        parameters = [
            parameter for parameter in self.far_heads.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("frozen TimesFM Q1 has no trainable parameters")
        return [{"params": parameters, "lr": learning_rate}]

    def direct_patch_loss(self, history, target):
        if self.mode != "generated":
            raise RuntimeError("only Generated mode has a training objective")
        history, batch, channels = self._to_channel_batch(history)
        target, target_batch, target_channels = self._to_channel_batch(target)
        if (batch, channels) != (target_batch, target_channels):
            raise ValueError("TimesFM history and target channels must match")
        if target.size(1) != self.train_pred_len:
            raise ValueError("TimesFM target does not match native K blocks")
        features, teacher_residual = self.trajectory_training_pair(
            history.squeeze(-1)
        )
        student_residual = torch.stack([
            head(features) for head in self.far_heads
        ], dim=1)
        loss = F.mse_loss(student_residual, teacher_residual)
        return {"loss": loss, "composition_loss": loss}

    @torch.no_grad()
    def trajectory_training_pair(self, channel_history):
        """Return frozen state features and exact recursive residual targets."""
        if channel_history.ndim != 2:
            raise ValueError("TimesFM trajectory cache expects [B*C, seq_len]")
        state = self._initial_rollout(
            channel_history, self.commit_patches
        )
        initial_hidden = state.hidden
        initial_first = state.first
        initial_sigma = state.sigma
        targets = [state.first]
        for _ in range(1, self.commit_patches):
            state = self._consume_blocks(state, state.first[:, None, :])
            targets.append(state.first)
        teacher = torch.stack(targets, dim=1)
        scale = torch.where(
            initial_sigma < _TOLERANCE,
            torch.ones_like(initial_sigma),
            initial_sigma,
        )[:, None, None]
        teacher_residual = (
            teacher[:, 1:] - initial_first[:, None, :]
        ) / scale
        return initial_hidden.detach(), teacher_residual.detach()

    def _rollout(self, x_enc, commit_patches, tangent=None):
        commit_patches = int(commit_patches)
        history, batch, channels = self._to_channel_batch(
            x_enc[:, -self.seq_len:]
        )
        total_blocks = (self.pred_len + self.patch_len - 1) // self.patch_len
        state = self._initial_rollout(history.squeeze(-1), total_blocks)
        outputs = []
        committed = 0
        template = None
        origin_mean = None
        origin_std = None
        if tangent is not None:
            period, _, _, _, _ = tangent
            origin_mean, origin_std = self._instance_stats(history)
            template = build_phase_template(self, history.clone(), int(period))
        while committed < total_blocks:
            width = min(commit_patches, total_blocks - committed)
            blocks = self._propose_blocks(state, width)
            if tangent is not None and float(tangent[1]) != 0.0:
                period, gamma, beta, band_edges, bands = tangent
                del period
                normalized = (
                    blocks - origin_mean
                ) / origin_std
                normalized = correct_far_commits(
                    normalized,
                    template,
                    committed,
                    self.patch_len,
                    gamma,
                    beta=beta,
                    band_edges=band_edges,
                    bands=bands,
                )
                blocks = normalized * origin_std + origin_mean
            outputs.append(blocks.reshape(history.size(0), -1))
            committed += width
            if committed < total_blocks:
                state = self._consume_blocks(state, blocks)
        channel_output = torch.cat(outputs, dim=1)[:, :self.pred_len]
        return from_channel_batch(channel_output, batch, channels)

    def forecast_with_commit_patches(self, x_enc, commit_patches):
        return self._rollout(x_enc, commit_patches)

    def forecast_with_explicit_tangent(
        self,
        x_enc,
        commit_patches,
        period,
        gamma,
        *,
        beta=DEFAULT_BETA,
        band_edges=DEFAULT_BAND_EDGES,
        bands=DEFAULT_BANDS,
    ):
        return self._rollout(
            x_enc,
            commit_patches,
            tangent=(period, gamma, beta, band_edges, bands),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.forecast_with_commit_patches(
            x_enc, self.eval_commit_patches
        )
