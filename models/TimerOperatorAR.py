"""Frozen Timer-base operator with Generated-K cached-token exits.

The public Timer ``generate`` path normalizes the initial context once, then
recursively normalizes and writes back one 96-point patch through its KV cache.
This adapter preserves that deployed recursion exactly for every non-constant
input.  Timer's raw ReVIN divides by zero on a constant window; here that
otherwise undefined case is extended by its continuous limit (zero normalized
input and constant de-normalized output).  Generated mode learns only
lightweight far exits against the patches produced by the same cached operator;
the immutable public checkpoint is not duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _RolloutState:
    past_key_values: object
    token_count: int
    first: torch.Tensor
    hidden: torch.Tensor
    scale: torch.Tensor


class _FarExit(nn.Module):
    def __init__(self, state_dim, hidden_dim, patch_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_len),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)


class Model(nn.Module):
    """Generated-trajectory compiler over a frozen Timer-base parent."""

    MODES = {"q1", "generated"}

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.mode = str(getattr(configs, "external_timer_mode", "q1"))
        if self.mode not in self.MODES:
            raise ValueError(
                f"external Timer mode must be one of {sorted(self.MODES)}"
            )

        factory = getattr(configs, "external_timer_backbone_factory", None)
        if factory is None:
            checkpoint = str(getattr(configs, "external_timer_checkpoint", ""))
            if not checkpoint:
                raise ValueError("a local Timer checkpoint directory is required")
            try:
                from transformers import AutoConfig, AutoModelForCausalLM
            except ImportError as error:
                raise ImportError(
                    "TimerOperatorAR requires the optional transformers package"
                ) from error
            timer_config = AutoConfig.from_pretrained(
                checkpoint, trust_remote_code=True
            )
            self.timer = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                config=timer_config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
        else:
            self.timer = factory()

        timer_config = getattr(self.timer, "config", SimpleNamespace())
        self.patch_len = int(getattr(timer_config, "input_token_len", 96))
        state_dim = int(getattr(timer_config, "hidden_size", 1024))
        if self.patch_len < 2 or self.seq_len % self.patch_len:
            raise ValueError("Timer patch length must divide the lookback")

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
            raise ValueError("Timer evaluation width must lie in [1, K]")

        hidden_dim = int(getattr(
            configs, "distilled_multi_commit_hidden", 512
        ))
        self.far_heads = nn.ModuleList([
            _FarExit(state_dim, hidden_dim, self.patch_len)
            for _ in range(self.commit_patches - 1)
        ])
        for parameter in self.timer.parameters():
            parameter.requires_grad = False
        if self.mode == "q1":
            for parameter in self.far_heads.parameters():
                parameter.requires_grad = False
        self.train_pred_len = self.commit_patches * self.patch_len

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        frozen_prefix = f"{prefix}timer."
        for name in [name for name in state if name.startswith(frozen_prefix)]:
            del state[name]
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        missing = [
            name for name in incompatible.missing_keys
            if not name.startswith("timer.")
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "incompatible Timer compiler checkpoint: "
                f"missing={missing} unexpected={unexpected}"
            )
        return type(incompatible)(missing, unexpected)

    @staticmethod
    def _instance_stats(history):
        mean = history.mean(dim=1, keepdim=True).detach()
        std = history.std(dim=1, keepdim=True).detach()
        return mean, Model._safe_scale(std)

    @staticmethod
    def _to_channel_batch(values):
        batch, length, channels = values.shape
        return (
            values.permute(0, 2, 1).reshape(batch * channels, length, 1),
            batch,
            channels,
        )

    @staticmethod
    def _safe_scale(scale):
        return torch.where(scale == 0, torch.ones_like(scale), scale)

    def _initial_rollout(self, history):
        if history.ndim != 2 or history.size(1) != self.seq_len:
            raise ValueError("Timer rollout expects [B*C, seq_len]")
        batch_size = history.size(0)
        tokens = self.seq_len // self.patch_len
        attention_mask = torch.ones(
            batch_size, tokens, dtype=torch.long, device=history.device
        )
        position_ids = torch.arange(
            tokens, dtype=torch.long, device=history.device
        )[None].expand(batch_size, -1)
        mean = history.mean(dim=-1, keepdim=True)
        raw_scale = history.std(dim=-1, keepdim=True)
        scale = self._safe_scale(raw_scale)
        normalized = (history - mean) / scale
        outputs = self.timer(
            input_ids=normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            revin=False,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None or outputs.past_key_values is None:
            raise RuntimeError("Timer did not expose hidden state and KV cache")
        return _RolloutState(
            past_key_values=outputs.past_key_values,
            token_count=tokens,
            first=outputs.logits * raw_scale + mean,
            hidden=outputs.hidden_states[-1][:, -1],
            scale=scale,
        )

    def _consume_patches(self, state, patches):
        batch_size, width, patch_len = patches.shape
        if patch_len != self.patch_len:
            raise ValueError("Timer writeback patch length changed")
        mean = patches.mean(dim=-1, keepdim=True)
        raw_scale = patches.std(dim=-1, keepdim=True)
        scale = self._safe_scale(raw_scale)
        normalized = ((patches - mean) / scale).reshape(batch_size, -1)
        new_tokens = int(width)
        total_tokens = int(state.token_count) + new_tokens
        attention_mask = torch.ones(
            batch_size, total_tokens, dtype=torch.long, device=patches.device
        )
        position_ids = torch.arange(
            state.token_count,
            total_tokens,
            dtype=torch.long,
            device=patches.device,
        )[None].expand(batch_size, -1)
        outputs = self.timer(
            input_ids=normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            revin=False,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        first = outputs.logits * raw_scale[:, -1] + mean[:, -1]
        return _RolloutState(
            past_key_values=outputs.past_key_values,
            token_count=total_tokens,
            first=first,
            hidden=outputs.hidden_states[-1][:, -1],
            scale=scale[:, -1],
        )

    def _propose_patches(self, state, width):
        width = int(width)
        if not 1 <= width <= self.commit_patches:
            raise ValueError("requested Timer commit width is unavailable")
        patches = [state.first]
        patches.extend(
            state.first + self.far_heads[index](state.hidden) * state.scale
            for index in range(width - 1)
        )
        return torch.stack(patches, dim=1)

    @torch.no_grad()
    def _recursive_q1_targets(self, history, width):
        state = self._initial_rollout(history)
        targets = [state.first]
        for _ in range(1, int(width)):
            state = self._consume_patches(state, state.first[:, None, :])
            targets.append(state.first)
        return torch.stack(targets, dim=1)

    @torch.no_grad()
    def trajectory_training_pair(self, channel_history):
        state = self._initial_rollout(channel_history)
        initial_hidden = state.hidden
        initial_first = state.first
        initial_scale = state.scale
        targets = [state.first]
        for _ in range(1, self.commit_patches):
            state = self._consume_patches(state, state.first[:, None, :])
            targets.append(state.first)
        teacher = torch.stack(targets, dim=1)
        residual = (
            teacher[:, 1:] - initial_first[:, None, :]
        ) / initial_scale[:, None, :]
        return initial_hidden.detach(), residual.detach()

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.timer.eval()
        for head in self.far_heads:
            head.train(mode and self.mode == "generated")
        return self

    def optimizer_parameter_groups(self, learning_rate):
        parameters = [
            parameter for parameter in self.far_heads.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("frozen Timer Q1 has no trainable parameters")
        return [{"params": parameters, "lr": learning_rate}]

    def direct_patch_loss(self, history, target):
        if self.mode != "generated":
            raise RuntimeError("only Generated mode has a training objective")
        history, batch, channels = self._to_channel_batch(history)
        target, target_batch, target_channels = self._to_channel_batch(target)
        if (batch, channels) != (target_batch, target_channels):
            raise ValueError("Timer history and target channels must match")
        if target.size(1) != self.train_pred_len:
            raise ValueError("Timer target does not match the commit width")
        features, teacher = self.trajectory_training_pair(history.squeeze(-1))
        student = torch.stack([
            head(features) for head in self.far_heads
        ], dim=1)
        loss = F.mse_loss(student, teacher)
        return {"loss": loss, "composition_loss": loss}

    def _rollout(self, x_enc, commit_patches, tangent=None):
        commit_patches = int(commit_patches)
        history, batch, channels = self._to_channel_batch(
            x_enc[:, -self.seq_len:]
        )
        total_patches = (self.pred_len + self.patch_len - 1) // self.patch_len
        state = self._initial_rollout(history.squeeze(-1))
        origin_mean = None
        origin_std = None
        template = None
        if tangent is not None:
            period, _, _, _, _ = tangent
            origin_mean, origin_std = self._instance_stats(history)
            template = build_phase_template(self, history.clone(), int(period))
        outputs = []
        committed = 0
        while committed < total_patches:
            width = min(commit_patches, total_patches - committed)
            patches = self._propose_patches(state, width)
            if tangent is not None and float(tangent[1]) != 0.0:
                period, gamma, beta, band_edges, bands = tangent
                del period
                normalized = (patches - origin_mean) / origin_std
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
                patches = normalized * origin_std + origin_mean
            outputs.append(patches.reshape(history.size(0), -1))
            committed += width
            if committed < total_patches:
                state = self._consume_patches(state, patches)
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
