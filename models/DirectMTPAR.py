"""Causal placeholder multi-token prediction initialized from an accepted Q1.

The model appends learned future placeholders to the observed parent-token
sequence and supervises only the outputs at those future positions.  It is a
clean Direct-MTP baseline: there is no historical next-token auxiliary loss
and no generated-trajectory teacher.  Causal attention makes every shorter
commit width the same mathematical prefix of the trained Kmax path
(kernel-length rounding can differ at floating-point precision).

``DirectMTPAR`` jointly updates the initialized Q1 and the placeholders.  The
otherwise identical frozen-trunk control lives in ``FrozenDirectMTPAR``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransformerAblationAR import Model as TransformerAblationARModel


class Model(TransformerAblationARModel):
    """Joint Direct-MTP with future-only placeholder supervision."""

    UPDATE_SCOPE = "joint"

    def __init__(self, configs):
        if int(getattr(configs, "dense_ar_roll_patches", 1)) != 1:
            raise ValueError("DirectMTPAR requires a Q1 parent configuration")
        super().__init__(configs)

        pretrained = str(getattr(
            configs, "distilled_multi_commit_pretrained", ""
        )).strip()
        if not pretrained:
            raise ValueError("DirectMTPAR requires an accepted Q1 checkpoint")
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        for name, value in self.state_dict().items():
            if name.endswith(".log_temperature") and name not in state:
                state[name] = value
        self.load_state_dict(state, strict=True)

        self.commit_patches = int(getattr(
            configs, "distilled_multi_commit_patches", 2
        ))
        if self.commit_patches < 2:
            raise ValueError("Direct-MTP commit width must be at least two")
        requested_eval = int(getattr(
            configs, "distilled_multi_commit_eval_patches", 0
        ))
        self.eval_commit_patches = (
            self.commit_patches if requested_eval == 0 else requested_eval
        )
        if not 1 <= self.eval_commit_patches <= self.commit_patches:
            raise ValueError("Direct-MTP evaluation width must lie in [1, K]")
        self.truth_loss_type = str(getattr(
            configs, "distilled_multi_commit_truth_loss", "mse"
        ))
        if self.truth_loss_type not in {"mse", "huber"}:
            raise ValueError("Direct-MTP truth loss must be mse or huber")

        # One independently learned row per future position is the complete
        # decoder-specific state.  Never expand/copy one shared placeholder
        # across slots: only the batch axis may be expanded below.  The parent
        # patch projection is deliberately bypassed because these are
        # mask/query tokens, not synthetic values in point space.
        self.future_tokens = nn.Parameter(torch.empty(
            1, self.commit_patches, int(configs.d_model)
        ))
        nn.init.normal_(self.future_tokens, mean=0.0, std=0.02)
        self.train_pred_len = self.commit_patches * self.patch_len

        if self.UPDATE_SCOPE not in {"joint", "frozen"}:
            raise ValueError("Direct-MTP update scope must be joint or frozen")
        if self.UPDATE_SCOPE == "frozen":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.future_tokens.requires_grad_(True)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if self.UPDATE_SCOPE == "frozen":
            # Frozen Q1 dropout must not turn a deterministic accepted parent
            # into a moving target while the placeholder rows are optimized.
            self.backbone.eval()
        return self

    def _history_tokens(self, history_patches):
        if self.patch_assembly != "none":
            return self._assemble_parent_patches(history_patches)
        return self.backbone.patch_projection(history_patches)

    def _predict_commit_normalized(self, history, commit_patches):
        commit_patches = int(commit_patches)
        if not 1 <= commit_patches <= self.commit_patches:
            raise ValueError("commit width must lie in [1, trained K]")
        mean, std = self._instance_stats(history)
        history_patches = self._patchify((history - mean) / std)
        history_tokens = self._history_tokens(history_patches)
        future_tokens = self.future_tokens[:, :commit_patches].expand(
            history.size(0), -1, -1
        )
        hidden = self.backbone.encode_projected(
            torch.cat([history_tokens, future_tokens], dim=1),
            state_mixer=self.variable_mixer,
        )
        predictions = self.backbone.output_head(hidden[:, -commit_patches:])
        return predictions, hidden, mean, std

    def _truth_regression_loss(self, prediction, target):
        if self.truth_loss_type == "mse":
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(
            prediction, target, beta=self.huber_delta
        )

    def direct_patch_loss(self, history, target):
        history, history_channels = self._to_channel_batch(history)
        target, target_channels = self._to_channel_batch(target)
        if history_channels != target_channels:
            raise ValueError("history and target channels must match")
        if target.size(1) != self.train_pred_len:
            raise ValueError(
                "Direct-MTP target must contain exactly K parent patches"
            )
        prediction, _, mean, std = self._predict_commit_normalized(
            history, self.commit_patches
        )
        truth = self._patchify((target - mean) / std)
        loss = self._truth_regression_loss(prediction, truth)
        return {
            "loss": loss,
            "tail_loss": loss.detach(),
            "truth_loss": loss.detach(),
        }

    @torch.no_grad()
    def forecast_with_commit_patches(self, x_enc, commit_patches):
        commit_patches = int(commit_patches)
        if not 1 <= commit_patches <= self.commit_patches:
            raise ValueError("commit width must lie in [1, trained K]")
        batch_size = x_enc.size(0)
        history, channels = self._to_channel_batch(
            x_enc[:, -self.seq_len:]
        )
        rollout_points = commit_patches * self.patch_len
        steps = math.ceil(self.pred_len / rollout_points)
        outputs = []
        for _ in range(steps):
            prediction, _, mean, std = self._predict_commit_normalized(
                history, commit_patches
            )
            points = (prediction * std + mean).reshape(
                history.size(0), rollout_points, 1
            )
            outputs.append(points)
            history = torch.cat([history, points], dim=1)[:, -self.seq_len:]
        output = torch.cat(outputs, dim=1)[:, :self.pred_len]
        return self._from_channel_batch(
            output.squeeze(-1), batch_size, channels
        )

    @torch.no_grad()
    def forecast(self, x_enc):
        return self.forecast_with_commit_patches(
            x_enc, self.eval_commit_patches
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
