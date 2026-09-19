"""Topology-matched clean-future control for :mod:`TimesFMOperatorAR`.

This class retains the native cached writeback and far-head checkpoint schema.
It exists separately so the already frozen Generated protocol and source
hashes remain immutable.
"""

import torch
import torch.nn.functional as F

from models.TimesFMOperatorAR import Model as _GeneratedTimesFM
from models.TimesFMOperatorAR import _TOLERANCE


class Model(_GeneratedTimesFM):
    """Frozen TimesFM parent with clean-future far-exit supervision."""

    @torch.no_grad()
    def clean_training_pair(self, channel_history, channel_target):
        if channel_history.ndim != 2:
            raise ValueError("TimesFM clean cache expects [B*C, seq_len]")
        if channel_target.ndim != 2:
            raise ValueError("TimesFM clean target expects [B*C, K*block_len]")
        if channel_target.size(1) != self.train_pred_len:
            raise ValueError("TimesFM clean target does not match native K blocks")

        state = self._initial_rollout(channel_history, self.commit_patches)
        clean = channel_target.reshape(
            channel_target.size(0), self.commit_patches, self.patch_len
        )
        scale = torch.where(
            state.sigma < _TOLERANCE,
            torch.ones_like(state.sigma),
            state.sigma,
        )[:, None, None]
        residual = (
            clean[:, 1:] - state.first[:, None, :]
        ) / scale
        return state.hidden.detach(), residual.detach()

    def direct_patch_loss(self, history, target):
        history, batch, channels = self._to_channel_batch(history)
        target, target_batch, target_channels = self._to_channel_batch(target)
        if (batch, channels) != (target_batch, target_channels):
            raise ValueError("TimesFM history and target channels must match")
        features, teacher = self.clean_training_pair(
            history.squeeze(-1), target.squeeze(-1)
        )
        student = torch.stack([
            head(features) for head in self.far_heads
        ], dim=1)
        loss = F.mse_loss(student, teacher)
        return {"loss": loss, "composition_loss": loss}
