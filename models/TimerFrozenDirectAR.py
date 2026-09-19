"""Topology-matched clean-future control for :mod:`TimerOperatorAR`.

The deployed model and checkpoint schema are identical to Generated-K8.  Only
the far-exit training target changes: offsets 2..K use clean future patches
from the initial observed history instead of the recursive Timer trajectory.
"""

import torch
import torch.nn.functional as F

from models.TimerOperatorAR import Model as _GeneratedTimer


class Model(_GeneratedTimer):
    """Frozen Timer parent with independent clean-future far exits."""

    def direct_patch_loss(self, history, target):
        history, batch, channels = self._to_channel_batch(history)
        target, target_batch, target_channels = self._to_channel_batch(target)
        if (batch, channels) != (target_batch, target_channels):
            raise ValueError("Timer history and target channels must match")
        if target.size(1) != self.train_pred_len:
            raise ValueError("Timer target does not match the commit width")

        with torch.no_grad():
            state = self._initial_rollout(history.squeeze(-1))
            clean = target.squeeze(-1).reshape(
                target.size(0), self.commit_patches, self.patch_len
            )
            teacher = (
                clean[:, 1:] - state.first[:, None, :]
            ) / state.scale[:, None, :]
            features = state.hidden.detach()

        student = torch.stack([
            head(features) for head in self.far_heads
        ], dim=1)
        loss = F.mse_loss(student, teacher)
        return {"loss": loss, "composition_loss": loss}
