"""Matched frozen-Q1 K-patch controls with independent far exits.

This model keeps the accepted Q1 operator immutable and exposes one isolated
residual MLP for every farther patch.  The final K4 topology therefore matches
the three independent exit paths in the protected dyadic K4 model.  Changing
``distilled_multi_commit_truth_weight`` between zero and one changes only the
target: repeated generated-history Q1 composition versus clean-future truth.
``distilled_multi_commit_geometry_components`` supplies a narrower matched
control that projects only selected truth-residual tangents into that target.
"""

import torch
import torch.nn as nn

from models.DistilledMultiCommitAR import (
    Model as DistilledMultiCommitARModel,
)


class _IndependentFarExitBundle(nn.Module):
    """Present independent patch heads through the legacy flat-head API."""

    def __init__(self, readout_dim, hidden_dim, patch_len, exit_count):
        super().__init__()
        if exit_count < 1:
            raise ValueError('matched frozen controls require a far exit')
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, patch_len),
            )
            for _ in range(exit_count)
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, state):
        return torch.cat([
            head(state)
            for head in self.heads
        ], dim=-1)

    def forward_prefix(self, state, exit_count):
        """Evaluate only the independent exits required by a commit width."""
        exit_count = int(exit_count)
        if not 1 <= exit_count <= len(self.heads):
            raise ValueError('exit prefix must lie in the trained range')
        return torch.cat([
            head(state)
            for head in self.heads[:exit_count]
        ], dim=-1)


class Model(DistilledMultiCommitARModel):
    """Frozen Q1 plus capacity-matched independent K4 exits."""

    def __init__(self, configs):
        super().__init__(configs)
        hidden_dim = int(getattr(
            configs,
            'distilled_multi_commit_hidden',
            configs.d_ff,
        ))
        self.second_residual = _IndependentFarExitBundle(
            readout_dim=self.commit_readout_dim,
            hidden_dim=hidden_dim,
            patch_len=self.patch_len,
            exit_count=self.commit_patches - 1,
        )

    def build_cached_training_batch(self, history, target):
        """Expose exact frozen-trajectory caching for matched exit heads."""
        return self._build_cached_training_batch(history, target)

    def cached_patch_loss(
            self,
            state,
            first,
            recursive_target,
            truth_future):
        return self._cached_patch_loss(
            state,
            first,
            recursive_target,
            truth_future,
        )

    def _student_commits_for_width(self, history, commit_patches):
        """Exploit independent exits without changing other subclasses."""
        return self._student_commits_normalized_prefix(
            history,
            commit_patches,
        )
