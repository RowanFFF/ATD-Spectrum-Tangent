"""Vectorized deployment form of the frozen AutoTimes spectrum tangent.

This module changes no statistical rule.  It replaces the per-phase Python
loop used to construct the immutable history template with one scatter-add,
then reuses the exact frozen tangent direction from the audited v1 adapter.
"""

from __future__ import annotations

import math

import torch

from models.AutoTimesSpectrumTangent import (
    DEFAULT_BAND_EDGES,
    DEFAULT_BANDS,
    DEFAULT_BETA,
    correct_far_commits,
    from_channel_batch,
)


def build_phase_template_vectorized(model, observed_history, period):
    """Vectorized equivalent of the oldest-occurrence-excluded template."""
    period = int(period)
    if period < 1:
        raise ValueError("tangent period must be positive")
    mean, std = model._instance_stats(observed_history)
    normalized = ((observed_history - mean) / std).squeeze(-1)
    length = int(normalized.size(1))
    history_time = torch.arange(
        -length, 0, device=normalized.device, dtype=torch.long
    )
    phases = history_time.remainder(period)
    counts = torch.bincount(phases, minlength=period)
    if int(counts.min()) < 2:
        phase = int(torch.argmin(counts))
        raise ValueError(
            f"period {period} has fewer than two observations at phase {phase}"
        )
    sums = normalized.new_zeros(normalized.size(0), period)
    sums.scatter_add_(
        1, phases[None, :].expand(normalized.size(0), -1), normalized
    )
    start_phase = (-length) % period
    first_positions = (
        torch.arange(period, device=normalized.device, dtype=torch.long)
        - int(start_phase)
    ).remainder(period)
    oldest = normalized.index_select(1, first_positions)
    return (sums - oldest) / (counts - 1).to(normalized.dtype)[None, :]


@torch.inference_mode()
def forecast_with_explicit_tangent_optimized(
    model,
    x_enc,
    commit_patches,
    period,
    gamma,
    *,
    beta=DEFAULT_BETA,
    band_edges=DEFAULT_BAND_EDGES,
    bands=DEFAULT_BANDS,
):
    """Roll out the same frozen correction with vectorized template setup."""
    commit_patches = int(commit_patches)
    if not 1 <= commit_patches <= int(model.commit_patches):
        raise ValueError("commit width must lie in [1, trained K]")
    batch_size = int(x_enc.size(0))
    history, _, channels = model._to_channel_batch(
        x_enc[:, -int(model.seq_len):]
    )
    template = build_phase_template_vectorized(
        model, history.clone(), int(period)
    )
    rollout_points = commit_patches * int(model.patch_len)
    outputs = []
    committed_patches = 0
    for _ in range(math.ceil(int(model.pred_len) / rollout_points)):
        commits, _, mean, std = model._protected_commits_normalized(
            history, commit_patches
        )
        commits = correct_far_commits(
            commits,
            template,
            committed_patches,
            int(model.patch_len),
            float(gamma),
            beta=beta,
            band_edges=band_edges,
            bands=bands,
        )
        channel_points = (commits * std + mean).reshape(
            history.size(0), rollout_points, 1
        )
        outputs.append(channel_points)
        history = torch.cat([history, channel_points], dim=1)[
            :, -int(model.seq_len):
        ]
        committed_patches += commit_patches
    output = torch.cat(outputs, dim=1)[:, :int(model.pred_len)]
    return from_channel_batch(output.squeeze(-1), batch_size, channels)
