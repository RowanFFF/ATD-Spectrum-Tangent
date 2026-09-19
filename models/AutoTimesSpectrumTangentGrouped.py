"""Reduction-order-preserving grouped deployment form of the AutoTimes tangent."""

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


def build_phase_template_grouped(model, observed_history, period):
    """Batch phases with equal counts while preserving each phase's order."""
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
    start_phase = (-length) % period
    first_positions = (
        torch.arange(period, device=normalized.device, dtype=torch.long)
        - int(start_phase)
    ).remainder(period)
    remaining = counts - 1
    template = normalized.new_empty(normalized.size(0), period)
    # For the frozen periods there are at most two count groups.  Within each
    # phase the gathered positions and mean reduction order exactly match the
    # audited reference loop's positions[1:].
    for count in torch.unique(remaining, sorted=True).tolist():
        count = int(count)
        phase_ids = torch.nonzero(
            remaining == count, as_tuple=False
        ).flatten()
        positions = (
            first_positions.index_select(0, phase_ids)[:, None]
            + period * torch.arange(
                1, count + 1,
                device=normalized.device, dtype=torch.long,
            )[None, :]
        )
        values = normalized[:, positions]
        template[:, phase_ids] = values.mean(dim=-1)
    return template


@torch.inference_mode()
def forecast_with_explicit_tangent_grouped(
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
    commit_patches = int(commit_patches)
    if not 1 <= commit_patches <= int(model.commit_patches):
        raise ValueError("commit width must lie in [1, trained K]")
    batch_size = int(x_enc.size(0))
    history, _, channels = model._to_channel_batch(
        x_enc[:, -int(model.seq_len):]
    )
    template = build_phase_template_grouped(
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
