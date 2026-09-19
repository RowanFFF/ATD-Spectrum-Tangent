"""Deterministic spectrum-tangent deployment adapter for AutoTimes operators.

The adapter owns no parameters and never mutates the wrapped Generated model.
It builds one immutable phase template from the initial observed history,
preserves the first (accepted Q1) patch of every macro proposal exactly, and
writes the corrected far patches back into the autoregressive history.

The correction is expressed in the Generated model's current instance-
normalized coordinates, matching the paper's Patch-AR tangent implementation::

    G_corrected = G + gamma * band(h) * beta * (T_tau - G)

Only slots 2..K are corrected.  ``period`` and ``gamma`` must be selected
outside this module from train-only deployment trajectories.
"""

from __future__ import annotations

import math

import torch


DEFAULT_BAND_EDGES = (96, 192, 336, 672)
DEFAULT_BANDS = (0.25, 0.5, 1.0, 2.0, 3.0)
DEFAULT_BETA = 0.0625


def _band_index(endpoint, edges):
    index = 0
    for edge in edges:
        if int(endpoint) >= int(edge):
            index += 1
    return index


def from_channel_batch(values, batch_size, channels):
    """Invert AutoTimes ``[B*C, T]`` channel batching."""
    return values.reshape(
        int(batch_size), int(channels), values.size(-1)
    ).permute(0, 2, 1)


def build_phase_template(model, observed_history, period):
    """Build an immutable normalized absolute-phase history template.

    The oldest occurrence of every phase is excluded, exactly as in the
    existing spectrum-tangent audit.  Consequently every phase needs at least
    two occurrences in the initial lookback.
    """
    period = int(period)
    if period < 1:
        raise ValueError("tangent period must be positive")
    mean, std = model._instance_stats(observed_history)
    normalized = ((observed_history - mean) / std).squeeze(-1)
    history_time = torch.arange(
        -normalized.size(1), 0,
        device=normalized.device, dtype=torch.long,
    )
    phases = history_time.remainder(period)
    template = normalized.new_empty(normalized.size(0), period)
    for phase in range(period):
        positions = torch.nonzero(phases == phase, as_tuple=False).flatten()
        if positions.numel() < 2:
            raise ValueError(
                f"period {period} has fewer than two observations at phase "
                f"{phase}"
            )
        template[:, phase] = normalized.index_select(
            1, positions[1:]
        ).mean(dim=1)
    return template


def normalized_tangent_direction(
    commits,
    template,
    committed_patches,
    patch_len,
    *,
    beta=DEFAULT_BETA,
    band_edges=DEFAULT_BAND_EDGES,
    bands=DEFAULT_BANDS,
):
    """Return the frozen tangent direction in normalized commit coordinates."""
    band_edges = tuple(int(value) for value in band_edges)
    bands = tuple(float(value) for value in bands)
    if tuple(sorted(band_edges)) != band_edges:
        raise ValueError("tangent band edges must be sorted")
    if len(bands) != len(band_edges) + 1:
        raise ValueError("tangent needs one multiplier per horizon band")
    if commits.ndim != 3:
        raise ValueError("commits must have shape [B*C, K, patch_len]")
    if commits.size(2) != int(patch_len):
        raise ValueError("commit patch dimension does not match patch_len")
    if template.ndim != 2 or template.size(0) != commits.size(0):
        raise ValueError("template must have shape [B*C, period]")

    offsets = torch.arange(
        commits.size(1), device=commits.device, dtype=torch.long
    ) + int(committed_patches)
    positions = (
        offsets[:, None] * int(patch_len)
        + torch.arange(
            int(patch_len), device=commits.device, dtype=torch.long
        )[None, :]
    )
    multipliers = commits.new_tensor([
        bands[_band_index(
            (global_patch + 1) * int(patch_len), band_edges
        )]
        for global_patch in range(
            int(committed_patches),
            int(committed_patches) + commits.size(1),
        )
    ])[None, :, None]
    direction = float(beta) * multipliers * (
        template[:, positions % template.size(1)] - commits
    )
    # This is an exact protected-prefix constraint, not a small coefficient.
    direction[:, 0].zero_()
    return direction


def correct_far_commits(
    commits,
    template,
    committed_patches,
    patch_len,
    gamma,
    **direction_kwargs,
):
    """Apply a frozen scalar coefficient while preserving local slot 1."""
    direction = normalized_tangent_direction(
        commits,
        template,
        committed_patches,
        patch_len,
        **direction_kwargs,
    )
    corrected = commits + float(gamma) * direction
    if commits.size(1) == 1:
        return commits
    return torch.cat([commits[:, :1], corrected[:, 1:]], dim=1)


@torch.inference_mode()
def forecast_with_explicit_tangent(
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
    """Roll out a frozen AutoTimes Generated operator with explicit correction."""
    commit_patches = int(commit_patches)
    if not 1 <= commit_patches <= int(model.commit_patches):
        raise ValueError("commit width must lie in [1, trained K]")
    if float(gamma) == 0.0 or commit_patches == 1:
        return model.forecast_with_commit_patches(x_enc, commit_patches)
    batch_size = int(x_enc.size(0))
    history, _, channels = model._to_channel_batch(
        x_enc[:, -int(model.seq_len):]
    )
    template = build_phase_template(model, history.clone(), int(period))
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
