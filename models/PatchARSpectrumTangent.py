"""Parameter-free Spectrum Tangent for the local channel-independent Patch-AR.

This uses the same phase template and endpoint ramp as the named-parent
adapters, but preserves the local parent's channel batching and exit API.
The coefficient is gamma; the paper's effective alpha is gamma * 0.0625.
"""

import math

import torch

from models.AutoTimesSpectrumTangent import build_phase_template, correct_far_commits


@torch.inference_mode()
def forecast(model, observed, width, period, gamma):
    """Roll out corrected writebacks; keep each call's first patch unchanged."""
    if not 1 <= int(width) <= model.commit_patches:
        raise ValueError("width must lie in [1, trained K]")
    if not math.isfinite(float(gamma)) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if model.training:
        raise ValueError("the frozen ATD model must be in evaluation mode")
    if gamma == 0 or width == 1:
        return model.forecast_with_commit_patches(observed, width)
    history, channels = model._to_channel_batch(observed[:, -model.seq_len:])
    template = build_phase_template(model, history, period)
    outputs = []
    macro_points = int(width) * model.patch_len
    for age in range(0, model.pred_len, macro_points):
        commits, _, mean, std = model._student_commits_for_width(history, width)
        corrected = correct_far_commits(
            commits, template, age // model.patch_len, model.patch_len, gamma
        )
        points = (corrected * std + mean).reshape(history.size(0), -1, 1)
        outputs.append(points)
        history = torch.cat([history, points], dim=1)[:, -model.seq_len:]
    output = torch.cat(outputs, dim=1)[:, :model.pred_len, 0]
    return model._from_channel_batch(output, observed.size(0), channels)
