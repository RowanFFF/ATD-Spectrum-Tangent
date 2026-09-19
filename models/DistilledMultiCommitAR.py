"""Distill repeated recursive Q1 transitions into one protected macro commit.

The pretrained one-step Transformer is immutable.  Its ordinary output is
the first committed patch.  A separate residual head reads the final history
state and learns the second patch produced by applying the frozen Q1 model
again after its own first prediction:

    student_k(H) ~= F^k(H).

Because only the second head is trainable, the far objective cannot alter the
anchor Q1 transition.  Inference emits two patches in one backbone pass.

Optional geometry-projected labels retain only selected level, aligned
periodic-shape, and phase-shift coordinates of the clean-future residual.  The
frame is history/Generated-only and is built independently per far patch, so
K-prefix nesting and the exact Q1 fallback remain protected.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransformerAblationAR import (
    Model as TransformerAblationARModel,
)

# Absolute-time band boundaries shared with the output-side closure
# (TemplateShrinkageAR.BAND_EDGES).  The far-target mixing schedule must use
# the same partition so the target-side contraction is comparable with the
# horizon-band closure; test_distilled_multi_commit_band_betas asserts the two
# stay identical.
TARGET_BAND_EDGES = (96, 192, 336, 672)
GEOMETRY_COMPONENTS = ('level', 'shape', 'phase')


def _band_index(t, edges=TARGET_BAND_EDGES):
    idx = 0
    for edge in edges:
        if t >= edge:
            idx += 1
    return idx


class Model(TransformerAblationARModel):
    """Frozen Q1 backbone with an isolated composition-distillation head."""

    def __init__(self, configs):
        if int(getattr(
                configs, 'dense_ar_roll_patches', 1)) != 1:
            raise ValueError(
                'DistilledMultiCommitAR requires a Q1 pretrained base')
        super().__init__(configs)
        pretrained = getattr(
            configs,
            'distilled_multi_commit_pretrained',
            '',
        )
        if not pretrained:
            raise ValueError(
                'a pretrained Q1 checkpoint is required')
        state = torch.load(
            pretrained,
            map_location='cpu',
            weights_only=True,
        )
        # A matched Q2/R1 checkpoint may contain one learned future token.
        # Causality keeps it from changing the Q1 map, and the distilled
        # model intentionally starts from the observed-history map only.
        state.pop('future_patches', None)
        # Older checkpoints used the same default temperature but did not
        # persist it.  Fill only this known compatibility key and retain
        # strict validation for every learned tensor.
        for name, value in self.state_dict().items():
            if (
                    name.endswith('.log_temperature')
                    and name not in state):
                state[name] = value
        self.load_state_dict(state, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad = False

        hidden_dim = int(getattr(
            configs,
            'distilled_multi_commit_hidden',
            configs.d_ff,
        ))
        if hidden_dim < 1:
            raise ValueError(
                'distillation hidden dimension must be positive')
        self.commit_readout = str(getattr(
            configs,
            'distilled_multi_commit_readout',
            'last',
        ))
        if self.commit_readout not in {
                'last', 'innovation', 'tail_mean'}:
            raise ValueError(
                'distilled readout must be last, innovation, '
                'or tail_mean')
        self.commit_readout_tail = int(getattr(
            configs,
            'distilled_multi_commit_readout_tail',
            4,
        ))
        if self.commit_readout_tail < 2:
            raise ValueError(
                'distilled readout tail must be at least two')
        self.commit_readout_dim = configs.d_model * (
            1 if self.commit_readout == 'last' else 2)
        self.truth_weight = float(getattr(
            configs,
            'distilled_multi_commit_truth_weight',
            0.0,
        ))
        self.truth_loss_type = str(getattr(
            configs,
            'distilled_multi_commit_truth_loss',
            'mse',
        ))
        if not 0.0 <= self.truth_weight <= 1.0:
            raise ValueError(
                'distillation truth weight must lie in [0, 1]')
        if self.truth_loss_type not in {'mse', 'huber'}:
            raise ValueError(
                'distillation truth loss must be mse or huber')
        # Per-band target-side contraction (the Generated x closure family):
        # y_hat + band_betas[band(t)] * (y - y_hat), one beta per absolute-time
        # band.  With MSE this is the exact interpolated target the mixed
        # truth_weight loss converges to, but per-patch rather than global.
        # Mutually exclusive with truth_weight mixing; negative betas (regime-B
        # anti-pull targets) require the explicit allow_negative flag.
        band_betas_text = str(getattr(
            configs,
            'distilled_multi_commit_band_betas',
            '',
        ) or '').strip()
        self.band_betas = None
        if band_betas_text:
            values = [float(v) for v in band_betas_text.split(',')]
            if len(values) != len(TARGET_BAND_EDGES) + 1:
                raise ValueError(
                    'distilled_multi_commit_band_betas must supply one beta '
                    f'per band ({len(TARGET_BAND_EDGES) + 1} values)')
            allow_negative = bool(getattr(
                configs,
                'allow_negative_truth_weight',
                False,
            ))
            if not allow_negative and any(v < 0.0 for v in values):
                raise ValueError(
                    'band betas must be non-negative unless '
                    'allow_negative_truth_weight is set')
            if self.truth_weight != 0.0:
                raise ValueError(
                    'band betas are mutually exclusive with truth_weight '
                    'mixing')
            self.band_betas = torch.tensor(
                values, dtype=torch.float32)
        geometry_text = str(getattr(
            configs,
            'distilled_multi_commit_geometry_components',
            '',
        ) or '').strip()
        requested_geometry = []
        for name in geometry_text.split(',') if geometry_text else ():
            name = name.strip().lower()
            # ``scale`` was the name used by the H720 geometry audit; the
            # direction is more precisely an aligned periodic-shape amplitude.
            if name == 'scale':
                name = 'shape'
            if name not in GEOMETRY_COMPONENTS:
                raise ValueError(
                    'geometry components must be a comma-separated subset '
                    'of level, shape, phase')
            if name in requested_geometry:
                raise ValueError('geometry components must be unique')
            requested_geometry.append(name)
        self.geometry_components = tuple(requested_geometry)
        self.geometry_period = int(getattr(
            configs,
            'distilled_multi_commit_geometry_period',
            0,
        ))
        self.geometry_strength = float(getattr(
            configs,
            'distilled_multi_commit_geometry_strength',
            1.0,
        ))
        geometry_strengths_text = str(getattr(
            configs,
            'distilled_multi_commit_geometry_component_strengths',
            '',
        ) or '').strip()
        self.geometry_component_strengths = None
        if geometry_strengths_text:
            values = [
                float(value)
                for value in geometry_strengths_text.split(',')
            ]
            if len(values) != len(GEOMETRY_COMPONENTS):
                raise ValueError(
                    'geometry component strengths must provide level, '
                    'shape, and phase values')
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(
                    'geometry component strengths must lie in [0, 1]')
            if self.geometry_strength != 1.0:
                raise ValueError(
                    'global and component-specific geometry strengths are '
                    'mutually exclusive')
            self.geometry_component_strengths = torch.tensor(
                values, dtype=torch.float32)
        if (
                self.geometry_component_strengths is not None
                and not self.geometry_components):
            raise ValueError(
                'geometry component strengths require enabled components')
        if self.geometry_components:
            if self.truth_weight != 0.0 or self.band_betas is not None:
                raise ValueError(
                    'geometry targets are mutually exclusive with '
                    'truth_weight and band-beta mixing')
            if self.truth_loss_type != 'mse':
                raise ValueError(
                    'geometry projection requires the MSE far loss')
            if not 2 <= self.geometry_period <= self.seq_len:
                raise ValueError(
                    'geometry period must lie in [2, seq_len]')
            if not 0.0 <= self.geometry_strength <= 1.0:
                raise ValueError(
                    'geometry target strength must lie in [0, 1]')
        self.commit_patches = int(getattr(
            configs,
            'distilled_multi_commit_patches',
            2,
        ))
        if self.commit_patches < 2:
            raise ValueError(
                'distilled commit width must be at least two')
        requested_eval_commit = int(getattr(
            configs,
            'distilled_multi_commit_eval_patches',
            0,
        ))
        self.eval_commit_patches = (
            self.commit_patches
            if requested_eval_commit == 0
            else requested_eval_commit
        )
        if not 1 <= self.eval_commit_patches <= self.commit_patches:
            raise ValueError(
                'distilled evaluation commit width must lie in [1, K]')
        self.second_residual = nn.Sequential(
            nn.Linear(self.commit_readout_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                (self.commit_patches - 1)
                * self.patch_len,
            ),
        )
        nn.init.zeros_(self.second_residual[-1].weight)
        nn.init.zeros_(self.second_residual[-1].bias)
        self.train_pred_len = (
            self.commit_patches * self.patch_len)

    def _commit_readout(self, hidden):
        """Expose a compact sufficient-state candidate to macro heads."""
        last = hidden[:, -1]
        if self.commit_readout == 'last':
            return last
        if hidden.size(1) < 2:
            raise ValueError(
                'innovation readout requires at least two hidden tokens')
        if self.commit_readout == 'innovation':
            context = last - hidden[:, -2]
        else:
            tail = hidden[:, -min(
                self.commit_readout_tail,
                hidden.size(1),
            ):]
            context = last - tail.mean(dim=1)
        return torch.cat([last, context], dim=-1)

    def train(self, mode=True):
        super().train(mode)
        # The teacher and anchor must be deterministic even while the
        # isolated second head is optimized.
        self.backbone.eval()
        self.second_residual.train(mode)
        return self

    def _base_hidden(self, history):
        mean, std = self._instance_stats(history)
        patches = self._patchify((history - mean) / std)
        tokens = (
            self._assemble_parent_patches(patches)
            if self.patch_assembly != 'none'
            else self.backbone.patch_projection(patches)
        )
        hidden = self.backbone.encode_projected(
            tokens,
            state_mixer=self.variable_mixer,
        )
        predictions = self.backbone.output_head(hidden)
        return predictions, hidden, mean, std

    def _student_commits_normalized(self, history):
        base, hidden, mean, std = self._base_hidden(history)
        first = base[:, -1:]
        future = (
            first.expand(
                -1,
                self.commit_patches - 1,
                -1,
            )
            + self.second_residual(
                self._commit_readout(hidden)).reshape(
                    history.size(0),
                    self.commit_patches - 1,
                    self.patch_len,
                )
        )
        return (
            torch.cat([first, future], dim=1),
            hidden,
            mean,
            std,
        )

    def _student_commits_normalized_prefix(
            self,
            history,
            commit_patches):
        """Evaluate the smallest available exit prefix for one width.

        The accepted first patch never needs a far exit.  Independent-exit
        variants additionally avoid evaluating heads beyond the requested
        width.  Monolithic legacy heads retain their original computation and
        are sliced only after evaluation.
        """
        commit_patches = int(commit_patches)
        if not 1 <= commit_patches <= self.commit_patches:
            raise ValueError(
                'commit_patches must lie between one and trained K')
        base, hidden, mean, std = self._base_hidden(history)
        first = base[:, -1:]
        if commit_patches == 1:
            return first, hidden, mean, std

        readout = self._commit_readout(hidden)
        far_count = commit_patches - 1
        if hasattr(self.second_residual, 'forward_prefix'):
            residual = self.second_residual.forward_prefix(
                readout,
                far_count,
            )
        else:
            residual = self.second_residual(readout)[
                :, :far_count * self.patch_len]
        future = (
            first.expand(-1, far_count, -1)
            + residual.reshape(
                history.size(0),
                far_count,
                self.patch_len,
            )
        )
        return (
            torch.cat([first, future], dim=1),
            hidden,
            mean,
            std,
        )

    def _student_commits_for_width(self, history, commit_patches):
        """Default width hook preserves subclass commit implementations."""
        commits, hidden, mean, std = self._student_commits_normalized(
            history
        )
        return commits[:, :int(commit_patches)], hidden, mean, std

    def _student_pair_normalized(self, history):
        if self.commit_patches != 2:
            raise ValueError(
                'pair helper is defined only for K2/R2')
        commits, hidden, mean, std = (
            self._student_commits_normalized(history)
        )
        return (
            commits[:, :1],
            commits[:, 1:2],
            hidden,
            mean,
            std,
        )

    def _far_regression_loss(self, prediction, target):
        """Apply one matched loss to clean and generated far targets.

        ``distilled_multi_commit_truth_loss`` predates the matched-target
        campaign, but its robust option must apply to generated-history
        targets as well.  Otherwise Direct and Generated differ in both the
        target and the regression loss, and low-variance raw-input channels
        can dominate only one side of the comparison.
        """
        if self.truth_loss_type == 'mse':
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(
            prediction,
            target,
            beta=self.huber_delta,
        )

    def _band_mixed_target(self, recursive_target, truth_future):
        """Interpolate generated and truth far targets per absolute-time band.

        Both inputs are (B, far_patches, patch_len) in origin-normalized
        coordinates.  Far patch m (0-indexed) commits at absolute time
        t = (m + 1) * patch_len, so its band is fixed by the shared band
        partition.  Returns (mixed, used): mixed is the per-patch target
        y = recursive + band_betas[band] * (truth - recursive), and used is
        False when the band schedule is inactive (callers keep their legacy
        loss structure).
        """
        if self.band_betas is None:
            return recursive_target, False
        if recursive_target.size(1) != truth_future.size(1):
            raise ValueError(
                'generated and truth far targets must have equal width')
        betas = torch.tensor(
            [
                float(self.band_betas[_band_index(
                    (patch + 1) * self.patch_len)])
                for patch in range(recursive_target.size(1))
            ],
            dtype=recursive_target.dtype,
            device=recursive_target.device,
        ).reshape(1, -1, 1)
        mixed = (1.0 - betas) * recursive_target + betas * truth_future
        return mixed, True

    def _periodic_far_templates(self, history, far_patches):
        """Build absolute-phase aligned shape and phase-shift directions.

        ``history`` is origin-normalized and indexed at absolute times
        ``[-W, ..., -1]``.  Far patch zero is the *second* forecast patch, so
        its first point has future time ``patch_len``.  Constructing the full
        future template before patchifying preserves this absolute phase and
        makes every K4 target an exact prefix of the corresponding K8 target.
        """
        if history.dim() != 2:
            raise ValueError('geometry history must have shape [series, W]')
        far_patches = int(far_patches)
        if far_patches < 1:
            raise ValueError('geometry target requires at least one far patch')
        period = self.geometry_period
        series, window = history.shape
        history_phase = torch.arange(
            -window, 0, device=history.device,
        ).remainder(period)
        phase_sum = history.new_zeros(series, period)
        phase_sum.scatter_add_(
            1,
            history_phase.view(1, -1).expand(series, -1),
            history,
        )
        phase_count = torch.bincount(
            history_phase, minlength=period,
        ).to(history.dtype).clamp_min(1.0)
        phase_mean = phase_sum / phase_count.view(1, -1)
        # Match the H720 geometry audit: remove the DC term from the complete
        # period frame before selecting future absolute phases.  Centering
        # each patch separately would erase sub-period low-frequency shape
        # (for example one 24-point quarter of an ETTm daily period) and is a
        # different tangent from the audited aligned-shape amplitude.
        phase_mean = phase_mean - phase_mean.mean(dim=1, keepdim=True)

        future_time = torch.arange(
            self.patch_len,
            (far_patches + 1) * self.patch_len,
            device=history.device,
        )
        future_phase = future_time.remainder(period)
        shape = phase_mean.index_select(1, future_phase)
        derivative_phase = -0.5 * (
            torch.roll(phase_mean, shifts=-1, dims=1)
            - torch.roll(phase_mean, shifts=1, dims=1)
        )
        phase = derivative_phase.index_select(1, future_phase)
        return tuple(
            values.reshape(series, far_patches, self.patch_len)
            for values in (shape, phase)
        )

    def _patchwise_geometry_basis(self, history, recursive_target):
        """Return ordered RMS-orthonormal level/shape/phase patch frames.

        The basis is a function only of the observed history and Generated
        teacher trajectory.  Future truth is used later only to label the
        coordinates in this frame.  Gram-Schmidt is independent per far
        patch, preserving protected K-prefix nesting.
        """
        if history.dim() != 2 or recursive_target.dim() != 3:
            raise ValueError('invalid patchwise geometry inputs')
        if history.size(0) != recursive_target.size(0):
            raise ValueError('geometry history and target batches differ')
        shape, phase = self._periodic_far_templates(
            history, recursive_target.size(1))
        raw = torch.stack((-recursive_target, shape, phase), dim=-1)
        directions = []
        minimum_rms = 1e-4
        for index in range(raw.size(-1)):
            vector = raw[..., index]
            for prior in directions:
                vector = vector - (
                    (vector * prior).mean(dim=-1, keepdim=True) * prior
                )
            rms = vector.square().mean(dim=-1).clamp_min(0.0).sqrt()
            active = rms > minimum_rms
            normalized = vector / rms.clamp_min(minimum_rms).unsqueeze(-1)
            directions.append(torch.where(
                active.unsqueeze(-1),
                normalized,
                torch.zeros_like(normalized),
            ))
        return torch.stack(directions, dim=-1)

    def _geometry_projected_target(
            self, history, recursive_target, truth_future):
        """Project truth residual onto selected history-computable tangents."""
        if not self.geometry_components:
            return recursive_target
        if recursive_target.shape != truth_future.shape:
            raise ValueError('geometry generated and truth targets must match')
        basis = self._patchwise_geometry_basis(history, recursive_target)
        enabled = basis.new_tensor([
            float(name in self.geometry_components)
            for name in GEOMETRY_COMPONENTS
        ]).reshape(1, 1, 1, -1)
        if self.geometry_component_strengths is None:
            strengths = enabled * self.geometry_strength
        else:
            strengths = enabled * self.geometry_component_strengths.to(
                dtype=basis.dtype,
                device=basis.device,
            ).reshape(1, 1, 1, -1)
        residual = truth_future - recursive_target
        coordinates = (
            basis * residual.unsqueeze(-1)
        ).mean(dim=2, keepdim=True)
        correction = (basis * coordinates * strengths).sum(dim=-1)
        return recursive_target + correction

    def direct_patch_loss(self, history, target):
        history, history_channels = self._to_channel_batch(
            history)
        target, target_channels = self._to_channel_batch(
            target)
        if history_channels != target_channels:
            raise ValueError(
                'history and target channels must match')
        if target.size(1) != self.train_pred_len:
            raise ValueError(
                'distillation target length does not match '
                'the macro commit width')

        with torch.no_grad():
            base, hidden, mean, std = self._base_hidden(
                history)
            first = base[:, -1:]
            recursive_target = None
            if (
                    self.truth_weight < 1.0
                    or self.band_betas is not None
                    or self.geometry_components):
                recursive_targets = []
                recursive_history = history
                recursive_prediction = first
                recursive_mean = mean
                recursive_std = std
                for _ in range(1, self.commit_patches):
                    recursive_points = (
                        recursive_prediction
                        * recursive_std
                        + recursive_mean
                    ).reshape(
                        history.size(0),
                        self.patch_len,
                        1,
                    )
                    recursive_history = torch.cat([
                        recursive_history,
                        recursive_points,
                    ], dim=1)[:, -self.seq_len:]
                    (
                        recursive,
                        _,
                        recursive_mean,
                        recursive_std,
                    ) = self._base_hidden(
                        recursive_history)
                    recursive_prediction = (
                        recursive[:, -1:])
                    recursive_points = (
                        recursive_prediction
                        * recursive_std
                        + recursive_mean
                    )
                    recursive_targets.append(
                        (recursive_points - mean) / std)
                recursive_target = torch.cat(
                    recursive_targets, dim=1)
            truth_future = self._patchify(
                (
                    target[:, self.patch_len:] - mean
                ) / std
            )

        if recursive_target is not None and truth_future is not None:
            recursive_target, mixed = self._band_mixed_target(
                recursive_target, truth_future)
            if self.geometry_components:
                normalized_history = ((history - mean) / std).squeeze(-1)
                recursive_target = self._geometry_projected_target(
                    normalized_history,
                    recursive_target,
                    truth_future,
                )
                mixed = True
        else:
            mixed = False

        student_future = (
            first.detach().expand(
                -1,
                self.commit_patches - 1,
                -1,
            )
            + self.second_residual(
                self._commit_readout(hidden).detach()).reshape(
                    history.size(0),
                    self.commit_patches - 1,
                    self.patch_len,
                )
        )
        distill_loss = (
            student_future.new_zeros(())
            if recursive_target is None
            else self._far_regression_loss(
                student_future,
                recursive_target,
            )
        )
        truth_loss = (
            student_future.new_zeros(())
            if truth_future is None
            else self._far_regression_loss(
                student_future,
                truth_future,
            )
        )
        if mixed:
            loss = distill_loss
        else:
            loss = (
                (1.0 - self.truth_weight) * distill_loss
                + self.truth_weight * truth_loss
            )
        return {
            'loss': loss,
            'distill_loss': distill_loss.detach(),
            'truth_loss': truth_loss.detach(),
            'tail_loss': truth_loss.detach(),
        }

    @torch.no_grad()
    def _build_cached_training_batch(self, history, target):
        """Materialize the exact frozen-teacher regression objective once."""
        batch_size = history.size(0)
        history, history_channels = self._to_channel_batch(history)
        target, target_channels = self._to_channel_batch(target)
        if history_channels != target_channels:
            raise ValueError(
                'history and target channels must match')
        if target.size(1) != self.train_pred_len:
            raise ValueError(
                'distillation target length does not match '
                'the macro commit width')

        base, hidden, mean, std = self._base_hidden(history)
        first = base[:, -1:]
        recursive_target = first.new_empty(
            history.size(0),
            0,
            self.patch_len,
        )
        if (
                self.truth_weight < 1.0
                or self.band_betas is not None
                or self.geometry_components):
            recursive_targets = []
            recursive_history = history
            recursive_prediction = first
            recursive_mean = mean
            recursive_std = std
            for _ in range(1, self.commit_patches):
                recursive_points = (
                    recursive_prediction
                    * recursive_std
                    + recursive_mean
                ).reshape(
                    history.size(0),
                    self.patch_len,
                    1,
                )
                recursive_history = torch.cat([
                    recursive_history,
                    recursive_points,
                ], dim=1)[:, -self.seq_len:]
                (
                    recursive,
                    _,
                    recursive_mean,
                    recursive_std,
                ) = self._base_hidden(recursive_history)
                recursive_prediction = recursive[:, -1:]
                recursive_points = (
                    recursive_prediction
                    * recursive_std
                    + recursive_mean
                )
                recursive_targets.append(
                    (recursive_points - mean) / std)
            recursive_target = torch.cat(
                recursive_targets,
                dim=1,
            )
        truth_future = first.new_empty(
            history.size(0),
            0,
            self.patch_len,
        )
        if (
                self.truth_weight > 0.0
                or self.band_betas is not None
                or self.geometry_components):
            truth_future = self._patchify(
                (target[:, self.patch_len:] - mean) / std)
        recursive_target, _ = self._band_mixed_target(
            recursive_target, truth_future)
        if self.geometry_components:
            normalized_history = ((history - mean) / std).squeeze(-1)
            recursive_target = self._geometry_projected_target(
                normalized_history,
                recursive_target,
                truth_future,
            )
        state = self._commit_readout(hidden)

        def restore_channels(tensor):
            return tensor.reshape(
                batch_size,
                history_channels,
                *tensor.shape[1:],
            )

        return tuple(restore_channels(tensor.detach()) for tensor in (
            state,
            first,
            recursive_target,
            truth_future,
        ))

    def _cached_patch_loss(
            self,
            state,
            first,
            recursive_target,
            truth_future):
        """Fit far exits from a cache without re-running the frozen Q1."""
        if state.dim() >= 3:
            state = state.flatten(0, 1)
            first = first.flatten(0, 1)
            recursive_target = recursive_target.flatten(0, 1)
            truth_future = truth_future.flatten(0, 1)
        student_future = (
            first.expand(
                -1,
                self.commit_patches - 1,
                -1,
            )
            + self.second_residual(state).reshape(
                state.size(0),
                self.commit_patches - 1,
                self.patch_len,
            )
        )
        distill_loss = (
            student_future.new_zeros(())
            if self.truth_weight >= 1.0
            else self._far_regression_loss(
                student_future,
                recursive_target,
            )
        )
        truth_loss = (
            student_future.new_zeros(())
            if self.truth_weight <= 0.0
            else self._far_regression_loss(
                student_future,
                truth_future,
            )
        )
        if self.band_betas is not None or self.geometry_components:
            # Per-band mixing already interpolated the far target inside the
            # cache; the cached truth column is informational only.
            loss = distill_loss
        else:
            loss = (
                (1.0 - self.truth_weight) * distill_loss
                + self.truth_weight * truth_loss
            )
        return {
            'loss': loss,
            'distill_loss': distill_loss.detach(),
            'truth_loss': truth_loss.detach(),
            'tail_loss': truth_loss.detach(),
        }

    @torch.no_grad()
    def forecast_with_commit_patches(self, x_enc, commit_patches):
        commit_patches = int(commit_patches)
        if not 1 <= commit_patches <= self.commit_patches:
            raise ValueError(
                'commit_patches must lie between one and trained K')
        batch_size = x_enc.size(0)
        history, channels = self._to_channel_batch(
            x_enc[:, -self.seq_len:])
        rollout_points = commit_patches * self.patch_len
        steps = math.ceil(
            self.pred_len / rollout_points)
        outputs = []

        for _ in range(steps):
            if hasattr(self, '_student_commits_for_width'):
                commits, _, mean, std = (
                    self._student_commits_for_width(
                        history,
                        commit_patches,
                    )
                )
            else:
                # Preserve the lightweight duck-typed protocol used by
                # external adapters and legacy audit fixtures.
                commits, _, mean, std = (
                    self._student_commits_normalized(history)
                )
                commits = commits[:, :commit_patches]
            points = (
                commits[:, :commit_patches] * std + mean).reshape(
                history.size(0),
                rollout_points,
                1,
            )
            outputs.append(points)
            history = torch.cat([
                history,
                points,
            ], dim=1)[:, -self.seq_len:]

        output = torch.cat(
            outputs, dim=1)[:, :self.pred_len]
        return self._from_channel_batch(
            output.squeeze(-1),
            batch_size,
            channels,
        )

    @torch.no_grad()
    def forecast(self, x_enc):
        return self.forecast_with_commit_patches(
            x_enc, self.eval_commit_patches)
