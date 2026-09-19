import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PatchAR_EncDec import PatchARBackbone


class Model(nn.Module):
    """Channel-independent dense next-patch autoregression."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != 'long_term_forecast':
            raise ValueError('DenseAR currently supports long_term_forecast only')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.features = configs.features
        self.patch_len = getattr(configs, 'ar_patch_len', 24)
        self.roll_patches = int(getattr(configs, 'dense_ar_roll_patches', 1))
        self.eval_commit_patches = int(getattr(
            configs, 'dense_ar_eval_commit_patches', 0))
        self.norm_eps = getattr(configs, 'ar_norm_eps', 1e-5)
        self.loss_space = getattr(
            configs, 'dense_ar_loss_space', 'normalized')
        self.loss_type = getattr(configs, 'dense_ar_loss_type', 'mse')
        self.far_patch_weight = float(getattr(
            configs, 'dense_ar_far_patch_weight', 1.0))
        self.onpolicy_train_patches = int(getattr(
            configs, 'dense_ar_onpolicy_train_patches', 1))
        self.onpolicy_loss_weight = float(getattr(
            configs, 'dense_ar_onpolicy_loss_weight', 0.0))
        self.onpolicy_detach_history = bool(getattr(
            configs, 'dense_ar_onpolicy_detach_history', True))
        occupancy_ages = str(getattr(
            configs, 'dense_ar_occupancy_ages', '')).strip()
        self.occupancy_ages = tuple(
            int(value.strip())
            for value in occupancy_ages.split(',')
            if value.strip()
        )
        self.occupancy_unroll_patches = int(getattr(
            configs, 'dense_ar_occupancy_unroll_patches', 2))
        self.occupancy_loss_weight = float(getattr(
            configs, 'dense_ar_occupancy_loss_weight', 0.0))
        self.occupancy_detach_local_history = bool(getattr(
            configs,
            'dense_ar_occupancy_detach_local_history',
            False,
        ))
        self.huber_delta = float(getattr(
            configs, 'dense_ar_huber_delta', 1.0))
        self.structural_loss_weight = float(getattr(
            configs, 'dense_ar_structural_loss_weight', 0.0))
        self.train_pred_len = self.patch_len * self.roll_patches

        if self.seq_len % self.patch_len != 0:
            raise ValueError('seq_len must be divisible by ar_patch_len')
        if self.roll_patches < 1:
            raise ValueError('dense_ar_roll_patches must be positive')
        if self.eval_commit_patches < 0 \
                or self.eval_commit_patches > self.roll_patches:
            raise ValueError(
                'dense_ar_eval_commit_patches must be zero or no wider '
                'than dense_ar_roll_patches')
        if self.loss_space not in {'normalized', 'point'}:
            raise ValueError(
                'dense_ar_loss_space must be normalized or point')
        if self.loss_type not in {'mse', 'huber'}:
            raise ValueError('dense_ar_loss_type must be mse or huber')
        if self.huber_delta <= 0:
            raise ValueError('dense_ar_huber_delta must be positive')
        if self.far_patch_weight < 0.0:
            raise ValueError(
                'dense_ar_far_patch_weight must be non-negative')
        if self.onpolicy_train_patches < 1:
            raise ValueError(
                'dense_ar_onpolicy_train_patches must be positive')
        if self.onpolicy_loss_weight < 0.0:
            raise ValueError(
                'dense_ar_onpolicy_loss_weight must be non-negative')
        if any(age < 0 for age in self.occupancy_ages):
            raise ValueError(
                'dense_ar_occupancy_ages must be non-negative')
        if len(set(self.occupancy_ages)) != len(self.occupancy_ages):
            raise ValueError(
                'dense_ar_occupancy_ages must not contain duplicates')
        if self.occupancy_unroll_patches < 1:
            raise ValueError(
                'dense_ar_occupancy_unroll_patches must be positive')
        if self.occupancy_loss_weight < 0.0:
            raise ValueError(
                'dense_ar_occupancy_loss_weight must be non-negative')
        if self.structural_loss_weight < 0.0:
            raise ValueError(
                'dense_ar_structural_loss_weight must be non-negative')
        if (
                self.structural_loss_weight > 0.0
                and self.loss_space != 'normalized'):
            raise ValueError(
                'patch structural loss currently requires normalized space')
        if self.onpolicy_train_patches > 1:
            if self.roll_patches != 1:
                raise ValueError(
                    'on-policy transition training requires Q1/R1')
            self.train_pred_len = (
                self.patch_len * self.onpolicy_train_patches)
        if self.occupancy_ages:
            if self.roll_patches != 1:
                raise ValueError(
                    'occupancy training requires a Q1/R1 operator')
            if self.onpolicy_train_patches > 1:
                raise ValueError(
                    'occupancy training cannot be combined with the '
                    'legacy on-policy objective')
            self.train_pred_len = self.patch_len * (
                max(self.occupancy_ages)
                + self.occupancy_unroll_patches
            )

        self.backbone = PatchARBackbone(
            patch_len=self.patch_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            norm_type=getattr(configs, 'ar_transformer_norm', 'post'),
        )
        if self.roll_patches > 1:
            self.future_patches = nn.Parameter(torch.empty(
                1, self.roll_patches - 1, self.patch_len))
            nn.init.normal_(self.future_patches, mean=0.0, std=0.02)
        else:
            self.register_parameter('future_patches', None)

    def _to_channel_batch(self, values):
        if self.features == 'MS':
            values = values[:, :, -1:]
        batch_size, length, channels = values.shape
        values = values.permute(0, 2, 1).reshape(batch_size * channels, length, 1)
        return values, channels

    @staticmethod
    def _from_channel_batch(values, batch_size, channels):
        length = values.size(1)
        return values.view(batch_size, channels, length).permute(0, 2, 1).contiguous()

    def _patchify(self, values):
        if values.size(1) % self.patch_len != 0:
            raise ValueError('the point dimension must be divisible by ar_patch_len')
        return values.squeeze(-1).reshape(values.size(0), -1, self.patch_len)

    def _instance_stats(self, history):
        if history.size(1) != self.seq_len:
            raise ValueError(f'DenseAR expects exactly {self.seq_len} history points')
        mean = history.mean(dim=1, keepdim=True).detach()
        variance = torch.mean(torch.square(history - mean), dim=1, keepdim=True)
        std = torch.sqrt(variance + self.norm_eps).detach()
        return mean, std

    def _predict_all_normalized(self, history):
        mean, std = self._instance_stats(history)
        history_normalized = (history - mean) / std
        history_patches = self._patchify(history_normalized)
        if self.future_patches is None:
            input_patches = history_patches
        else:
            forecast_queries = self.future_patches.expand(
                history.size(0), -1, -1)
            input_patches = torch.cat(
                [history_patches, forecast_queries], dim=1)
        hidden = self.backbone(input_patches)
        predictions = self.backbone.output_head(hidden)
        return predictions, history_patches, mean, std

    def _regression_loss(self, predictions, targets):
        if self.loss_type == 'mse':
            return F.mse_loss(predictions, targets)
        return F.smooth_l1_loss(
            predictions, targets, beta=self.huber_delta)

    @staticmethod
    def _patch_structural_components(predictions, targets):
        """Correlation, shape-distribution, and mean losses from PS Loss."""
        predictions = predictions.float()
        targets = targets.float()
        target_mean = targets.mean(dim=-1, keepdim=True)
        prediction_mean = predictions.mean(dim=-1, keepdim=True)
        target_centered = targets - target_mean
        prediction_centered = predictions - prediction_mean
        target_variance = target_centered.square().mean(
            dim=-1, keepdim=True)
        prediction_variance = prediction_centered.square().mean(
            dim=-1, keepdim=True)
        target_std = torch.sqrt(target_variance)
        prediction_std = torch.sqrt(prediction_variance)
        covariance = (
            target_centered * prediction_centered
        ).mean(dim=-1, keepdim=True)
        correlation = (
            covariance + 1e-5
        ) / (target_std * prediction_std + 1e-5)
        correlation_loss = (1.0 - correlation).mean()
        distribution_loss = F.kl_div(
            F.log_softmax(predictions, dim=-1),
            F.softmax(targets, dim=-1),
            reduction='none',
        ).sum(dim=-1).mean()
        mean_loss = torch.abs(
            target_mean - prediction_mean).mean()
        return correlation_loss, distribution_loss, mean_loss

    def _patch_structural_loss(self, predictions, targets):
        components = self._patch_structural_components(
            predictions, targets)
        head_weight = self.backbone.output_head.weight
        gradients = [
            torch.autograd.grad(
                component,
                head_weight,
                retain_graph=True,
                create_graph=False,
            )[0]
            for component in components
        ]
        average_gradient = sum(gradients) / len(gradients)
        average_norm = average_gradient.norm().detach()
        weights = [
            (
                average_norm
                / gradient.norm().detach().clamp_min(1e-8)
            ).clamp(max=100.0)
            for gradient in gradients
        ]

        flat_predictions = predictions.float().flatten(1)
        flat_targets = targets.float().flatten(1)
        prediction_mean = flat_predictions.mean(dim=-1, keepdim=True)
        target_mean = flat_targets.mean(dim=-1, keepdim=True)
        prediction_centered = flat_predictions - prediction_mean
        target_centered = flat_targets - target_mean
        prediction_variance = prediction_centered.square().mean(
            dim=-1, keepdim=True)
        target_variance = target_centered.square().mean(
            dim=-1, keepdim=True)
        prediction_std = torch.sqrt(prediction_variance)
        target_std = torch.sqrt(target_variance)
        covariance = (
            prediction_centered * target_centered
        ).mean(dim=-1, keepdim=True)
        linear_similarity = 0.5 * (1.0 + (
            covariance + 1e-5
        ) / (prediction_std * target_std + 1e-5))
        variance_similarity = (
            2.0 * prediction_std * target_std + 1e-5
        ) / (
            prediction_variance + target_variance + 1e-5
        )
        weights[2] = weights[2] * (
            linear_similarity * variance_similarity
        ).mean().detach()
        structural_loss = sum(
            weight * component
            for weight, component in zip(weights, components)
        )
        return structural_loss, components, weights

    def direct_patch_loss(self, history, target):
        if self.occupancy_ages:
            return self._occupancy_patch_loss(history, target)
        if self.onpolicy_train_patches > 1:
            return self._onpolicy_patch_loss(
                history, target)
        history, history_channels = self._to_channel_batch(history)
        target, target_channels = self._to_channel_batch(target)
        if history_channels != target_channels:
            raise ValueError('history and target channel counts must match')
        if target.size(1) != self.train_pred_len:
            raise ValueError(
                f'DenseAR training target must contain exactly {self.train_pred_len} points'
            )

        predictions, history_patches, mean, std = self._predict_all_normalized(history)
        future_patches = self._patchify((target - mean) / std)
        next_patch_targets = torch.cat(
            [history_patches[:, 1:], future_patches], dim=1)
        if self.loss_space == 'normalized':
            loss_predictions = predictions
            loss_targets = next_patch_targets
            tail_loss = F.mse_loss(
                predictions[:, -self.roll_patches:], future_patches).detach()
        else:
            point_predictions = predictions * std + mean
            point_targets = next_patch_targets * std + mean
            loss_predictions = point_predictions
            loss_targets = point_targets
            tail_loss = F.mse_loss(
                point_predictions[:, -self.roll_patches:],
                point_targets[:, -self.roll_patches:],
            ).detach()
        far_count = self.roll_patches - 1
        if far_count == 0 or self.far_patch_weight == 1.0:
            loss = self._regression_loss(
                loss_predictions,
                loss_targets,
            )
            base_loss = loss.detach()
            far_loss = loss.new_zeros(())
        else:
            base_count = loss_predictions.size(1) - far_count
            base_objective = self._regression_loss(
                loss_predictions[:, :base_count],
                loss_targets[:, :base_count],
            )
            far_objective = self._regression_loss(
                loss_predictions[:, base_count:],
                loss_targets[:, base_count:],
            )
            denominator = (
                base_count
                + self.far_patch_weight * far_count
            )
            loss = (
                base_count * base_objective
                + self.far_patch_weight
                * far_count * far_objective
            ) / denominator
            base_loss = base_objective.detach()
            far_loss = far_objective.detach()
        optimization_loss = loss
        structural_loss = loss.new_zeros(())
        structural_components = [loss.new_zeros(())] * 3
        structural_weights = [loss.new_zeros(())] * 3
        if (
                self.structural_loss_weight > 0.0
                and self.training
                and torch.is_grad_enabled()):
            (
                structural_loss,
                structural_components,
                structural_weights,
            ) = self._patch_structural_loss(
                loss_predictions,
                loss_targets,
            )
            optimization_loss = (
                loss
                + self.structural_loss_weight * structural_loss
            )
        return {
            'loss': loss,
            'optimization_loss': optimization_loss,
            'dense_loss': loss.detach(),
            'tail_loss': tail_loss,
            'base_loss': base_loss,
            'far_loss': far_loss,
            'structural_loss': structural_loss.detach(),
            'structural_corr_loss': structural_components[0].detach(),
            'structural_distribution_loss':
                structural_components[1].detach(),
            'structural_mean_loss': structural_components[2].detach(),
            'structural_corr_weight': structural_weights[0].detach(),
            'structural_distribution_weight':
                structural_weights[1].detach(),
            'structural_mean_weight': structural_weights[2].detach(),
        }

    def _occupancy_patch_loss(self, history, target):
        """Optimize Q1 after a sampled amount of generated-history roll-in.

        The prefix is generated in inference mode without a computation graph.
        Only the short suffix is differentiable, which exposes late rollout
        states without retaining the complete long-horizon graph.
        """
        history, history_channels = self._to_channel_batch(history)
        target, target_channels = self._to_channel_batch(target)
        if history_channels != target_channels:
            raise ValueError(
                'history and target channel counts must match')
        if target.size(1) != self.train_pred_len:
            raise ValueError(
                'occupancy target must contain exactly '
                f'{self.train_pred_len} points')

        # Preserve the ordinary observed-history next-transition objective.
        predictions, history_patches, mean, std = (
            self._predict_all_normalized(history)
        )
        first_points = target[:, :self.patch_len]
        first_patch = self._patchify(
            (first_points - mean) / std)
        dense_targets = torch.cat([
            history_patches[:, 1:],
            first_patch,
        ], dim=1)
        if self.loss_space == 'normalized':
            dense_loss = self._regression_loss(
                predictions,
                dense_targets,
            )
            tail_loss = self._regression_loss(
                predictions[:, -1:],
                first_patch,
            ).detach()
        else:
            point_predictions = predictions * std + mean
            point_targets = dense_targets * std + mean
            dense_loss = self._regression_loss(
                point_predictions,
                point_targets,
            )
            tail_loss = self._regression_loss(
                point_predictions[:, -1:],
                point_targets[:, -1:],
            ).detach()

        age_index = torch.randint(
            len(self.occupancy_ages),
            (1,),
            device=history.device,
        ).item()
        age = self.occupancy_ages[age_index]
        current_history = history
        occupancy_losses = []
        was_training = self.training
        outer_grad_enabled = torch.is_grad_enabled()
        try:
            # The visited-state path should match deterministic inference,
            # including disabled dropout, while remaining differentiable in
            # the local suffix.
            self.eval()
            for step in range(age + self.occupancy_unroll_patches):
                local_step = step >= age
                differentiable = outer_grad_enabled and local_step
                with torch.set_grad_enabled(differentiable):
                    (
                        step_predictions,
                        _,
                        step_mean,
                        step_std,
                    ) = self._predict_all_normalized(current_history)
                    predicted_patch = step_predictions[:, -1:]
                    predicted_points = (
                        predicted_patch * step_std + step_mean
                    ).reshape(
                        current_history.size(0),
                        self.patch_len,
                        1,
                    )
                    if local_step:
                        target_start = step * self.patch_len
                        target_points = target[
                            :,
                            target_start:target_start + self.patch_len,
                        ]
                        if self.loss_space == 'normalized':
                            target_patch = self._patchify(
                                (target_points - step_mean) / step_std)
                            occupancy_losses.append(
                                self._regression_loss(
                                    predicted_patch,
                                    target_patch,
                                )
                            )
                        else:
                            occupancy_losses.append(
                                self._regression_loss(
                                    predicted_points,
                                    target_points,
                                )
                            )
                    if (
                            not differentiable
                            or self.occupancy_detach_local_history):
                        predicted_points = predicted_points.detach()
                    current_history = torch.cat([
                        current_history,
                        predicted_points,
                    ], dim=1)[:, -self.seq_len:]
        finally:
            self.train(was_training)

        occupancy_loss = torch.stack(occupancy_losses).mean()
        optimization_loss = (
            dense_loss
            + self.occupancy_loss_weight * occupancy_loss
        )
        return {
            'loss': optimization_loss,
            'optimization_loss': optimization_loss,
            'dense_loss': dense_loss.detach(),
            'tail_loss': tail_loss,
            'occupancy_loss': occupancy_loss.detach(),
            'occupancy_age': dense_loss.new_tensor(float(age)),
        }

    def _onpolicy_patch_loss(self, history, target):
        """Train Q1 on the generated-history distribution it induces."""
        history, history_channels = self._to_channel_batch(
            history)
        target, target_channels = self._to_channel_batch(target)
        if history_channels != target_channels:
            raise ValueError(
                'history and target channel counts must match')
        expected = (
            self.patch_len * self.onpolicy_train_patches)
        if target.size(1) != expected:
            raise ValueError(
                'on-policy target must contain exactly '
                f'{expected} points')

        predictions, history_patches, mean, std = (
            self._predict_all_normalized(history)
        )
        first_points = target[:, :self.patch_len]
        first_patch = self._patchify(
            (first_points - mean) / std)
        dense_targets = torch.cat([
            history_patches[:, 1:],
            first_patch,
        ], dim=1)
        if self.loss_space == 'normalized':
            dense_loss = self._regression_loss(
                predictions,
                dense_targets,
            )
        else:
            dense_loss = self._regression_loss(
                predictions * std + mean,
                dense_targets * std + mean,
            )

        current_history = history
        onpolicy_losses = []
        for step in range(self.onpolicy_train_patches):
            if step == 0:
                step_predictions = predictions
                step_mean = mean
                step_std = std
            else:
                step_predictions, _, step_mean, step_std = (
                    self._predict_all_normalized(
                        current_history)
                )
            predicted_patch = step_predictions[:, -1:]
            target_start = step * self.patch_len
            target_points = target[
                :, target_start:target_start + self.patch_len]
            target_patch = self._patchify(
                (target_points - step_mean) / step_std)
            if step > 0:
                if self.loss_space == 'normalized':
                    onpolicy_losses.append(
                        self._regression_loss(
                            predicted_patch,
                            target_patch,
                        )
                    )
                else:
                    onpolicy_losses.append(
                        self._regression_loss(
                            predicted_patch * step_std
                            + step_mean,
                            target_patch * step_std
                            + step_mean,
                        )
                    )
            predicted_points = (
                predicted_patch * step_std + step_mean
            ).reshape(
                current_history.size(0),
                self.patch_len,
                1,
            )
            if self.onpolicy_detach_history:
                predicted_points = predicted_points.detach()
            current_history = torch.cat([
                current_history,
                predicted_points,
            ], dim=1)[:, -self.seq_len:]

        onpolicy_loss = torch.stack(
            onpolicy_losses).mean()
        loss = (
            dense_loss
            + self.onpolicy_loss_weight * onpolicy_loss
        )
        return {
            'loss': loss,
            'dense_loss': dense_loss.detach(),
            'tail_loss': F.mse_loss(
                predictions[:, -1:],
                first_patch,
            ).detach(),
            'onpolicy_loss': onpolicy_loss.detach(),
        }

    @torch.no_grad()
    def forecast(self, x_enc):
        batch_size = x_enc.size(0)
        history, channels = self._to_channel_batch(x_enc[:, -self.seq_len:, :])
        commit_patches = (
            self.eval_commit_patches
            if self.eval_commit_patches
            else self.roll_patches
        )
        rollout_points = self.patch_len * commit_patches
        steps = math.ceil(self.pred_len / rollout_points)
        predictions = []

        for _ in range(steps):
            if bool(getattr(
                    self, 'last_query_inference', False)):
                normalized, _, mean, std = (
                    self._predict_last_normalized(history))
            else:
                normalized, _, mean, std = (
                    self._predict_all_normalized(history))
            query_predictions = normalized[:, -self.roll_patches:]
            next_patches = (
                query_predictions[:, :commit_patches] * std + mean)
            next_points = next_patches.reshape(
                history.size(0), rollout_points, 1)
            predictions.append(next_points)
            history = torch.cat([history, next_points], dim=1)[:, -self.seq_len:, :]

        output = torch.cat(predictions, dim=1)[:, :self.pred_len, :]
        return self._from_channel_batch(output.squeeze(-1), batch_size, channels)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
