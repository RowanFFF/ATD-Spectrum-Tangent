from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import StreamingMetric, metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math
import numpy as np
import copy
import json
from torch.utils.data import DataLoader, TensorDataset
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_args = self.args
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        train_pred_len = getattr(model, 'train_pred_len', None)
        if flag == 'train' and train_pred_len is not None \
                and not self._point_space_supervision():
            data_args = copy.copy(self.args)
            data_args.pred_len = train_pred_len
        data_set, data_loader = data_provider(data_args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        parameters = (
            model.optimizer_parameter_groups(
                self.args.learning_rate)
            if hasattr(
                model,
                'optimizer_parameter_groups',
            )
            else self.model.parameters()
        )
        model_optim = optim.Adam(
            parameters,
            lr=self.args.learning_rate,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        json_path = getattr(self.args, 'band_norm_json', '')
        if json_path:
            with open(json_path, encoding='utf-8') as handle:
                signal_energy = json.load(handle)
            from models.band_normalized_loss import BandNormalizedMSELoss
            criterion = BandNormalizedMSELoss(
                self.args.pred_len,
                self.args.ar_patch_len,
                signal_energy,
            ).to(self.device)
        return criterion

    def _sam_perturb(self, rho):
        """Move parameters to the first-order worst-case SAM neighbor."""
        if rho <= 0:
            raise ValueError('SAM rho must be positive')
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not parameters:
            raise RuntimeError('SAM requires at least one gradient')
        gradient_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().float().norm(2)
            for parameter in parameters
        ]), 2)
        scale = float(rho) / (gradient_norm + 1e-12)
        perturbations = []
        with torch.no_grad():
            for parameter in parameters:
                perturbation = parameter.grad.detach() * scale.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter.add_(perturbation)
                perturbations.append((parameter, perturbation))
        return perturbations

    @staticmethod
    def _sam_restore(perturbations):
        with torch.no_grad():
            for parameter, perturbation in perturbations:
                parameter.sub_(perturbation)

    def _checkpoint_score(self, vali_loss, test_loss):
        selection = getattr(self.args, 'checkpoint_selection', 'vali')
        if selection == 'vali':
            return vali_loss
        if selection == 'test_oracle':
            if not np.isfinite(test_loss):
                raise ValueError(
                    'test_oracle selection requires per-epoch test scoring')
            return test_loss
        raise ValueError(f'unknown checkpoint selection: {selection}')

    @staticmethod
    def _set_active_channel_indices(model, indices):
        if hasattr(model, 'set_active_channel_indices'):
            model.set_active_channel_indices(indices)

    def _point_space_supervision(self):
        """Point-space training supervises the AR forecast output (model
        forward == forecast) instead of the patch-space DenseAR objective.
        Activated explicitly by --point_space_train or implicitly by
        --band_norm_json (mechanism 3 normalizes the forecast's DCT bands).
        """
        return bool(
            getattr(self.args, 'band_norm_json', '')
            or getattr(self.args, 'point_space_train', False))

    def _point_space_forecast(self, model, batch_x):
        """Gradient-carrying copy of the DenseAR rollout.

        Model.forecast is wrapped in @torch.no_grad() (it is the inference
        path), so point-space supervision cannot call it for training: the
        loss would carry no graph.  This replicates the same rollout through
        the model's trainable _predict_all_normalized so gradients flow.
        """
        batch_size = batch_x.size(0)
        history, channels = model._to_channel_batch(
            batch_x[:, -model.seq_len:, :])
        commit_patches = (
            model.eval_commit_patches
            if model.eval_commit_patches
            else model.roll_patches
        )
        rollout_points = model.patch_len * commit_patches
        steps = math.ceil(model.pred_len / rollout_points)
        predictions = []
        for _ in range(steps):
            if bool(getattr(model, 'last_query_inference', False)):
                normalized, _, mean, std = (
                    model._predict_last_normalized(history))
            else:
                normalized, _, mean, std = (
                    model._predict_all_normalized(history))
            query_predictions = normalized[:, -model.roll_patches:]
            next_patches = (
                query_predictions[:, :commit_patches] * std + mean)
            next_points = next_patches.reshape(
                history.size(0), rollout_points, 1)
            predictions.append(next_points)
            history = torch.cat(
                [history, next_points], dim=1)[:, -model.seq_len:, :]
        output = torch.cat(predictions, dim=1)[:, :model.pred_len, :]
        return model._from_channel_batch(
            output.squeeze(-1), batch_size, channels)

    def _training_objective(self, batch_x, batch_y, batch_x_mark,
                            batch_y_mark, dec_inp, criterion):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'direct_patch_loss_with_marks') \
                and not self._point_space_supervision():
            self._set_active_channel_indices(model, None)
            target = batch_y[:, -model.train_pred_len:, :]
            future_marks = batch_y_mark[:, -model.train_pred_len:, :]
            losses = model.direct_patch_loss_with_marks(
                batch_x,
                target,
                batch_x_mark,
                future_marks,
            )
            return losses.get('optimization_loss', losses['loss']), losses
        if hasattr(model, 'direct_patch_loss') \
                and not self._point_space_supervision():
            target = batch_y[:, -model.train_pred_len:, :]
            channel_indices = None
            channel_limit = int(getattr(
                self.args, 'direct_patch_train_channels', 0))
            if channel_limit > 0 and batch_x.size(-1) > channel_limit:
                channel_indices = torch.randperm(
                    batch_x.size(-1),
                    device=batch_x.device,
                )[:channel_limit]
                batch_x = batch_x.index_select(-1, channel_indices)
                target = target.index_select(-1, channel_indices)
            self._set_active_channel_indices(model, channel_indices)
            losses = model.direct_patch_loss(batch_x, target)
            return losses.get('optimization_loss', losses['loss']), losses

        self._set_active_channel_indices(model, None)
        if self._point_space_supervision():
            outputs = self._point_space_forecast(model, batch_x)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        return criterion(outputs, target), None

    def _build_direct_patch_cache(
            self,
            train_loader,
            cache_index):
        model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if not hasattr(model, 'build_cached_training_batch') \
                or not hasattr(model, 'cached_patch_loss'):
            raise ValueError(
                'cached direct-patch training requires model cache hooks')
        started = time.time()
        chunks = None
        tensors = None
        tensor_offset = 0
        generic_cache = bool(getattr(
            self.args,
            'direct_patch_cached_training',
            False,
        ))
        cache_prefix = (
            'direct_patch_cache'
            if generic_cache
            else 'protected_pushforward_cache'
        )
        cache_fraction = float(getattr(
            self.args,
            f'{cache_prefix}_fraction',
            1.0,
        ))
        if not 0.0 < cache_fraction <= 1.0:
            raise ValueError(
                'protected pushforward cache fraction '
                'must lie in (0, 1]')
        cache_batches = max(
            1,
            int(np.ceil(
                len(train_loader) * cache_fraction)),
        )
        model.eval()
        with torch.no_grad():
            for batch_index, (
                    batch_x,
                    batch_y,
                    _,
                    _) in enumerate(train_loader):
                if batch_index >= cache_batches:
                    break
                batch_x = batch_x.float().to(self.device)
                target = batch_y[
                    :, -model.train_pred_len:, :
                ].float().to(self.device)
                channel_limit = int(getattr(
                    self.args,
                    'direct_patch_train_channels',
                    0,
                ))
                channel_indices = None
                if (
                        channel_limit > 0
                        and batch_x.size(-1) > channel_limit):
                    channel_indices = torch.randperm(
                        batch_x.size(-1),
                        device=batch_x.device,
                    )[:channel_limit]
                    batch_x = batch_x.index_select(
                        -1, channel_indices)
                    target = target.index_select(
                        -1, channel_indices)
                self._set_active_channel_indices(
                    model, channel_indices)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        cached = (
                            model.build_cached_training_batch(
                                batch_x,
                                target,
                            ))
                else:
                    cached = model.build_cached_training_batch(
                        batch_x,
                        target,
                    )
                cached = tuple(
                    tensor.detach().float().cpu()
                    for tensor in cached
                )
                if chunks is None and tensors is None:
                    loader_batch_size = getattr(
                        train_loader, 'batch_size', None)
                    loader_dataset = getattr(
                        train_loader, 'dataset', None)
                    if (
                            isinstance(loader_batch_size, int)
                            and loader_batch_size > 0
                            and loader_dataset is not None):
                        dataset_windows = len(loader_dataset)
                        if bool(getattr(
                                train_loader, 'drop_last', False)):
                            dataset_windows = (
                                dataset_windows // loader_batch_size
                            ) * loader_batch_size
                        cached_windows = min(
                            dataset_windows,
                            cache_batches * loader_batch_size,
                        )
                        tensors = tuple(torch.empty(
                            (cached_windows, *tensor.shape[1:]),
                            dtype=tensor.dtype,
                        ) for tensor in cached)
                    else:
                        # Small synthetic iterables used by tests and custom
                        # callers do not expose enough size information for
                        # safe preallocation.
                        chunks = [[] for _ in cached]
                destinations = tensors if tensors is not None else chunks
                if len(cached) != len(destinations):
                    raise ValueError(
                        'cached model returned an inconsistent '
                        'number of tensors')
                if tensors is not None:
                    next_offset = tensor_offset + cached[0].size(0)
                    if next_offset > tensors[0].size(0):
                        raise ValueError(
                            'direct-patch cache preallocation was too small')
                    for destination, tensor in zip(tensors, cached):
                        destination[tensor_offset:next_offset].copy_(tensor)
                    tensor_offset = next_offset
                else:
                    for destination, tensor in zip(chunks, cached):
                        destination.append(tensor)
        if chunks is None and tensors is None:
            raise ValueError(
                'cannot build a direct-patch cache from '
                'an empty training loader')
        if tensors is not None:
            tensors = tuple(
                tensor[:tensor_offset] for tensor in tensors
            )
        else:
            tensors = tuple(
                torch.cat(parts, dim=0)
                for parts in chunks
            )
        window_count = tensors[0].size(0)
        channel_count = tensors[0].size(1)
        shuffle_unit = str(getattr(
            self.args,
            f'{cache_prefix}_shuffle_unit',
            'window',
        ))
        if shuffle_unit == 'channel':
            tensors = tuple(
                tensor.flatten(0, 1)
                for tensor in tensors
            )
            cache_batch_size = (
                self.args.batch_size * channel_count)
            cache_description = (
                f'{window_count * channel_count} '
                'channel records')
        elif shuffle_unit == 'window':
            cache_batch_size = self.args.batch_size
            cache_description = (
                f'{window_count} windows x '
                f'{channel_count} channels')
        else:
            raise ValueError(
                'protected pushforward cache shuffle unit '
                'must be window or channel')
        generator = torch.Generator()
        loader_seed = int(getattr(
            self.args, 'loader_seed', -1))
        generator.manual_seed(
            (loader_seed if loader_seed >= 0 else 0)
            + 1009 + cache_index
        )
        cached_loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=cache_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=generator,
        )
        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )
        elapsed = time.time() - started
        print(
            f'Cached direct-patch pass {cache_index}: '
            f'{cache_description}, '
            f'fraction {cache_fraction:.3f}, '
            f'{cache_bytes / (1024 ** 2):.1f} MiB, '
            f'{elapsed:.3f}s'
        )
        model.train()
        return cached_loader

    def _backward_anchor_pcgrad(self, objectives):
        """Protect the first objective from conflicting far-horizon gradients.

        The model identifies parameters that carry shared transition
        semantics. For those parameters, a far-objective gradient with a
        negative inner product against the anchor has only that conflicting
        component removed. Other parameters receive the ordinary sum.
        """
        model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if self.args.use_amp and not getattr(
                model, 'supports_amp_gradient_routing', False):
            raise ValueError(
                'anchor_pcgrad currently requires AMP to be disabled')
        if not hasattr(model, 'gradient_projection_parameters'):
            raise ValueError(
                'gradient objectives require '
                'gradient_projection_parameters()')
        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        protected_ids = {
            id(parameter)
            for parameter in model.gradient_projection_parameters()
            if parameter.requires_grad
        }
        # A fixed AMP scale is unsafe for long closed-loop objectives: models
        # with many autoregressive commits can have perfectly finite unscaled
        # gradients that overflow only after multiplying the loss by 1024.
        # Allow such models to request an unscaled routing pass while keeping
        # the historical default for existing gradient-routed models.
        gradient_scale = float(getattr(
            model,
            'gradient_routing_scale',
            1024.0 if self.args.use_amp else 1.0,
        ))
        if not np.isfinite(gradient_scale) or gradient_scale <= 0.0:
            raise ValueError('gradient routing scale must be finite and positive')
        gradients = []
        for index, objective in enumerate(objectives):
            objective_gradients = torch.autograd.grad(
                objective * gradient_scale,
                parameters,
                retain_graph=index + 1 < len(objectives),
                allow_unused=True,
            )
            if any(
                    gradient is not None
                    and not bool(torch.isfinite(gradient).all())
                    for gradient in objective_gradients):
                raise FloatingPointError(
                    'non-finite gradient in anchor_pcgrad routing; refusing '
                    'to update or checkpoint the model'
                )
            gradients.append(objective_gradients)

        anchor = gradients[0]
        projected = [list(group) for group in gradients]
        normalize_far = (
            getattr(model, 'gradient_mode', '')
            == 'anchor_norm_pcgrad_amp'
        )
        conflicts = 0
        far_count = max(len(gradients) - 1, 1)
        for group_index in range(1, len(gradients)):
            dot = None
            anchor_square = None
            far_square = None
            for parameter, anchor_grad, far_grad in zip(
                    parameters, anchor, gradients[group_index]):
                if id(parameter) not in protected_ids \
                        or anchor_grad is None or far_grad is None:
                    continue
                product = torch.sum(anchor_grad * far_grad)
                square = torch.sum(anchor_grad.square())
                far_norm_square = torch.sum(far_grad.square())
                dot = product if dot is None else dot + product
                anchor_square = (
                    square if anchor_square is None
                    else anchor_square + square
                )
                far_square = (
                    far_norm_square if far_square is None
                    else far_square + far_norm_square
                )
            if dot is None or anchor_square is None:
                continue
            if normalize_far and far_square is not None:
                norm_scale = torch.minimum(
                    torch.ones_like(anchor_square),
                    torch.sqrt(
                        anchor_square.clamp_min(1e-12)
                        / far_square.clamp_min(1e-12)
                    ),
                )
                dot = dot * norm_scale
                for parameter_index, (
                        parameter, anchor_grad, far_grad) in enumerate(zip(
                            parameters, anchor, gradients[group_index])):
                    if id(parameter) in protected_ids \
                            and anchor_grad is not None \
                            and far_grad is not None:
                        projected[group_index][parameter_index] = (
                            far_grad * norm_scale)
            if dot.detach().item() < 0:
                conflicts += 1
                coefficient = dot / anchor_square.clamp_min(1e-12)
                for parameter_index, (
                        parameter, anchor_grad, far_grad) in enumerate(zip(
                            parameters, anchor, gradients[group_index])):
                    if id(parameter) in protected_ids \
                            and anchor_grad is not None \
                            and far_grad is not None:
                        projected[group_index][parameter_index] = (
                            projected[group_index][parameter_index]
                            - coefficient * anchor_grad
                        )

        for parameter_index, parameter in enumerate(parameters):
            total = None
            for group in projected:
                gradient = group[parameter_index]
                if gradient is not None:
                    total = gradient if total is None else total + gradient
            if total is not None and not bool(torch.isfinite(total).all()):
                raise FloatingPointError(
                    'non-finite projected anchor_pcgrad gradient; refusing '
                    'to update or checkpoint the model'
                )
            parameter.grad = (
                None if total is None else total / gradient_scale)
        return conflicts / far_count

    def _backward_isolated_far(self, objectives):
        """Route far-horizon gradients only to horizon-specific parameters."""
        model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if self.args.use_amp and not getattr(
                model, 'supports_amp_gradient_routing', False):
            raise ValueError(
                'isolated_far currently requires AMP to be disabled')
        if not hasattr(model, 'far_horizon_parameters'):
            raise ValueError(
                'isolated objectives require far_horizon_parameters()')
        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        far_parameter_ids = {
            id(parameter)
            for parameter in model.far_horizon_parameters()
            if parameter.requires_grad
        }
        gradient_scale = 1024.0 if self.args.use_amp else 1.0
        gradients = [
            torch.autograd.grad(
                objective * gradient_scale,
                parameters,
                retain_graph=index + 1 < len(objectives),
                allow_unused=True,
            )
            for index, objective in enumerate(objectives)
        ]
        for parameter_index, parameter in enumerate(parameters):
            total = gradients[0][parameter_index]
            if id(parameter) in far_parameter_ids:
                for group in gradients[1:]:
                    gradient = group[parameter_index]
                    if gradient is not None:
                        total = (
                            gradient
                            if total is None
                            else total + gradient
                        )
            parameter.grad = (
                None if total is None else total / gradient_scale)

    def _scaler_stats(self, data_set, width):
        mean = np.asarray(data_set.scaler.mean_)
        scale = np.asarray(data_set.scaler.scale_)
        if mean.size == width:
            return mean, scale
        if self.args.features == 'MS' and width == 1:
            return mean[-1:], scale[-1:]
        if width < mean.size:
            return mean[-width:], scale[-width:]
        raise ValueError('scaler statistics do not match forecast channels')

    def _benchmark_tensors(self, data_set, pred, true):
        """Put raw-input model outputs in TSLib's standardized metric space."""
        if data_set.scale or not hasattr(data_set, 'scaler') \
                or not hasattr(data_set.scaler, 'mean_'):
            return pred, true
        mean, scale = self._scaler_stats(data_set, pred.size(-1))
        mean = torch.as_tensor(mean, device=pred.device, dtype=pred.dtype)
        scale = torch.as_tensor(scale, device=pred.device, dtype=pred.dtype)
        return (pred - mean) / scale, (true - mean) / scale

    def _evaluation_arrays(self, data_set, pred, true):
        """Match the existing protocol: standardized by default, raw with --inverse."""
        if (data_set.scale and not self.args.inverse) \
                or (not data_set.scale and self.args.inverse):
            return pred, true
        if not hasattr(data_set, 'scaler') or not hasattr(data_set.scaler, 'mean_'):
            return pred, true
        mean, scale = self._scaler_stats(data_set, pred.shape[-1])
        shape = (1,) * (pred.ndim - 1) + (-1,)
        mean = mean.reshape(shape)
        scale = scale.reshape(shape)
        if data_set.scale:
            return pred * scale + mean, true * scale + mean
        return (pred - mean) / scale, (true - mean) / scale

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_weight = []
        self.model.eval()
        model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        direct_patch_validation = (
            getattr(self.args, 'direct_patch_validation', False)
            and hasattr(model, 'direct_patch_loss')
            and not self._point_space_supervision()
        )
        self._set_active_channel_indices(model, None)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if direct_patch_validation:
                    target_start = self.args.label_len
                    target_end = target_start + model.train_pred_len
                    target = batch_y[:, target_start:target_end, :].to(
                        self.device)
                    channel_limit = int(getattr(
                        self.args,
                        'direct_patch_validation_channels',
                        -1,
                    ))
                    if channel_limit < 0:
                        channel_limit = int(getattr(
                            self.args, 'direct_patch_train_channels', 0))
                    channel_indices = None
                    if channel_limit > 0 \
                            and batch_x.size(-1) > channel_limit:
                        channel_indices = torch.div(
                            torch.arange(
                                channel_limit,
                                device=batch_x.device,
                            ) * batch_x.size(-1),
                            channel_limit,
                            rounding_mode='floor',
                        )
                        batch_x = batch_x.index_select(
                            -1, channel_indices)
                        target = target.index_select(
                            -1, channel_indices)
                    self._set_active_channel_indices(
                        model, channel_indices)
                    if hasattr(model, 'direct_patch_loss_with_marks'):
                        future_marks = batch_y_mark[
                            :, target_start:target_end, :
                        ]
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                loss = model.direct_patch_loss_with_marks(
                                    batch_x,
                                    target,
                                    batch_x_mark,
                                    future_marks,
                                )['loss']
                        else:
                            loss = model.direct_patch_loss_with_marks(
                                batch_x,
                                target,
                                batch_x_mark,
                                future_marks,
                            )['loss']
                    elif self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            loss = model.direct_patch_loss(
                                batch_x, target)['loss']
                    else:
                        loss = model.direct_patch_loss(
                            batch_x, target)['loss']
                    total_loss.append(loss.item())
                    total_weight.append(batch_x.size(0))
                    continue

                # decoder input
                self._set_active_channel_indices(model, None)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()
                pred, true = self._benchmark_tensors(vali_data, pred, true)
                loss = criterion(pred, true)

                total_loss.append(loss.item())
                total_weight.append(batch_x.size(0))
        total_loss = np.average(
            total_loss,
            weights=total_weight,
        )
        self._set_active_channel_indices(model, None)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        raw_train_loader = train_loader
        generic_cached_training = bool(getattr(
            self.args,
            'direct_patch_cached_training',
            False,
        ))
        protected_cached_training = bool(getattr(
            self.args,
            'protected_pushforward_cached_training',
            False,
        ))
        cached_training = (
            generic_cached_training
            or protected_cached_training
        )
        cache_prefix = (
            'direct_patch_cache'
            if generic_cached_training
            else 'protected_pushforward_cache'
        )
        cached_training_active = cached_training
        cache_epochs = int(getattr(
            self.args,
            f'{cache_prefix}_epochs',
            0,
        ))
        if cache_epochs < 0:
            raise ValueError(
                'protected pushforward cache epochs '
                'must be non-negative')
        requested_cache_refreshes = int(getattr(
            self.args,
            f'{cache_prefix}_refreshes',
            0,
        ))
        if requested_cache_refreshes < 0:
            raise ValueError(
                'protected pushforward cache refreshes '
                'must be non-negative')
        completed_cache_refreshes = 0
        if cached_training:
            train_loader = self._build_direct_patch_cache(
                raw_train_loader,
                cache_index=0,
            )

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if getattr(
                self.args,
                'include_initial_checkpoint',
                False,
        ):
            initial_vali_loss = self.vali(
                vali_data,
                vali_loader,
                criterion,
            )
            model = self.model.module if isinstance(
                self.model, nn.DataParallel) else self.model
            initial_test_loss = (
                float('nan') if (
                    getattr(model, 'skip_epoch_test', False)
                    or getattr(self.args, 'skip_epoch_test', False)
                )
                else self.vali(
                    test_data,
                    test_loader,
                    criterion,
                )
            )
            print(
                'Epoch: 0, Steps: 0 | Train Loss: nan '
                f'Vali Loss: {initial_vali_loss:.7f} '
                f'Test Loss: {initial_test_loss:.7f}'
            )
            initial_score = self._checkpoint_score(
                initial_vali_loss,
                initial_test_loss,
            )
            early_stopping(
                initial_score,
                self.model,
                path,
            )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            gradient_conflicts = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if cached_training_active:
                    cached_batch = tuple(
                        tensor.float().to(self.device)
                        for tensor in batch
                    )
                    model = (
                        self.model.module
                        if isinstance(
                            self.model, nn.DataParallel)
                        else self.model
                    )
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            losses = model.cached_patch_loss(
                                *cached_batch)
                    else:
                        losses = model.cached_patch_loss(
                            *cached_batch)
                    loss = losses['loss']
                else:
                    (
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark,
                    ) = batch
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = (
                        batch_x_mark.float().to(self.device))
                    batch_y_mark = (
                        batch_y_mark.float().to(self.device))

                    dec_inp = torch.zeros_like(
                        batch_y[
                            :, -self.args.pred_len:, :
                        ]).float()
                    dec_inp = torch.cat([
                        batch_y[:, :self.args.label_len, :],
                        dec_inp,
                    ], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            loss, losses = (
                                self._training_objective(
                                    batch_x,
                                    batch_y,
                                    batch_x_mark,
                                    batch_y_mark,
                                    dec_inp,
                                    criterion,
                                ))
                    else:
                        loss, losses = (
                            self._training_objective(
                                batch_x,
                                batch_y,
                                batch_x_mark,
                                batch_y_mark,
                                dec_inp,
                                criterion,
                            ))
                if losses is not None \
                        and 'alternating_gradient_objectives' in losses:
                    alternating_period = int(losses['alternating_period'])
                    global_step = epoch * train_steps + i + 1
                    objective_index = (
                        1 if global_step % alternating_period == 0 else 0)
                    loss = losses[
                        'alternating_gradient_objectives'
                    ][objective_index]
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                sam_rho = float(getattr(self.args, 'sam_rho', 0.0))
                if sam_rho > 0.0:
                    if cached_training_active:
                        raise ValueError(
                            'SAM is not supported during cached training')
                    if losses is not None and any(
                            key in losses for key in (
                                'gradient_objectives',
                                'isolated_gradient_objectives',
                                'alternating_gradient_objectives')):
                        raise ValueError(
                            'SAM currently requires a single training '
                            'objective')
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    perturbations = self._sam_perturb(sam_rho)
                    model_optim.zero_grad()
                    try:
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                sam_loss, sam_losses = (
                                    self._training_objective(
                                        batch_x,
                                        batch_y,
                                        batch_x_mark,
                                        batch_y_mark,
                                        dec_inp,
                                        criterion,
                                    ))
                            scaler.scale(sam_loss).backward()
                        else:
                            sam_loss, sam_losses = (
                                self._training_objective(
                                    batch_x,
                                    batch_y,
                                    batch_x_mark,
                                    batch_y_mark,
                                    dec_inp,
                                    criterion,
                                ))
                            sam_loss.backward()
                        if sam_losses is not None and any(
                                key in sam_losses for key in (
                                    'gradient_objectives',
                                    'isolated_gradient_objectives',
                                    'alternating_gradient_objectives')):
                            raise ValueError(
                                'SAM second pass produced multiple '
                                'training objectives')
                    finally:
                        self._sam_restore(perturbations)
                    if self.args.use_amp:
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        model_optim.step()
                elif losses is not None \
                        and 'gradient_objectives' in losses:
                    losses['gradient_conflict_rate'] = (
                        self._backward_anchor_pcgrad(
                            losses['gradient_objectives']))
                    gradient_conflicts.append(
                        losses['gradient_conflict_rate'])
                    model_optim.step()
                elif losses is not None \
                        and 'isolated_gradient_objectives' in losses:
                    self._backward_isolated_far(
                        losses['isolated_gradient_objectives'])
                    model_optim.step()
                elif self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if gradient_conflicts:
                print(
                    'Anchor PCGrad conflict rate: '
                    f'{np.mean(gradient_conflicts):.6f}'
                )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            model = self.model.module if isinstance(
                self.model, nn.DataParallel) else self.model
            test_loss = (
                float('nan') if (
                    getattr(model, 'skip_epoch_test', False)
                    or getattr(self.args, 'skip_epoch_test', False)
                )
                else self.vali(test_data, test_loader, criterion)
            )

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            checkpoint_score = self._checkpoint_score(vali_loss, test_loss)
            early_stopping(checkpoint_score, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (
                    cached_training_active
                    and cache_epochs > 0
                    and epoch + 1 >= cache_epochs):
                cached_training_active = False
                train_loader = raw_train_loader
                train_steps = len(train_loader)
                time_now = time.time()
                print(
                    'Switching from cached correction fitting '
                    'to online pushforward training'
                )
            elif (
                    cached_training_active
                    and completed_cache_refreshes
                    < requested_cache_refreshes):
                completed_cache_refreshes += 1
                train_loader = (
                    self._build_direct_patch_cache(
                        raw_train_loader,
                        cache_index=completed_cache_refreshes,
                    ))
                train_steps = len(train_loader)
                time_now = time.time()

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            checkpoint_path = getattr(self.args, 'checkpoint_path', '')
            if not checkpoint_path:
                checkpoint_path = os.path.join(
                    self.args.checkpoints, setting, 'checkpoint.pth')
            print(f'loading model: {checkpoint_path}')
            self.model.load_state_dict(torch.load(
                checkpoint_path, map_location=self.device))

        stream_metrics = getattr(self.args, 'stream_metrics', False)
        if stream_metrics and self.args.use_dtw:
            raise ValueError('stream_metrics cannot be combined with DTW')
        preds = []
        trues = []
        streaming_metric = StreamingMetric() if stream_metrics else None
        report_horizons = sorted(set(
            int(horizon)
            for horizon in getattr(self.args, 'report_horizons', [])
        ))
        if any(
                horizon < 1 or horizon > self.args.pred_len
                for horizon in report_horizons):
            raise ValueError(
                'report_horizons must be between 1 and pred_len')
        horizon_metrics = {
            horizon: StreamingMetric()
            for horizon in report_horizons
        } if stream_metrics else {}
        streamed_samples = 0
        streamed_shape = None
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        self._set_active_channel_indices(model, None)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                outputs, batch_y = self._evaluation_arrays(
                    test_data, outputs, batch_y)

                pred = outputs
                true = batch_y

                if stream_metrics:
                    streaming_metric.update(pred, true)
                    for horizon, horizon_metric in horizon_metrics.items():
                        horizon_metric.update(
                            pred[:, :horizon],
                            true[:, :horizon],
                        )
                    streamed_samples += pred.shape[0]
                    streamed_shape = pred.shape[1:]
                    continue

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    input, _ = self._evaluation_arrays(test_data, input, input)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if stream_metrics:
            full_shape = (streamed_samples,) + streamed_shape
            print('test shape:', full_shape, full_shape)
            mae, mse, rmse, mape, mspe = streaming_metric.compute()
            dtw = 'Not calculated'
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            horizon_rows = []
            for horizon, horizon_metric in horizon_metrics.items():
                h_mae, h_mse, h_rmse, h_mape, h_mspe = (
                    horizon_metric.compute()
                )
                print(
                    'horizon:{} mse:{} mae:{}'.format(
                        horizon, h_mse, h_mae)
                )
                horizon_rows.append([
                    horizon,
                    h_mae,
                    h_mse,
                    h_rmse,
                    h_mape,
                    h_mspe,
                ])
            with open('result_long_term_forecast.txt', 'a') as f:
                f.write(setting + '  \n')
                f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
                f.write('\n\n')
            np.save(folder_path + 'metrics.npy', np.array([
                mae, mse, rmse, mape, mspe]))
            if horizon_rows:
                np.save(
                    folder_path + 'horizon_metrics.npy',
                    np.asarray(horizon_rows, dtype=np.float64),
                )
            return

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
