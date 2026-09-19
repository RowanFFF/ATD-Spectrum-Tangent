import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PatchAR_EncDec import RotaryEmbedding


class RMSNorm(nn.Module):
    """Minimal RMSNorm for matched LayerNorm/RMSNorm block comparisons."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, inputs):
        normalized = inputs * torch.rsqrt(
            inputs.float().square().mean(
                dim=-1, keepdim=True) + self.eps
        ).to(inputs.dtype)
        return normalized * self.weight


class CausalDepthwiseMixer(nn.Module):
    """Small zero-initialized local residual over the token axis."""

    def __init__(self, d_model, kernel_size, dropout, bias=True):
        super().__init__()
        if kernel_size < 2:
            raise ValueError('local kernel size must be at least two')
        self.kernel_size = kernel_size
        self.convolution = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            groups=d_model,
            bias=bias,
        )
        nn.init.zeros_(self.convolution.weight)
        if self.convolution.bias is not None:
            nn.init.zeros_(self.convolution.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        values = inputs.transpose(1, 2)
        values = F.pad(values, (self.kernel_size - 1, 0))
        return self.dropout(
            self.convolution(values).transpose(1, 2))

    def forward_last(self, inputs):
        """Return only the final causal convolution output."""
        values = inputs[:, -self.kernel_size:].transpose(1, 2)
        if values.size(-1) < self.kernel_size:
            values = F.pad(
                values,
                (self.kernel_size - values.size(-1), 0),
            )
        return self.dropout(
            self.convolution(values).transpose(1, 2))


class SwiGLUFeedForward(nn.Module):
    """Parameter-matched gated FFN used in the architecture experiments."""

    def __init__(self, d_model, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(d_model, 2 * hidden_size)
        self.output_projection = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        values, gates = self.input_projection(inputs).chunk(2, dim=-1)
        hidden = values * F.silu(gates)
        hidden = self.dropout(hidden)
        return self.dropout(self.output_projection(hidden))


class AblationSelfAttention(nn.Module):
    """Causal token mixer with explicit attention-mechanism ablations.

    The modes isolate three questions:

    * ``softmax`` / ``sigmoid_norm`` / ``relu_norm`` / ``raw`` retain
      input-dependent QK scores and only change score normalization.
    * ``uniform`` and ``static`` remove content-dependent QK selection while
      retaining causal temporal mixing.
    * ``identity`` and ``none`` remove cross-token mixing.

    ``raw`` uses signed scaled dot-product scores divided by the number of
    visible keys.  This keeps its magnitude comparable across causal prefix
    lengths without introducing a row-normalizing nonlinearity.
    """

    MODES = {
        'softmax',
        'shrinkage',
        'lag_bias',
        'sigmoid_norm',
        'relu_norm',
        'raw',
        'uniform',
        'static',
        'identity',
        'none',
    }
    CONTENT_MODES = {
        'softmax',
        'shrinkage',
        'lag_bias',
        'sigmoid_norm',
        'relu_norm',
        'raw',
    }
    SHRINKAGE_TYPES = {
        'fixed',
        'hybrid',
        'learned_head',
        'learned_query',
    }
    VALUE_TOPOLOGIES = {
        'state',
        'past_state',
        'past_anchor',
        'successor',
        'successor_delta',
    }
    SELF_HISTORY_MODES = {
        'joint',
        'fixed',
        'learned_scalar',
        'learned_head',
        'learned_query',
        'evidence_sum',
        'evidence_count025',
        'evidence_count05',
        'evidence_count075',
        'evidence_mean',
        'evidence_max',
    }
    QK_INPUT_MODES = {
        'state',
        'delta',
        'detrended',
        'query_delta',
        'key_delta',
        'query_detrended',
        'key_detrended',
    }
    HEAD_SHARING_MODES = {
        'none',
        'query',
        'key',
        'value',
        'query_key',
        'key_value',
    }

    def __init__(self, d_model, n_heads, dropout, mode='softmax',
                 position_encoding='rope', max_tokens=None,
                 shrinkage_type='fixed', shrinkage_alpha=1.0,
                 dynamic_heads=None, qk_norm='dot',
                 attention_temperature=1.0,
                 attention_softcap=0.0,
                 learnable_temperature=False,
                 value_topology='state',
                 self_logit_bias=0.0,
                 self_edge_heads=-1,
                 self_history_mode='joint',
                 self_history_init=0.5,
                 qk_input_mode='state',
                 head_sharing='none',
                 attention_window=0,
                 coarse_memory_group=0,
                 coarse_memory_gate_init=0.0,
                 address_geometry='single',
                 geometry_gate_init=0.0,
                 dyadic_router_mode='none',
                 dyadic_router_gate_init=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if mode not in self.MODES:
            raise ValueError(
                f'attention mode must be one of {sorted(self.MODES)}')
        if position_encoding not in {'rope', 'none'}:
            raise ValueError('position_encoding must be rope or none')
        if mode in {'static', 'lag_bias'} and (
                max_tokens is None or max_tokens < 1):
            raise ValueError(
                f'{mode} attention requires a positive max_tokens')
        if shrinkage_type not in self.SHRINKAGE_TYPES:
            raise ValueError(
                'shrinkage_type must be one of '
                f'{sorted(self.SHRINKAGE_TYPES)}')
        if not 0.0 <= shrinkage_alpha <= 1.0:
            raise ValueError('shrinkage_alpha must be in [0, 1]')
        if dynamic_heads is None:
            dynamic_heads = n_heads
        if not 0 <= dynamic_heads <= n_heads:
            raise ValueError('dynamic_heads must be between 0 and n_heads')
        if qk_norm not in {'dot', 'cosine'}:
            raise ValueError('qk_norm must be dot or cosine')
        if attention_temperature <= 0:
            raise ValueError('attention_temperature must be positive')
        if attention_softcap < 0:
            raise ValueError('attention_softcap cannot be negative')
        if value_topology not in self.VALUE_TOPOLOGIES:
            raise ValueError(
                'value_topology must be one of '
                f'{sorted(self.VALUE_TOPOLOGIES)}')
        if value_topology != 'state' and mode not in self.CONTENT_MODES:
            raise ValueError(
                'non-state value topology requires content attention')
        if self_edge_heads < -1 or self_edge_heads > n_heads:
            raise ValueError(
                'self_edge_heads must be -1 or between 0 and n_heads')
        if self_edge_heads >= 0 and (
                value_topology != 'state'
                or mode not in self.CONTENT_MODES):
            raise ValueError(
                'headwise self edges require state-value content attention')
        if self_history_mode not in self.SELF_HISTORY_MODES:
            raise ValueError(
                'self_history_mode must be one of '
                f'{sorted(self.SELF_HISTORY_MODES)}')
        if not 0.0 <= self_history_init <= 1.0:
            raise ValueError('self_history_init must be in [0, 1]')
        if self_history_mode != 'joint' and (
                mode != 'softmax'
                or value_topology != 'state'
                or self_edge_heads != -1
                or self_logit_bias != 0.0):
            raise ValueError(
                'factorized self/history routing requires ordinary '
                'state-value softmax attention')
        if qk_input_mode not in self.QK_INPUT_MODES:
            raise ValueError(
                'qk_input_mode must be one of '
                f'{sorted(self.QK_INPUT_MODES)}')
        if qk_input_mode != 'state' and mode not in self.CONTENT_MODES:
            raise ValueError(
                'non-state QK inputs require content attention')
        if head_sharing not in self.HEAD_SHARING_MODES:
            raise ValueError(
                'head_sharing must be one of '
                f'{sorted(self.HEAD_SHARING_MODES)}')
        if head_sharing != 'none' and mode not in self.CONTENT_MODES:
            raise ValueError(
                'head sharing requires content attention')
        if attention_window < 0:
            raise ValueError('attention_window cannot be negative')
        if coarse_memory_group not in {0, 2, 4, 8}:
            raise ValueError(
                'coarse_memory_group must be zero, two, four, or eight')
        if not -1.0 < coarse_memory_gate_init < 1.0:
            raise ValueError(
                'coarse_memory_gate_init must be strictly between -1 and 1')
        if address_geometry not in {
                'single', 'level_change', 'query_change',
                'query_change_shared', 'query_change_fixed',
                'query_change_eval_shared',
                'query_change_distribution',
                'query_change_independent',
                'query_change_distribution_independent',
                'query_change_dynamic', 'query_multiscale',
                'query_local_change', 'query_short_multiscale',
                'query_reference_mix'}:
            raise ValueError(
                'address_geometry must be single, level_change, '
                'query_change, query_change_shared, query_change_fixed, '
                'query_change_eval_shared, '
                'query_change_distribution, '
                'query_change_independent, '
                'query_change_distribution_independent, '
                'query_change_dynamic, '
                'query_multiscale, query_local_change, '
                'query_short_multiscale, or query_reference_mix')
        if not -1.0 < geometry_gate_init < 1.0:
            raise ValueError(
                'geometry_gate_init must be strictly between -1 and 1')
        if dyadic_router_mode not in {
                'none', 'count', 'innovation',
                'soft_count', 'soft_innovation'}:
            raise ValueError(
                'dyadic router mode must be none, count, innovation, '
                'soft_count, or soft_innovation')
        if not -1.0 < dyadic_router_gate_init < 1.0:
            raise ValueError(
                'dyadic router gate init must be strictly between -1 and 1')
        if coarse_memory_group and (
                mode != 'softmax'
                or value_topology != 'state'
                or self_history_mode != 'joint'
                or self_edge_heads != -1
                or self_logit_bias != 0.0):
            raise ValueError(
                'coarse memory requires ordinary state-value softmax')
        if address_geometry != 'single' and (
                mode not in {'softmax', 'shrinkage'}
                or qk_input_mode != 'state'
                or head_sharing not in {'none', 'query'}
                or self_history_mode != 'joint'):
            raise ValueError(
                'level-change geometry requires ordinary state-input '
                'softmax/shrinkage attention with at most query sharing')
        if dyadic_router_mode != 'none' and (
                mode != 'softmax'
                or value_topology != 'state'
                or qk_input_mode != 'state'
                or head_sharing != 'none'
                or self_history_mode != 'joint'
                or self_edge_heads != -1
                or self_logit_bias != 0.0
                or address_geometry != 'single'):
            raise ValueError(
                'dyadic routing requires ordinary state-input/state-value '
                'softmax without other attention topology changes')

        self.mode = mode
        self.position_encoding = position_encoding
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = float(dropout)
        self.shrinkage_type = shrinkage_type
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.dynamic_heads = int(dynamic_heads)
        self.qk_norm = qk_norm
        self.attention_softcap = float(attention_softcap)
        self.value_topology = value_topology
        self.self_logit_bias = float(self_logit_bias)
        self.self_edge_heads = int(self_edge_heads)
        self.self_history_mode = self_history_mode
        self.self_history_init = float(self_history_init)
        self.qk_input_mode = qk_input_mode
        self.head_sharing = head_sharing
        self.attention_window = int(attention_window)
        self.coarse_memory_group = int(coarse_memory_group)
        self.address_geometry = address_geometry
        self.dyadic_router_mode = dyadic_router_mode

        if mode in self.CONTENT_MODES:
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.value_projection = None
        elif mode in {'uniform', 'static', 'identity'}:
            self.qkv = None
            self.value_projection = nn.Linear(d_model, d_model)
        else:
            self.qkv = None
            self.value_projection = None

        if mode == 'static':
            self.static_logits = nn.Parameter(torch.zeros(
                n_heads, max_tokens, max_tokens))
        else:
            self.register_parameter('static_logits', None)

        if mode == 'lag_bias':
            # A Toeplitz causal prior: index 0 is the current token, index k
            # is a key k patches in the past.  Zero initialization makes this
            # mode exactly ordinary softmax attention at initialization.
            self.lag_bias = nn.Parameter(torch.zeros(n_heads, max_tokens))
        else:
            self.register_parameter('lag_bias', None)

        if mode == 'shrinkage' and shrinkage_type == 'learned_head':
            probability = min(max(shrinkage_alpha, 1e-4), 1.0 - 1e-4)
            initial_logit = math.log(probability / (1.0 - probability))
            self.shrinkage_logits = nn.Parameter(torch.full(
                (n_heads,), initial_logit))
        else:
            self.register_parameter('shrinkage_logits', None)

        if mode == 'shrinkage' and shrinkage_type == 'learned_query':
            probability = min(max(shrinkage_alpha, 1e-4), 1.0 - 1e-4)
            initial_logit = math.log(probability / (1.0 - probability))
            self.query_gate_weight = nn.Parameter(torch.zeros(
                n_heads, self.head_dim))
            self.query_gate_bias = nn.Parameter(torch.full(
                (n_heads,), initial_logit))
        else:
            self.register_parameter('query_gate_weight', None)
            self.register_parameter('query_gate_bias', None)

        temperature = torch.tensor(
            math.log(float(attention_temperature)))
        if mode in self.CONTENT_MODES and learnable_temperature:
            self.log_temperature = nn.Parameter(temperature)
        else:
            self.register_buffer(
                'log_temperature',
                temperature,
                persistent=mode in self.CONTENT_MODES,
            )

        if mode == 'none':
            self.out_projection = None
            self.output_dropout = nn.Identity()
        else:
            self.out_projection = nn.Linear(d_model, d_model)
            self.output_dropout = nn.Dropout(dropout)

        if self.coarse_memory_group:
            self.coarse_memory_gate = nn.Parameter(torch.full(
                (n_heads,),
                math.atanh(float(coarse_memory_gate_init)),
            ))
        else:
            self.register_parameter('coarse_memory_gate', None)
        if self.address_geometry == 'query_change_fixed':
            self.register_parameter('geometry_gate', None)
            self.register_buffer(
                'fixed_geometry_gate',
                torch.tensor(math.atanh(float(
                    geometry_gate_init))),
                persistent=False,
            )
        elif self.address_geometry in {
                'level_change', 'query_change', 'query_change_shared',
                'query_change_eval_shared',
                'query_change_distribution',
                'query_change_independent',
                'query_change_distribution_independent',
                'query_change_dynamic', 'query_multiscale',
                'query_local_change', 'query_short_multiscale',
                'query_reference_mix'}:
            geometry_count = {
                'level_change': 2,
                'query_change': 1,
                'query_change_shared': 1,
                'query_change_eval_shared': 1,
                'query_change_distribution': 1,
                'query_change_independent': 1,
                'query_change_distribution_independent': 1,
                'query_change_dynamic': 1,
                'query_multiscale': 3,
                'query_local_change': 1,
                'query_short_multiscale': 2,
                'query_reference_mix': 1,
            }[self.address_geometry]
            geometry_heads = (
                1
                if self.address_geometry == 'query_change_shared'
                else n_heads
            )
            initial_gates = torch.zeros(
                geometry_heads, geometry_count)
            if self.address_geometry in {
                    'query_multiscale', 'query_short_multiscale'}:
                initial_gates[:, -1] = math.atanh(
                    float(geometry_gate_init))
            else:
                initial_gates.fill_(
                    math.atanh(float(geometry_gate_init)))
            self.geometry_gate = nn.Parameter(initial_gates)
        else:
            self.register_parameter('geometry_gate', None)
        if self.address_geometry != 'query_change_fixed':
            self.register_buffer(
                'fixed_geometry_gate',
                None,
                persistent=False,
            )
        if self.address_geometry == 'query_reference_mix':
            self.geometry_reference_logits = nn.Parameter(
                torch.zeros(3))
        else:
            self.register_parameter(
                'geometry_reference_logits', None)
        if self.address_geometry in {
                'query_change_independent',
                'query_change_distribution_independent'}:
            self.innovation_query_projection = nn.Linear(
                d_model, d_model)
            with torch.no_grad():
                query_weight = self.qkv.weight[:d_model]
                self.innovation_query_projection.weight.copy_(
                    query_weight)
                if self.qkv.bias is not None:
                    self.innovation_query_projection.bias.copy_(
                        self.qkv.bias[:d_model])
        else:
            self.innovation_query_projection = None
        if self.address_geometry == 'query_change_dynamic':
            self.geometry_gate_slope = nn.Parameter(
                torch.zeros(n_heads))
        else:
            self.register_parameter('geometry_gate_slope', None)

        max_dyadic_groups = (
            (int(max_tokens) - 1).bit_length() + 1
            if max_tokens is not None else 1
        )
        if self.dyadic_router_mode != 'none':
            router_value = math.atanh(float(
                dyadic_router_gate_init))
            self.dyadic_count_gate = nn.Parameter(torch.full(
                (n_heads, max_dyadic_groups),
                router_value,
            ))
        else:
            self.register_parameter('dyadic_count_gate', None)
        if self.dyadic_router_mode in {
                'innovation', 'soft_innovation'}:
            self.dyadic_innovation_gate = nn.Parameter(torch.full(
                (n_heads, max_dyadic_groups),
                math.atanh(float(dyadic_router_gate_init)),
            ))
        else:
            self.register_parameter('dyadic_innovation_gate', None)

        gate_probability = min(
            max(float(self_history_init), 1e-4),
            1.0 - 1e-4,
        )
        gate_logit = math.log(
            gate_probability / (1.0 - gate_probability))
        if self_history_mode == 'learned_scalar':
            self.self_history_logit = nn.Parameter(torch.tensor(gate_logit))
        elif self_history_mode == 'learned_head':
            self.self_history_logit = nn.Parameter(torch.full(
                (n_heads,), gate_logit))
        else:
            self.register_parameter('self_history_logit', None)
        if self_history_mode == 'learned_query':
            self.self_history_gate_weight = nn.Parameter(torch.zeros(
                n_heads, self.head_dim))
            self.self_history_gate_bias = nn.Parameter(torch.full(
                (n_heads,), gate_logit))
        else:
            self.register_parameter('self_history_gate_weight', None)
            self.register_parameter('self_history_gate_bias', None)

        self.rope = (
            RotaryEmbedding(self.head_dim)
            if position_encoding == 'rope' and mode in self.CONTENT_MODES
            else None
        )

    @staticmethod
    def _causal_mask(token_count, device):
        return torch.ones(
            token_count, token_count, device=device, dtype=torch.bool).tril()

    def _project_values(self, inputs):
        batch_size, token_count, d_model = inputs.shape
        projected = self.value_projection(inputs)
        return projected.view(
            batch_size, token_count, self.n_heads, self.head_dim
        ).transpose(1, 2)

    @staticmethod
    def _delta_inputs(inputs):
        differences = torch.empty_like(inputs)
        differences[:, :1] = inputs[:, :1]
        differences[:, 1:] = (
            inputs[:, 1:] - inputs[:, :-1])
        return differences

    @staticmethod
    def _detrended_inputs(inputs):
        previous_sum = inputs.cumsum(
            dim=1) - inputs
        previous_count = torch.arange(
            inputs.size(1),
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, -1, 1)
        previous_mean = previous_sum / previous_count.clamp_min(1.0)
        return torch.where(
            previous_count > 0,
            inputs - previous_mean,
            inputs,
        )

    @staticmethod
    def _local_detrended_inputs(inputs, window):
        token_count = inputs.size(1)
        prefix_sum = torch.cat([
            torch.zeros_like(inputs[:, :1]),
            inputs.cumsum(dim=1),
        ], dim=1)
        ends = torch.arange(
            token_count, device=inputs.device)
        starts = (ends - window).clamp_min(0)
        previous_sum = (
            prefix_sum[:, ends] - prefix_sum[:, starts])
        previous_count = (ends - starts).to(
            inputs.dtype).view(1, token_count, 1)
        previous_mean = previous_sum / previous_count.clamp_min(1.0)
        return torch.where(
            previous_count > 0,
            inputs - previous_mean,
            inputs,
        )

    def _qk_inputs(self, inputs):
        if self.qk_input_mode == 'state':
            return inputs, inputs
        if 'delta' in self.qk_input_mode:
            transformed = self._delta_inputs(inputs)
        elif 'detrended' in self.qk_input_mode:
            transformed = self._detrended_inputs(inputs)
        else:
            raise RuntimeError(
                f'unsupported QK input mode: {self.qk_input_mode}')
        if self.qk_input_mode.startswith('query_'):
            return transformed, inputs
        if self.qk_input_mode.startswith('key_'):
            return inputs, transformed
        return transformed, transformed

    def _share_heads(self, queries, keys, values):
        if self.head_sharing in {'query', 'query_key'}:
            queries = queries.mean(
                dim=1, keepdim=True).expand_as(queries)
        if self.head_sharing in {
                'key', 'query_key', 'key_value'}:
            keys = keys.mean(
                dim=1, keepdim=True).expand_as(keys)
        if self.head_sharing in {'value', 'key_value'}:
            values = values.mean(
                dim=1, keepdim=True).expand_as(values)
        return queries, keys, values

    def _content_qkv(
            self, inputs, positions, address_inputs=None,
            value_inputs=None):
        batch_size, token_count, d_model = inputs.shape
        if address_inputs is not None or value_inputs is not None:
            if self.qk_input_mode != 'state':
                raise ValueError(
                    'external role inputs require state QK mode')
            if (
                    address_inputs is not None
                    and address_inputs.shape != inputs.shape):
                raise ValueError(
                    'external address inputs must match token shape')
            if (
                    value_inputs is not None
                    and value_inputs.shape != inputs.shape):
                raise ValueError(
                    'external value inputs must match token shape')
            qk_inputs = (
                inputs
                if address_inputs is None
                else address_inputs
            )
            value_inputs = (
                inputs
                if value_inputs is None
                else value_inputs
            )
            weight = self.qkv.weight.view(3, d_model, d_model)
            bias = (
                self.qkv.bias.view(3, d_model)
                if self.qkv.bias is not None else (None, None, None)
            )
            queries = F.linear(
                qk_inputs, weight[0],
                None if self.qkv.bias is None else bias[0])
            keys = F.linear(
                qk_inputs, weight[1],
                None if self.qkv.bias is None else bias[1])
            values = F.linear(
                value_inputs, weight[2],
                None if self.qkv.bias is None else bias[2])
            queries = queries.view(
                batch_size, token_count,
                self.n_heads, self.head_dim,
            ).transpose(1, 2)
            keys = keys.view(
                batch_size, token_count,
                self.n_heads, self.head_dim,
            ).transpose(1, 2)
            values = values.view(
                batch_size, token_count,
                self.n_heads, self.head_dim,
            ).transpose(1, 2)
        elif self.qk_input_mode == 'state':
            qkv = self.qkv(inputs).view(
                batch_size,
                token_count,
                3,
                self.n_heads,
                self.head_dim,
            )
            queries, keys, values = qkv.permute(
                2, 0, 3, 1, 4).unbind(0)
        else:
            query_inputs, key_inputs = self._qk_inputs(inputs)
            weight = self.qkv.weight.view(3, d_model, d_model)
            bias = (
                self.qkv.bias.view(3, d_model)
                if self.qkv.bias is not None else (None, None, None)
            )
            queries = F.linear(
                query_inputs, weight[0],
                None if self.qkv.bias is None else bias[0])
            keys = F.linear(
                key_inputs, weight[1],
                None if self.qkv.bias is None else bias[1])
            values = F.linear(
                inputs, weight[2],
                None if self.qkv.bias is None else bias[2])
            queries = queries.view(
                batch_size, token_count,
                self.n_heads, self.head_dim,
            ).transpose(1, 2)
            keys = keys.view(
                batch_size, token_count,
                self.n_heads, self.head_dim,
            ).transpose(1, 2)
            values = values.view(
                batch_size, token_count,
                self.n_heads, self.head_dim,
            ).transpose(1, 2)
        if self.rope is not None:
            queries, keys = self.rope(queries, keys, positions)
        return self._share_heads(queries, keys, values)

    def _project_content_role(self, inputs, role):
        batch_size, token_count, d_model = inputs.shape
        weight = self.qkv.weight.view(3, d_model, d_model)[role]
        bias = (
            None
            if self.qkv.bias is None
            else self.qkv.bias.view(3, d_model)[role]
        )
        return F.linear(inputs, weight, bias).view(
            batch_size,
            token_count,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _project_innovation_query(self, inputs):
        if self.innovation_query_projection is None:
            return self._project_content_role(inputs, 0)
        batch_size, token_count, _ = inputs.shape
        return self.innovation_query_projection(inputs).view(
            batch_size,
            token_count,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _geometry_qkv(
            self, inputs, positions, address_inputs=None,
            value_inputs=None):
        queries, keys, values = self._content_qkv(
            inputs,
            positions,
            address_inputs=address_inputs,
            value_inputs=value_inputs,
        )
        if self.address_geometry == 'single':
            return queries, keys, values, None

        state_scores = self._content_scores(queries, keys)

        if self.address_geometry in {
                'query_multiscale', 'query_short_multiscale',
                'query_reference_mix'}:
            innovations = [
                self._delta_inputs(inputs),
                self._local_detrended_inputs(inputs, 4),
            ]
            if self.address_geometry in {
                    'query_multiscale', 'query_reference_mix'}:
                innovations.append(
                    self._detrended_inputs(inputs))
            innovation_scores = []
            for innovation in innovations:
                innovation_queries = self._project_content_role(
                    innovation, 0)
                if self.rope is not None:
                    innovation_queries, _ = self.rope(
                        innovation_queries,
                        innovation_queries,
                        positions,
                    )
                innovation_scores.append(self._content_scores(
                    innovation_queries, keys))
            score_residuals = torch.stack([
                score - state_scores
                for score in innovation_scores
            ], dim=-1)
            if self.address_geometry == 'query_reference_mix':
                reference_weights = torch.softmax(
                    self.geometry_reference_logits,
                    dim=0,
                ).to(state_scores.dtype).view(
                    1, 1, 1, 1, len(innovations))
                mixed_residual = (
                    reference_weights * score_residuals).sum(dim=-1)
                gate = torch.tanh(self.geometry_gate).to(
                    state_scores.dtype).view(
                        1, self.n_heads, 1, 1)
                scores = state_scores + gate * mixed_residual
            else:
                gates = torch.tanh(self.geometry_gate).to(
                    state_scores.dtype).view(
                        1, self.n_heads, 1, 1, len(innovations))
                scores = state_scores + (
                    gates * score_residuals).sum(dim=-1)
            return queries, keys, values, scores

        changes = (
            self._local_detrended_inputs(inputs, 4)
            if self.address_geometry == 'query_local_change'
            else self._detrended_inputs(inputs)
        )
        change_queries = self._project_innovation_query(
            changes)
        if self.head_sharing == 'query':
            change_queries = change_queries.mean(
                dim=1, keepdim=True).expand_as(change_queries)
        if self.rope is not None:
            change_queries, _ = self.rope(
                change_queries,
                change_queries,
                positions,
            )
        change_query_scores = self._content_scores(
            change_queries, keys)
        if self.address_geometry in {
                'query_change', 'query_change_shared',
                'query_change_fixed',
                'query_change_eval_shared',
                'query_change_independent',
                'query_change_dynamic',
                'query_local_change'}:
            raw_gate = (
                self.fixed_geometry_gate.view(1, 1, 1, 1)
                if self.address_geometry == 'query_change_fixed'
                else self.geometry_gate.view(
                    1, self.geometry_gate.size(0), 1, 1)
            )
            if self.address_geometry == 'query_change_dynamic':
                state_rms = inputs.float().square().mean(
                    dim=-1).sqrt().clamp_min(1e-6)
                change_rms = changes.float().square().mean(
                    dim=-1).sqrt().clamp_min(1e-6)
                feature = (change_rms / state_rms).log().clamp(
                    min=-4.0, max=4.0)
                raw_gate = (
                    raw_gate
                    + self.geometry_gate_slope.view(
                        1, self.n_heads, 1, 1)
                    * feature[:, None, :, None]
                )
            gate = torch.tanh(raw_gate)
            if (
                    self.address_geometry
                    == 'query_change_eval_shared'
                    and not self.training):
                gate = gate.mean(dim=1, keepdim=True)
            gate = gate.to(state_scores.dtype)
            scores = state_scores + gate * (
                change_query_scores - state_scores)
            return queries, keys, values, scores

        change_keys = self._project_content_role(changes, 1)
        if self.rope is not None:
            _, change_keys = self.rope(
                change_keys,
                change_keys,
                positions,
            )
        change_key_scores = self._content_scores(
            queries, change_keys)
        gates = torch.tanh(self.geometry_gate).to(
            state_scores.dtype).view(1, self.n_heads, 1, 1, 2)
        scores = (
            state_scores
            + gates[..., 0] * (
                change_query_scores - state_scores)
            + gates[..., 1] * (
                change_key_scores - state_scores)
        )
        return queries, keys, values, scores

    @staticmethod
    def _normalize_positive(weights, mask):
        weights = weights * mask.to(weights.dtype)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights / denominator

    def _shrinkage_gate(self, queries):
        if self.shrinkage_type == 'fixed':
            return queries.new_tensor(self.shrinkage_alpha)
        if self.shrinkage_type == 'hybrid':
            gate = torch.zeros(
                self.n_heads,
                device=queries.device,
                dtype=queries.dtype,
            )
            gate[:self.dynamic_heads] = 1.0
            return gate.view(1, self.n_heads, 1, 1)
        if self.shrinkage_type == 'learned_head':
            return torch.sigmoid(self.shrinkage_logits).to(
                queries.dtype).view(1, self.n_heads, 1, 1)
        if self.shrinkage_type == 'learned_query':
            logits = torch.einsum(
                'bhtd,hd->bht',
                queries,
                self.query_gate_weight.to(queries.dtype),
            ) / math.sqrt(self.head_dim)
            logits = logits + self.query_gate_bias.to(
                queries.dtype).view(1, self.n_heads, 1)
            return torch.sigmoid(logits).unsqueeze(-1)
        raise RuntimeError(
            f'unsupported shrinkage type: {self.shrinkage_type}')

    def _lag_bias_matrix(self, query_count, key_count, device):
        if max(query_count, key_count) > self.lag_bias.size(-1):
            raise ValueError(
                'token count exceeds configured lag-bias attention size')
        query_indices = torch.arange(
            key_count - query_count,
            key_count,
            device=device,
        )
        key_indices = torch.arange(key_count, device=device)
        lags = (query_indices[:, None] - key_indices[None, :]).clamp_min(0)
        return self.lag_bias[:, lags]

    def _content_weights(
            self, queries, keys, causal_mask, scores=None):
        if scores is None:
            scores = self._content_scores(queries, keys)
        if self.self_logit_bias != 0.0:
            diagonal = torch.eye(
                scores.size(-2),
                scores.size(-1),
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = scores + self.self_logit_bias * diagonal
        if causal_mask.ndim == 2:
            mask = causal_mask.view(
                1, 1, causal_mask.size(0), causal_mask.size(1))
        elif causal_mask.ndim == 3:
            mask = causal_mask.unsqueeze(0)
        else:
            raise ValueError(
                'causal attention mask must have two or three dimensions')

        if self.mode in {'softmax', 'shrinkage', 'lag_bias'}:
            if self.mode == 'lag_bias':
                scores = scores + self._lag_bias_matrix(
                    scores.size(-2),
                    scores.size(-1),
                    scores.device,
                ).unsqueeze(0).to(scores.dtype)
            has_memory = mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~mask, float('-inf'))
            scores = torch.where(
                has_memory, scores, torch.zeros_like(scores))
            content_weights = torch.softmax(scores, dim=-1)
            content_weights = content_weights * mask.to(
                content_weights.dtype)
            if self.mode != 'shrinkage':
                return content_weights
            uniform_weights = mask.to(scores.dtype)
            uniform_weights = uniform_weights / uniform_weights.sum(
                dim=-1, keepdim=True).clamp_min(1.0)
            gate = self._shrinkage_gate(queries)
            return uniform_weights + gate * (
                content_weights - uniform_weights)
        if self.mode == 'sigmoid_norm':
            return self._normalize_positive(torch.sigmoid(scores), mask)
        if self.mode == 'relu_norm':
            return self._normalize_positive(F.relu(scores), mask)
        if self.mode == 'raw':
            visible_count = mask.sum(
                dim=-1, keepdim=True).to(scores.dtype).clamp_min(1.0)
            return scores.masked_fill(~mask, 0.0) / visible_count
        raise RuntimeError(f'unsupported content attention mode: {self.mode}')

    @staticmethod
    def _dyadic_group_ids(query_count, key_count, device):
        query_indices = torch.arange(
            key_count - query_count,
            key_count,
            device=device,
        ).view(-1, 1)
        key_indices = torch.arange(
            key_count,
            device=device,
        ).view(1, -1)
        lags = (query_indices - key_indices).clamp_min(0)
        groups = torch.zeros_like(lags)
        positive = lags > 0
        groups[positive] = (
            torch.floor(torch.log2(
                lags[positive].float())).long() + 1
        )
        return groups

    def _dyadic_routed_weights(
            self, state_scores, change_scores, causal_mask):
        if causal_mask.ndim != 2:
            raise ValueError(
                'dyadic routing requires a shared two-dimensional mask')
        mask = causal_mask.view(
            1, 1, causal_mask.size(0), causal_mask.size(1))
        group_ids = self._dyadic_group_ids(
            state_scores.size(-2),
            state_scores.size(-1),
            state_scores.device,
        )
        group_count = int(group_ids.max().item()) + 1
        if group_count > self.dyadic_count_gate.size(-1):
            raise ValueError(
                'dyadic group count exceeds configured maximum')
        if self.dyadic_router_mode.startswith('soft_'):
            coordinates = torch.log2(
                (
                    torch.arange(
                        state_scores.size(-2),
                        device=state_scores.device,
                    ).view(-1, 1)
                    - torch.arange(
                        state_scores.size(-1),
                        device=state_scores.device,
                    ).view(1, -1)
                ).clamp_min(0).to(state_scores.dtype) + 1.0
            )
            centers = torch.arange(
                group_count,
                device=state_scores.device,
                dtype=state_scores.dtype,
            )
            membership = torch.softmax(
                -2.0 * (
                    coordinates.unsqueeze(-1)
                    - centers.view(1, 1, -1)
                ).square(),
                dim=-1,
            ).view(
                1, 1,
                group_ids.size(0),
                group_ids.size(1),
                group_count,
            )
            group_mask = mask.unsqueeze(-1).expand_as(
                membership)
        else:
            membership = F.one_hot(
                group_ids,
                num_classes=group_count,
            ).to(state_scores.dtype).view(
                1, 1,
                group_ids.size(0),
                group_ids.size(1),
                group_count,
            )
            group_mask = mask.unsqueeze(-1) & membership.bool()
        weighted_membership = (
            membership
            * group_mask.to(membership.dtype)
        )
        visible = weighted_membership.sum(dim=-2)
        has_group = visible > 0
        component_bias = membership.clamp_min(
            torch.finfo(membership.dtype).tiny).log()
        masked_state = (
            state_scores.unsqueeze(-1) + component_bias
        ).masked_fill(
            ~group_mask, float('-inf'))
        state_lse = torch.logsumexp(
            masked_state, dim=-2)
        state_lse = torch.where(
            has_group,
            state_lse,
            torch.zeros_like(state_lse),
        )
        state_within = torch.softmax(
            torch.where(
                has_group.unsqueeze(-2),
                masked_state,
                torch.zeros_like(masked_state),
            ),
            dim=-2,
        )
        state_within = state_within * group_mask.to(
            state_within.dtype)

        count_gate = torch.tanh(
            self.dyadic_count_gate[:, :group_count]
        ).to(state_scores.dtype).view(
            1, self.n_heads, 1, group_count)
        router_logits = (
            state_lse
            - count_gate * visible.clamp_min(1.0).log()
        )

        if change_scores is not None:
            masked_change = (
                change_scores.unsqueeze(-1) + component_bias
            ).masked_fill(
                ~group_mask, float('-inf'))
            change_lse = torch.logsumexp(
                masked_change, dim=-2)
            change_lse = torch.where(
                has_group,
                change_lse,
                torch.zeros_like(change_lse),
            )
            innovation_gate = torch.tanh(
                self.dyadic_innovation_gate[:, :group_count]
            ).to(state_scores.dtype).view(
                1, self.n_heads, 1, group_count)
            router_logits = router_logits + innovation_gate * (
                change_lse - state_lse)

        router_logits = router_logits.masked_fill(
            ~has_group, float('-inf'))
        router_weights = torch.softmax(
            router_logits, dim=-1)
        return (
            state_within
            * router_weights.unsqueeze(-2)
        ).sum(dim=-1)

    def _coarse_memory_output(
            self, queries, keys, values, token_mask=None):
        group_size = self.coarse_memory_group
        complete_groups = keys.size(-2) // group_size
        if complete_groups == 0:
            return torch.zeros_like(queries)

        used_tokens = complete_groups * group_size
        grouped_keys = keys[..., :used_tokens, :].reshape(
            keys.size(0),
            self.n_heads,
            complete_groups,
            group_size,
            self.head_dim,
        )
        grouped_values = values[..., :used_tokens, :].reshape(
            values.size(0),
            self.n_heads,
            complete_groups,
            group_size,
            self.head_dim,
        )
        group_valid = None
        if token_mask is None:
            coarse_keys = grouped_keys.mean(dim=-2)
            coarse_values = grouped_values.mean(dim=-2)
        else:
            valid = token_mask[:, :used_tokens].reshape(
                token_mask.size(0),
                complete_groups,
                group_size,
            )
            weights = valid[:, None, :, :, None].to(keys.dtype)
            denominator = weights.sum(dim=-2).clamp_min(1.0)
            coarse_keys = (
                grouped_keys * weights).sum(dim=-2) / denominator
            coarse_values = (
                grouped_values * weights).sum(dim=-2) / denominator
            group_valid = valid.any(dim=-1)

        group_ends = (
            torch.arange(
                complete_groups,
                device=queries.device,
            ) + 1
        ) * group_size - 1
        query_indices = torch.arange(
            queries.size(-2),
            device=queries.device,
        )
        mask = group_ends.view(1, -1) <= query_indices.view(-1, 1)
        mask = mask.view(
            1, 1, queries.size(-2), complete_groups)
        if group_valid is not None:
            mask = mask & group_valid[:, None, None, :]

        scores = self._content_scores(queries, coarse_keys)
        has_memory = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, float('-inf'))
        scores = torch.where(
            has_memory, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.to(weights.dtype)
        weights = F.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(weights, coarse_values)
        gate = torch.tanh(self.coarse_memory_gate).to(
            output.dtype).view(1, self.n_heads, 1, 1)
        return gate * output

    def _value_memory(self, values):
        if self.value_topology in {
                'state', 'past_state', 'past_anchor'}:
            return values
        shifted = torch.zeros_like(values)
        if self.value_topology == 'successor':
            shifted[..., :-1, :] = values[..., 1:, :]
        elif self.value_topology == 'successor_delta':
            shifted[..., :-1, :] = (
                values[..., 1:, :] - values[..., :-1, :])
        else:
            raise RuntimeError(
                f'unsupported value topology: {self.value_topology}')
        return shifted

    def _content_scores(self, queries, keys):
        if self.qk_norm == 'cosine':
            queries = F.normalize(queries.float(), dim=-1).to(queries.dtype)
            keys = F.normalize(keys.float(), dim=-1).to(keys.dtype)
            scores = torch.matmul(
                queries, keys.transpose(-2, -1))
        else:
            scores = torch.matmul(
                queries, keys.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)
        temperature = self.log_temperature.float().exp().clamp(
            min=0.05, max=20.0).to(scores.dtype)
        scores = scores * temperature
        if self.attention_softcap > 0:
            cap = scores.new_tensor(self.attention_softcap)
            scores = cap * torch.tanh(scores / cap)
        return scores

    def _self_history_gate(self, queries, keys=None, history_mask=None):
        if self.self_history_mode == 'fixed':
            return queries.new_tensor(self.self_history_init)
        if self.self_history_mode == 'learned_scalar':
            return torch.sigmoid(
                self.self_history_logit).to(queries.dtype)
        if self.self_history_mode == 'learned_head':
            return torch.sigmoid(
                self.self_history_logit
            ).to(queries.dtype).view(1, self.n_heads, 1, 1)
        if self.self_history_mode == 'learned_query':
            logits = torch.einsum(
                'bhtd,hd->bht',
                queries,
                self.self_history_gate_weight.to(queries.dtype),
            ) / math.sqrt(self.head_dim)
            logits = logits + self.self_history_gate_bias.to(
                queries.dtype).view(1, self.n_heads, 1)
            return torch.sigmoid(logits).unsqueeze(-1)
        if self.self_history_mode in {
                'evidence_sum',
                'evidence_count025',
                'evidence_count05',
                'evidence_count075',
                'evidence_mean',
                'evidence_max'}:
            if keys is None or history_mask is None:
                raise ValueError(
                    'evidence routing requires keys and a history mask')
            scores = self._content_scores(queries, keys)
            mask = history_mask.view(
                1, 1, history_mask.size(-2), history_mask.size(-1))
            masked_scores = scores.masked_fill(
                ~mask, float('-inf'))
            if self.self_history_mode == 'evidence_max':
                history_evidence = masked_scores.amax(
                    dim=-1, keepdim=True)
            else:
                history_evidence = torch.logsumexp(
                    masked_scores, dim=-1, keepdim=True)
                count_power = {
                    'evidence_sum': 0.0,
                    'evidence_count025': 0.25,
                    'evidence_count05': 0.5,
                    'evidence_count075': 0.75,
                    'evidence_mean': 1.0,
                }[self.self_history_mode]
                if count_power:
                    visible = mask.sum(
                        dim=-1, keepdim=True
                    ).to(scores.dtype).clamp_min(1.0)
                    history_evidence = (
                        history_evidence
                        - count_power * visible.log())
            self_score = torch.diagonal(
                scores, dim1=-2, dim2=-1).unsqueeze(-1)
            return torch.sigmoid(
                self_score - history_evidence)
        raise RuntimeError(
            f'unsupported factorized self/history mode: '
            f'{self.self_history_mode}')

    def forward(
            self, inputs, token_mask=None, positions=None,
            address_inputs=None, value_inputs=None,
            value_residuals=None):
        if self.mode == 'none':
            return torch.zeros_like(inputs)
        if (
                address_inputs is not None
                or value_inputs is not None) and (
                self.mode not in self.CONTENT_MODES
                or self.address_geometry != 'single'
                or self.dyadic_router_mode != 'none'
                or self.self_history_mode != 'joint'):
            raise ValueError(
                'external address inputs require ordinary joint '
                'content attention')
        if value_residuals is not None:
            if (
                    self.mode not in self.CONTENT_MODES
                    or self.value_topology != 'state'):
                raise ValueError(
                    'external value residuals require state-value '
                    'content attention')
            if value_residuals.shape != inputs.shape:
                raise ValueError(
                    'external value residuals must match token shape')

        batch_size, token_count, d_model = inputs.shape
        causal_mask = self._causal_mask(token_count, inputs.device)
        if self.attention_window:
            query_indices = torch.arange(
                token_count, device=inputs.device).view(-1, 1)
            key_indices = torch.arange(
                token_count, device=inputs.device).view(1, -1)
            causal_mask = causal_mask & (
                key_indices
                >= query_indices - self.attention_window + 1
            )
        if self.self_history_mode != 'joint':
            causal_mask = causal_mask & ~torch.eye(
                token_count,
                device=inputs.device,
                dtype=torch.bool,
            )
        elif self.value_topology != 'state':
            causal_mask = causal_mask & ~torch.eye(
                token_count,
                device=inputs.device,
                dtype=torch.bool,
            )
            if self.value_topology == 'past_anchor':
                causal_mask[0, 0] = True
        elif self.self_edge_heads >= 0:
            strict_past = causal_mask & ~torch.eye(
                token_count,
                device=inputs.device,
                dtype=torch.bool,
            )
            causal_mask = strict_past.unsqueeze(0).expand(
                self.n_heads, -1, -1).clone()
            if self.self_edge_heads > 0:
                diagonal = torch.eye(
                    token_count,
                    device=inputs.device,
                    dtype=torch.bool,
                )
                causal_mask[:self.self_edge_heads] |= diagonal
        if token_mask is not None:
            if token_mask.shape != (batch_size, token_count):
                raise ValueError('token_mask must have shape [batch, tokens]')
            key_mask = token_mask.bool()[:, None, None, :]
        else:
            key_mask = None

        if (
                self.mode in self.CONTENT_MODES
                and self.dyadic_router_mode != 'none'):
            queries, keys, values = self._content_qkv(
                inputs, positions)
            state_scores = self._content_scores(
                queries, keys)
            change_scores = None
            if self.dyadic_router_mode in {
                    'innovation', 'soft_innovation'}:
                changes = self._detrended_inputs(inputs)
                change_queries = self._project_innovation_query(
                    changes)
                if self.rope is not None:
                    change_queries, _ = self.rope(
                        change_queries,
                        change_queries,
                        positions,
                    )
                change_scores = self._content_scores(
                    change_queries, keys)
            values = self._value_memory(values)
            weights = self._dyadic_routed_weights(
                state_scores,
                change_scores,
                causal_mask,
            )
        elif (
                self.mode in self.CONTENT_MODES
                and self.address_geometry in {
                    'query_change_distribution',
                    'query_change_distribution_independent'}):
            queries, keys, values = self._content_qkv(
                inputs, positions)
            state_scores = self._content_scores(
                queries, keys)
            changes = self._detrended_inputs(inputs)
            change_queries = self._project_innovation_query(
                changes)
            if self.rope is not None:
                change_queries, _ = self.rope(
                    change_queries,
                    change_queries,
                    positions,
                )
            change_scores = self._content_scores(
                change_queries, keys)
            values = self._value_memory(values)
            state_weights = self._content_weights(
                queries,
                keys,
                causal_mask,
                scores=state_scores,
            )
            change_weights = self._content_weights(
                change_queries,
                keys,
                causal_mask,
                scores=change_scores,
            )
            distribution_gate = torch.tanh(
                self.geometry_gate).to(
                    state_weights.dtype).view(
                        1, self.n_heads, 1, 1)
            weights = state_weights + distribution_gate * (
                change_weights - state_weights)
        elif self.mode in self.CONTENT_MODES:
            queries, keys, values, geometry_scores = self._geometry_qkv(
                inputs,
                positions,
                address_inputs=address_inputs,
                value_inputs=value_inputs,
            )
            values = self._value_memory(values)
            weights = self._content_weights(
                queries,
                keys,
                causal_mask,
                scores=geometry_scores,
            )
        else:
            values = self._project_values(inputs)
            mask = causal_mask.view(1, 1, token_count, token_count)
            if self.mode == 'uniform':
                weights = mask.to(inputs.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True)
            elif self.mode == 'static':
                if token_count > self.static_logits.size(-1):
                    raise ValueError(
                        'token count exceeds configured static attention size')
                logits = self.static_logits[
                    :, :token_count, :token_count
                ].unsqueeze(0)
                weights = torch.softmax(
                    logits.masked_fill(~mask, float('-inf')), dim=-1)
            elif self.mode == 'identity':
                weights = torch.eye(
                    token_count, device=inputs.device, dtype=inputs.dtype
                ).view(1, 1, token_count, token_count)
            else:
                raise RuntimeError(
                    f'unsupported value-only attention mode: {self.mode}')

        if value_residuals is not None:
            residual_values = value_residuals.view(
                batch_size,
                token_count,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)
            values = values + residual_values.to(values.dtype)

        if key_mask is not None:
            weights = weights * key_mask.to(weights.dtype)
            if self.mode != 'raw':
                denominator = weights.sum(
                    dim=-1, keepdim=True).clamp_min(1e-6)
                weights = weights / denominator

        if self.self_history_mode != 'joint':
            gate = self._self_history_gate(
                queries,
                keys=keys,
                history_mask=causal_mask,
            )
            has_history = causal_mask.any(dim=-1).view(
                1, 1, token_count, 1)
            gate = torch.where(
                has_history,
                gate,
                torch.ones_like(gate),
            )
            diagonal = torch.eye(
                token_count,
                device=inputs.device,
                dtype=weights.dtype,
            ).view(1, 1, token_count, token_count)
            weights = (1.0 - gate) * weights + gate * diagonal

        weights = F.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(weights, values)
        if self.coarse_memory_group:
            output = output + self._coarse_memory_output(
                queries,
                keys,
                values,
                token_mask=token_mask,
            )
        output = output.transpose(1, 2).reshape(
            batch_size, token_count, d_model)
        output = self.output_dropout(self.out_projection(output))
        if (
                self.value_topology != 'state'
                or self.self_edge_heads >= 0):
            has_memory = causal_mask.any(dim=-1)
            if has_memory.ndim == 2:
                has_memory = has_memory.any(dim=0)
            has_memory = has_memory.view(1, token_count, 1)
            output = output * has_memory.to(output.dtype)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(output.dtype)
        return output

    def forward_last(self, inputs, positions=None):
        """Exact final-query path for the supported one-layer AR models.

        A single Transformer layer computes every causal query in parallel;
        its final query does not depend on the outputs of earlier queries.
        This path therefore retains all keys/values but materializes only the
        last attention row and its output projection.
        """
        if self.training:
            raise ValueError(
                'last-query attention is an evaluation-only path')
        if self.mode not in {'softmax', 'shrinkage'}:
            raise ValueError(
                'last-query attention currently supports softmax/shrinkage')
        if self.qk_input_mode != 'state' or self.head_sharing != 'none':
            raise ValueError(
                'last-query attention requires unshared state QK inputs')
        if (
                self.value_topology != 'state'
                or self.self_edge_heads != -1
                or self.self_history_mode != 'joint'):
            raise ValueError(
                'last-query attention requires ordinary state values')
        if self.coarse_memory_group or self.dyadic_router_mode != 'none':
            raise ValueError(
                'last-query attention does not support memory routing')
        if self.self_logit_bias != 0.0:
            raise ValueError(
                'last-query attention does not support self-logit bias')
        if self.address_geometry not in {
                'single', 'query_change', 'query_change_shared',
                'query_change_fixed', 'query_change_eval_shared'}:
            raise ValueError(
                'last-query attention supports ordinary or QIA geometry')

        batch_size, token_count, d_model = inputs.shape
        if token_count < 1:
            raise ValueError('last-query attention requires a token')
        if positions is None:
            key_positions = torch.arange(
                token_count,
                device=inputs.device,
                dtype=torch.float32,
            )
        else:
            key_positions = positions.to(
                device=inputs.device,
                dtype=torch.float32,
            )
            if key_positions.ndim != 1 \
                    or key_positions.numel() != token_count:
                raise ValueError(
                    'last-query positions must match the token count')

        # Keep the fused QKV projection used by the full path.  The primary
        # savings come from the N-by-N score/value products and the FFN; using
        # the same projection also minimizes numerical drift.
        qkv = self.qkv(inputs).view(
            batch_size,
            token_count,
            3,
            self.n_heads,
            self.head_dim,
        )
        queries, keys, values = qkv.permute(
            2, 0, 3, 1, 4).unbind(0)
        queries = queries[..., -1:, :]
        if self.rope is not None:
            queries, _ = self.rope(
                queries,
                queries,
                key_positions[-1:],
            )
            _, keys = self.rope(
                keys,
                keys,
                key_positions,
            )

        scores = self._content_scores(queries, keys)
        if self.address_geometry != 'single':
            changes = self._detrended_inputs(inputs)
            change_queries = self._project_innovation_query(
                changes[:, -1:])
            if self.rope is not None:
                change_queries, _ = self.rope(
                    change_queries,
                    change_queries,
                    key_positions[-1:],
                )
            change_scores = self._content_scores(
                change_queries,
                keys,
            )
            raw_gate = (
                self.fixed_geometry_gate.view(1, 1, 1, 1)
                if self.address_geometry == 'query_change_fixed'
                else self.geometry_gate.view(
                    1, self.geometry_gate.size(0), 1, 1)
            )
            gate = torch.tanh(raw_gate)
            if self.address_geometry == 'query_change_eval_shared':
                gate = gate.mean(dim=1, keepdim=True)
            scores = scores + gate.to(scores.dtype) * (
                change_scores - scores)

        visible_start = 0
        if self.attention_window:
            visible_start = max(
                token_count - self.attention_window,
                0,
            )
        keys = keys[..., visible_start:, :]
        values = values[..., visible_start:, :]
        scores = scores[..., visible_start:]
        weights = torch.softmax(scores, dim=-1)
        if self.mode == 'shrinkage':
            uniform = torch.full_like(
                weights,
                1.0 / weights.size(-1),
            )
            gate = self._shrinkage_gate(queries)
            weights = uniform + gate * (weights - uniform)
        weights = F.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(weights, values)
        output = output.transpose(1, 2).reshape(
            batch_size,
            1,
            d_model,
        )
        return self.output_dropout(self.out_projection(output))

    def forward_step(
            self,
            inputs,
            cache=None,
            position=None,
            max_cache_tokens=None):
        """Append one causal token while reusing projected keys and values.

        The streaming research path deliberately supports content attention
        only.  Those modes are the ones for which an ordinary Transformer KV
        cache is well defined, and ``softmax`` is the strong Patch-AR base used
        by the endogenous-statistics experiments.
        """
        if self.mode not in self.CONTENT_MODES:
            raise ValueError(
                'streaming cache currently requires content attention')
        if self.qk_input_mode != 'state':
            raise ValueError(
                'streaming cache currently requires state QK inputs')
        if (
                self.coarse_memory_group
                or self.address_geometry != 'single'):
            raise ValueError(
                'streaming cache does not support structural attention '
                'branches')
        if (
                self.value_topology != 'state'
                or self.self_edge_heads >= 0
                or self.self_history_mode != 'joint'):
            raise ValueError(
                'streaming cache currently requires state values')
        if inputs.ndim != 3 or inputs.size(1) != 1:
            raise ValueError(
                'streaming attention expects exactly one input token')
        if max_cache_tokens is not None and max_cache_tokens < 1:
            raise ValueError('max_cache_tokens must be positive')

        if position is None:
            cached_tokens = 0 if cache is None else cache[0].size(-2)
            position = cached_tokens
        positions = torch.as_tensor(
            [position],
            device=inputs.device,
            dtype=torch.float32,
        )
        queries, new_keys, new_values = self._content_qkv(
            inputs,
            positions,
        )
        if cache is None:
            keys = new_keys
            values = new_values
        else:
            cached_keys, cached_values = cache
            keys = torch.cat([cached_keys, new_keys], dim=-2)
            values = torch.cat([cached_values, new_values], dim=-2)
        if max_cache_tokens is not None:
            keys = keys[..., -max_cache_tokens:, :]
            values = values[..., -max_cache_tokens:, :]
        if self.attention_window:
            keys = keys[..., -self.attention_window:, :]
            values = values[..., -self.attention_window:, :]

        scores = self._content_scores(queries, keys)
        if self.mode in {'softmax', 'shrinkage', 'lag_bias'}:
            if self.mode == 'lag_bias':
                scores = scores + self._lag_bias_matrix(
                    1, keys.size(-2), scores.device
                ).unsqueeze(0).to(scores.dtype)
            weights = torch.softmax(scores, dim=-1)
            if self.mode == 'shrinkage':
                uniform = torch.full_like(
                    weights, 1.0 / keys.size(-2))
                gate = self._shrinkage_gate(queries)
                weights = uniform + gate * (weights - uniform)
        elif self.mode == 'sigmoid_norm':
            weights = torch.sigmoid(scores)
            weights = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
        elif self.mode == 'relu_norm':
            weights = F.relu(scores)
            weights = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
        elif self.mode == 'raw':
            weights = scores / keys.size(-2)
        else:
            raise RuntimeError(
                f'unsupported streaming attention mode: {self.mode}')
        weights = F.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(weights, values)
        output = output.transpose(1, 2).reshape(
            inputs.size(0),
            1,
            inputs.size(-1),
        )
        output = self.output_dropout(
            self.out_projection(output))
        return output, (keys, values)


class AblationTransformerBlock(nn.Module):
    """Transformer block whose attention, FFN, norm, and residuals are separable."""

    def __init__(self, d_model, n_heads, d_ff, dropout, activation='gelu',
                 attention_mode='softmax', ffn_mode='mlp',
                 norm_type='post', position_encoding='rope',
                 attention_residual=True, ffn_residual=True,
                 max_tokens=None, shrinkage_type='fixed',
                 shrinkage_alpha=1.0, dynamic_heads=None,
                 qk_norm='dot', attention_temperature=1.0,
                 attention_softcap=0.0,
                 learnable_temperature=False, norm_kind='layer',
                 layer_scale_init=None, local_kernel=0,
                 local_bias=True, local_mode='state',
                 value_topology='state',
                 self_logit_bias=0.0, self_edge_heads=-1,
                 self_history_mode='joint',
                 self_history_init=0.5,
                 qk_input_mode='state',
                 head_sharing='none',
                 attention_window=0,
                 coarse_memory_group=0,
                 coarse_memory_gate_init=0.0,
                 address_geometry='single',
                 geometry_gate_init=0.0,
                 dyadic_router_mode='none',
                 dyadic_router_gate_init=0.0):
        super().__init__()
        if ffn_mode not in {'mlp', 'swiglu', 'linear', 'none'}:
            raise ValueError(
                'ffn_mode must be mlp, swiglu, linear, or none')
        if norm_type not in {'pre', 'post', 'none'}:
            raise ValueError('norm_type must be pre, post, or none')
        if norm_kind not in {'layer', 'rms'}:
            raise ValueError('norm_kind must be layer or rms')
        if layer_scale_init is not None and layer_scale_init < 0:
            raise ValueError('layer_scale_init cannot be negative')
        if local_kernel != 0 and (
                local_kernel < 3 or local_kernel % 2 == 0):
            raise ValueError(
                'local_kernel must be zero or an odd integer >= 3')
        if local_mode not in {
                'state', 'innovation', 'state_innovation',
                'attention', 'state_attention'}:
            raise ValueError(
                'local_mode must be state, innovation, state_innovation, '
                'attention, or state_attention')

        self.norm_type = norm_type
        self.ffn_mode = ffn_mode
        self.attention_residual = bool(attention_residual)
        self.ffn_residual = bool(ffn_residual)
        self.local_mode = local_mode
        if norm_type == 'none':
            norm = lambda _: nn.Identity()
        elif norm_kind == 'layer':
            norm = nn.LayerNorm
        else:
            norm = RMSNorm
        self.attention_norm = norm(d_model)
        self.feed_forward_norm = norm(d_model)
        self.attention = AblationSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            mode=attention_mode,
            position_encoding=position_encoding,
            max_tokens=max_tokens,
            shrinkage_type=shrinkage_type,
            shrinkage_alpha=shrinkage_alpha,
            dynamic_heads=dynamic_heads,
            qk_norm=qk_norm,
            attention_temperature=attention_temperature,
            attention_softcap=attention_softcap,
            learnable_temperature=learnable_temperature,
            value_topology=value_topology,
            self_logit_bias=self_logit_bias,
            self_edge_heads=self_edge_heads,
            self_history_mode=self_history_mode,
            self_history_init=self_history_init,
            qk_input_mode=qk_input_mode,
            head_sharing=head_sharing,
            attention_window=attention_window,
            coarse_memory_group=coarse_memory_group,
            coarse_memory_gate_init=coarse_memory_gate_init,
            address_geometry=address_geometry,
            geometry_gate_init=geometry_gate_init,
            dyadic_router_mode=dyadic_router_mode,
            dyadic_router_gate_init=dyadic_router_gate_init,
        )
        self.local_mixer = (
            CausalDepthwiseMixer(
                d_model=d_model,
                kernel_size=local_kernel,
                dropout=dropout,
                bias=local_bias,
            )
            if local_kernel > 0
            else None
        )
        self.innovation_local_mixer = (
            CausalDepthwiseMixer(
                d_model=d_model,
                kernel_size=local_kernel,
                dropout=dropout,
                bias=local_bias,
            )
            if local_kernel > 0 and local_mode in {
                'state_innovation', 'state_attention'}
            else None
        )
        if layer_scale_init is None:
            self.register_parameter('attention_scale', None)
            self.register_parameter('ffn_scale', None)
        else:
            self.attention_scale = nn.Parameter(torch.full(
                (d_model,), float(layer_scale_init)))
            self.ffn_scale = nn.Parameter(torch.full(
                (d_model,), float(layer_scale_init)))

        if ffn_mode == 'mlp':
            activation_layer = nn.GELU if activation == 'gelu' else nn.ReLU
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                activation_layer(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
        elif ffn_mode == 'swiglu':
            # Match the ordinary two-layer FFN's parameter count rather than
            # granting the gated variant a hidden-width advantage.
            gated_width = max(
                1,
                round(
                    (2 * d_model * d_ff + d_ff)
                    / (3 * d_model + 2)
                ),
            )
            self.feed_forward = SwiGLUFeedForward(
                d_model=d_model,
                hidden_size=gated_width,
                dropout=dropout,
            )
        elif ffn_mode == 'linear':
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.feed_forward = None

    @staticmethod
    def _merge(inputs, update, use_residual):
        return inputs + update if use_residual else update

    def _ffn(self, inputs):
        if self.feed_forward is None:
            return torch.zeros_like(inputs)
        return self.feed_forward(inputs)

    @staticmethod
    def _local_innovations(inputs):
        return torch.cat([
            torch.zeros_like(inputs[:, :1]),
            inputs[:, 1:] - inputs[:, :-1],
        ], dim=1)

    def _local_update(
            self, inputs, attention_update=None, last_only=False):
        if self.local_mixer is None:
            return None
        if self.local_mode in {'attention', 'state_attention'}:
            if last_only:
                raise ValueError(
                    'post-attention local fusion requires all query outputs')
            if attention_update is None:
                raise ValueError(
                    'post-attention local fusion requires attention outputs')
            if self.local_mode == 'attention':
                return self.local_mixer(attention_update)
            return (
                self.local_mixer(inputs)
                + self.innovation_local_mixer(attention_update)
            )
        mixer_forward = (
            self.local_mixer.forward_last
            if last_only else self.local_mixer)
        if self.local_mode == 'state':
            return mixer_forward(inputs)
        innovations = self._local_innovations(inputs)
        if self.local_mode == 'state_innovation':
            innovation_forward = (
                self.innovation_local_mixer.forward_last
                if last_only else self.innovation_local_mixer)
            return (
                mixer_forward(inputs)
                + innovation_forward(innovations)
            )
        return mixer_forward(innovations)

    def _attention_update(
            self, inputs, token_mask, positions, address_inputs=None,
            value_inputs=None, value_residuals=None):
        update = self.attention(
            inputs,
            token_mask,
            positions,
            address_inputs=address_inputs,
            value_inputs=value_inputs,
            value_residuals=value_residuals,
        )
        if self.local_mixer is not None:
            update = update + self._local_update(
                inputs, attention_update=update)
        if self.attention_scale is not None:
            update = update * self.attention_scale
        return update

    def _ffn_update(self, inputs):
        update = self._ffn(inputs)
        if self.ffn_scale is not None:
            update = update * self.ffn_scale
        return update

    def forward(
            self, inputs, token_mask=None, positions=None,
            address_inputs=None, value_inputs=None,
            value_residuals=None, state_mixer=None):
        if (
                address_inputs is not None
                or value_inputs is not None
                or value_residuals is not None
                ) and self.norm_type == 'pre':
            raise ValueError(
                'external address inputs currently require post/no norm')
        if state_mixer is not None:
            inputs = state_mixer('pre_attention', inputs)
        if self.norm_type == 'pre':
            attention_update = self._attention_update(
                self.attention_norm(inputs), token_mask, positions)
            hidden = self._merge(
                inputs, attention_update, self.attention_residual)
            if state_mixer is not None:
                hidden = state_mixer('post_attention', hidden)
            ffn_update = self._ffn_update(
                self.feed_forward_norm(hidden))
            hidden = self._merge(hidden, ffn_update, self.ffn_residual)
        else:
            attention_update = self._attention_update(
                inputs,
                token_mask,
                positions,
                address_inputs=address_inputs,
                value_inputs=value_inputs,
                value_residuals=value_residuals,
            )
            hidden = self.attention_norm(self._merge(
                inputs, attention_update, self.attention_residual))
            if state_mixer is not None:
                hidden = state_mixer('post_attention', hidden)
            ffn_update = self._ffn_update(hidden)
            hidden = self.feed_forward_norm(self._merge(
                hidden, ffn_update, self.ffn_residual))

        if state_mixer is not None:
            hidden = state_mixer('post_ffn', hidden)
        if token_mask is not None:
            hidden = hidden * token_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden

    def forward_last(self, inputs, positions=None):
        """Compute the exact final token of a single Transformer block."""
        if self.norm_type == 'pre':
            normalized = self.attention_norm(inputs)
            attention_update = self.attention.forward_last(
                normalized,
                positions=positions,
            )
            if self.local_mixer is not None:
                attention_update = (
                    attention_update
                    + self._local_update(normalized, last_only=True)
                )
            if self.attention_scale is not None:
                attention_update = (
                    attention_update * self.attention_scale)
            hidden = self._merge(
                inputs[:, -1:],
                attention_update,
                self.attention_residual,
            )
            ffn_update = self._ffn_update(
                self.feed_forward_norm(hidden))
            return self._merge(
                hidden,
                ffn_update,
                self.ffn_residual,
            )

        attention_update = self.attention.forward_last(
            inputs,
            positions=positions,
        )
        if self.local_mixer is not None:
            attention_update = (
                attention_update
                + self._local_update(inputs, last_only=True)
            )
        if self.attention_scale is not None:
            attention_update = attention_update * self.attention_scale
        hidden = self.attention_norm(self._merge(
            inputs[:, -1:],
            attention_update,
            self.attention_residual,
        ))
        ffn_update = self._ffn_update(hidden)
        return self.feed_forward_norm(self._merge(
            hidden,
            ffn_update,
            self.ffn_residual,
        ))

    def forward_step(
            self,
            inputs,
            cache=None,
            position=None,
            max_cache_tokens=None):
        """Streaming counterpart of ``forward`` for one appended token."""
        if self.local_mixer is not None:
            raise ValueError(
                'streaming cache does not support the local token mixer')
        if self.norm_type == 'pre':
            attention_update, cache = self.attention.forward_step(
                self.attention_norm(inputs),
                cache=cache,
                position=position,
                max_cache_tokens=max_cache_tokens,
            )
            hidden = self._merge(
                inputs,
                attention_update,
                self.attention_residual,
            )
            ffn_update = self._ffn_update(
                self.feed_forward_norm(hidden))
            hidden = self._merge(
                hidden,
                ffn_update,
                self.ffn_residual,
            )
        else:
            attention_update, cache = self.attention.forward_step(
                inputs,
                cache=cache,
                position=position,
                max_cache_tokens=max_cache_tokens,
            )
            hidden = self.attention_norm(self._merge(
                inputs,
                attention_update,
                self.attention_residual,
            ))
            ffn_update = self._ffn_update(hidden)
            hidden = self.feed_forward_norm(self._merge(
                hidden,
                ffn_update,
                self.ffn_residual,
            ))
        return hidden, cache


class AblationPatchARBackbone(nn.Module):
    """Patch AR backbone used for controlled Transformer component ablations."""

    def __init__(self, patch_len, d_model, n_heads, e_layers, d_ff, dropout,
                 activation, attention_mode='softmax', ffn_mode='mlp',
                 norm_type='post', position_encoding='rope',
                 attention_residual=True, ffn_residual=True,
                 max_tokens=None, shrinkage_type='fixed',
                 shrinkage_alpha=1.0, dynamic_heads=None,
                 qk_norm='dot', attention_temperature=1.0,
                 attention_softcap=0.0,
                 learnable_temperature=False, norm_kind='layer',
                 layer_scale_init=None, local_kernel=0,
                 local_bias=True, local_mode='state',
                 value_topology='state',
                 self_logit_bias=0.0, self_edge_heads=-1,
                 self_history_mode='joint',
                 self_history_init=0.5,
                 self_history_pattern='all',
                 qk_input_mode='state',
                 head_sharing='none',
                 attention_window=0,
                 coarse_memory_group=0,
                 coarse_memory_gate_init=0.0,
                 address_geometry='single',
                 geometry_gate_init=0.0,
                 dyadic_router_mode='none',
                 dyadic_router_gate_init=0.0):
        super().__init__()
        if e_layers < 0:
            raise ValueError('e_layers cannot be negative')
        if self_history_pattern not in {'all', 'first', 'last'}:
            raise ValueError(
                'self_history_pattern must be all, first, or last')

        self.patch_len = patch_len
        # Define the common input/output maps before variant-specific blocks so
        # equal seeds give every ablation identical shared-map initialization.
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.output_head = nn.Linear(d_model, patch_len)
        def layer_routing_mode(layer_index):
            if self_history_mode == 'joint' \
                    or self_history_pattern == 'all':
                return self_history_mode
            if self_history_pattern == 'first':
                return (
                    self_history_mode
                    if layer_index == 0 else 'joint'
                )
            return (
                self_history_mode
                if layer_index == e_layers - 1 else 'joint'
            )

        self.blocks = nn.ModuleList([
            AblationTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                attention_mode=attention_mode,
                ffn_mode=ffn_mode,
                norm_type=norm_type,
                position_encoding=position_encoding,
                attention_residual=attention_residual,
                ffn_residual=ffn_residual,
                max_tokens=max_tokens,
                shrinkage_type=shrinkage_type,
                shrinkage_alpha=shrinkage_alpha,
                dynamic_heads=dynamic_heads,
                qk_norm=qk_norm,
                attention_temperature=attention_temperature,
                attention_softcap=attention_softcap,
                learnable_temperature=learnable_temperature,
                norm_kind=norm_kind,
                layer_scale_init=layer_scale_init,
                local_kernel=local_kernel,
                local_bias=local_bias,
                local_mode=local_mode,
                value_topology=value_topology,
                self_logit_bias=self_logit_bias,
                self_edge_heads=self_edge_heads,
                self_history_mode=layer_routing_mode(layer_index),
                self_history_init=self_history_init,
                qk_input_mode=qk_input_mode,
                head_sharing=head_sharing,
                attention_window=attention_window,
                coarse_memory_group=coarse_memory_group,
                coarse_memory_gate_init=coarse_memory_gate_init,
                address_geometry=address_geometry,
                geometry_gate_init=geometry_gate_init,
                dyadic_router_mode=dyadic_router_mode,
                dyadic_router_gate_init=dyadic_router_gate_init,
            )
            for layer_index in range(e_layers)
        ])
        self.output_norm = (
            (
                nn.LayerNorm(d_model)
                if norm_kind == 'layer'
                else RMSNorm(d_model)
            )
            if norm_type == 'pre' and e_layers > 0
            else nn.Identity()
        )

    def encode_projected(
            self, tokens, token_mask=None, positions=None,
            attention_mask_override=None, address_tokens=None,
            value_tokens=None, value_residuals=None, state_mixer=None):
        if attention_mask_override is not None:
            raise ValueError(
                'TransformerAblationAR does not accept an attention mask override')
        if (
                address_tokens is not None
                or value_tokens is not None
                or value_residuals is not None
                ) and len(self.blocks) != 1:
            raise ValueError(
                'external address tokens currently require one layer')
        for layer_index, block in enumerate(self.blocks):
            layer_mixer = (
                None
                if state_mixer is None
                else lambda stage, state, index=layer_index: state_mixer(
                    index, stage, state)
            )
            tokens = block(
                tokens,
                token_mask,
                positions,
                address_inputs=address_tokens,
                value_inputs=value_tokens,
                value_residuals=value_residuals,
                state_mixer=layer_mixer,
            )
        tokens = self.output_norm(tokens)
        if token_mask is not None:
            tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens

    def forward(self, patches):
        return self.encode_projected(self.patch_projection(patches))

    def encode_projected_last(self, tokens, positions=None):
        """Return the exact final hidden state for a one-layer backbone."""
        if len(self.blocks) != 1:
            raise ValueError(
                'last-query inference requires exactly one Transformer layer')
        token = self.blocks[0].forward_last(
            tokens,
            positions=positions,
        )
        return self.output_norm(token)

    def encode_projected_step(
            self,
            token,
            caches=None,
            position=None,
            max_cache_tokens=None):
        """Encode one projected token and return updated per-layer caches."""
        if token.ndim != 3 or token.size(1) != 1:
            raise ValueError(
                'streaming backbone expects one projected token')
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError(
                'one streaming cache is required per Transformer layer')
        next_caches = []
        for block, cache in zip(self.blocks, caches):
            token, cache = block.forward_step(
                token,
                cache=cache,
                position=position,
                max_cache_tokens=max_cache_tokens,
            )
            next_caches.append(cache)
        return self.output_norm(token), next_caches
