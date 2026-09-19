import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to every patch token."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError('the attention head dimension must be even for RoPE')
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(values):
        even = values[..., 0::2]
        odd = values[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(self, queries, keys, positions=None):
        token_count = queries.size(-2)
        if positions is None:
            positions = torch.arange(
                token_count, device=queries.device, dtype=torch.float32)
        else:
            positions = positions.to(device=queries.device, dtype=torch.float32)
            if positions.shape[-1] != token_count:
                raise ValueError('RoPE positions must match the token dimension')
        angles = positions.unsqueeze(-1) * self.inv_freq.float()
        cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=queries.dtype)
        sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=queries.dtype)
        if positions.ndim == 1:
            cos = cos.view(1, 1, token_count, -1)
            sin = sin.view(1, 1, token_count, -1)
        elif positions.ndim == 2:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            raise ValueError('RoPE positions must be one- or two-dimensional')
        queries = queries * cos + self._rotate_half(queries) * sin
        keys = keys * cos + self._rotate_half(keys) * sin
        return queries, keys


class DyadicRotaryEmbedding(RotaryEmbedding):
    """RoPE whose periods cover the finite token range on powers of two.

    Standard RoPE inherits its frequencies from a very long language-model
    context.  In the short patch-token sequences used here, most of those
    frequencies barely rotate.  This variant assigns one regularly spaced
    power-of-two period to every rotary pair, ending at ``max_period``.
    """

    def __init__(self, head_dim, max_period):
        nn.Module.__init__(self)
        if head_dim % 2 != 0:
            raise ValueError('the attention head dimension must be even for RoPE')
        if max_period <= 0:
            raise ValueError('max_period must be positive')
        pair_count = head_dim // 2
        exponents = torch.arange(
            pair_count - 1, -1, -1, dtype=torch.float32)
        periods = float(max_period) / torch.pow(2.0, exponents)
        inv_freq = (2.0 * torch.pi) / periods
        self.register_buffer('periods', periods, persistent=False)
        self.register_buffer('inv_freq', inv_freq, persistent=False)


class ValuePairMixer(nn.Module):
    """Content-gated, ordered composition of every Value with its predecessor.

    The same tiny MLP is shared across attention heads.  All positions and
    heads are processed as batch dimensions, so the operation remains fully
    parallel and adds no token-pair loop or quadratic term.
    """

    def __init__(self, head_dim, hidden_multiplier=1):
        super().__init__()
        if hidden_multiplier < 1:
            raise ValueError('hidden_multiplier must be positive')
        pair_dim = 4 * head_dim
        hidden_dim = hidden_multiplier * head_dim
        self.gate_projection = nn.Linear(pair_dim, 1)
        self.compose_in = nn.Linear(pair_dim, hidden_dim)
        self.compose_out = nn.Linear(hidden_dim, head_dim)

        # Begin close to ordinary attention while allowing every component to
        # receive gradient immediately.  sigmoid(-2) is approximately 0.12.
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -2.0)
        nn.init.normal_(self.compose_out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.compose_out.bias)

    def forward(self, values):
        previous = torch.cat([
            torch.zeros_like(values[..., :1, :]),
            values[..., :-1, :],
        ], dim=-2)
        pair_features = torch.cat([
            previous,
            values,
            values - previous,
            values * previous,
        ], dim=-1)
        gate = torch.sigmoid(self.gate_projection(pair_features))
        # There is no predecessor at token zero, so it must remain atomic.
        first_mask = torch.ones_like(gate)
        first_mask[..., 0, :] = 0.0
        gate = gate * first_mask
        correction = self.compose_out(F.silu(
            self.compose_in(pair_features)))
        return values + gate * correction, gate


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, attention_mode='causal',
                 attention_block_size=1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if attention_mode not in ('causal', 'bidirectional', 'block_causal'):
            raise ValueError(
                'attention_mode must be causal, bidirectional, or block_causal')
        if attention_block_size < 1:
            raise ValueError('attention_block_size must be positive')
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout
        self.attention_mode = attention_mode
        self.attention_block_size = attention_block_size
        self.rope = RotaryEmbedding(self.head_dim)
        # Optional experimental source-local Value transformation.  Models
        # that do not explicitly attach one retain the original computation.
        self.value_pair_mixer = None

    def forward(self, values, token_mask=None, positions=None,
                attention_mask_override=None):
        batch_size, token_count, d_model = values.shape
        qkv = self.qkv(values).view(
            batch_size, token_count, 3, self.n_heads, self.head_dim
        )
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rope is not None:
            queries, keys = self.rope(queries, keys, positions)
        if self.value_pair_mixer is not None:
            values, _ = self.value_pair_mixer(values)
        attention_mask = None
        is_causal = self.attention_mode == 'causal'
        if attention_mask_override is not None:
            if attention_mask_override.shape == (token_count, token_count):
                attention_mask = attention_mask_override.view(
                    1, 1, token_count, token_count)
            elif attention_mask_override.shape == (
                    batch_size, token_count, token_count):
                attention_mask = attention_mask_override.unsqueeze(1)
            elif attention_mask_override.shape == (
                    1, self.n_heads, token_count, token_count):
                attention_mask = attention_mask_override
            elif attention_mask_override.shape == (
                    batch_size, self.n_heads, token_count, token_count):
                attention_mask = attention_mask_override
            else:
                raise ValueError(
                    'attention_mask_override must have shape [tokens, tokens] '
                    'or [batch, tokens, tokens], [1, heads, tokens, tokens], '
                    'or [batch, heads, tokens, tokens]')
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(device=values.device)
            elif torch.is_floating_point(attention_mask):
                attention_mask = attention_mask.to(
                    device=values.device, dtype=queries.dtype)
            else:
                raise ValueError(
                    'attention_mask_override must be boolean or floating point')
            is_causal = False
        elif self.attention_mode == 'block_causal':
            token_indices = torch.arange(token_count, device=values.device)
            block_indices = token_indices // self.attention_block_size
            attention_mask = (
                block_indices.view(1, -1)
                <= block_indices.view(-1, 1)
            ).view(1, 1, token_count, token_count)
            is_causal = False
        if token_mask is not None:
            if token_mask.shape != (batch_size, token_count):
                raise ValueError('token_mask must have shape [batch, tokens]')
            token_mask = token_mask.bool()
            if attention_mask_override is None \
                    and self.attention_mode == 'causal':
                causal_mask = torch.ones(
                    token_count, token_count, device=values.device,
                    dtype=torch.bool).tril()
                attention_mask = (
                    causal_mask.view(1, 1, token_count, token_count)
                    & token_mask[:, None, None, :]
                )
            elif attention_mask is None:
                attention_mask = token_mask[:, None, None, :]
            elif attention_mask.dtype == torch.bool:
                attention_mask = (
                    attention_mask & token_mask[:, None, None, :])
            else:
                attention_mask = attention_mask.masked_fill(
                    ~token_mask[:, None, None, :],
                    float('-inf'),
                )
            is_causal = False
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).reshape(batch_size, token_count, d_model)
        output = self.output_dropout(self.out_projection(output))
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(output.dtype)
        return output


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation='gelu',
                 norm_type='post', attention_mode='causal',
                 attention_block_size=1):
        super().__init__()
        if norm_type not in ('pre', 'post'):
            raise ValueError('norm_type must be pre or post')
        self.norm_type = norm_type
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, n_heads, dropout, attention_mode,
            attention_block_size)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        activation_layer = nn.GELU if activation == 'gelu' else nn.ReLU
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values, token_mask=None, positions=None,
                attention_mask_override=None):
        if self.norm_type == 'pre':
            values = values + self.attention(
                self.attention_norm(values), token_mask, positions,
                attention_mask_override)
            values = values + self.feed_forward(self.feed_forward_norm(values))
        else:
            values = self.attention_norm(
                values + self.attention(
                    values, token_mask, positions, attention_mask_override))
            values = self.feed_forward_norm(
                values + self.feed_forward(values))
        if token_mask is not None:
            values = values * token_mask.unsqueeze(-1).to(values.dtype)
        return values


class PatchARBackbone(nn.Module):
    """Shared channel-independent causal Transformer for DenseAR and TailAR."""

    def __init__(self, patch_len, d_model, n_heads, e_layers, d_ff, dropout,
                 activation, norm_type='post', attention_mode='causal',
                 attention_block_size=1):
        super().__init__()
        if norm_type not in ('pre', 'post'):
            raise ValueError('norm_type must be pre or post')
        self.patch_len = patch_len
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model, n_heads, d_ff, dropout, activation, norm_type,
                attention_mode, attention_block_size)
            for _ in range(e_layers)
        ])
        self.output_norm = (
            nn.LayerNorm(d_model) if norm_type == 'pre' else nn.Identity()
        )
        self.output_head = nn.Linear(d_model, patch_len)

    def encode_projected(self, tokens, token_mask=None, positions=None,
                         attention_mask_override=None):
        for block in self.blocks:
            tokens = block(
                tokens,
                token_mask,
                positions,
                attention_mask_override,
            )
        tokens = self.output_norm(tokens)
        if token_mask is not None:
            tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens

    def forward(self, patches):
        return self.encode_projected(self.patch_projection(patches))
