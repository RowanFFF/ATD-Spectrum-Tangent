import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.PatchARAblation import AblationPatchARBackbone
from models.DenseAR import Model as DenseARModel


class DiagonalAtomProjection(nn.Module):
    """Minimal trainable gauge over canonical fine-patch coordinates."""

    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(int(width)))

    @property
    def weight(self):
        return torch.diag(self.scale)

    def forward(self, inputs):
        return inputs * self.scale


class FactorizedPatchProjection(nn.Module):
    """Shared fine-patch stem with fixed-width positional aggregation.

    A parent patch is split into ``slots`` fine patches.  Every fine patch is
    encoded by the same ``atom_projection`` and then mapped into the fixed
    ``d_model`` space through a DCT position basis.  ``basis_dim`` controls
    the number of shared basis maps and is independent of the active slot
    count, so changing the parent patch length does not change this module's
    parameter count.

    When ``atom_dim >= fine_patch_len`` and ``basis_dim >= slots``, the module
    can represent every affine ``patch_len -> d_model`` projection.  The
    ``initialize_from_native`` helper realizes that equivalence explicitly.
    """

    def __init__(self, patch_len, fine_patch_len, atom_dim, d_model,
                 basis_dim=8, stem='linear'):
        super().__init__()
        if fine_patch_len < 1 or patch_len % fine_patch_len:
            raise ValueError(
                'factorized fine-patch length must divide the parent patch')
        if atom_dim < 1:
            raise ValueError('factorized atom dimension must be positive')
        if basis_dim < 1:
            raise ValueError('factorized basis dimension must be positive')
        if stem not in {'linear', 'diagonal', 'identity'}:
            raise ValueError(
                'factorized stem must be linear, diagonal, or identity')
        self.patch_len = int(patch_len)
        self.fine_patch_len = int(fine_patch_len)
        self.atom_dim = int(atom_dim)
        self.d_model = int(d_model)
        self.basis_dim = int(basis_dim)
        self.slots = self.patch_len // self.fine_patch_len
        if self.basis_dim < self.slots:
            raise ValueError(
                'factorized basis dimension must be at least the number '
                f'of fine-patch slots ({self.slots})')
        if stem in {'diagonal', 'identity'}:
            if self.atom_dim != self.fine_patch_len:
                raise ValueError(
                    f'{stem} factorized stem requires atom_dim to equal '
                    'the fine-patch length')
            # Reserve the same RNG draws consumed by the matched learned
            # ``Linear(..., bias=False)`` stem.  The learned stem is reset to
            # identity below, so this keeps its basis-map initialization
            # exactly paired with the parameter-free identity control.
            reserved_weight = torch.empty(
                self.atom_dim, self.fine_patch_len)
            nn.init.kaiming_uniform_(reserved_weight, a=math.sqrt(5))
            self.atom_projection = (
                nn.Identity()
                if stem == 'identity'
                else DiagonalAtomProjection(self.atom_dim)
            )
        else:
            self.atom_projection = nn.Linear(
                self.fine_patch_len, self.atom_dim, bias=False)

        positions = (
            torch.arange(self.slots, dtype=torch.float32) + 0.5
        ) / self.slots
        frequencies = torch.arange(
            self.basis_dim, dtype=torch.float32)
        coefficients = torch.cos(
            math.pi * positions[:, None] * frequencies[None, :])
        if self.basis_dim > 1:
            coefficients[:, 1:] *= math.sqrt(2.0)
        self.register_buffer(
            'position_coefficients', coefficients, persistent=True)
        self.basis_maps = nn.Parameter(torch.empty(
            self.basis_dim, self.d_model, self.atom_dim))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.register_parameter('protected_scale_coordinates', None)
        self.register_buffer(
            'protected_scale_null_basis', None, persistent=True)
        self.protected_scale_slots = None
        self.register_parameter('refinement_scale_coordinates', None)
        self.register_buffer(
            'refinement_scale_null_basis', None, persistent=True)
        self.refinement_scale_protected_slots = ()
        self.scale = 1.0 / math.sqrt(self.slots * self.basis_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.atom_projection, nn.Linear):
            # An identity lift starts full-column-rank whenever possible and
            # avoids an unnecessary random bottleneck before the slot mixer.
            if self.atom_dim >= self.fine_patch_len:
                nn.init.zeros_(self.atom_projection.weight)
                with torch.no_grad():
                    self.atom_projection.weight[
                        :self.fine_patch_len
                    ].copy_(torch.eye(self.fine_patch_len))
            else:
                nn.init.orthogonal_(self.atom_projection.weight)
        for basis_map in self.basis_maps:
            nn.init.xavier_uniform_(basis_map)
        nn.init.zeros_(self.bias)

    def _atom_weight(self):
        if isinstance(self.atom_projection, nn.Identity):
            return torch.eye(
                self.fine_patch_len,
                device=self.basis_maps.device,
                dtype=self.basis_maps.dtype,
            )
        return self.atom_projection.weight

    def effective_weight(self):
        """Return the equivalent dense ``[d_model, patch_len]`` matrix."""
        atom_weight = self._atom_weight()
        slot_maps = self.scale * torch.einsum(
            'km,mdr->kdr',
            self.position_coefficients.to(self.basis_maps.dtype),
            self._active_basis_maps(),
        )
        blocks = torch.einsum('kdr,rp->kdp', slot_maps, atom_weight)
        return blocks.permute(1, 0, 2).reshape(
            self.d_model, self.patch_len)

    @staticmethod
    def _position_basis(slots, basis_dim):
        positions = (
            torch.arange(slots, dtype=torch.float64) + 0.5
        ) / slots
        frequencies = torch.arange(basis_dim, dtype=torch.float64)
        coefficients = torch.cos(
            math.pi * positions[:, None] * frequencies[None, :])
        if basis_dim > 1:
            coefficients[:, 1:] *= math.sqrt(2.0)
        return coefficients

    @classmethod
    def _joint_null_basis(
            cls, protected_slots, basis_dim, device, dtype):
        protected_slots = tuple(sorted(set(map(int, protected_slots))))
        if not protected_slots:
            raise ValueError('at least one protected scale is required')
        coefficients = torch.cat([
            cls._position_basis(slots, basis_dim)
            for slots in protected_slots
        ])
        _, singular, right = torch.linalg.svd(
            coefficients, full_matrices=True)
        tolerance = max(coefficients.shape) * torch.finfo(
            coefficients.dtype).eps * singular.max()
        rank = int((singular > tolerance).sum().item())
        return right[rank:].T.to(device=device, dtype=dtype)

    @classmethod
    def _null_basis(cls, slots, basis_dim, device, dtype):
        return cls._joint_null_basis(
            (slots,), basis_dim, device, dtype)

    def enable_protected_scale_residual(self, protected_slots):
        """Add trainable basis directions invisible at ``protected_slots``."""
        protected_slots = int(protected_slots)
        if not 1 <= protected_slots < self.basis_dim:
            raise ValueError(
                'protected slots must be positive and smaller than basis_dim')
        if self.protected_scale_coordinates is not None:
            raise RuntimeError('protected scale residual is already enabled')
        null_basis = self._null_basis(
            protected_slots,
            self.basis_dim,
            self.basis_maps.device,
            self.basis_maps.dtype,
        )
        self.protected_scale_null_basis = null_basis
        self.protected_scale_slots = protected_slots
        self.protected_scale_coordinates = nn.Parameter(torch.zeros(
            null_basis.size(1), self.d_model, self.atom_dim,
            device=self.basis_maps.device,
            dtype=self.basis_maps.dtype,
        ))
        return self.protected_scale_coordinates

    def enable_refinement_scale_residual(self, protected_slots):
        """Add a later residual invisible to every protected scale."""
        protected_slots = tuple(sorted(set(map(int, protected_slots))))
        if not protected_slots or protected_slots[-1] >= self.basis_dim:
            raise ValueError(
                'protected slots must be positive and smaller than basis_dim')
        if protected_slots[0] < 1:
            raise ValueError('protected slots must be positive')
        if self.refinement_scale_coordinates is not None:
            raise RuntimeError('refinement scale residual is already enabled')
        null_basis = self._joint_null_basis(
            protected_slots,
            self.basis_dim,
            self.basis_maps.device,
            self.basis_maps.dtype,
        )
        if null_basis.size(1) == 0:
            raise ValueError('protected scales leave no refinement direction')
        self.refinement_scale_null_basis = null_basis
        self.refinement_scale_protected_slots = protected_slots
        self.refinement_scale_coordinates = nn.Parameter(torch.zeros(
            null_basis.size(1), self.d_model, self.atom_dim,
            device=self.basis_maps.device,
            dtype=self.basis_maps.dtype,
        ))
        return self.refinement_scale_coordinates

    def _active_basis_maps(self):
        active = self.basis_maps
        if self.protected_scale_coordinates is not None \
                and self.slots > self.protected_scale_slots:
            active = active + torch.einsum(
                'mr,rda->mda',
                self.protected_scale_null_basis.to(self.basis_maps.dtype),
                self.protected_scale_coordinates,
            )
        if self.refinement_scale_coordinates is not None \
                and self.slots > max(
                    self.refinement_scale_protected_slots):
            active = active + torch.einsum(
                'mr,rda->mda',
                self.refinement_scale_null_basis.to(self.basis_maps.dtype),
                self.refinement_scale_coordinates,
            )
        return active

    def initialize_from_native(self, projection):
        """Set the factorization to exactly reproduce an affine projection."""
        if not isinstance(projection, nn.Linear):
            raise TypeError('native patch projection must be nn.Linear')
        if (
                projection.in_features != self.patch_len
                or projection.out_features != self.d_model):
            raise ValueError('native projection shape does not match')
        atom_weight = self._atom_weight().detach().float()
        if torch.linalg.matrix_rank(atom_weight) < self.fine_patch_len:
            raise ValueError(
                'exact native initialization requires a full-column-rank '
                'atom projection')
        target_blocks = projection.weight.detach().float().reshape(
            self.d_model, self.slots, self.fine_patch_len
        ).permute(1, 0, 2)
        atom_right_inverse = torch.linalg.pinv(atom_weight)
        slot_maps = torch.einsum(
            'kdp,pr->kdr', target_blocks, atom_right_inverse)
        coefficient_inverse = torch.linalg.pinv(
            self.position_coefficients.float())
        basis_maps = torch.einsum(
            'mk,kdr->mdr', coefficient_inverse, slot_maps
        ) / self.scale
        with torch.no_grad():
            self.basis_maps.copy_(basis_maps.to(self.basis_maps.dtype))
            if projection.bias is None:
                self.bias.zero_()
            else:
                self.bias.copy_(projection.bias)

    def forward(self, parent_patches):
        if parent_patches.size(-1) != self.patch_len:
            raise ValueError(
                f'expected parent patches of length {self.patch_len}')
        fine = parent_patches.reshape(
            *parent_patches.shape[:-1],
            self.slots,
            self.fine_patch_len,
        )
        atoms = self.atom_projection(fine)
        projected = self.scale * torch.einsum(
            '...kr,km,mdr->...d',
            atoms,
            self.position_coefficients.to(atoms.dtype),
            self._active_basis_maps(),
        )
        return projected + self.bias


class FrequencyBandProjection(nn.Module):
    """Per-patch DCT band decomposition with independent per-band widths.

    A parent patch is DCT-transformed with the same orthonormal DCT-II
    basis used by the band diagnostics (``dct_basis``), and its
    coefficients are grouped into the patch-relative bands level=[0,1),
    macro=[1,c), patch_scale=[c,8c), fine=[8c,patch_len) with
    c=ceil(patch_len/fine_patch_len) — the ``temporal_dct_bands``
    partition anchored per patch.  Each band gets an independent
    projection of width ``w_b``; outputs concatenate to one token of
    width ``sum(widths)`` (required to equal ``d_model``).  The width
    vector (w_1..w_4) is the per-band capacity dial: the partition is
    lossless whenever w_b >= group size for every band (per-band
    injectivity), so width above that floor is pure capacity allocation
    with no representational cost.
    """

    def __init__(self, patch_len, d_model, widths, fine_patch_len=12):
        super().__init__()
        patch_len = int(patch_len)
        d_model = int(d_model)
        widths = tuple(int(w) for w in widths)
        if len(widths) != 4:
            raise ValueError(
                'frequency-band assembly requires exactly four widths '
                f'(level, macro, patch_scale, fine), got {widths}')
        if sum(widths) != d_model:
            raise ValueError(
                'frequency-band widths must sum to d_model: '
                f'{widths} vs d_model={d_model}')
        if fine_patch_len < 1:
            raise ValueError('frequency-band fine-patch length must be positive')
        self.patch_len = patch_len
        self.d_model = d_model
        self.fine_patch_len = int(fine_patch_len)
        self.widths = widths
        commits = (patch_len + fine_patch_len - 1) // fine_patch_len
        starts = (0, min(1, patch_len), min(commits, patch_len),
                  min(8 * commits, patch_len))
        stops = starts[1:] + (patch_len,)
        self.bands = tuple(
            (name, start, stop)
            for name, start, stop in zip(
                ("level", "macro", "patch_scale", "fine"), starts, stops)
            if stop > start
        )
        self.projections = nn.ModuleList(
            nn.Linear(stop - start, w, bias=True)
            for (_, start, stop), w in zip(self.bands, widths)
        )
        # Imported lazily: ScaleSelectiveHeadsAR imports this module, so a
        # module-level import here would form a cycle.
        from models.ScaleSelectiveHeadsAR import dct_basis  # noqa: PLC0415
        self.register_buffer(
            'basis', dct_basis(patch_len, patch_len), persistent=True)

    @property
    def lossless(self):
        return all(
            w >= stop - start
            for (_, start, stop), w in zip(self.bands, self.widths))

    def forward(self, parent_patches):
        if parent_patches.size(-1) != self.patch_len:
            raise ValueError(
                f'expected parent patches of length {self.patch_len}')
        basis = self.basis.to(parent_patches)
        coefficients = parent_patches @ basis
        outputs = [
            projection(coefficients[..., start:stop])
            for (_, start, stop), projection in zip(
                self.bands, self.projections)
        ]
        return torch.cat(outputs, dim=-1)


class FactorizedAtomOutputHead(nn.Module):
    """Decode a variable number of fine patches with fixed parameter count."""

    def __init__(self, d_model, patch_len, fine_patch_len, basis_dim=8):
        super().__init__()
        if fine_patch_len < 1 or patch_len % fine_patch_len:
            raise ValueError(
                'factorized output fine-patch length must divide patch_len')
        if basis_dim < 1:
            raise ValueError('factorized output basis dimension must be positive')
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.fine_patch_len = int(fine_patch_len)
        self.basis_dim = int(basis_dim)
        self.slots = self.patch_len // self.fine_patch_len
        if self.basis_dim < self.slots:
            raise ValueError(
                'factorized output basis dimension must be at least the '
                f'number of fine-patch slots ({self.slots})')
        positions = (
            torch.arange(self.slots, dtype=torch.float32) + 0.5
        ) / self.slots
        frequencies = torch.arange(
            self.basis_dim, dtype=torch.float32)
        coefficients = torch.cos(
            math.pi * positions[:, None] * frequencies[None, :])
        if self.basis_dim > 1:
            coefficients[:, 1:] *= math.sqrt(2.0)
        self.register_buffer(
            'position_coefficients', coefficients, persistent=True)
        self.basis_weights = nn.Parameter(torch.empty(
            self.basis_dim, self.fine_patch_len, self.d_model))
        self.basis_biases = nn.Parameter(torch.zeros(
            self.basis_dim, self.fine_patch_len))
        self.register_parameter(
            'protected_scale_weight_coordinates', None)
        self.register_parameter(
            'protected_scale_bias_coordinates', None)
        self.register_buffer(
            'protected_scale_null_basis', None, persistent=True)
        self.protected_scale_slots = None
        self.register_parameter(
            'refinement_scale_weight_coordinates', None)
        self.register_parameter(
            'refinement_scale_bias_coordinates', None)
        self.register_buffer(
            'refinement_scale_null_basis', None, persistent=True)
        self.refinement_scale_protected_slots = ()
        self.scale = 1.0 / math.sqrt(self.basis_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for basis_weight in self.basis_weights:
            nn.init.xavier_uniform_(basis_weight)
        nn.init.zeros_(self.basis_biases)

    def effective_weight(self):
        basis_weights, _ = self._active_basis_parameters()
        weights = self.scale * torch.einsum(
            'km,mpd->kpd',
            self.position_coefficients.to(self.basis_weights.dtype),
            basis_weights,
        )
        return weights.reshape(self.patch_len, self.d_model)

    def effective_bias(self):
        _, basis_biases = self._active_basis_parameters()
        biases = self.scale * torch.einsum(
            'km,mp->kp',
            self.position_coefficients.to(self.basis_biases.dtype),
            basis_biases,
        )
        return biases.reshape(self.patch_len)

    def initialize_from_native(self, head):
        if not isinstance(head, nn.Linear):
            raise TypeError('native output head must be nn.Linear')
        if (
                head.in_features != self.d_model
                or head.out_features != self.patch_len):
            raise ValueError('native output-head shape does not match')
        coefficient_inverse = torch.linalg.pinv(
            self.position_coefficients.float())
        target_weights = head.weight.detach().float().reshape(
            self.slots, self.fine_patch_len, self.d_model)
        target_biases = (
            torch.zeros(
                self.slots, self.fine_patch_len, dtype=torch.float32)
            if head.bias is None
            else head.bias.detach().float().reshape(
                self.slots, self.fine_patch_len)
        )
        basis_weights = torch.einsum(
            'mk,kpd->mpd', coefficient_inverse, target_weights
        ) / self.scale
        basis_biases = torch.einsum(
            'mk,kp->mp', coefficient_inverse, target_biases
        ) / self.scale
        with torch.no_grad():
            self.basis_weights.copy_(
                basis_weights.to(self.basis_weights.dtype))
            self.basis_biases.copy_(
                basis_biases.to(self.basis_biases.dtype))

    def enable_protected_scale_residual(self, protected_slots):
        """Add output-basis directions invisible at ``protected_slots``."""
        protected_slots = int(protected_slots)
        if not 1 <= protected_slots < self.basis_dim:
            raise ValueError(
                'protected slots must be positive and smaller than basis_dim')
        if self.protected_scale_weight_coordinates is not None:
            raise RuntimeError('protected scale residual is already enabled')
        null_basis = FactorizedPatchProjection._null_basis(
            protected_slots,
            self.basis_dim,
            self.basis_weights.device,
            self.basis_weights.dtype,
        )
        rank = null_basis.size(1)
        self.protected_scale_null_basis = null_basis
        self.protected_scale_slots = protected_slots
        self.protected_scale_weight_coordinates = nn.Parameter(torch.zeros(
            rank, self.fine_patch_len, self.d_model,
            device=self.basis_weights.device,
            dtype=self.basis_weights.dtype,
        ))
        self.protected_scale_bias_coordinates = nn.Parameter(torch.zeros(
            rank, self.fine_patch_len,
            device=self.basis_biases.device,
            dtype=self.basis_biases.dtype,
        ))
        return (
            self.protected_scale_weight_coordinates,
            self.protected_scale_bias_coordinates,
        )

    def enable_refinement_scale_residual(self, protected_slots):
        """Add a later output residual invisible to protected scales."""
        protected_slots = tuple(sorted(set(map(int, protected_slots))))
        if not protected_slots or protected_slots[-1] >= self.basis_dim:
            raise ValueError(
                'protected slots must be positive and smaller than basis_dim')
        if protected_slots[0] < 1:
            raise ValueError('protected slots must be positive')
        if self.refinement_scale_weight_coordinates is not None:
            raise RuntimeError('refinement scale residual is already enabled')
        null_basis = FactorizedPatchProjection._joint_null_basis(
            protected_slots,
            self.basis_dim,
            self.basis_weights.device,
            self.basis_weights.dtype,
        )
        rank = null_basis.size(1)
        if rank == 0:
            raise ValueError('protected scales leave no refinement direction')
        self.refinement_scale_null_basis = null_basis
        self.refinement_scale_protected_slots = protected_slots
        self.refinement_scale_weight_coordinates = nn.Parameter(torch.zeros(
            rank, self.fine_patch_len, self.d_model,
            device=self.basis_weights.device,
            dtype=self.basis_weights.dtype,
        ))
        self.refinement_scale_bias_coordinates = nn.Parameter(torch.zeros(
            rank, self.fine_patch_len,
            device=self.basis_biases.device,
            dtype=self.basis_biases.dtype,
        ))
        return (
            self.refinement_scale_weight_coordinates,
            self.refinement_scale_bias_coordinates,
        )

    def _active_basis_parameters(self):
        active_weights = self.basis_weights
        active_biases = self.basis_biases
        if self.protected_scale_weight_coordinates is not None \
                and self.slots > self.protected_scale_slots:
            null_basis = self.protected_scale_null_basis.to(
                self.basis_weights.dtype)
            active_weights = active_weights + torch.einsum(
                'mr,rpd->mpd',
                null_basis,
                self.protected_scale_weight_coordinates,
            )
            active_biases = active_biases + torch.einsum(
                'mr,rp->mp',
                null_basis,
                self.protected_scale_bias_coordinates,
            )
        if self.refinement_scale_weight_coordinates is not None \
                and self.slots > max(
                    self.refinement_scale_protected_slots):
            null_basis = self.refinement_scale_null_basis.to(
                self.basis_weights.dtype)
            active_weights = active_weights + torch.einsum(
                'mr,rpd->mpd',
                null_basis,
                self.refinement_scale_weight_coordinates,
            )
            active_biases = active_biases + torch.einsum(
                'mr,rp->mp',
                null_basis,
                self.refinement_scale_bias_coordinates,
            )
        return active_weights, active_biases

    def forward(self, hidden):
        basis_weights, basis_biases = self._active_basis_parameters()
        outputs = self.scale * torch.einsum(
            '...d,km,mpd->...kp',
            hidden,
            self.position_coefficients.to(hidden.dtype),
            basis_weights,
        )
        biases = self.scale * torch.einsum(
            'km,mp->kp',
            self.position_coefficients.to(hidden.dtype),
            basis_biases,
        )
        return (outputs + biases).reshape(
            *hidden.shape[:-1], self.patch_len)


class FineSlotOutputHead(nn.Module):
    """Decode one canonical parent patch from ordered hidden subspaces.

    The module changes only the output parameterization.  Every call still
    returns one complete parent patch, so AR commit width and DPOD semantics
    are unchanged.  ``split_haar`` applies a fixed orthogonal DCT mixing over
    slot coordinates before the same independent heads; it is the alignment
    control for a direct input-slot/output-slot correspondence.
    """

    MODES = {'split_independent', 'split_shared', 'split_haar'}

    def __init__(self, d_model, patch_len, fine_patch_len, mode):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f'fine-slot output mode must be one of {sorted(self.MODES)}')
        if fine_patch_len < 1 or patch_len % fine_patch_len:
            raise ValueError(
                'output fine-patch length must divide the parent patch')
        self.mode = mode
        self.patch_len = int(patch_len)
        self.fine_patch_len = int(fine_patch_len)
        self.slots = self.patch_len // self.fine_patch_len
        if self.slots < 2:
            raise ValueError('fine-slot output requires at least two slots')
        if d_model % self.slots:
            raise ValueError(
                'd_model must be divisible by the number of output slots')
        self.slot_dim = d_model // self.slots
        if mode == 'split_shared':
            self.shared_head = nn.Linear(
                self.slot_dim, self.fine_patch_len)
            self.heads = None
        else:
            self.shared_head = None
            self.heads = nn.ModuleList([
                nn.Linear(self.slot_dim, self.fine_patch_len)
                for _ in range(self.slots)
            ])

        if mode == 'split_haar':
            # Orthonormal DCT-II works for two slots (the Haar sum/difference
            # basis) and remains a deterministic alignment control for P48+
            # without consuming RNG state.
            slot = torch.arange(self.slots, dtype=torch.float32)
            frequency = slot[:, None]
            mixing = torch.cos(
                math.pi / self.slots
                * (slot[None, :] + 0.5)
                * frequency
            )
            mixing[0] *= 1.0 / math.sqrt(self.slots)
            if self.slots > 1:
                mixing[1:] *= math.sqrt(2.0 / self.slots)
            self.register_buffer(
                'slot_mixing', mixing, persistent=True)
        else:
            self.register_buffer(
                'slot_mixing', None, persistent=False)

    def slot_states(self, hidden):
        states = hidden.reshape(
            *hidden.shape[:-1], self.slots, self.slot_dim)
        if self.slot_mixing is not None:
            states = torch.einsum(
                'ij,...jd->...id',
                self.slot_mixing.to(states.dtype),
                states,
            )
        return states

    def forward(self, hidden):
        states = self.slot_states(hidden)
        if self.shared_head is not None:
            decoded = self.shared_head(states)
            return decoded.reshape(*hidden.shape[:-1], self.patch_len)
        return torch.cat([
            head(states[..., slot, :])
            for slot, head in enumerate(self.heads)
        ], dim=-1)


class StaticVariableMixer(nn.Module):
    """Protected per-patch interaction across variable hidden states.

    DenseAR keeps channels in a flattened ``[B*C, T, D]`` layout.  This
    module temporarily restores ``[B, C, T, D]``.  Most modes apply the same
    static variable operator at every patch position; ``dynamic_slots`` uses
    content-conditioned latent summaries without variable identities.  The
    zero residual gate makes every topology exactly equal to the
    channel-independent parent at initialization.
    """

    MODES = {
        'full', 'row_softmax', 'low_rank',
        'low_rank_orthogonal', 'mean', 'dynamic_slots'}
    SOURCES = {'state', 'temporal_innovation', 'channel_centered'}
    PLACEMENTS = {'pre_attention', 'post_attention', 'post_ffn'}
    PATTERNS = {'first', 'last', 'all_shared', 'all_independent'}

    def __init__(self, channels, layers, mode, source, placement, pattern,
                 rank=8, gate_init=0.0, gate_limit=1.0, hidden_dim=None):
        super().__init__()
        if channels < 1:
            raise ValueError('variable mixer requires at least one channel')
        if layers < 1:
            raise ValueError('variable mixer requires at least one layer')
        if mode not in self.MODES:
            raise ValueError(
                f'variable mixer mode must be one of {sorted(self.MODES)}')
        if source not in self.SOURCES:
            raise ValueError(
                f'variable mixer source must be one of '
                f'{sorted(self.SOURCES)}')
        if placement not in self.PLACEMENTS:
            raise ValueError(
                f'variable mixer placement must be one of '
                f'{sorted(self.PLACEMENTS)}')
        if pattern not in self.PATTERNS:
            raise ValueError(
                f'variable mixer pattern must be one of '
                f'{sorted(self.PATTERNS)}')
        if rank < 1:
            raise ValueError('variable mixer rank must be positive')
        if not -1.0 < gate_init < 1.0:
            raise ValueError('variable mixer gate init must be in (-1, 1)')
        if not 0.0 < gate_limit <= 1.0:
            raise ValueError('variable mixer gate limit must be in (0, 1]')

        self.channels = int(channels)
        self.layers = int(layers)
        self.mode = mode
        self.source = source
        self.placement = placement
        self.pattern = pattern
        self.rank = min(int(rank), self.channels)
        self.hidden_dim = (
            None if hidden_dim is None else int(hidden_dim))
        if self.mode == 'dynamic_slots' \
                and (self.hidden_dim is None or self.hidden_dim < 1):
            raise ValueError(
                'dynamic slot mixing requires a positive hidden_dim')
        self.parameter_sets = (
            self.layers if pattern == 'all_independent' else 1)
        self.gates = nn.Parameter(torch.full(
            (self.parameter_sets,), math.atanh(gate_init)))
        self.register_buffer(
            'gate_limit', torch.tensor(float(gate_limit)), persistent=True)
        # Do not advance the experiment-global RNG: matched parent and mixer
        # runs must see the same dropout/data stochasticity after model build.
        initialization_generator = torch.Generator(device='cpu')
        initialization_generator.manual_seed(1729)

        if mode in {'full', 'row_softmax'}:
            self.weights = nn.Parameter(torch.empty(
                self.parameter_sets, self.channels, self.channels))
            nn.init.normal_(
                self.weights, mean=0.0,
                std=1.0 / math.sqrt(max(self.channels, 1)),
                generator=initialization_generator)
            self.register_parameter('left_factors', None)
            self.register_parameter('right_factors', None)
            self.register_parameter('singular_values', None)
            self.register_parameter('route_queries', None)
            self.register_parameter('route_keys', None)
        elif mode in {'low_rank', 'low_rank_orthogonal'}:
            self.register_parameter('weights', None)
            self.left_factors = nn.Parameter(torch.empty(
                self.parameter_sets, self.channels, self.rank))
            self.right_factors = nn.Parameter(torch.empty(
                self.parameter_sets, self.channels, self.rank))
            nn.init.normal_(
                self.left_factors, mean=0.0,
                std=1.0 / math.sqrt(max(self.rank, 1)),
                generator=initialization_generator)
            nn.init.normal_(
                self.right_factors, mean=0.0,
                std=1.0 / math.sqrt(max(self.channels, 1)),
                generator=initialization_generator)
            if mode == 'low_rank_orthogonal':
                self.singular_values = nn.Parameter(torch.full(
                    (self.parameter_sets, self.rank), math.atanh(0.5)))
            else:
                self.register_parameter('singular_values', None)
            self.register_parameter('route_queries', None)
            self.register_parameter('route_keys', None)
        elif mode == 'dynamic_slots':
            self.register_parameter('weights', None)
            self.register_parameter('left_factors', None)
            self.register_parameter('right_factors', None)
            self.register_parameter('singular_values', None)
            self.route_queries = nn.Parameter(torch.empty(
                self.parameter_sets, self.hidden_dim, self.rank))
            self.route_keys = nn.Parameter(torch.empty(
                self.parameter_sets, self.hidden_dim, self.rank))
            route_scale = 1.0 / math.sqrt(self.hidden_dim)
            nn.init.normal_(
                self.route_queries,
                mean=0.0,
                std=route_scale,
                generator=initialization_generator,
            )
            nn.init.normal_(
                self.route_keys,
                mean=0.0,
                std=route_scale,
                generator=initialization_generator,
            )
        else:
            self.register_parameter('weights', None)
            self.register_parameter('left_factors', None)
            self.register_parameter('right_factors', None)
            self.register_parameter('singular_values', None)
            self.register_parameter('route_queries', None)
            self.register_parameter('route_keys', None)

        self.register_buffer(
            'off_diagonal',
            ~torch.eye(self.channels, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            'active_channel_indices', None, persistent=False)

    def set_active_channel_indices(self, indices):
        """Retain original variable identities for a sampled channel batch."""
        if indices is None:
            self.active_channel_indices = None
            return
        indices = indices.detach().to(
            device=self.gates.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError(
                'active channel indices must be a non-empty vector')
        if indices.min().item() < 0 or indices.max().item() >= self.channels:
            raise ValueError('active channel index is out of range')
        if indices.unique().numel() != indices.numel():
            raise ValueError('active channel indices must be unique')
        self.active_channel_indices = indices

    def _parameter_index(self, layer_index):
        return layer_index if self.pattern == 'all_independent' else 0

    def _active(self, layer_index):
        if self.pattern == 'first':
            return layer_index == 0
        if self.pattern == 'last':
            return layer_index == self.layers - 1
        return True

    def _source_values(self, states):
        if self.source == 'state':
            return states
        if self.source == 'channel_centered':
            return states - states.mean(dim=1, keepdim=True)
        innovations = torch.zeros_like(states)
        innovations[:, :, 1:] = (
            states[:, :, 1:] - states[:, :, :-1])
        return innovations

    def _correction(self, values, parameter_index, channel_indices):
        active_channels = values.size(1)
        if self.mode == 'dynamic_slots':
            if active_channels == 1:
                return torch.zeros_like(values)
            # Content-dependent, permutation-equivariant variable routing.
            # Each latent slot is a normalized summary over variables.  The
            # receiver's own contribution is removed before it reads the
            # slots, so the branch cannot collapse to a hidden self-MLP.
            query_logits = torch.einsum(
                'bctd,dr->bctr',
                values,
                self.route_queries[parameter_index],
            )
            key_logits = torch.einsum(
                'bctd,dr->bctr',
                values,
                self.route_keys[parameter_index],
            )
            key_weights = torch.softmax(
                key_logits.float(), dim=1).to(values.dtype)
            query_weights = torch.softmax(
                query_logits.float(), dim=-1).to(values.dtype)
            slots = torch.einsum(
                'bctr,bctd->btrd', key_weights, values)
            self_contribution = (
                key_weights[..., None] * values[..., None, :])
            denominator = (1.0 - key_weights).clamp_min(1e-4)
            leave_one_out = (
                slots[:, None] - self_contribution
            ) / denominator[..., None]
            return torch.einsum(
                'bctr,bctrd->bctd', query_weights, leave_one_out)
        if self.mode == 'full':
            weight = self.weights[parameter_index]
            if channel_indices is not None:
                weight = weight.index_select(
                    0, channel_indices).index_select(1, channel_indices)
            weight = weight.masked_fill(
                torch.eye(
                    active_channels,
                    device=weight.device,
                    dtype=torch.bool,
                ),
                0.0,
            )
            return torch.einsum('ij,bjtd->bitd', weight, values)
        if self.mode == 'row_softmax':
            if active_channels == 1:
                return torch.zeros_like(values)
            logits = self.weights[parameter_index]
            if channel_indices is not None:
                logits = logits.index_select(
                    0, channel_indices).index_select(1, channel_indices)
            logits = logits.masked_fill(
                torch.eye(
                    active_channels,
                    device=logits.device,
                    dtype=torch.bool,
                ),
                -torch.inf,
            )
            weight = torch.softmax(logits, dim=-1)
            return torch.einsum('ij,bjtd->bitd', weight, values)
        if self.mode == 'low_rank':
            right = self.right_factors[parameter_index]
            left = self.left_factors[parameter_index]
            if channel_indices is not None:
                right = right.index_select(0, channel_indices)
                left = left.index_select(0, channel_indices)
            factors = torch.einsum('jr,bjtd->brtd', right, values)
            return torch.einsum('ir,brtd->bitd', left, factors)
        if self.mode == 'low_rank_orthogonal':
            left = torch.linalg.qr(
                self.left_factors[parameter_index],
                mode='reduced').Q
            right = torch.linalg.qr(
                self.right_factors[parameter_index],
                mode='reduced').Q
            if channel_indices is not None:
                left = left.index_select(0, channel_indices)
                right = right.index_select(0, channel_indices)
            singular = torch.tanh(
                self.singular_values[parameter_index])
            factors = torch.einsum('jr,bjtd->brtd', right, values)
            factors = factors * singular[None, :, None, None]
            return torch.einsum('ir,brtd->bitd', left, factors)
        if active_channels == 1:
            return torch.zeros_like(values)
        total = values.sum(dim=1, keepdim=True)
        return (total - values) / (active_channels - 1)

    def forward(self, layer_index, stage, hidden):
        if stage != self.placement or not self._active(layer_index):
            return hidden
        channel_indices = self.active_channel_indices
        active_channels = (
            self.channels
            if channel_indices is None
            else channel_indices.numel()
        )
        if hidden.size(0) % active_channels:
            raise ValueError(
                'channel-batched hidden states do not align with enc_in')
        batch_size = hidden.size(0) // active_channels
        states = hidden.view(
            batch_size,
            active_channels,
            hidden.size(1),
            hidden.size(2),
        )
        parameter_index = self._parameter_index(layer_index)
        values = self._source_values(states)
        correction = self._correction(
            values, parameter_index, channel_indices)
        gate = (
            self.gate_limit
            * torch.tanh(self.gates[parameter_index])
        ).to(hidden.dtype)
        return (states + gate * correction).reshape_as(hidden)


class Model(DenseARModel):
    """Channel-independent dense patch AR with fully ablatable Transformer."""

    def __init__(self, configs):
        super().__init__(configs)
        max_tokens = (
            self.seq_len // self.patch_len
            + max(self.roll_patches - 1, 0)
        )
        self.backbone = AblationPatchARBackbone(
            patch_len=self.patch_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            attention_mode=getattr(
                configs, 'ablation_attention', 'softmax'),
            ffn_mode=getattr(configs, 'ablation_ffn', 'mlp'),
            norm_type=getattr(configs, 'ablation_norm', 'post'),
            position_encoding=getattr(
                configs, 'ablation_position_encoding', 'rope'),
            attention_residual=bool(getattr(
                configs, 'ablation_attention_residual', True)),
            ffn_residual=bool(getattr(
                configs, 'ablation_ffn_residual', True)),
            max_tokens=max_tokens,
            shrinkage_type=getattr(
                configs, 'ablation_shrinkage_type', 'fixed'),
            shrinkage_alpha=float(getattr(
                configs, 'ablation_shrinkage_alpha', 1.0)),
            dynamic_heads=getattr(
                configs, 'ablation_dynamic_heads', None),
            qk_norm=getattr(
                configs, 'ablation_qk_norm', 'dot'),
            attention_temperature=float(getattr(
                configs, 'ablation_attention_temperature', 1.0)),
            attention_softcap=float(getattr(
                configs, 'ablation_attention_softcap', 0.0)),
            learnable_temperature=bool(getattr(
                configs, 'ablation_learnable_temperature', False)),
            norm_kind=getattr(
                configs, 'ablation_norm_kind', 'layer'),
            layer_scale_init=(
                None
                if float(getattr(
                    configs, 'ablation_layer_scale_init', -1.0)) < 0
                else float(getattr(
                    configs, 'ablation_layer_scale_init', -1.0))
            ),
            local_kernel=int(getattr(
                configs, 'ablation_local_kernel', 0)),
            local_bias=not bool(getattr(
                configs, 'ablation_local_no_bias', False)),
            local_mode=getattr(
                configs, 'ablation_local_mode', 'state'),
            value_topology=getattr(
                configs, 'ablation_value_topology', 'state'),
            self_logit_bias=float(getattr(
                configs, 'ablation_self_logit_bias', 0.0)),
            self_edge_heads=int(getattr(
                configs, 'ablation_self_edge_heads', -1)),
            self_history_mode=getattr(
                configs, 'ablation_self_history_mode', 'joint'),
            self_history_init=float(getattr(
                configs, 'ablation_self_history_init', 0.5)),
            self_history_pattern=getattr(
                configs, 'ablation_self_history_pattern', 'all'),
            qk_input_mode=getattr(
                configs, 'ablation_qk_input_mode', 'state'),
            head_sharing=getattr(
                configs, 'ablation_head_sharing', 'none'),
            attention_window=int(getattr(
                configs, 'ablation_attention_window', 0)),
            coarse_memory_group=int(getattr(
                configs, 'ablation_coarse_memory_group', 0)),
            coarse_memory_gate_init=float(getattr(
                configs, 'ablation_coarse_memory_gate_init', 0.0)),
            address_geometry=getattr(
                configs, 'ablation_address_geometry', 'single'),
            geometry_gate_init=float(getattr(
                configs, 'ablation_geometry_gate_init', 0.0)),
            dyadic_router_mode=getattr(
                configs, 'ablation_dyadic_router_mode', 'none'),
            dyadic_router_gate_init=float(getattr(
                configs, 'ablation_dyadic_router_gate_init', 0.0)),
        )
        self.cross_channel_mixer = bool(getattr(
            configs, 'ablation_cross_channel_mixer', False))
        self.channel_specific_head = bool(getattr(
            configs, 'ablation_channel_specific_head', False))
        self.raw_patch_skip_kernel = int(getattr(
            configs, 'ablation_raw_patch_skip_kernel', 0))
        self.raw_patch_skip_mode = getattr(
            configs, 'ablation_raw_patch_skip_mode', 'full')
        if self.raw_patch_skip_mode not in {
                'full', 'phase', 'dyadic_phase'}:
            raise ValueError(
                'raw patch skip mode must be full, phase, or dyadic_phase')
        if self.raw_patch_skip_kernel < 0:
            raise ValueError(
                'raw patch skip kernel cannot be negative')
        if self.raw_patch_skip_kernel:
            if self.roll_patches != 1:
                raise ValueError(
                    'raw patch skip currently requires Q1 predictions')
            self.raw_patch_skip = nn.Conv1d(
                self.patch_len,
                self.patch_len,
                self.raw_patch_skip_kernel,
                groups=(
                    self.patch_len
                    if self.raw_patch_skip_mode in {
                        'phase', 'dyadic_phase'
                    }
                    else 1
                ),
            )
            nn.init.zeros_(self.raw_patch_skip.weight)
            nn.init.zeros_(self.raw_patch_skip.bias)
            if self.raw_patch_skip_mode == 'dyadic_phase':
                skip_mask = torch.zeros_like(
                    self.raw_patch_skip.weight)
                causal_lags = {
                    (1 << scale) - 1
                    for scale in range(
                        self.raw_patch_skip_kernel.bit_length())
                    if (1 << scale) - 1
                    < self.raw_patch_skip_kernel
                }
                for lag in causal_lags:
                    skip_mask[
                        ..., self.raw_patch_skip_kernel - 1 - lag
                    ] = 1.0
                self.register_buffer(
                    'raw_patch_skip_mask',
                    skip_mask,
                    persistent=False,
                )
            else:
                self.register_buffer(
                    'raw_patch_skip_mask',
                    None,
                    persistent=False,
                )
        else:
            self.raw_patch_skip = None
            self.register_buffer(
                'raw_patch_skip_mask',
                None,
                persistent=False,
            )
        recurrent_state_init = float(getattr(
            configs, 'ablation_recurrent_state_gate_init', -2.0))
        self.recurrent_state_path = recurrent_state_init > -1.0
        if self.recurrent_state_path:
            if not -1.0 < recurrent_state_init < 1.0:
                raise ValueError(
                    'recurrent state gate init must be in (-1, 1)')
            if self.roll_patches != 1:
                raise ValueError(
                    'recurrent state path currently requires Q1 predictions')
            self.context_state_gru = nn.GRU(
                configs.d_model,
                configs.d_model,
                batch_first=True,
            )
            self.context_state_norm = nn.LayerNorm(
                configs.d_model)
            self.recurrent_state_gate = nn.Parameter(torch.tensor(
                math.atanh(recurrent_state_init)))
        else:
            self.context_state_gru = None
            self.context_state_norm = None
            self.register_parameter(
                'recurrent_state_gate', None)
        patch_shape_init = float(getattr(
            configs, 'ablation_patch_shape_gate_init', -2.0))
        patch_moment_init = float(getattr(
            configs, 'ablation_patch_moment_gate_init', -2.0))
        patch_shape_address_init = float(getattr(
            configs, 'ablation_patch_shape_address_gate_init', -2.0))
        patch_shape_value_init = float(getattr(
            configs, 'ablation_patch_shape_value_gate_init', -2.0))
        patch_shape_sidecar_init = float(getattr(
            configs, 'ablation_patch_shape_sidecar_gate_init', -2.0))
        patch_moment_sidecar_init = float(getattr(
            configs, 'ablation_patch_moment_sidecar_gate_init', -2.0))
        causal_low_value_init = float(getattr(
            configs, 'ablation_causal_low_value_gate_init', -2.0))
        causal_detail_value_init = float(getattr(
            configs, 'ablation_causal_detail_value_gate_init', -2.0))
        self.patch_shape_path = patch_shape_init > -1.0
        self.patch_moment_path = patch_moment_init > -1.0
        self.patch_shape_address_path = (
            patch_shape_address_init > -1.0)
        self.patch_shape_value_path = (
            patch_shape_value_init > -1.0)
        self.patch_shape_sidecar_path = (
            patch_shape_sidecar_init > -1.0)
        self.patch_moment_sidecar_path = (
            patch_moment_sidecar_init > -1.0)
        self.causal_low_value_path = causal_low_value_init > -1.0
        self.causal_detail_value_path = (
            causal_detail_value_init > -1.0)
        if self.patch_shape_path or self.patch_moment_path:
            if self.roll_patches != 1:
                raise ValueError(
                    'factorized patch embedding currently requires Q1 '
                    'predictions')
        if self.patch_shape_path:
            if not -1.0 < patch_shape_init < 1.0:
                raise ValueError(
                    'patch shape gate init must be in (-1, 1)')
            self.patch_shape_projection = nn.Linear(
                self.patch_len,
                configs.d_model,
                bias=False,
            )
            self.patch_shape_norm = nn.LayerNorm(configs.d_model)
            self.patch_shape_gate = nn.Parameter(torch.tensor(
                math.atanh(patch_shape_init)))
        else:
            self.patch_shape_projection = None
            self.patch_shape_norm = None
            self.register_parameter('patch_shape_gate', None)
        if self.patch_moment_path:
            if not -1.0 < patch_moment_init < 1.0:
                raise ValueError(
                    'patch moment gate init must be in (-1, 1)')
            self.patch_moment_projection = nn.Linear(
                2,
                configs.d_model,
                bias=False,
            )
            self.patch_moment_norm = nn.LayerNorm(configs.d_model)
            self.patch_moment_gate = nn.Parameter(torch.tensor(
                math.atanh(patch_moment_init)))
        else:
            self.patch_moment_projection = None
            self.patch_moment_norm = None
            self.register_parameter('patch_moment_gate', None)
        if self.patch_shape_address_path:
            if not -1.0 < patch_shape_address_init < 1.0:
                raise ValueError(
                    'patch shape address gate init must be in (-1, 1)')
            if self.roll_patches != 1:
                raise ValueError(
                    'shape-addressed attention currently requires Q1')
            if configs.e_layers != 1:
                raise ValueError(
                    'shape-addressed attention currently requires one layer')
            if getattr(configs, 'ablation_norm', 'post') == 'pre':
                raise ValueError(
                    'shape-addressed attention does not support pre-norm')
            if getattr(
                    configs, 'ablation_address_geometry', 'single'
                    ) != 'single':
                raise ValueError(
                    'shape-addressed attention requires single address '
                    'geometry')
            if getattr(
                    configs, 'ablation_qk_input_mode', 'state') != 'state':
                raise ValueError(
                    'shape-addressed attention requires state QK mode')
            self.patch_shape_address_projection = nn.Linear(
                self.patch_len,
                configs.d_model,
                bias=False,
            )
            self.patch_shape_address_norm = nn.LayerNorm(
                configs.d_model)
            self.patch_shape_address_gate = nn.Parameter(torch.tensor(
                math.atanh(patch_shape_address_init)))
        else:
            self.patch_shape_address_projection = None
            self.patch_shape_address_norm = None
            self.register_parameter(
                'patch_shape_address_gate', None)
        if self.patch_shape_value_path:
            if not -1.0 < patch_shape_value_init < 1.0:
                raise ValueError(
                    'patch shape value gate init must be in (-1, 1)')
            if self.roll_patches != 1:
                raise ValueError(
                    'shape-value attention currently requires Q1')
            if configs.e_layers != 1:
                raise ValueError(
                    'shape-value attention currently requires one layer')
            if getattr(configs, 'ablation_norm', 'post') == 'pre':
                raise ValueError(
                    'shape-value attention does not support pre-norm')
            if getattr(
                    configs, 'ablation_address_geometry', 'single'
                    ) != 'single':
                raise ValueError(
                    'shape-value attention requires single address '
                    'geometry')
            if getattr(
                    configs, 'ablation_qk_input_mode', 'state') != 'state':
                raise ValueError(
                    'shape-value attention requires state QK mode')
            self.patch_shape_value_projection = nn.Linear(
                self.patch_len,
                configs.d_model,
                bias=False,
            )
            self.patch_shape_value_norm = nn.LayerNorm(
                configs.d_model)
            self.patch_shape_value_gate = nn.Parameter(torch.tensor(
                math.atanh(patch_shape_value_init)))
        else:
            self.patch_shape_value_projection = None
            self.patch_shape_value_norm = None
            self.register_parameter(
                'patch_shape_value_gate', None)
        if self.patch_shape_sidecar_path \
                or self.patch_moment_sidecar_path:
            if self.roll_patches != 1:
                raise ValueError(
                    'patch value sidecars currently require Q1')
            if configs.e_layers != 1:
                raise ValueError(
                    'patch value sidecars currently require one layer')
            if getattr(configs, 'ablation_norm', 'post') == 'pre':
                raise ValueError(
                    'patch value sidecars do not support pre-norm')
            if getattr(
                    configs, 'ablation_value_topology', 'state'
                    ) != 'state':
                raise ValueError(
                    'patch value sidecars require state values')
            if getattr(
                    configs, 'ablation_head_sharing', 'none'
                    ) in {'value', 'key_value'}:
                raise ValueError(
                    'patch value sidecars do not support shared V heads')
        if self.patch_shape_sidecar_path:
            if not -1.0 < patch_shape_sidecar_init < 1.0:
                raise ValueError(
                    'patch shape sidecar gate init must be in (-1, 1)')
            self.patch_shape_sidecar_projection = nn.Linear(
                self.patch_len,
                configs.d_model,
                bias=False,
            )
            self.patch_shape_sidecar_norm = nn.LayerNorm(
                configs.d_model)
            self.patch_shape_sidecar_gate = nn.Parameter(torch.tensor(
                math.atanh(patch_shape_sidecar_init)))
        else:
            self.patch_shape_sidecar_projection = None
            self.patch_shape_sidecar_norm = None
            self.register_parameter(
                'patch_shape_sidecar_gate', None)
        if self.patch_moment_sidecar_path:
            if not -1.0 < patch_moment_sidecar_init < 1.0:
                raise ValueError(
                    'patch moment sidecar gate init must be in (-1, 1)')
            self.patch_moment_sidecar_projection = nn.Linear(
                2,
                configs.d_model,
                bias=False,
            )
            self.patch_moment_sidecar_norm = nn.LayerNorm(
                configs.d_model)
            self.patch_moment_sidecar_gate = nn.Parameter(torch.tensor(
                math.atanh(patch_moment_sidecar_init)))
        else:
            self.patch_moment_sidecar_projection = None
            self.patch_moment_sidecar_norm = None
            self.register_parameter(
                'patch_moment_sidecar_gate', None)
        if self.causal_low_value_path \
                or self.causal_detail_value_path:
            if self.roll_patches != 1:
                raise ValueError(
                    'causal band value sidecars currently require Q1')
            if configs.e_layers != 1:
                raise ValueError(
                    'causal band value sidecars require one layer')
            if getattr(configs, 'ablation_norm', 'post') == 'pre':
                raise ValueError(
                    'causal band value sidecars do not support pre-norm')
            if getattr(
                    configs, 'ablation_value_topology', 'state'
                    ) != 'state':
                raise ValueError(
                    'causal band value sidecars require state values')
        if self.causal_low_value_path:
            if not -1.0 < causal_low_value_init < 1.0:
                raise ValueError(
                    'causal low value gate init must be in (-1, 1)')
            self.causal_low_value_projection = nn.Linear(
                configs.d_model,
                configs.d_model,
                bias=False,
            )
            self.causal_low_value_norm = nn.LayerNorm(
                configs.d_model)
            self.causal_low_value_gate = nn.Parameter(torch.tensor(
                math.atanh(causal_low_value_init)))
        else:
            self.causal_low_value_projection = None
            self.causal_low_value_norm = None
            self.register_parameter(
                'causal_low_value_gate', None)
        if self.causal_detail_value_path:
            if not -1.0 < causal_detail_value_init < 1.0:
                raise ValueError(
                    'causal detail value gate init must be in (-1, 1)')
            self.causal_detail_value_projection = nn.Linear(
                configs.d_model,
                configs.d_model,
                bias=False,
            )
            self.causal_detail_value_norm = nn.LayerNorm(
                configs.d_model)
            self.causal_detail_value_gate = nn.Parameter(torch.tensor(
                math.atanh(causal_detail_value_init)))
        else:
            self.causal_detail_value_projection = None
            self.causal_detail_value_norm = None
            self.register_parameter(
                'causal_detail_value_gate', None)
        persistence_init = float(getattr(
            configs, 'ablation_persistence_gate_init', -2.0))
        self.persistence_blend = persistence_init > -1.0
        if self.persistence_blend:
            if not -1.0 < persistence_init < 1.0:
                raise ValueError(
                    'persistence gate init must be in (-1, 1)')
            self.persistence_gate = nn.Parameter(torch.tensor(
                math.atanh(persistence_init)))
        else:
            self.register_parameter('persistence_gate', None)
        if self.channel_specific_head:
            if self.features != 'M':
                raise ValueError(
                    'channel-specific heads currently require M features')
            self.channel_head_weight = nn.Parameter(torch.zeros(
                self.enc_in,
                self.patch_len,
                self.patch_len,
            ))
            self.channel_head_bias = nn.Parameter(torch.zeros(
                self.enc_in, self.patch_len))
        else:
            self.register_parameter('channel_head_weight', None)
            self.register_parameter('channel_head_bias', None)
        if self.cross_channel_mixer:
            if self.features != 'M':
                raise ValueError(
                    'cross-channel mixing currently requires M features')
            self.cross_channel_weight = nn.Parameter(torch.zeros(
                self.enc_in, self.enc_in))
            self.register_buffer(
                'cross_channel_off_diagonal',
                ~torch.eye(self.enc_in, dtype=torch.bool),
                persistent=False,
            )
        else:
            self.register_parameter('cross_channel_weight', None)
            self.register_buffer(
                'cross_channel_off_diagonal',
                None,
                persistent=False,
            )
        self.patch_assembly = str(getattr(
            configs, 'ablation_patch_assembly', 'none'))
        self.patch_assembly_stem = str(getattr(
            configs, 'ablation_patch_assembly_stem', 'linear'))
        self.assembly_fine_patch_len = int(getattr(
            configs, 'ablation_patch_assembly_fine_len', 12))
        self.assembly_atom_dim = int(getattr(
            configs, 'ablation_patch_assembly_atom_dim', 0))
        self.assembly_basis_dim = int(getattr(
            configs, 'ablation_patch_assembly_basis_dim', 8))
        self.assembly_interface_dim = int(getattr(
            configs, 'ablation_patch_assembly_interface_dim', 0))
        if self.assembly_interface_dim < 0:
            raise ValueError(
                'assembly interface dimension must be non-negative')
        if self.patch_assembly not in {
                'none', 'chronological', 'reverse', 'mean_sorted',
                'factorized', 'fd'}:
            raise ValueError(
                'patch assembly must be none, chronological, reverse, '
                'mean_sorted, factorized, or fd')
        if self.patch_assembly_stem not in {
                'linear', 'diagonal', 'identity'}:
            raise ValueError(
                'patch assembly stem must be linear, diagonal, or identity')
        if self.assembly_atom_dim < 0:
            raise ValueError(
                'patch assembly atom dimension must be non-negative')
        if self.assembly_basis_dim < 1:
            raise ValueError(
                'patch assembly basis dimension must be positive')
        if self.patch_assembly == 'none':
            self.assembly_slots = 1
            self.assembly_fine_dim = configs.d_model
            self.assembly_fine_projection = None
            self.assembly_factorized_projection = None
        else:
            if self.assembly_fine_patch_len < 1:
                raise ValueError(
                    'assembly fine-patch length must be positive')
            if self.patch_len % self.assembly_fine_patch_len:
                raise ValueError(
                    'ar_patch_len must be divisible by the assembly '
                    'fine-patch length')
            self.assembly_slots = (
                self.patch_len // self.assembly_fine_patch_len)
            if self.patch_assembly == 'factorized':
                self.assembly_fine_dim = (
                    self.assembly_atom_dim
                    if self.assembly_atom_dim
                    else self.assembly_fine_patch_len
                )
                self.assembly_fine_projection = None
                self.assembly_factorized_projection = (
                    FactorizedPatchProjection(
                        patch_len=self.patch_len,
                        fine_patch_len=self.assembly_fine_patch_len,
                        atom_dim=self.assembly_fine_dim,
                        d_model=configs.d_model,
                        basis_dim=self.assembly_basis_dim,
                        stem=self.patch_assembly_stem,
                    )
                )
            elif self.patch_assembly == 'fd':
                raw_widths = str(getattr(
                    configs, 'ablation_patch_assembly_fd_widths', ''))
                self.assembly_fine_dim = configs.d_model
                self.assembly_fine_projection = None
                self.assembly_factorized_projection = None
                self.assembly_fd_projection = FrequencyBandProjection(
                    patch_len=self.patch_len,
                    d_model=configs.d_model,
                    widths=(
                        int(part)
                        for part in raw_widths.split(',') if part
                    ),
                    fine_patch_len=self.assembly_fine_patch_len,
                )
            elif self.assembly_atom_dim:
                expected_width = self.assembly_slots * self.assembly_atom_dim
                if (not self.assembly_interface_dim
                        and configs.d_model != expected_width):
                    raise ValueError(
                        'fixed patch assembly atom dimension requires '
                        f'd_model={expected_width}, got {configs.d_model}')
                self.assembly_fine_dim = self.assembly_atom_dim
                self.assembly_factorized_projection = None
            else:
                if (not self.assembly_interface_dim
                        and configs.d_model % self.assembly_slots):
                    raise ValueError(
                        'd_model must be divisible by the number of assembly '
                        'slots')
                self.assembly_fine_dim = (
                    configs.d_model // self.assembly_slots)
                self.assembly_factorized_projection = None
            if self.patch_assembly in {'factorized', 'fd'}:
                pass
            elif self.patch_assembly_stem in {'diagonal', 'identity'}:
                if (
                        self.assembly_fine_dim
                        != self.assembly_fine_patch_len):
                    raise ValueError(
                        f'{self.patch_assembly_stem} assembly requires '
                        'd_model / slots to '
                        'equal the fine-patch length')
                self.assembly_fine_projection = (
                    nn.Identity()
                    if self.patch_assembly_stem == 'identity'
                    else DiagonalAtomProjection(self.assembly_fine_dim)
                )
            else:
                self.assembly_fine_projection = nn.Linear(
                    self.assembly_fine_patch_len,
                    self.assembly_fine_dim,
                )
            # Concatenation already has d_model coordinates; an additional
            # parent-token projection would erase the direct packing test.
            # The interface mode instead keeps a fixed concat width and
            # projects it to d_model, so the stem width (and therefore the
            # per-dataset atom budget) stays decoupled from the backbone.
            if self.assembly_interface_dim:
                concat_width = (
                    self.assembly_slots * self.assembly_fine_dim)
                self.assembly_interface_projection = (
                    None
                    if concat_width == configs.d_model
                    else nn.Linear(concat_width, configs.d_model)
                )
            else:
                self.assembly_interface_projection = None
            self.backbone.patch_projection = nn.Identity()
            incompatible_paths = {
                'patch_shape': self.patch_shape_path,
                'patch_moment': self.patch_moment_path,
                'shape_address': self.patch_shape_address_path,
                'shape_value': self.patch_shape_value_path,
                'shape_sidecar': self.patch_shape_sidecar_path,
                'moment_sidecar': self.patch_moment_sidecar_path,
                'causal_low_value': self.causal_low_value_path,
                'causal_detail_value': self.causal_detail_value_path,
            }
            active = [
                name for name, enabled in incompatible_paths.items()
                if enabled
            ]
            if active:
                raise ValueError(
                    'direct patch assembly cannot be combined with '
                    f'projection-side branches: {active}')
        self.output_head_mode = str(getattr(
            configs, 'ablation_output_head', 'full'))
        self.output_fine_patch_len = int(getattr(
            configs, 'ablation_output_fine_patch_len', 12))
        self.output_basis_dim = int(getattr(
            configs, 'ablation_output_basis_dim', 8))
        if self.output_head_mode not in {
                'full', 'factorized_atoms', *FineSlotOutputHead.MODES}:
            raise ValueError(
                'ablation output head must be full, split_independent, '
                'split_shared, split_haar, or factorized_atoms')
        if self.output_head_mode != 'full':
            if self.structural_loss_weight > 0.0:
                raise ValueError(
                    'structured patch loss currently requires the full '
                    'output head')
            if self.output_head_mode == 'factorized_atoms':
                self.backbone.output_head = FactorizedAtomOutputHead(
                    configs.d_model,
                    self.patch_len,
                    self.output_fine_patch_len,
                    self.output_basis_dim,
                )
            else:
                self.backbone.output_head = FineSlotOutputHead(
                    configs.d_model,
                    self.patch_len,
                    self.output_fine_patch_len,
                    self.output_head_mode,
                )
        # Construct the optional mixer after every shared parent parameter.
        # This preserves identical parent initialization across matched runs
        # even though mixer topology changes its own random parameter count.
        variable_mixer_mode = str(getattr(
            configs, 'ablation_variable_mixer', 'none'))
        self.variable_mixer_enabled = variable_mixer_mode != 'none'
        if self.variable_mixer_enabled:
            if self.features != 'M':
                raise ValueError(
                    'hidden variable mixing currently requires M features')
            self.variable_mixer = StaticVariableMixer(
                channels=self.enc_in,
                layers=len(self.backbone.blocks),
                mode=variable_mixer_mode,
                source=str(getattr(
                    configs, 'ablation_variable_mixer_source', 'state')),
                placement=str(getattr(
                    configs, 'ablation_variable_mixer_placement',
                    'post_attention')),
                pattern=str(getattr(
                    configs, 'ablation_variable_mixer_pattern',
                    'all_shared')),
                rank=int(getattr(
                    configs, 'ablation_variable_mixer_rank', 8)),
                gate_init=float(getattr(
                    configs, 'ablation_variable_mixer_gate_init', 0.0)),
                gate_limit=float(getattr(
                    configs, 'ablation_variable_mixer_gate_limit', 1.0)),
                hidden_dim=configs.d_model,
            )
        else:
            self.variable_mixer = None
        self.last_query_inference = bool(getattr(
            configs, 'ablation_last_query_inference', False))
        if self.last_query_inference:
            if self.roll_patches != 1:
                raise ValueError(
                    'last-query inference currently requires Q1')
            if len(self.backbone.blocks) != 1:
                raise ValueError(
                    'last-query inference requires one Transformer layer')
        self._load_protected_initialization(configs)
        self._maybe_expand_initialize(configs)
        self._maybe_inject_lora(configs)

    def _maybe_expand_initialize(self, configs):
        """Zero-pad a narrower trained checkpoint into this wider model.

        --expand_init_from points at a trained checkpoint of the *same
        architecture family at a smaller width* (e.g. plan_a16_d64 when
        d_model=128).  Shape-driven zero padding (see
        models/width_expansion.py) makes the wide model start from the
        narrow model's behavior, so extra capacity is inert until the
        dataset's data activates it — the control for whether from-scratch
        random width initialization is the source of over-capacity damage.
        """
        checkpoint = str(getattr(configs, 'expand_init_from', '')).strip()
        if not checkpoint:
            return
        from models.width_expansion import expand_initialize
        state = torch.load(
            checkpoint, map_location='cpu', weights_only=True)
        records = expand_initialize(
            self, state,
            n_heads=int(configs.n_heads),
            d_model_new=int(configs.d_model),
        )
        copied = sum(1 for _, action in records if action == 'copy')
        skipped = [r for r in records if r[1].startswith(
            ('shape-mismatch-skip', 'skip-'))]
        print(
            f'expand_init: {len(records)} params '
            f'({copied} copied, {len(records) - copied - len(skipped)} '
            f'padded, {len(skipped)} skipped)', flush=True)

    def _maybe_inject_lora(self, configs):
        """Attach low-rank adapters over the shared backbone (plan_a16).

        --lora_rank r > 0 replaces the block projections (attention qkv /
        out_projection, FFN in/out) with LoRALinear wrappers that share the
        base weight and add a zero-initialized low-rank bypass, so the model
        is bit-identical to the plain d64 backbone at init.  When
        --lora_init_from points at a trained backbone checkpoint it is
        loaded first (the only missing keys are the fresh lora bypasses);
        --lora_freeze_backbone then pins the shared weights and the
        optimizer only ever sees the adapter parameters.
        """
        rank = int(getattr(configs, 'lora_rank', 0))
        if rank <= 0:
            return
        from models.lora_adapter import freeze_backbone, inject_lora
        injected = inject_lora(self, rank)
        if injected == 0:
            raise ValueError(
                f'lora_rank={rank} but no block projections were found')
        init_from = str(getattr(configs, 'lora_init_from', '')).strip()
        if init_from:
            state = torch.load(
                init_from, map_location='cpu', weights_only=True)
            self.load_state_dict(state, strict=False)
        if bool(getattr(configs, 'lora_freeze_backbone', False)):
            freeze_backbone(self)

    def initialize_factorized_io_from_native(self, native):
        """Analytically transplant a matched native patch-I/O model.

        Every shape-compatible non-I/O state is copied directly.  The native
        dense input projection and output head are then represented exactly by
        the factorized operators.  No data or optimization is involved.
        """
        if self.assembly_factorized_projection is None:
            raise ValueError(
                'native transplant requires factorized patch assembly')
        if not isinstance(
                self.backbone.output_head, FactorizedAtomOutputHead):
            raise ValueError(
                'native transplant requires a factorized atom output head')
        if not isinstance(native.backbone.patch_projection, nn.Linear):
            raise TypeError('native model must use a dense patch projection')
        if not isinstance(native.backbone.output_head, nn.Linear):
            raise TypeError('native model must use a dense output head')

        target_state = self.state_dict()
        copied = []
        for name, value in native.state_dict().items():
            if name in target_state and target_state[name].shape == value.shape:
                target_state[name] = value.detach().clone()
                copied.append(name)
        self.load_state_dict(target_state, strict=True)
        self.assembly_factorized_projection.initialize_from_native(
            native.backbone.patch_projection)
        self.backbone.output_head.initialize_from_native(
            native.backbone.output_head)
        return tuple(copied)

    def load_factorized_scale_state(self, source_state):
        """Load every learned parameter while rebuilding K-specific buffers."""
        if self.assembly_factorized_projection is None:
            raise ValueError(
                'scale-state loading requires factorized patch assembly')
        if not isinstance(
                self.backbone.output_head, FactorizedAtomOutputHead):
            raise ValueError(
                'scale-state loading requires a factorized atom output head')
        if isinstance(source_state, nn.Module):
            source_state = source_state.state_dict()
        target_state = self.state_dict()
        loaded = set()
        for name, value in source_state.items():
            # Position coefficients are deterministic functions of active K.
            if name.endswith('position_coefficients'):
                continue
            if name in target_state and target_state[name].shape == value.shape:
                target_state[name] = value.detach().clone()
                loaded.add(name)
        missing_parameters = sorted(
            name for name, _ in self.named_parameters()
            if name not in loaded
        )
        if missing_parameters:
            raise ValueError(
                'source state does not cover target learned parameters: '
                f'{missing_parameters}')
        self.load_state_dict(target_state, strict=True)
        return tuple(sorted(loaded))

    def set_active_channel_indices(self, indices):
        if self.variable_mixer is not None:
            self.variable_mixer.set_active_channel_indices(indices)

    def _assemble_parent_patches(self, parent_patches):
        """Compose fine patches into fixed parent-token coordinates.

        The operation never changes the parent-token count or the AR patch
        boundary.  ``factorized`` uses a shared atom stem plus fixed-width DCT
        position aggregation.  ``reverse`` is the fixed-permutation control;
        ``mean_sorted`` gives every slot a content-rank identity without a
        learned router.  Sorting is performed on raw normalized P12 values;
        the selected values still receive ordinary gradients.
        """
        if self.patch_assembly == 'none':
            return self.backbone.patch_projection(parent_patches)
        if parent_patches.size(-1) != self.patch_len:
            raise ValueError(
                f'expected parent patches of length {self.patch_len}')
        if self.patch_assembly == 'factorized':
            return self.assembly_factorized_projection(parent_patches)
        if self.patch_assembly == 'fd':
            return self.assembly_fd_projection(parent_patches)
        fine = parent_patches.reshape(
            *parent_patches.shape[:-1],
            self.assembly_slots,
            self.assembly_fine_patch_len,
        )
        if self.patch_assembly == 'reverse':
            fine = fine.flip(-2)
        elif self.patch_assembly == 'mean_sorted':
            order = torch.argsort(
                fine.mean(dim=-1),
                dim=-1,
                stable=True,
            )
            gather_index = order.unsqueeze(-1).expand_as(fine)
            fine = torch.gather(fine, -2, gather_index)
        projected = self.assembly_fine_projection(fine)
        projected = projected.reshape(
            *parent_patches.shape[:-1], -1)
        if self.assembly_interface_projection is not None:
            projected = self.assembly_interface_projection(projected)
        return projected

    def _load_protected_initialization(self, configs):
        checkpoint = str(getattr(
            configs, 'ablation_pretrained', '')).strip()
        trainable_scope = getattr(
            configs, 'ablation_trainable_scope', 'all')
        if trainable_scope not in {
                'all', 'local_mixer', 'variable_mixer'}:
            raise ValueError(
                'ablation trainable scope must be all, local_mixer, or '
                'variable_mixer')
        if not checkpoint:
            if trainable_scope != 'all':
                raise ValueError(
                    'a restricted ablation trainable scope requires a '
                    'pretrained checkpoint')
            return

        state = torch.load(
            checkpoint, map_location='cpu', weights_only=True)
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected:
            raise ValueError(
                'pretrained ablation checkpoint has unexpected keys: '
                f'{unexpected}')
        if trainable_scope == 'all' and missing:
            raise ValueError(
                'pretrained ablation checkpoint is incomplete: '
                f'{missing}')
        if trainable_scope in {'local_mixer', 'variable_mixer'}:
            marker = (
                '.local_mixer.'
                if trainable_scope == 'local_mixer'
                else 'variable_mixer.')
            invalid_missing = [
                name for name in missing
                if marker not in name
            ]
            if invalid_missing:
                raise ValueError(
                    'protected initialization is missing parent '
                    f'parameters: {invalid_missing}')
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            trainable = []
            for name, parameter in self.named_parameters():
                if marker in name:
                    parameter.requires_grad_(True)
                    trainable.append(name)
            if not trainable:
                raise ValueError(
                    f'protected fitting requires a {trainable_scope}')

    def _project_factorized_patches(self, history_patches):
        """Add protected shape/moment coordinates to the raw patch token."""
        projected = self.backbone.patch_projection(history_patches)
        patch_mean = history_patches.mean(dim=-1, keepdim=True)
        centered = history_patches - patch_mean
        patch_rms = torch.sqrt(
            centered.float().square().mean(
                dim=-1, keepdim=True) + self.norm_eps
        ).to(history_patches.dtype)
        if self.patch_shape_path:
            shape = centered / patch_rms
            shape_token = self.patch_shape_norm(
                self.patch_shape_projection(shape))
            shape_gate = torch.tanh(
                self.patch_shape_gate).to(projected.dtype)
            projected = projected + shape_gate * shape_token
        if self.patch_moment_path:
            moments = torch.cat([
                patch_mean,
                torch.log(patch_rms.clamp_min(self.norm_eps)),
            ], dim=-1)
            moment_token = self.patch_moment_norm(
                self.patch_moment_projection(moments))
            moment_gate = torch.tanh(
                self.patch_moment_gate).to(projected.dtype)
            projected = projected + moment_gate * moment_token
        return projected

    def _shape_address_tokens(
            self, history_patches, projected):
        patch_mean = history_patches.mean(dim=-1, keepdim=True)
        centered = history_patches - patch_mean
        patch_rms = torch.sqrt(
            centered.float().square().mean(
                dim=-1, keepdim=True) + self.norm_eps
        ).to(history_patches.dtype)
        shape = centered / patch_rms
        shape_token = self.patch_shape_address_norm(
            self.patch_shape_address_projection(shape))
        gate = torch.tanh(
            self.patch_shape_address_gate).to(projected.dtype)
        return projected + gate * shape_token

    def _shape_value_tokens(
            self, history_patches, projected):
        patch_mean = history_patches.mean(dim=-1, keepdim=True)
        centered = history_patches - patch_mean
        patch_rms = torch.sqrt(
            centered.float().square().mean(
                dim=-1, keepdim=True) + self.norm_eps
        ).to(history_patches.dtype)
        shape = centered / patch_rms
        shape_token = self.patch_shape_value_norm(
            self.patch_shape_value_projection(shape))
        gate = torch.tanh(
            self.patch_shape_value_gate).to(projected.dtype)
        return projected + gate * shape_token

    def _patch_value_sidecar(self, history_patches):
        patch_mean = history_patches.mean(dim=-1, keepdim=True)
        centered = history_patches - patch_mean
        patch_rms = torch.sqrt(
            centered.float().square().mean(
                dim=-1, keepdim=True) + self.norm_eps
        ).to(history_patches.dtype)
        value_residuals = None
        if self.patch_shape_sidecar_path:
            shape = centered / patch_rms
            shape_values = self.patch_shape_sidecar_norm(
                self.patch_shape_sidecar_projection(shape))
            shape_gate = torch.tanh(
                self.patch_shape_sidecar_gate).to(shape_values.dtype)
            value_residuals = shape_gate * shape_values
        if self.patch_moment_sidecar_path:
            moments = torch.cat([
                patch_mean,
                torch.log(patch_rms.clamp_min(self.norm_eps)),
            ], dim=-1)
            moment_values = self.patch_moment_sidecar_norm(
                self.patch_moment_sidecar_projection(moments))
            moment_gate = torch.tanh(
                self.patch_moment_sidecar_gate).to(moment_values.dtype)
            moment_values = moment_gate * moment_values
            value_residuals = (
                moment_values
                if value_residuals is None
                else value_residuals + moment_values
            )
        return value_residuals

    def _causal_band_value_sidecar(self, projected, width=4):
        token_count = projected.size(1)
        prefix = torch.cat([
            torch.zeros_like(projected[:, :1]),
            projected.cumsum(dim=1),
        ], dim=1)
        ends = torch.arange(
            1,
            token_count + 1,
            device=projected.device,
        )
        starts = (ends - width).clamp_min(0)
        counts = (ends - starts).to(projected.dtype).view(
            1, token_count, 1)
        low = (
            prefix[:, ends] - prefix[:, starts]
        ) / counts
        detail = projected - low
        value_residuals = None
        if self.causal_low_value_path:
            low_values = self.causal_low_value_norm(
                self.causal_low_value_projection(low))
            low_gate = torch.tanh(
                self.causal_low_value_gate).to(low_values.dtype)
            value_residuals = low_gate * low_values
        if self.causal_detail_value_path:
            detail_values = self.causal_detail_value_norm(
                self.causal_detail_value_projection(detail))
            detail_gate = torch.tanh(
                self.causal_detail_value_gate).to(
                    detail_values.dtype)
            detail_values = detail_gate * detail_values
            value_residuals = (
                detail_values
                if value_residuals is None
                else value_residuals + detail_values
            )
        return value_residuals

    def _predict_all_normalized(self, history):
        factorized_patch_path = (
            self.patch_shape_path or self.patch_moment_path)
        assembly_path = self.patch_assembly != 'none'
        custom_projection_path = (
            assembly_path
            or factorized_patch_path
            or self.variable_mixer_enabled
            or self.patch_shape_address_path
            or self.patch_shape_value_path
            or self.patch_shape_sidecar_path
            or self.patch_moment_sidecar_path
            or self.causal_low_value_path
            or self.causal_detail_value_path)
        if not self.recurrent_state_path and not custom_projection_path:
            outputs = super()._predict_all_normalized(history)
        else:
            mean, std = self._instance_stats(history)
            history_normalized = (history - mean) / std
            history_patches = self._patchify(
                history_normalized)
            input_patches = history_patches
            if (
                    (assembly_path or self.variable_mixer_enabled)
                    and self.future_patches is not None):
                forecast_queries = self.future_patches.expand(
                    history.size(0), -1, -1)
                input_patches = torch.cat([
                    history_patches, forecast_queries,
                ], dim=1)
            if assembly_path:
                projected = self._assemble_parent_patches(
                    input_patches)
            elif factorized_patch_path:
                projected = self._project_factorized_patches(
                    history_patches)
            else:
                projected = self.backbone.patch_projection(
                    input_patches)
            address_tokens = (
                self._shape_address_tokens(
                    history_patches, projected)
                if self.patch_shape_address_path
                else None
            )
            value_tokens = (
                self._shape_value_tokens(
                    history_patches, projected)
                if self.patch_shape_value_path
                else None
            )
            value_residuals = (
                self._patch_value_sidecar(history_patches)
                if (
                    self.patch_shape_sidecar_path
                    or self.patch_moment_sidecar_path)
                else None
            )
            if (
                    self.causal_low_value_path
                    or self.causal_detail_value_path):
                band_residuals = self._causal_band_value_sidecar(
                    projected)
                value_residuals = (
                    band_residuals
                    if value_residuals is None
                    else value_residuals + band_residuals
                )
            hidden = self.backbone.encode_projected(
                projected,
                address_tokens=address_tokens,
                value_tokens=value_tokens,
                value_residuals=value_residuals,
                state_mixer=self.variable_mixer,
            )
            if self.recurrent_state_path:
                recurrent, _ = self.context_state_gru(
                    projected)
                gate = torch.tanh(
                    self.recurrent_state_gate).to(hidden.dtype)
                hidden = hidden + gate * self.context_state_norm(
                    recurrent)
            predictions = self.backbone.output_head(hidden)
            outputs = predictions, history_patches, mean, std
        predictions, history_patches, mean, std = outputs
        if self.raw_patch_skip is not None:
            patch_values = history_patches.transpose(1, 2)
            patch_values = F.pad(
                patch_values,
                (self.raw_patch_skip_kernel - 1, 0),
            )
            if self.raw_patch_skip_mask is None:
                correction = self.raw_patch_skip(
                    patch_values)
            else:
                correction = F.conv1d(
                    patch_values,
                    self.raw_patch_skip.weight
                    * self.raw_patch_skip_mask,
                    self.raw_patch_skip.bias,
                    groups=self.patch_len,
                )
            correction = correction.transpose(1, 2)
            if correction.shape != predictions.shape:
                raise ValueError(
                    'raw patch skip requires one prediction per '
                    'observed patch')
            predictions = predictions + correction
            outputs = predictions, history_patches, mean, std
        if self.persistence_blend:
            if predictions.shape != history_patches.shape:
                raise ValueError(
                    'persistence blending currently requires Q1 predictions')
            gate = torch.tanh(
                self.persistence_gate).to(predictions.dtype)
            predictions = predictions + gate * (
                history_patches - predictions)
            outputs = predictions, history_patches, mean, std
        if (
                not self.cross_channel_mixer
                and not self.channel_specific_head):
            return outputs
        if predictions.size(0) % self.enc_in:
            raise ValueError(
                'channel-batched predictions do not align with enc_in')
        batch_size = predictions.size(0) // self.enc_in
        shaped = predictions.view(
            batch_size,
            self.enc_in,
            predictions.size(1),
            predictions.size(2),
        )
        if self.channel_specific_head:
            correction = torch.einsum(
                'cpq,bctq->bctp',
                self.channel_head_weight,
                shaped,
            )
            correction = (
                correction
                + self.channel_head_bias[None, :, None, :]
            )
            shaped = shaped + correction
        if not self.cross_channel_mixer:
            return (
                shaped.reshape_as(predictions),
                history_patches,
                mean,
                std,
            )
        weight = self.cross_channel_weight.masked_fill(
            ~self.cross_channel_off_diagonal, 0.0)
        correction = torch.einsum(
            'ij,bjtp->bitp',
            weight,
            shaped,
        )
        predictions = (
            shaped + correction).reshape_as(predictions)
        return predictions, history_patches, mean, std

    def _predict_last_normalized(self, history):
        """Numerically exact one-layer final-query inference path."""
        unsupported = (
            self.recurrent_state_path
            or self.patch_shape_path
            or self.patch_moment_path
            or self.patch_shape_address_path
            or self.patch_shape_value_path
            or self.patch_shape_sidecar_path
            or self.patch_moment_sidecar_path
            or self.causal_low_value_path
            or self.causal_detail_value_path
            or self.raw_patch_skip is not None
            or self.persistence_blend
            or self.cross_channel_mixer
            or self.channel_specific_head
            or self.variable_mixer_enabled
        )
        if unsupported:
            raise ValueError(
                'last-query inference currently supports the plain '
                'one-layer Transformer/QIA/local paths')
        mean, std = self._instance_stats(history)
        history_normalized = (history - mean) / std
        history_patches = self._patchify(history_normalized)
        projected = (
            self._assemble_parent_patches(history_patches)
            if self.patch_assembly != 'none'
            else self.backbone.patch_projection(history_patches)
        )
        hidden = self.backbone.encode_projected_last(projected)
        predictions = self.backbone.output_head(hidden)
        return predictions, history_patches, mean, std
