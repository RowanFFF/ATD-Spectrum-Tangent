"""AutoTimes-GPT2 operator with Generated-K and protected DPOD exits.

The base follows the official AutoTimes formulation: continuous non-overlapping
patches are projected into a frozen GPT-2, and a learned detokenizer predicts
the next patch.  Parent training uses patch next-token prediction.  Wider modes
freeze the complete accepted operator and learn only offset-specific far exits.

Official source used for the interface audit:
https://github.com/thuml/AutoTimes (MIT, pinned separately in the paper ledger).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers, dropout):
        super().__init__()
        if layers < 0:
            raise ValueError("AutoTimes MLP layer count must be non-negative")
        if layers == 0:
            self.layers = nn.Linear(input_dim, output_dim)
            return
        modules = [
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        ]
        for _ in range(max(layers - 2, 0)):
            modules.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            ])
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.layers(inputs)


class _FarExit(nn.Module):
    def __init__(self, state_dim, hidden_dim, patch_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_len),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)


class Model(nn.Module):
    """Frozen-operator hierarchy over an AutoTimes-GPT2 Q1 parent."""

    MODES = {"q1", "clean", "generated", "staged"}

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.patch_len = int(getattr(
            configs, "external_autotimes_token_len", 96
        ))
        if self.patch_len < 1 or self.seq_len % self.patch_len:
            raise ValueError(
                "AutoTimes token length must divide the lookback length"
            )
        self.mode = str(getattr(configs, "external_autotimes_mode", "q1"))
        if self.mode not in self.MODES:
            raise ValueError(
                f"external AutoTimes mode must be one of {sorted(self.MODES)}"
            )
        self.commit_patches = (
            1 if self.mode == "q1" else int(getattr(
                configs, "distilled_multi_commit_patches", 2
            ))
        )
        if self.commit_patches < 1:
            raise ValueError("AutoTimes commit width must be positive")
        requested_eval_patches = int(getattr(
            configs,
            "distilled_multi_commit_eval_patches",
            self.commit_patches,
        ))
        self.eval_commit_patches = (
            self.commit_patches
            if requested_eval_patches == 0
            else requested_eval_patches
        )
        if not 1 <= self.eval_commit_patches <= self.commit_patches:
            raise ValueError("AutoTimes evaluation width must lie in [1, K]")
        self.protected_patches = (
            1 if self.mode != "staged" else int(getattr(
                configs, "progressive_nested_protected_patches", 1
            ))
        )
        if self.mode == "staged" and not (
            1 <= self.protected_patches < self.commit_patches
            and self.commit_patches == 2 * self.protected_patches
        ):
            raise ValueError("staged AutoTimes requires a protected K to 2K expansion")

        factory = getattr(
            configs, "external_autotimes_backbone_factory", None
        )
        if factory is None:
            checkpoint = str(getattr(
                configs, "external_autotimes_llm_checkpoint", ""
            ))
            if not checkpoint:
                raise ValueError("an AutoTimes GPT-2 checkpoint is required")
            try:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Model
            except ImportError as error:
                raise ImportError(
                    "AutoTimesOperatorAR requires the optional transformers package"
                ) from error
            self.gpt2 = GPT2Model.from_pretrained(checkpoint)
        else:
            self.gpt2 = factory()
        state_dim = int(getattr(
            getattr(self.gpt2, "config", SimpleNamespace()),
            "hidden_size",
            getattr(configs, "d_model", 768),
        ))
        mlp_hidden = int(getattr(
            configs, "external_autotimes_mlp_hidden", 512
        ))
        mlp_layers = int(getattr(
            configs, "external_autotimes_mlp_layers", 2
        ))
        dropout = float(getattr(configs, "dropout", 0.1))
        self.encoder = _MLP(
            self.patch_len, state_dim, mlp_hidden, mlp_layers, dropout
        )
        self.decoder = _MLP(
            state_dim, self.patch_len, mlp_hidden, mlp_layers, dropout
        )
        for parameter in self.gpt2.parameters():
            parameter.requires_grad = False

        pretrained = str(getattr(
            configs, "external_autotimes_pretrained", ""
        ))
        if self.mode != "q1" and not pretrained:
            raise ValueError("non-Q1 AutoTimes modes require an accepted checkpoint")
        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            base_state = {
                name: value for name, value in state.items()
                if not name.startswith("far_heads.")
            }
            incompatible = self.load_state_dict(base_state, strict=False)
            unexpected = [
                name for name in incompatible.unexpected_keys
                if not name.startswith("far_heads.")
            ]
            missing = [
                name for name in incompatible.missing_keys
                if not name.startswith("far_heads.")
            ]
            if unexpected or missing:
                raise RuntimeError(
                    "incompatible AutoTimes base checkpoint: "
                    f"missing={missing} unexpected={unexpected}"
                )

        hidden_dim = int(getattr(
            configs, "distilled_multi_commit_hidden", mlp_hidden
        ))
        self.far_heads = nn.ModuleList([
            _FarExit(state_dim, hidden_dim, self.patch_len)
            for _ in range(self.commit_patches - 1)
        ])
        if pretrained and self.mode == "staged":
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            incompatible = self.load_state_dict(state, strict=False)
            allowed_missing = {
                name for name in self.state_dict()
                if name.startswith("far_heads.")
                and int(name.split(".")[1]) >= self.protected_patches - 1
            }
            if incompatible.unexpected_keys or set(incompatible.missing_keys) != allowed_missing:
                raise RuntimeError(
                    "incompatible protected AutoTimes checkpoint: "
                    f"missing={incompatible.missing_keys} "
                    f"unexpected={incompatible.unexpected_keys}"
                )

        if self.mode != "q1":
            for parameter in self.parameters():
                parameter.requires_grad = False
            first_trainable = (
                self.protected_patches - 1 if self.mode == "staged" else 0
            )
            for head in self.far_heads[first_trainable:]:
                for parameter in head.parameters():
                    parameter.requires_grad = True
        self.train_pred_len = self.commit_patches * self.patch_len

    def state_dict(self, *args, **kwargs):
        """Save only the lightweight operator adapter and DPOD exits.

        The frozen GPT-2 is restored from the explicitly pinned Hugging Face
        checkpoint on every construction.  Repeating it in every dataset,
        seed, and width checkpoint would add hundreds of megabytes without
        carrying any trained state.
        """
        state = super().state_dict(*args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        frozen_prefix = f"{prefix}gpt2."
        for name in [name for name in state if name.startswith(frozen_prefix)]:
            del state[name]
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        missing = [
            name for name in incompatible.missing_keys
            if not name.startswith("gpt2.")
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "incompatible AutoTimes operator checkpoint: "
                f"missing={missing} unexpected={unexpected}"
            )
        return type(incompatible)(missing, unexpected)

    @staticmethod
    def _instance_stats(history):
        mean = history.mean(dim=1, keepdim=True).detach()
        centered = history - mean
        std = torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return mean, std

    @staticmethod
    def _to_channel_batch(values):
        batch, length, channels = values.shape
        return (
            values.permute(0, 2, 1).reshape(batch * channels, length, 1),
            batch,
            channels,
        )

    def _base_all_normalized(self, history):
        mean, std = self._instance_stats(history)
        normalized = (history - mean) / std
        patches = normalized.squeeze(-1).unfold(
            dimension=-1, size=self.patch_len, step=self.patch_len
        )
        embeddings = self.encoder(patches)
        hidden = self.gpt2(inputs_embeds=embeddings).last_hidden_state
        predictions = self.decoder(hidden)
        return predictions, hidden, mean, std

    def _base_hidden(self, history):
        predictions, hidden, mean, std = self._base_all_normalized(history)
        return predictions[:, -1:], hidden[:, -1], mean, std

    def _protected_commits_normalized(self, history, width):
        first, state, mean, std = self._base_hidden(history)
        commits = [first]
        commits.extend(
            first + self.far_heads[index](state).unsqueeze(1)
            for index in range(width - 1)
        )
        return torch.cat(commits, dim=1), state, mean, std

    def _student_commits_normalized(self, history):
        return self._protected_commits_normalized(history, self.commit_patches)

    @torch.no_grad()
    def _recursive_q1_targets(self, history, width, origin_mean, origin_std):
        targets = []
        current = history
        for _ in range(width):
            first, _, mean, std = self._base_hidden(current)
            points = first * std + mean
            targets.append((points - origin_mean) / origin_std)
            current = torch.cat([
                current,
                points.reshape(history.size(0), self.patch_len, 1),
            ], dim=1)[:, -self.seq_len:]
        return torch.cat(targets, dim=1)

    @torch.no_grad()
    def _protected_composition_targets(
        self, history, origin_mean, origin_std
    ):
        first, _, mean, std = self._protected_commits_normalized(
            history, self.protected_patches
        )
        points = (first * std + mean).reshape(
            history.size(0), self.protected_patches * self.patch_len, 1
        )
        next_history = torch.cat([history, points], dim=1)[:, -self.seq_len:]
        second, _, next_mean, next_std = self._protected_commits_normalized(
            next_history, self.protected_patches
        )
        return torch.cat([
            first[:, 1:],
            (second * next_std + next_mean - origin_mean) / origin_std,
        ], dim=1)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if self.mode == "q1":
            # Match the official AutoTimes training path: although GPT-2 is
            # weight-frozen, its dropout follows the parent model's mode.
            self.gpt2.train(mode)
            self.encoder.train(mode)
            self.decoder.train(mode)
        else:
            # Distillation targets must come from the accepted deployed
            # operator, so the frozen teacher always remains deterministic.
            self.gpt2.eval()
            self.encoder.eval()
            self.decoder.eval()
            first_trainable = (
                self.protected_patches - 1 if self.mode == "staged" else 0
            )
            for index, head in enumerate(self.far_heads):
                head.train(mode and index >= first_trainable)
        return self

    def optimizer_parameter_groups(self, learning_rate):
        parameters = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        ]
        return [{"params": parameters, "lr": learning_rate}]

    def direct_patch_loss(self, history, target):
        history, batch, channels = self._to_channel_batch(history)
        target, target_batch, target_channels = self._to_channel_batch(target)
        if (batch, channels) != (target_batch, target_channels):
            raise ValueError("AutoTimes history and target channels must match")
        if target.size(1) != self.train_pred_len:
            raise ValueError("AutoTimes target does not match the commit width")
        if self.mode == "q1":
            predictions, _, mean, std = self._base_all_normalized(history)
            normalized_history = (history - mean) / std
            history_patches = normalized_history.squeeze(-1).unfold(
                dimension=-1, size=self.patch_len, step=self.patch_len
            )
            future = ((target[:, :self.patch_len] - mean) / std).squeeze(-1)
            shifted = torch.cat([history_patches[:, 1:], future.unsqueeze(1)], dim=1)
            loss = F.mse_loss(predictions, shifted)
            return {"loss": loss, "next_token_loss": loss}

        first, state, mean, std = self._base_hidden(history)
        with torch.no_grad():
            if self.mode == "clean":
                teacher = ((target - mean) / std).squeeze(-1).reshape(
                    target.size(0), self.commit_patches, self.patch_len
                )
            elif self.mode == "generated":
                teacher = self._recursive_q1_targets(
                    history, self.commit_patches, mean, std
                )
            else:
                teacher = self._protected_composition_targets(
                    history, mean, std
                )
        first_trainable = (
            self.protected_patches - 1 if self.mode == "staged" else 0
        )
        student = torch.cat([
            first.detach()
            + self.far_heads[index](state.detach()).unsqueeze(1)
            for index in range(first_trainable, self.commit_patches - 1)
        ], dim=1)
        teacher_start = (
            1 if self.mode in {"clean", "generated"}
            else self.protected_patches - 1
        )
        loss = F.mse_loss(student, teacher[:, teacher_start:])
        return {"loss": loss, "composition_loss": loss}

    def forecast_with_commit_patches(self, x_enc, commit_patches):
        if not 1 <= commit_patches <= self.commit_patches:
            raise ValueError("requested AutoTimes commit width is unavailable")
        history, batch, channels = self._to_channel_batch(
            x_enc[:, -self.seq_len:]
        )
        outputs = []
        generated = 0
        while generated < self.pred_len:
            commits, _, mean, std = self._protected_commits_normalized(
                history, commit_patches
            )
            points = (commits * std + mean).reshape(
                batch, channels, commit_patches * self.patch_len
            ).permute(0, 2, 1)
            outputs.append(points)
            channel_points = points.permute(0, 2, 1).reshape(
                batch * channels, commit_patches * self.patch_len, 1
            )
            history = torch.cat([history, channel_points], dim=1)[
                :, -self.seq_len:
            ]
            generated += commit_patches * self.patch_len
        return torch.cat(outputs, dim=1)[:, :self.pred_len]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.forecast_with_commit_patches(
            x_enc, self.eval_commit_patches
        )
