import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.TimesFMOperatorAR import Model


class _FakeTimesFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.o = 8
        self.m = 2
        self.q = 3
        self.aridx = 1
        self.md = 12
        self.x = 1
        self.h = 1
        self.hd = 12
        self.tokenizer = nn.Linear(8, 12)
        self.projection = nn.Linear(12, self.o * self.q)

    def forward(self, inputs, masks, decode_caches=None):
        tokens = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
        hidden = torch.tanh(self.tokenizer(tokens))
        outputs = self.projection(hidden)
        if decode_caches is not None:
            for cache in decode_caches:
                cache.next_index += inputs.size(1)
        spreads = torch.zeros_like(outputs)
        return (hidden, hidden, outputs, spreads), decode_caches


def _fake_timesfm_factory():
    with torch.random.fork_rng():
        torch.manual_seed(307)
        return _FakeTimesFM()


def make_config(**overrides):
    values = dict(
        seq_len=16,
        pred_len=32,
        external_timesfm_mode="q1",
        external_timesfm_source="",
        external_timesfm_checkpoint="",
        external_timesfm_backbone_factory=_fake_timesfm_factory,
        distilled_multi_commit_hidden=16,
        distilled_multi_commit_patches=4,
        distilled_multi_commit_eval_patches=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TimesFMOperatorARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(311)
        self.q1 = Model(make_config()).eval()
        self.generated = Model(make_config(
            external_timesfm_mode="generated",
            distilled_multi_commit_eval_patches=4,
        ))

    def test_checkpoint_contains_only_far_exits(self):
        state = self.generated.state_dict()
        self.assertTrue(state)
        self.assertTrue(all(name.startswith("far_heads.") for name in state))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.pth"
            torch.save(state, path)
            restored = Model(make_config(
                external_timesfm_mode="generated",
                distilled_multi_commit_eval_patches=4,
            )).eval()
            restored.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True),
                strict=True,
            )
            history = torch.randn(2, 16, 2)
            torch.testing.assert_close(
                restored.forecast_with_commit_patches(history, 4),
                self.generated.eval().forecast_with_commit_patches(history, 4),
            )

    def test_k1_is_exact_cached_timesfm_rollout(self):
        history = torch.randn(3, 16, 2)
        torch.testing.assert_close(
            self.generated.eval().forecast_with_commit_patches(history, 1),
            self.q1.forecast_with_commit_patches(history, 1),
            atol=0.0,
            rtol=0.0,
        )

    def test_generated_targets_are_recursive_q1_blocks(self):
        history = torch.randn(2, 16, 1)
        channel_history, _, _ = self.generated._to_channel_batch(history)
        teacher = self.generated.eval()._recursive_q1_targets(
            channel_history.squeeze(-1), 4
        )
        expected = self.q1.forecast_with_commit_patches(history, 1)
        torch.testing.assert_close(
            teacher.reshape(2, 32, 1), expected, atol=0.0, rtol=0.0
        )

    def test_generated_trains_only_far_exits(self):
        trainable = [
            name for name, parameter in self.generated.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("far_heads.") for name in trainable))
        history = torch.randn(2, 16, 2)
        target = torch.randn(2, 32, 2)
        loss = self.generated.train().direct_patch_loss(history, target)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in self.generated.far_heads.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in self.generated.timesfm.parameters()
        ))

    def test_zero_tangent_is_bit_exact_generated_rollout(self):
        history = torch.randn(2, 16, 2)
        for width in (1, 2, 4):
            torch.testing.assert_close(
                self.generated.eval().forecast_with_explicit_tangent(
                    history, width, period=4, gamma=0.0
                ),
                self.generated.forecast_with_commit_patches(history, width),
                atol=0.0,
                rtol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
