import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.TimerOperatorAR import Model


class _FakeTimer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(input_token_len=4, hidden_size=12)
        self.embedding = nn.Linear(4, 12)
        self.head = nn.Linear(12, 4, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        revin=False,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    ):
        del attention_mask, position_ids, return_dict
        mean = input_ids.mean(dim=-1, keepdim=True)
        std = input_ids.std(dim=-1, keepdim=True)
        if revin:
            input_ids = (input_ids - mean) / std
        patches = input_ids.unfold(-1, 4, 4)
        hidden = torch.tanh(self.embedding(patches))
        logits = self.head(hidden[:, -1])
        if revin:
            logits = logits * std + mean
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            past_key_values=(object(),) if use_cache else past_key_values,
        )


def _fake_timer_factory():
    with torch.random.fork_rng():
        torch.manual_seed(211)
        return _FakeTimer()


def make_config(**overrides):
    values = dict(
        seq_len=16,
        pred_len=16,
        external_timer_mode="q1",
        external_timer_checkpoint="",
        external_timer_backbone_factory=_fake_timer_factory,
        distilled_multi_commit_hidden=16,
        distilled_multi_commit_patches=4,
        distilled_multi_commit_eval_patches=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TimerOperatorARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(223)
        self.q1 = Model(make_config()).eval()
        self.generated = Model(make_config(
            external_timer_mode="generated",
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
                external_timer_mode="generated",
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

    def test_k1_is_exact_original_timer_rollout(self):
        history = torch.randn(3, 16, 2)
        torch.testing.assert_close(
            self.generated.eval().forecast_with_commit_patches(history, 1),
            self.q1.forecast_with_commit_patches(history, 1),
            atol=0.0,
            rtol=0.0,
        )

    def test_constant_history_uses_finite_revin_limit(self):
        history = torch.full((2, 16, 3), 2.75)
        forecast = self.q1.forecast_with_commit_patches(history, 1)
        self.assertTrue(torch.isfinite(forecast).all())
        torch.testing.assert_close(
            forecast,
            torch.full_like(forecast, 2.75),
            atol=0.0,
            rtol=0.0,
        )

    def test_generated_trains_only_far_exits(self):
        trainable = [
            name for name, parameter in self.generated.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("far_heads.") for name in trainable))
        history = torch.randn(2, 16, 2)
        target = torch.randn(2, 16, 2)
        loss = self.generated.train().direct_patch_loss(history, target)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in self.generated.far_heads.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in self.generated.timer.parameters()
        ))

    def test_generated_targets_are_recursive_timer_q1(self):
        model = self.generated.eval()
        history = torch.randn(2, 16, 1)
        channel_history, _, _ = model._to_channel_batch(history)
        targets = model._recursive_q1_targets(
            channel_history.squeeze(-1), 4
        )
        expected = self.q1.forecast_with_commit_patches(history, 1)
        torch.testing.assert_close(
            targets.reshape(2, 16, 1), expected, atol=0.0, rtol=0.0
        )

    def test_zero_tangent_is_bit_exact_generated_rollout(self):
        model = self.generated.eval()
        history = torch.randn(2, 16, 2)
        for width in (1, 2, 4):
            torch.testing.assert_close(
                model.forecast_with_explicit_tangent(
                    history, width, period=4, gamma=0.0
                ),
                model.forecast_with_commit_patches(history, width),
                atol=0.0,
                rtol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
