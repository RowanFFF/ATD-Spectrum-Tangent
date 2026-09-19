import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.AutoTimesOperatorAR import Model
from models.AutoTimesSpectrumTangent import (
    build_phase_template,
    correct_far_commits,
    forecast_with_explicit_tangent,
)
from models.AutoTimesSpectrumTangentOptimized import (
    build_phase_template_vectorized,
    forecast_with_explicit_tangent_optimized,
)
from models.AutoTimesSpectrumTangentGrouped import (
    build_phase_template_grouped,
    forecast_with_explicit_tangent_grouped,
)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds):
        return SimpleNamespace(
            last_hidden_state=torch.tanh(self.projection(inputs_embeds))
        )


def _fake_backbone_factory():
    # Stand in for a pinned from_pretrained checkpoint without perturbing the
    # adapter initialization stream.
    with torch.random.fork_rng():
        torch.manual_seed(101)
        return _FakeBackbone()


def make_config(**overrides):
    values = dict(
        seq_len=16,
        pred_len=16,
        d_model=12,
        dropout=0.0,
        external_autotimes_token_len=4,
        external_autotimes_mode='q1',
        external_autotimes_backbone_factory=_fake_backbone_factory,
        external_autotimes_llm_checkpoint='',
        external_autotimes_pretrained='',
        external_autotimes_mlp_hidden=16,
        external_autotimes_mlp_layers=2,
        distilled_multi_commit_hidden=16,
        distilled_multi_commit_patches=2,
        distilled_multi_commit_eval_patches=0,
        progressive_nested_protected_patches=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class AutoTimesOperatorARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.q1_path = self.root / 'q1.pth'
        self.q1 = Model(make_config()).eval()
        torch.save(self.q1.state_dict(), self.q1_path)

    def tearDown(self):
        self.temporary.cleanup()

    def generated(self, width=4):
        return Model(make_config(
            external_autotimes_mode='generated',
            external_autotimes_pretrained=str(self.q1_path),
            distilled_multi_commit_patches=width,
            distilled_multi_commit_eval_patches=width,
        ))

    def clean(self, width=4):
        return Model(make_config(
            external_autotimes_mode='clean',
            external_autotimes_pretrained=str(self.q1_path),
            distilled_multi_commit_patches=width,
            distilled_multi_commit_eval_patches=width,
        ))

    def test_q1_uses_full_next_token_objective_and_rolls_out(self):
        model = Model(make_config()).train()
        history = torch.randn(3, 16, 2)
        target = torch.randn(3, 4, 2)
        losses = model.direct_patch_loss(history, target)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.encoder.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.gpt2.parameters()
        ))
        prediction = model.eval()(history, None, None, None)
        self.assertEqual(prediction.shape, (3, 16, 2))

    def test_checkpoint_omits_frozen_gpt2_but_restores_operator_exactly(self):
        state = self.q1.state_dict()
        self.assertFalse(any(name.startswith('gpt2.') for name in state))
        torch.manual_seed(67)
        restored = Model(make_config()).eval()
        restored.load_state_dict(state, strict=True)
        history = torch.randn(2, 16, 2)
        torch.testing.assert_close(
            restored.forecast_with_commit_patches(history, 1),
            self.q1.forecast_with_commit_patches(history, 1),
        )

    def test_generated_freezes_q1_and_trains_only_far_exits(self):
        model = self.generated().train()
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('far_heads.') for name in trainable))
        history = torch.randn(2, 16, 2)
        torch.testing.assert_close(
            model.eval().forecast_with_commit_patches(history, 1),
            self.q1.forecast_with_commit_patches(history, 1),
        )
        losses = model.train().direct_patch_loss(
            history, torch.randn(2, 16, 2)
        )
        losses['loss'].backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in model.far_heads.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder.parameters()
        ))

    def test_generated_targets_are_exact_recursive_q1_writeback(self):
        model = self.generated(width=4).eval()
        history = torch.randn(2, 16, 1)
        channel_history, _, _ = model._to_channel_batch(history)
        _, _, mean, std = model._base_hidden(channel_history)
        expected = model._recursive_q1_targets(
            channel_history, 4, mean, std
        )
        recursive = []
        current = history
        for _ in range(4):
            next_patch = self.q1.forecast_with_commit_patches(current, 1)[:, :4]
            recursive.append(next_patch)
            current = torch.cat([current, next_patch], dim=1)[:, -16:]
        recursive = torch.cat(recursive, dim=1)
        expected_points = expected.reshape(2, 16, 1) * std.reshape(2, 1, 1) \
            + mean.reshape(2, 1, 1)
        torch.testing.assert_close(expected_points, recursive)

    def test_clean_control_uses_truth_far_patches_not_generated_targets(self):
        model = self.clean(width=4).train()
        history = torch.randn(2, 16, 1)
        target = torch.randn(2, 16, 1)
        loss = model.direct_patch_loss(history, target)['loss']
        channel_history, _, _ = model._to_channel_batch(history)
        channel_target, _, _ = model._to_channel_batch(target)
        first, state, mean, std = model._base_hidden(channel_history)
        student = torch.cat([
            first.detach() + head(state.detach()).unsqueeze(1)
            for head in model.far_heads
        ], dim=1)
        truth = ((channel_target - mean) / std).squeeze(-1).reshape(2, 4, 4)
        expected = torch.nn.functional.mse_loss(student, truth[:, 1:])
        torch.testing.assert_close(loss, expected)

    def test_staged_k2_to_k4_preserves_k1_and_k2_exactly(self):
        k2 = self.generated(width=2).eval()
        with torch.no_grad():
            k2.far_heads[0].net[-1].weight.normal_(std=0.03)
            k2.far_heads[0].net[-1].bias.normal_(std=0.03)
        k2_path = self.root / 'k2.pth'
        torch.save(k2.state_dict(), k2_path)
        k4 = Model(make_config(
            external_autotimes_mode='staged',
            external_autotimes_pretrained=str(k2_path),
            distilled_multi_commit_patches=4,
            distilled_multi_commit_eval_patches=4,
            progressive_nested_protected_patches=2,
        )).eval()
        history = torch.randn(3, 16, 2)
        for width in (1, 2):
            torch.testing.assert_close(
                k4.forecast_with_commit_patches(history, width),
                k2.forecast_with_commit_patches(history, width),
            )
        trainable = [
            name for name, parameter in k4.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith(('far_heads.1.', 'far_heads.2.'))
            for name in trainable
        ))

    def test_staged_composition_loss_aligns_new_k2_to_k4_targets(self):
        k2 = self.generated(width=2).eval()
        k2_path = self.root / 'k2_for_loss.pth'
        torch.save(k2.state_dict(), k2_path)
        k4 = Model(make_config(
            external_autotimes_mode='staged',
            external_autotimes_pretrained=str(k2_path),
            distilled_multi_commit_patches=4,
            progressive_nested_protected_patches=2,
        )).train()
        history = torch.randn(2, 16, 1)
        loss = k4.direct_patch_loss(
            history, torch.randn(2, 16, 1)
        )['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None
            for head in k4.far_heads[1:]
            for parameter in head.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in k4.far_heads[0].parameters()
        ))

    def test_invalid_sequence_and_staged_width_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'divide'):
            Model(make_config(seq_len=15))
        with self.assertRaisesRegex(ValueError, 'K to 2K'):
            Model(make_config(
                external_autotimes_mode='staged',
                external_autotimes_pretrained=str(self.q1_path),
                distilled_multi_commit_patches=4,
                progressive_nested_protected_patches=1,
            ))

    def test_spectrum_tangent_zero_gamma_is_exact_generated_rollout(self):
        model = self.generated(width=4).eval()
        history = torch.randn(3, 16, 2)
        for width in (1, 2, 4):
            torch.testing.assert_close(
                forecast_with_explicit_tangent(
                    model, history, width, period=4, gamma=0.0
                ),
                model.forecast_with_commit_patches(history, width),
                atol=0.0,
                rtol=0.0,
            )

    def test_spectrum_tangent_hard_preserves_every_local_q1_slot(self):
        model = self.generated(width=4).eval()
        history = torch.randn(2, 16, 1)
        channel_history, _, _ = model._to_channel_batch(history)
        commits, _, _, _ = model._protected_commits_normalized(
            channel_history, 4
        )
        template = build_phase_template(model, channel_history, period=4)
        corrected = correct_far_commits(
            commits, template, committed_patches=8,
            patch_len=model.patch_len, gamma=3.0,
        )
        torch.testing.assert_close(
            corrected[:, :1], commits[:, :1], atol=0.0, rtol=0.0
        )
        self.assertGreater(
            float((corrected[:, 1:] - commits[:, 1:]).abs().max().detach()),
            0.0,
        )

    def test_spectrum_tangent_does_not_mutate_generated_parameters(self):
        model = self.generated(width=4).eval()
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        forecast_with_explicit_tangent(
            model, torch.randn(2, 16, 2), 4, period=4, gamma=1.0
        )
        after = model.state_dict()
        self.assertEqual(set(before), set(after))
        for name in before:
            torch.testing.assert_close(
                before[name], after[name], atol=0.0, rtol=0.0
            )

    def test_vectorized_tangent_matches_audited_reference(self):
        model = self.generated(width=4).eval()
        history = torch.randn(3, 16, 2)
        channel_history, _, _ = model._to_channel_batch(history)
        torch.testing.assert_close(
            build_phase_template_vectorized(model, channel_history, 4),
            build_phase_template(model, channel_history, 4),
            atol=2e-7,
            rtol=2e-7,
        )
        for width in (2, 4):
            torch.testing.assert_close(
                forecast_with_explicit_tangent_optimized(
                    model, history, width, period=4, gamma=1.7
                ),
                forecast_with_explicit_tangent(
                    model, history, width, period=4, gamma=1.7
                ),
                atol=2e-6,
                rtol=2e-6,
            )

    def test_grouped_tangent_preserves_reference_reduction(self):
        model = self.generated(width=4).eval()
        history = torch.randn(3, 16, 2)
        channel_history, _, _ = model._to_channel_batch(history)
        torch.testing.assert_close(
            build_phase_template_grouped(model, channel_history, 4),
            build_phase_template(model, channel_history, 4),
            atol=0.0,
            rtol=0.0,
        )
        for width in (2, 4):
            torch.testing.assert_close(
                forecast_with_explicit_tangent_grouped(
                    model, history, width, period=4, gamma=1.7
                ),
                forecast_with_explicit_tangent(
                    model, history, width, period=4, gamma=1.7
                ),
                atol=0.0,
                rtol=0.0,
            )


if __name__ == '__main__':
    unittest.main()
