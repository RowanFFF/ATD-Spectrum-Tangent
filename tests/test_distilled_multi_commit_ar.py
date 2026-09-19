import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch

from models.DistilledMultiCommitAR import Model
from models.TransformerAblationAR import (
    Model as BaseModel,
)


def make_config(checkpoint='', **overrides):
    values = dict(
        task_name='long_term_forecast',
        seq_len=16,
        pred_len=12,
        enc_in=1,
        features='S',
        ar_patch_len=4,
        dense_ar_roll_patches=1,
        dense_ar_eval_commit_patches=1,
        dense_ar_loss_space='normalized',
        dense_ar_loss_type='mse',
        dense_ar_huber_delta=1.0,
        dense_ar_far_patch_weight=1.0,
        dense_ar_onpolicy_train_patches=1,
        dense_ar_onpolicy_loss_weight=0.0,
        dense_ar_onpolicy_detach_history=True,
        ar_norm_eps=1e-5,
        ar_transformer_norm='post',
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_ff=16,
        dropout=0.0,
        activation='gelu',
        ablation_attention='softmax',
        ablation_ffn='mlp',
        ablation_norm='post',
        ablation_position_encoding='rope',
        ablation_attention_residual=True,
        ablation_ffn_residual=True,
        distilled_multi_commit_pretrained=checkpoint,
        distilled_multi_commit_hidden=16,
        distilled_multi_commit_truth_weight=0.1,
        distilled_multi_commit_truth_loss='mse',
        distilled_multi_commit_eval_patches=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class DistilledMultiCommitARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021)

    def make_model_pair(self, directory):
        base = BaseModel(make_config()).eval()
        checkpoint = Path(directory) / 'base.pth'
        torch.save(base.state_dict(), checkpoint)
        distilled = Model(
            make_config(str(checkpoint))).eval()
        return base, distilled

    def test_first_patch_is_exact_frozen_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            base, distilled = self.make_model_pair(directory)
            history = torch.randn(3, 16, 1)
            base_prediction = base._predict_all_normalized(
                history)[0][:, -1:]
            first = distilled._student_pair_normalized(
                history)[0]
            torch.testing.assert_close(
                first, base_prediction)
            for name, parameter in (
                    distilled.named_parameters()):
                self.assertEqual(
                    parameter.requires_grad,
                    name.startswith('second_residual.'),
                )

    def test_first_patch_is_exact_with_hidden_variable_mixer(self):
        config = dict(
            enc_in=3,
            features='M',
            ablation_variable_mixer='low_rank',
            ablation_variable_mixer_source='temporal_innovation',
            ablation_variable_mixer_placement='post_attention',
            ablation_variable_mixer_pattern='all_shared',
            ablation_variable_mixer_rank=2,
            ablation_variable_mixer_gate_init=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            base = BaseModel(make_config(**config)).eval()
            base.variable_mixer.gates.data.fill_(0.25)
            checkpoint = Path(directory) / 'variable_parent.pth'
            torch.save(base.state_dict(), checkpoint)
            distilled = Model(make_config(
                str(checkpoint), **config)).eval()
            raw_history = torch.randn(2, 16, 3)
            history, _ = base._to_channel_batch(raw_history)
            expected = base._predict_all_normalized(
                history)[0][:, -1:]
            actual = distilled._student_pair_normalized(
                history)[0]
            torch.testing.assert_close(
                actual, expected, atol=0.0, rtol=0.0)

    def test_loss_only_trains_isolated_second_head(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model = self.make_model_pair(directory)
            model.train()
            history = torch.randn(3, 16, 1)
            target = torch.randn(3, 8, 1)
            losses = model.direct_patch_loss(
                history, target)
            self.assertIn('distill_loss', losses)
            self.assertIn('truth_loss', losses)
            losses['loss'].backward()
            self.assertGreater(
                model.second_residual[-1]
                .weight.grad.abs().sum().item(),
                0,
            )
            self.assertTrue(all(
                parameter.grad is None
                for name, parameter
                in model.named_parameters()
                if not name.startswith(
                    'second_residual.')
            ))

    def test_forecast_commits_two_patches_and_truncates(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model = self.make_model_pair(directory)
            history = torch.randn(2, 16, 1)
            output = model(history, None, None, None)
            self.assertEqual(
                output.shape,
                torch.Size([2, 12, 1]),
            )

    def test_huber_truth_loss_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model = self.make_model_pair(directory)
            model.truth_loss_type = 'huber'
            losses = model.direct_patch_loss(
                torch.randn(3, 16, 1),
                torch.randn(3, 8, 1),
            )
            self.assertTrue(torch.isfinite(losses['truth_loss']))

    def test_huber_is_shared_by_generated_and_clean_far_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model = self.make_model_pair(directory)
            model.truth_loss_type = 'huber'
            prediction = torch.tensor([0.0, 10.0])
            target = torch.zeros_like(prediction)
            actual = model._far_regression_loss(prediction, target)
            expected = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
                beta=model.huber_delta,
            )
            torch.testing.assert_close(actual, expected)
            self.assertLess(actual, torch.nn.functional.mse_loss(
                prediction, target))

    def test_commit_width_generalizes_to_three(self):
        with tempfile.TemporaryDirectory() as directory:
            base = BaseModel(make_config()).eval()
            checkpoint = Path(directory) / 'base.pth'
            torch.save(base.state_dict(), checkpoint)
            model = Model(make_config(
                str(checkpoint),
                distilled_multi_commit_patches=3,
            ))
            self.assertEqual(model.train_pred_len, 12)
            history = torch.randn(2, 16, 1)
            target = torch.randn(2, 12, 1)
            model.direct_patch_loss(
                history, target)['loss'].backward()
            output = model.eval()(
                history, None, None, None)
            self.assertEqual(
                output.shape,
                torch.Size([2, 12, 1]),
            )

    def test_k4_checkpoint_can_evaluate_its_k2_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            base = BaseModel(make_config()).eval()
            checkpoint = Path(directory) / 'base.pth'
            torch.save(base.state_dict(), checkpoint)
            model = Model(make_config(
                str(checkpoint),
                distilled_multi_commit_patches=4,
                distilled_multi_commit_eval_patches=2,
            )).eval()
            history = torch.randn(2, 16, 1)
            expected = model.forecast_with_commit_patches(history, 2)
            actual = model.forecast(history)
            torch.testing.assert_close(actual, expected)

    def test_innovation_readout_exposes_level_and_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base = BaseModel(make_config()).eval()
            checkpoint = Path(directory) / 'base.pth'
            torch.save(base.state_dict(), checkpoint)
            model = Model(make_config(
                str(checkpoint),
                distilled_multi_commit_readout='innovation',
            ))
            hidden = torch.randn(3, 4, 8)
            readout = model._commit_readout(hidden)
            self.assertEqual(readout.shape, torch.Size([3, 16]))
            torch.testing.assert_close(readout[:, :8], hidden[:, -1])
            torch.testing.assert_close(
                readout[:, 8:], hidden[:, -1] - hidden[:, -2]
            )
            self.assertEqual(
                model.second_residual[0].in_features,
                16,
            )

    def test_pure_truth_control_skips_recursive_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            base = BaseModel(make_config()).eval()
            checkpoint = Path(directory) / 'base.pth'
            torch.save(base.state_dict(), checkpoint)
            model = Model(make_config(
                str(checkpoint),
                distilled_multi_commit_patches=4,
                distilled_multi_commit_truth_weight=1.0,
            ))
            with mock.patch.object(
                    model, '_base_hidden', wraps=model._base_hidden) as wrapped:
                losses = model.direct_patch_loss(
                    torch.randn(2, 16, 1),
                    torch.randn(2, 16, 1),
                )
            self.assertEqual(wrapped.call_count, 1)
            self.assertEqual(losses['distill_loss'].item(), 0.0)
            self.assertTrue(torch.isfinite(losses['loss']))


if __name__ == '__main__':
    unittest.main()
