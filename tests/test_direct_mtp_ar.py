import tempfile
import unittest
from pathlib import Path

import torch

from models.DirectMTPAR import Model as DirectMTP
from models.FrozenDirectMTPAR import Model as FrozenDirectMTP
from models.TransformerAblationAR import Model as ParentModel
from tests.test_distilled_multi_commit_ar import make_config


def mtp_config(checkpoint, **overrides):
    values = dict(
        pred_len=32,
        distilled_multi_commit_pretrained=str(checkpoint),
        distilled_multi_commit_patches=4,
        distilled_multi_commit_eval_patches=4,
        distilled_multi_commit_truth_loss="mse",
        ablation_patch_assembly="chronological",
        ablation_patch_assembly_fine_len=2,
        ablation_patch_assembly_atom_dim=3,
        ablation_patch_assembly_interface_dim=6,
        ar_patch_len=4,
        d_model=8,
    )
    values.update(overrides)
    return make_config(**values)


class DirectMTPARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "parent.pth"
        parent = ParentModel(mtp_config(self.checkpoint))
        torch.save(parent.state_dict(), self.checkpoint)

    def test_a16_style_interface_and_future_only_loss(self):
        model = DirectMTP(mtp_config(self.checkpoint))
        self.assertEqual(model.assembly_fine_projection.out_features, 3)
        self.assertEqual(model.assembly_interface_projection.in_features, 6)
        history = torch.randn(3, 16, 2)
        target = torch.randn(3, 16, 2)
        losses = model.direct_patch_loss(history, target)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(model.future_tokens.grad)
        self.assertIsNotNone(model.backbone.output_head.weight.grad)

    def test_every_future_patch_has_an_independent_learned_row(self):
        model = DirectMTP(mtp_config(self.checkpoint))
        self.assertEqual(tuple(model.future_tokens.shape), (1, 4, 8))
        self.assertNotEqual(model.future_tokens.stride(1), 0)
        self.assertTrue(all(
            not torch.equal(
                model.future_tokens[0, left],
                model.future_tokens[0, right],
            )
            for left in range(4)
            for right in range(left + 1, 4)
        ))
        history = torch.randn(3, 16, 1)
        target = torch.randn(3, 16, 1)
        model.direct_patch_loss(history, target)["loss"].backward()
        per_slot_gradient = model.future_tokens.grad[0].abs().sum(dim=-1)
        self.assertTrue(bool((per_slot_gradient > 0).all()))

    def test_frozen_control_updates_only_future_tokens(self):
        model = FrozenDirectMTP(mtp_config(self.checkpoint)).train()
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(trainable, ["future_tokens"])
        history = torch.randn(2, 16, 1)
        target = torch.randn(2, 16, 1)
        model.direct_patch_loss(history, target)["loss"].backward()
        self.assertGreater(model.future_tokens.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if name != "future_tokens"
        ))
        self.assertFalse(model.backbone.training)

    def test_causal_widths_are_numerical_prefixes(self):
        model = DirectMTP(mtp_config(self.checkpoint)).eval()
        history, _ = model._to_channel_batch(torch.randn(2, 16, 1))
        full = model._predict_commit_normalized(history, 4)[0]
        for width in (1, 2, 3):
            prefix = model._predict_commit_normalized(history, width)[0]
            torch.testing.assert_close(
                prefix, full[:, :width], atol=2e-6, rtol=2e-6
            )

    def test_rollout_shape_for_joint_and_frozen(self):
        history = torch.randn(2, 16, 3)
        for model_class in (DirectMTP, FrozenDirectMTP):
            model = model_class(mtp_config(
                self.checkpoint, enc_in=3, features="M"
            )).eval()
            prediction = model.forecast_with_commit_patches(history, 2)
            self.assertEqual(prediction.shape, (2, 32, 3))
            self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
