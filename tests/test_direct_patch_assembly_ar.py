import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from models.TransformerAblationAR import Model
from models.DistilledMultiCommitAR import Model as DistilledModel
from tests.test_distilled_multi_commit_ar import (
    make_config as make_distilled_config,
)


def assembly_config(mode='chronological', stem='identity', **overrides):
    values = dict(
        seq_len=8,
        pred_len=8,
        ar_patch_len=4,
        d_model=4,
        n_heads=2,
        d_ff=8,
        ablation_patch_assembly=mode,
        ablation_patch_assembly_fine_len=2,
        ablation_patch_assembly_stem=stem,
    )
    values.update(overrides)
    return make_distilled_config(**values)


class DirectPatchAssemblyARTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_identity_chronological_concat_is_exact_raw_parent(self):
        model = Model(assembly_config())
        parent = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        assembled = model._assemble_parent_patches(parent)
        torch.testing.assert_close(assembled, parent)
        self.assertIsInstance(
            model.backbone.patch_projection, nn.Identity)

    def test_reverse_and_content_sorted_orders_are_exact(self):
        reverse = Model(assembly_config('reverse'))
        parent = torch.tensor([[[10.0, 12.0, -2.0, 0.0]]])
        torch.testing.assert_close(
            reverse._assemble_parent_patches(parent),
            torch.tensor([[[-2.0, 0.0, 10.0, 12.0]]]),
        )
        sorted_model = Model(assembly_config('mean_sorted'))
        torch.testing.assert_close(
            sorted_model._assemble_parent_patches(parent),
            torch.tensor([[[-2.0, 0.0, 10.0, 12.0]]]),
        )

    def test_assembly_is_parent_local_and_causal(self):
        model = Model(assembly_config('mean_sorted')).eval()
        parents = torch.randn(2, 3, 4)
        changed = parents.clone()
        changed[:, 2] = torch.randn_like(changed[:, 2]) * 100.0
        first = model._assemble_parent_patches(parents)
        second = model._assemble_parent_patches(changed)
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_orders_have_identical_parameter_count(self):
        counts = []
        for mode in ('chronological', 'reverse', 'mean_sorted'):
            model = Model(assembly_config(mode, stem='linear'))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_q1_forecast_and_loss_keep_original_ar_shape(self):
        model = Model(assembly_config('mean_sorted'))
        history = torch.randn(3, 8, 2)
        target = torch.randn(3, 4, 2)
        loss = model.direct_patch_loss(history, target)
        loss['optimization_loss'].backward()
        self.assertTrue(torch.isfinite(loss['loss']))
        self.assertIsNotNone(
            model.backbone.output_head.weight.grad)
        forecast = model.eval().forecast(history)
        self.assertEqual(forecast.shape, (3, 8, 2))
        self.assertTrue(torch.isfinite(forecast).all())

    def test_direct_k_queries_keep_dense_ar_interface(self):
        model = Model(assembly_config(
            'mean_sorted',
            dense_ar_roll_patches=2,
        ))
        history = torch.randn(2, 8, 1)
        predictions, history_patches, _, _ = (
            model._predict_all_normalized(history))
        self.assertEqual(history_patches.shape, (2, 2, 4))
        self.assertEqual(predictions.shape, (2, 3, 4))
        target = torch.randn(2, 8, 1)
        self.assertTrue(torch.isfinite(
            model.direct_patch_loss(history, target)['loss']))

    def test_last_query_path_matches_full_assembly_path(self):
        model = Model(assembly_config('mean_sorted')).eval()
        history = torch.randn(3, 8, 1)
        expected = model._predict_all_normalized(history)[0][:, -1:]
        actual = model._predict_last_normalized(history)[0]
        torch.testing.assert_close(
            actual, expected, atol=2e-6, rtol=2e-6)

    def test_dpod_protocol_loads_assembly_parent_exactly(self):
        assembly = dict(
            ablation_patch_assembly='mean_sorted',
            ablation_patch_assembly_fine_len=2,
            ablation_patch_assembly_stem='linear',
        )
        base = Model(make_distilled_config(**assembly)).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'assembly_parent.pth'
            torch.save(base.state_dict(), checkpoint)
            distilled = DistilledModel(make_distilled_config(
                str(checkpoint), **assembly)).eval()
            history = torch.randn(3, 16, 1)
            expected = base._predict_all_normalized(history)[0][:, -1:]
            actual = distilled._student_pair_normalized(history)[0]
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
