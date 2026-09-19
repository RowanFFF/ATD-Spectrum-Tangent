import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.MatchedFrozenMultiExitAR import Model
from models.TransformerAblationAR import Model as Parent
from models.PatchARSpectrumTangent import forecast
from scripts import local_spectrum_tangent as runner
from scripts import check_release
from test_distilled_multi_commit_ar import make_config


class LocalTangentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        checkpoint = Path(self.directory.name) / "parent.pth"
        options = dict(seq_len=24, pred_len=28, features="M", enc_in=3,
                       distilled_multi_commit_patches=4,
                       distilled_multi_commit_eval_patches=4)
        parent = Parent(make_config(**options)).eval()
        torch.save(parent.state_dict(), checkpoint)
        self.model = Model(make_config(str(checkpoint), **options)).eval()
        self.observed = torch.randn(2, 24, 3)

    def test_exact_fallbacks(self):
        for width, gamma in ((4, 0.0), (1, 2.0)):
            expected = self.model.forecast_with_commit_patches(self.observed, width)
            actual = forecast(self.model, self.observed, width, 8, gamma)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shape_prefix_and_no_parameter_mutation(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        raw = self.model.forecast_with_commit_patches(self.observed, 4)
        corrected = forecast(self.model, self.observed, 4, 8, 1.0)
        self.assertEqual(corrected.shape, (2, 28, 3))
        torch.testing.assert_close(raw[:, :4], corrected[:, :4], rtol=0, atol=0)
        self.assertFalse(torch.equal(raw[:, 4:], corrected[:, 4:]))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_invalid_strength_and_training_mode(self):
        for gamma in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                forecast(self.model, self.observed, 4, 8, gamma)
        self.model.train()
        with self.assertRaises(ValueError):
            forecast(self.model, self.observed, 4, 8, 1.0)

    def test_zero_residual_moments(self):
        raw = self.model.forecast_with_commit_patches(self.observed, 4).detach()
        moments = runner.origin_moments(self.model, self.observed, raw, 4, (4, 8))
        np.testing.assert_allclose(moments[..., 0], 0, atol=1e-12)
        np.testing.assert_allclose(moments[..., 2], 0, atol=1e-12)
        self.assertTrue((moments[..., 1] > 0).all())

    def test_holdout_does_not_select_and_alpha_units(self):
        blocks = np.ones((3, 4, 2, 3))
        blocks[:, :3, 1, 2] = 2
        first = runner.select_period(blocks, (12, 24))
        blocks[:, 3, :, 2] = -100
        second = runner.select_period(blocks, (12, 24))
        self.assertEqual(first["period"], 24)
        self.assertEqual(first["gamma"], 2)
        self.assertEqual(first["alpha"], 0.125)
        self.assertEqual(first["period"], second["period"])
        self.assertEqual(first["gamma"], second["gamma"])
        self.assertLess(second["confirmation_r2"], 0)

    def test_nonnegative_fit_and_period_tie(self):
        blocks = np.ones((1, 4, 2, 3))
        blocks[..., 2] = -2
        result = runner.select_period(blocks, (24, 12))
        self.assertEqual(result["period"], 12)
        self.assertEqual(result["gamma"], 0)

    def test_manifest_change_is_rejected(self):
        root = Path(self.directory.name)
        path = root / "input.csv"
        path.write_text("original")
        with mock.patch.object(runner, "ROOT", root):
            manifest = {"input.csv": runner.sha256(path)}
            runner.verify_manifest(manifest)
            path.write_text("changed")
            with self.assertRaises(RuntimeError):
                runner.verify_manifest(manifest)

    def test_test_gate_precedes_data_access(self):
        with mock.patch.object(runner, "data_provider") as provider:
            with self.assertRaisesRegex(ValueError, "open-test"):
                runner.evaluate(SimpleNamespace(open_test=False), torch.device("cpu"))
            provider.assert_not_called()

    def test_select_to_test_synthetic_workflow(self):
        x, y = torch.randn(8, 24, 3), torch.randn(8, 28, 3)
        data = TensorDataset(x, y, torch.zeros(8, 24, 1), torch.zeros(8, 28, 1))
        args = SimpleNamespace(datasets=["ETTh1"], seeds=[2021], widths=[4], batch_size=2,
                               train_origins=8, open_test=True,
                               output_dir=Path(self.directory.name) / "output")
        flags = []

        def provider(config, flag):
            flags.append(flag)
            return data, DataLoader(data, batch_size=2)

        with mock.patch.object(runner, "PERIODS", (4, 8)), \
             mock.patch.object(runner, "HORIZONS", (4, 12, 28)), \
             mock.patch.object(runner, "input_manifest", return_value={}), \
             mock.patch.object(runner, "data_provider", side_effect=provider), \
             mock.patch.object(runner, "load_model", return_value=(self.model, SimpleNamespace(pred_len=28))):
            runner.select(args, torch.device("cpu"))
            self.assertEqual(flags, ["train"])
            lock = json.loads((args.output_dir / "selection_lock.json").read_text())
            self.assertEqual(lock["seeds"], [2021])
            runner.evaluate(args, torch.device("cpu"))
            self.assertEqual(flags, ["train", "test"])
            self.assertTrue((args.output_dir / "test_metrics.csv").is_file())
            self.assertTrue((args.output_dir / "test_summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                runner.select(args, torch.device("cpu"))
            with self.assertRaises(FileExistsError):
                runner.evaluate(args, torch.device("cpu"))

    def test_paper_grid_matches_runtime(self):
        grid = json.loads((runner.ROOT / "configs/paper_grid.json").read_text())
        for name, cell in grid["datasets"].items():
            cfg = runner.model_config(name, 2021, 4, 8)
            self.assertEqual(cfg.ar_patch_len, cell["parent_patch"])
            self.assertEqual(cfg.d_model, cell["d_model"])
            self.assertEqual(cfg.e_layers, cell["layers"])
            self.assertEqual(cfg.enc_in, cell["channels"])
            self.assertEqual(cfg.ar_norm_eps, 0.001)

    def test_real_csv_loader_and_paper_config_checkpoint_roundtrip(self):
        """CPU smoke with synthetic CSV/random weights, never benchmark data."""
        root = Path(self.directory.name)
        folder = root / "dataset/weather"
        folder.mkdir(parents=True)
        # 20% test = 720 points: precisely one complete H720 test origin.
        frame = pd.DataFrame(np.random.default_rng(7).normal(size=(3600, 21)),
                             columns=[f"v{i}" for i in range(20)] + ["OT"])
        frame.insert(0, "date", pd.date_range("2020-01-01", periods=3600, freq="10min"))
        frame.to_csv(folder / "weather.csv", index=False)
        parent_path, atd_path = root / "paper_parent.pth", root / "paper_atd.pth"
        args = SimpleNamespace(datasets=["Weather"], seeds=[2021], widths=[4], batch_size=2,
                               train_origins=4, open_test=True, output_dir=root / "paper_smoke")
        with mock.patch.object(runner, "ROOT", root), \
             mock.patch.object(runner, "parent_checkpoint", return_value=parent_path), \
             mock.patch.object(runner, "method_checkpoint", return_value=atd_path), \
             mock.patch.object(runner, "input_manifest", return_value={}):
            config = runner.model_config("Weather", 2021, 4, 2)
            parent = Parent(config).eval()
            torch.save(parent.state_dict(), parent_path)
            compiled = Model(config).eval()
            torch.save(compiled.state_dict(), atd_path)
            loaded, _ = runner.load_model("Weather", 2021, 4, 2, torch.device("cpu"))
            for key, value in compiled.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            runner.select(args, torch.device("cpu"))
            runner.evaluate(args, torch.device("cpu"))
        metrics = pd.read_csv(args.output_dir / "test_metrics.csv")
        self.assertEqual(len(metrics), 8)
        self.assertEqual(metrics.origins.unique().tolist(), [1])
        self.assertTrue(np.isfinite(metrics.mse).all())
        self.assertEqual(metrics.first_patch_max_abs_delta.max(), 0)

    def test_summary_single_seed_does_not_claim_zero_uncertainty(self):
        rows = [dict(dataset="ETTh1", width=4, method="atd", horizon=720, mse=1.0, mae=0.5)]
        self.assertIsNone(runner.summarize_metrics(rows)[0]["mse_std"])

    def test_release_scan_excludes_environment_not_source(self):
        root = Path(self.directory.name)
        (root / ".venv").mkdir()
        (root / ".venv" / "weights.bin").write_bytes(b"not publishable")
        (root / "source.py").write_text("pass\n")
        with mock.patch.object(check_release, "ROOT", root):
            names = {path.relative_to(root).as_posix() for path in check_release.release_files()}
        self.assertNotIn(".venv/weights.bin", names)
        self.assertIn("source.py", names)


if __name__ == "__main__":
    unittest.main()
