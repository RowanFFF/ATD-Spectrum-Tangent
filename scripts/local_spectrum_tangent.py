#!/usr/bin/env python3
"""Select and evaluate the local seven-dataset ATD + Spectrum Tangent path.

Selection: FP32 raw-ATD trajectories, 512 train origins per seed, chronological
blocks 1--3 pooled across seeds. Block 4 confirms but never changes selection.
Evaluation: full test split, explicit --open-test, verified immutable inputs.
No training, historical ledgers, plotting, or absorption experiments required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data_provider.data_factory import data_provider
from models.AutoTimesSpectrumTangent import (
    DEFAULT_BETA, build_phase_template, normalized_tangent_direction,
)
from models.MatchedFrozenMultiExitAR import Model
from models.PatchARSpectrumTangent import forecast
from paper_clean_a16_campaign import (
    DATASETS, SEEDS, WIDTHS, method_checkpoint, parent_checkpoint,
)
from paper_four_dataset_parent_campaign import DATASETS as LOADER_DATASETS

PERIODS = tuple(range(12, 217, 12))
HORIZONS = (96, 192, 336, 720)


def model_config(dataset, seed, width, batch_size):
    spec, loader = DATASETS[dataset], LOADER_DATASETS[dataset]
    atoms = spec["atoms"]
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=672, label_len=0, pred_len=720,
        features="M", target="OT", embed="timeF", seasonal_patterns="Monthly",
        root_path=str(ROOT / loader["root_path"]), data_path=loader["data_path"],
        data=loader["data"], freq=loader["freq"], data_scale=True,
        enc_in=loader["channels"], dec_in=loader["channels"], c_out=loader["channels"],
        batch_size=batch_size, eval_batch_size=batch_size, num_workers=0,
        loader_seed=seed, augmentation_ratio=0,
        ar_patch_len=12 * atoms, dense_ar_roll_patches=1,
        dense_ar_eval_commit_patches=1, dense_ar_loss_space="point",
        dense_ar_loss_type="mse", ar_norm_eps=0.001, ar_transformer_norm="post",
        d_model=spec["d_model"], n_heads=8, e_layers=spec["layers"],
        d_layers=1, d_ff=4 * spec["d_model"], dropout=0.1, activation="gelu",
        ablation_attention="softmax", ablation_ffn="mlp", ablation_norm="post",
        ablation_position_encoding="rope", ablation_attention_residual=True,
        ablation_ffn_residual=True, ablation_qk_norm="dot",
        ablation_attention_temperature=1.0, ablation_norm_kind="layer",
        ablation_patch_assembly="chronological", ablation_patch_assembly_fine_len=12,
        ablation_patch_assembly_stem="linear", ablation_patch_assembly_atom_dim=16,
        ablation_patch_assembly_interface_dim=atoms * 16, ablation_output_head="full",
        distilled_multi_commit_pretrained=str(parent_checkpoint(dataset, seed)),
        distilled_multi_commit_hidden=4 * spec["d_model"],
        distilled_multi_commit_patches=width, distilled_multi_commit_eval_patches=width,
        distilled_multi_commit_truth_weight=0.0, distilled_multi_commit_truth_loss="mse",
    )


def load_model(dataset, seed, width, batch_size, device):
    args = model_config(dataset, seed, width, batch_size)
    checkpoint = method_checkpoint(dataset, seed, "generated", width)
    if parent_checkpoint(dataset, seed) is None or checkpoint is None:
        raise FileNotFoundError(f"train parent and ATD first: {dataset}, seed={seed}, K={width}")
    model = Model(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, value in model.state_dict().items():
        if name.endswith(".log_temperature") and name not in state:
            state[name] = value
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(datasets, seeds, widths):
    paths = {Path(__file__), ROOT / "configs/paper_grid.json"}
    for directory in ("models", "layers", "data_provider", "utils"):
        paths.update((ROOT / directory).glob("*.py"))
    paths.update(ROOT / "scripts" / name for name in (
        "paper_clean_a16_campaign.py", "paper_four_dataset_parent_campaign.py",
    ))
    for dataset in datasets:
        spec = LOADER_DATASETS[dataset]
        paths.add(ROOT / spec["root_path"] / spec["data_path"])
        for seed in seeds:
            parent = parent_checkpoint(dataset, seed)
            if parent is None:
                raise FileNotFoundError(f"missing parent: {dataset}, seed={seed}")
            paths.add(parent)
            for width in widths:
                path = method_checkpoint(dataset, seed, "generated", width)
                if path is None:
                    raise FileNotFoundError(f"missing ATD: {dataset}, seed={seed}, K={width}")
                paths.add(path)
    return {str(path.resolve().relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def verify_manifest(manifest):
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"selection input changed or missing: {relative}")


@torch.inference_mode()
def origin_moments(model, observed, truth, width, periods):
    """Per-origin (V, A, B) in data coordinates along uncorrected rollouts."""
    history, channels = model._to_channel_batch(observed[:, -model.seq_len:])
    templates = [build_phase_template(model, history, p) for p in periods]
    moments = torch.zeros(observed.size(0), len(periods), 3,
                          dtype=torch.float64, device=observed.device)
    macro_points = width * model.patch_len
    for age in range(0, model.pred_len, macro_points):
        commits, _, mean, std = model._student_commits_for_width(history, width)
        points = (commits * std + mean).reshape(history.size(0), -1, 1)
        raw = model._from_channel_batch(points[..., 0], observed.size(0), channels)
        valid = min(macro_points, model.pred_len - age)
        error = (truth[:, age:age + valid] - raw[:, :valid]).double()
        for index, template in enumerate(templates):
            direction = normalized_tangent_direction(
                commits, template, age // model.patch_len, model.patch_len
            )
            direction = model._from_channel_batch(
                (direction * std).reshape(history.size(0), -1), observed.size(0), channels
            )[:, :valid].double()
            moments[:, index, 0] += error.square().sum((1, 2))
            moments[:, index, 1] += direction.square().sum((1, 2))
            moments[:, index, 2] += (direction * error).sum((1, 2))
        history = torch.cat([history, points], dim=1)[:, -model.seq_len:]
    return (moments / (model.pred_len * channels)).cpu().numpy()


def select_period(blocks, periods):
    """blocks: [seed, four chronological blocks, period, (V,A,B)]."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (4, len(periods), 3):
        raise ValueError("expected seed x 4 blocks x periods x 3 moments")
    if not np.isfinite(values).all():
        raise ValueError("non-finite selection moments")
    fit = values[:, :3].mean(axis=(0, 1))
    candidates = []
    for index, (v, a, b) in enumerate(fit):
        gamma = max(float(b / max(a, 1e-30)), 0.0)
        gain = 2 * gamma * b - gamma * gamma * a
        candidates.append((-gain / max(v, 1e-30), periods[index], index, gamma))
    negative_score, period, index, gamma = min(candidates)
    v, a, b = values[:, 3, index].mean(axis=0)
    return dict(period=int(period), gamma=float(gamma), alpha=float(gamma * DEFAULT_BETA),
                fit_r2=float(-negative_score),
                confirmation_r2=float((2 * gamma * b - gamma * gamma * a) / max(v, 1e-30)))


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows):
    groups = {}
    for row in rows:
        key = (row["dataset"], row["width"], row["method"], row["horizon"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (dataset, width, method, horizon), group in sorted(groups.items()):
        row = dict(dataset=dataset, width=width, method=method, horizon=horizon, seeds=len(group))
        for metric in ("mse", "mae"):
            values = [cell[metric] for cell in group]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    return summary


def select(args, device):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use an empty output directory; selection is never overwritten")
    manifest = input_manifest(args.datasets, args.seeds, args.widths)
    selected, records = [], []
    for dataset in args.datasets:
        batch_size = min(args.batch_size, LOADER_DATASETS[dataset]["eval_batch_size"])
        for width in args.widths:
            seed_blocks = []
            for seed in args.seeds:
                model, config = load_model(dataset, seed, width, batch_size, device)
                train, _ = data_provider(config, "train")
                if len(train) < args.train_origins:
                    raise ValueError("not enough train origins for the requested protocol")
                indices = torch.linspace(0, len(train) - 1, args.train_origins).round().long().tolist()
                loader = DataLoader(Subset(train, indices), batch_size=batch_size, shuffle=False)
                chunks = [origin_moments(model, x.float().to(device), y.float().to(device),
                                         width, PERIODS) for x, y, _, _ in loader]
                moments = np.concatenate(chunks)
                blocks = np.stack([chunk.mean(axis=0) for chunk in np.array_split(moments, 4)])
                seed_blocks.append(blocks)
                for block, values in enumerate(blocks, 1):
                    for period, (v, a, b) in zip(PERIODS, values):
                        records.append(dict(dataset=dataset, width=width, seed=seed, block=block,
                                            period=period, V=v, A=a, B=b))
                del model
            row = dict(dataset=dataset, width=width, **select_period(seed_blocks, PERIODS))
            selected.append(row)
            print(row, flush=True)
    verify_manifest(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "train_moments.csv", records)
    lock = dict(schema=1, datasets=args.datasets, seeds=args.seeds, widths=args.widths,
                train_origins=args.train_origins, periods=PERIODS, selections=selected,
                selection_precision="float32", test_precision="CUDA autocast FP16; CPU FP32",
                batch_size=args.batch_size, inputs=manifest,
                environment=dict(torch=torch.__version__, python=sys.version.split()[0]))
    with (args.output_dir / "selection_lock.json").open("x") as handle:
        json.dump(lock, handle, indent=2)
        handle.write("\n")


@torch.inference_mode()
def evaluate(args, device):
    if not args.open_test:
        raise ValueError("test requires --open-test after train-only selection")
    lock_path = args.output_dir / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    lock_hash = sha256(lock_path)
    verify_manifest(lock["inputs"])
    output = args.output_dir / "test_metrics.csv"
    summary_path = args.output_dir / "test_summary.csv"
    if output.exists() or summary_path.exists():
        raise FileExistsError("test output already exists; it will not be overwritten")
    rows = []
    for selection in lock["selections"]:
        dataset, width = selection["dataset"], selection["width"]
        batch_size = min(lock["batch_size"], LOADER_DATASETS[dataset]["eval_batch_size"])
        for seed in lock["seeds"]:
            model, config = load_model(dataset, seed, width, batch_size, device)
            _, loader = data_provider(config, "test")
            sums = {(method, h): np.zeros(3) for method in ("atd", "tangent") for h in HORIZONS}
            origins = 0
            prefix_error = 0.0
            for x, y, _, _ in loader:
                x, truth = x.float().to(device), y[:, -config.pred_len:].float().to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    raw = model.forecast_with_commit_patches(x, width)
                    corrected = forecast(model, x, width, selection["period"], selection["gamma"])
                prefix_error = max(prefix_error, float((raw[:, :model.patch_len]
                                                       - corrected[:, :model.patch_len]).abs().max()))
                for method, prediction in (("atd", raw), ("tangent", corrected)):
                    for horizon in HORIZONS:
                        error = (prediction[:, :horizon].float() - truth[:, :horizon]).double()
                        sums[method, horizon] += [float(error.square().sum()),
                                                 float(error.abs().sum()), error.numel()]
                origins += x.size(0)
            for (method, horizon), (squared, absolute, count) in sums.items():
                rows.append(dict(dataset=dataset, seed=seed, width=width, method=method,
                                 horizon=horizon, mse=squared / count, mae=absolute / count,
                                 origins=origins, first_patch_max_abs_delta=prefix_error,
                                 device=str(device), precision="fp16_autocast" if device.type == "cuda" else "fp32",
                                 selection_sha256=lock_hash))
            del model
    verify_manifest(lock["inputs"])
    if sha256(lock_path) != lock_hash:
        raise RuntimeError("selection lock changed during evaluation")
    write_csv(output, rows)
    write_csv(summary_path, summarize_metrics(rows))
    print(f"Wrote {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--widths", nargs="+", type=int, choices=WIDTHS)
    parser.add_argument("--train-origins", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/local_spectrum_tangent")
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "test" and any(getattr(args, key) is not None for key in ("datasets", "seeds", "widths")):
        parser.error("test scope comes from the selection lock; omit --datasets/--seeds/--widths")
    args.datasets = args.datasets or list(DATASETS)
    args.seeds = args.seeds or list(SEEDS)
    args.widths = args.widths or list(WIDTHS)
    if args.batch_size < 1 or args.train_origins < 4 or args.train_origins % 4:
        parser.error("batch size must be positive; train origins must be a positive multiple of four")
    for key in ("datasets", "seeds", "widths"):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, datasets=args.datasets, seeds=args.seeds,
                              widths=args.widths, train_origins=args.train_origins,
                              test_uses_selection_lock=True, output_dir=str(args.output_dir)), indent=2))
        return
    device = torch.device(args.device)
    (select if args.stage == "select" else evaluate)(args, device)


if __name__ == "__main__":
    main()
