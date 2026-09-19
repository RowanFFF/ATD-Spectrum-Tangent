#!/usr/bin/env python3
"""Validation-only multi-exit audit for the AutoTimes ATD-K8 pilot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from autotimes_atd_ds1_pilot import (  # noqa: E402
    DATASET,
    SEED as DEFAULT_SEED,
    TOKEN_LEN,
    WIDTH,
    checkpoint,
)
from data_provider.data_factory import data_provider  # noqa: E402
from models.AutoTimesOperatorAR import Model  # noqa: E402
from paper_four_dataset_parent_campaign import DATASETS  # noqa: E402


HORIZONS = (96, 192, 336, 720)
OUTPUT = ROOT / "analysis_outputs" / "autotimes_atd_ds1_validation.csv"


def model_config(llm_checkpoint, mode, parent=None):
    return SimpleNamespace(
        seq_len=672,
        pred_len=720,
        d_model=768,
        dropout=0.1,
        external_autotimes_llm_checkpoint=str(llm_checkpoint),
        external_autotimes_pretrained=str(parent or ""),
        external_autotimes_mode=mode,
        external_autotimes_token_len=TOKEN_LEN,
        external_autotimes_mlp_hidden=512,
        external_autotimes_mlp_layers=2,
        distilled_multi_commit_hidden=512,
        distilled_multi_commit_patches=1 if mode == "q1" else WIDTH,
        distilled_multi_commit_eval_patches=1 if mode == "q1" else WIDTH,
        progressive_nested_protected_patches=WIDTH // 2,
    )


def data_config(eval_batch_size, seed=DEFAULT_SEED):
    spec = DATASETS[DATASET]
    return SimpleNamespace(
        task_name="long_term_forecast",
        root_path=spec["root_path"],
        data_path=spec["data_path"],
        data=spec["data"],
        features="M",
        target="OT",
        freq=spec["freq"],
        embed="timeF",
        seasonal_patterns="Monthly",
        seq_len=672,
        label_len=576,
        pred_len=720,
        data_scale=True,
        batch_size=eval_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=0,
        loader_seed=seed,
        augmentation_ratio=0,
    )


def load_model(path, config, device):
    model = Model(config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def empty_metric():
    return {
        horizon: {"square": 0.0, "absolute": 0.0, "count": 0}
        for horizon in HORIZONS
    }


def update_metric(metric, prediction, truth):
    for horizon in HORIZONS:
        error = (prediction[:, :horizon] - truth[:, :horizon]).double()
        metric[horizon]["square"] += float(error.square().sum().cpu())
        metric[horizon]["absolute"] += float(error.abs().sum().cpu())
        metric[horizon]["count"] += error.numel()


def finish_metric(metric):
    result = {}
    for horizon in HORIZONS:
        count = metric[horizon]["count"]
        result[f"h{horizon}_mse"] = metric[horizon]["square"] / count
        result[f"h{horizon}_mae"] = metric[horizon]["absolute"] / count
    return result


def zero_metric_fields(prefix=""):
    return {
        f"{prefix}h{horizon}_{name}": 0.0
        for horizon in HORIZONS
        for name in ("mse", "mae")
    }


@torch.inference_mode()
def evaluate(q1, atd, args, split, seed, device):
    _, loader = data_provider(args, split)
    truth_metrics = {width: empty_metric() for width in (1, 2, 4, 8)}
    q1_metric = empty_metric()
    fidelity = {width: empty_metric() for width in (1, 2, 4, 8)}
    k1_max_abs = 0.0
    values = 0
    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float().to(device)
        truth = batch_y[:, -args.pred_len:].float().to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=(
                device.type == "cuda"
                and not bool(getattr(args, "disable_autocast", False))
            ),
        ):
            q1_prediction = q1.forecast_with_commit_patches(batch_x, 1)
            predictions = {
                width: atd.forecast_with_commit_patches(batch_x, width)
                for width in (1, 2, 4, 8)
            }
        q1_prediction = q1_prediction.float()
        update_metric(q1_metric, q1_prediction, truth)
        for width, prediction in predictions.items():
            prediction = prediction.float()
            update_metric(truth_metrics[width], prediction, truth)
            update_metric(fidelity[width], prediction, q1_prediction)
        delta = (predictions[1].float() - q1_prediction).abs()
        k1_max_abs = max(k1_max_abs, float(delta.max().cpu()))
        values += delta.numel()
    rows = [{
        "dataset": DATASET,
        "seed": seed,
        "data_scale": 1,
        "split": "validation" if split == "val" else "test",
        "artifact": "q1",
        "commit_patches": 1,
        "backbone_calls": 8,
        **finish_metric(q1_metric),
        **zero_metric_fields("fidelity_"),
        "q1_max_abs_delta": 0.0,
        "compared_values": values,
    }]
    for width in (1, 2, 4, 8):
        row = {
            "dataset": DATASET,
            "seed": seed,
            "data_scale": 1,
            "split": "validation" if split == "val" else "test",
            "artifact": "atd_k8",
            "commit_patches": width,
            "backbone_calls": (720 + width * TOKEN_LEN - 1) // (
                width * TOKEN_LEN
            ),
            **finish_metric(truth_metrics[width]),
            **{
                f"fidelity_{name}": value
                for name, value in finish_metric(fidelity[width]).items()
            },
            "q1_max_abs_delta": k1_max_abs if width == 1 else "",
            "compared_values": values,
        }
        rows.append(row)
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-checkpoint", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        raise ValueError("eval batch size must be positive")
    q1_path = checkpoint("q1", seed=args.seed)
    atd_path = checkpoint(
        "generated_k8", args.learning_rate, seed=args.seed)
    if q1_path is None or atd_path is None:
        raise FileNotFoundError("Q1 or ATD-K8 checkpoint is missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_model(
        q1_path,
        model_config(args.llm_checkpoint, "q1"),
        device,
    )
    atd = load_model(
        atd_path,
        model_config(args.llm_checkpoint, "generated", q1_path),
        device,
    )
    output = args.output
    if args.seed != DEFAULT_SEED and output == OUTPUT:
        output = ROOT / "analysis_outputs" / (
            f"autotimes_atd_ds1_validation_s{args.seed}.csv"
        )
    if args.split == "test" and output == OUTPUT:
        output = ROOT / "analysis_outputs" / "autotimes_atd_ds1_test.csv"
    elif args.split == "test" and args.output == OUTPUT:
        output = ROOT / "analysis_outputs" / (
            f"autotimes_atd_ds1_test_s{args.seed}.csv"
        )
    rows = evaluate(
        q1, atd, data_config(args.eval_batch_size, args.seed),
        args.split, args.seed, device
    )
    write_rows(output, rows)
    for row in rows:
        print(
            f"{row['artifact']} K{row['commit_patches']}: "
            f"H720 MSE={row['h720_mse']:.6f} "
            f"MAE={row['h720_mae']:.6f}",
            flush=True,
        )
    print(f"output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
