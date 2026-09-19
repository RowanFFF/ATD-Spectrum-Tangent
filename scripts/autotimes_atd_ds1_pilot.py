#!/usr/bin/env python3
"""Validation-only AutoTimes ATD-K8 pilot under the unified ds1 protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CHECKPOINT_ROOT = ROOT / "checkpoints"
LOG_ROOT = ROOT / "experiment_logs" / "autotimes_atd_ds1"
TOKEN_LEN = 96
WIDTH = 8
DATASET = "ETTh1"
SEED = 2021
LEARNING_RATES = (3e-5, 1e-4, 3e-4)

sys.path.insert(0, str(ROOT / "scripts"))
from paper_four_dataset_parent_campaign import DATASETS  # noqa: E402


def learning_rate_tag(value):
    return f"{float(value):g}".replace(".", "p")


def model_id(variant, learning_rate=None, seed=SEED):
    if variant == "q1":
        return f"PATD_EXT_AUTOTIMES_q1_{DATASET}_T96_ds1_s{seed}"
    if variant != "generated_k8" or learning_rate is None:
        raise ValueError("generated_k8 requires a learning rate")
    return (
        f"PATD_EXT_AUTOTIMES_generated_k8_{DATASET}_T96_"
        f"lr{learning_rate_tag(learning_rate)}_ds1_s{seed}")


def log_path(variant, learning_rate=None, seed=SEED):
    return LOG_ROOT / f"{model_id(variant, learning_rate, seed)}.log"


def checkpoint(variant, learning_rate=None, seed=SEED):
    identifier = model_id(variant, learning_rate, seed)
    matches = [
        directory / "checkpoint.pth"
        for directory in CHECKPOINT_ROOT.glob(f"*{identifier}*")
        if (directory / "checkpoint.pth").is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple checkpoints for {identifier}: {matches}")
    return matches[0] if matches else None


def common_flags(
        llm_checkpoint, identifier, learning_rate, epochs, patience, seed):
    spec = DATASETS[DATASET]
    return [
        str(PYTHON), "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", spec["root_path"],
        "--data_path", spec["data_path"],
        "--model_id", identifier,
        "--model", "AutoTimesOperatorAR",
        "--data", spec["data"],
        "--features", "M",
        "--seq_len", "672",
        "--label_len", "576",
        "--pred_len", "720",
        "--freq", spec["freq"],
        "--enc_in", str(spec["channels"]),
        "--dec_in", str(spec["channels"]),
        "--c_out", str(spec["channels"]),
        "--d_model", "768",
        "--dropout", "0.1",
        "--external_autotimes_llm_checkpoint", str(llm_checkpoint),
        "--external_autotimes_token_len", str(TOKEN_LEN),
        "--external_autotimes_mlp_hidden", "512",
        "--external_autotimes_mlp_layers", "2",
        "--distilled_multi_commit_hidden", "512",
        "--batch_size", "256",
        "--eval_batch_size", "128",
        "--num_workers", "0",
        "--direct_patch_train_channels", "0",
        "--direct_patch_validation_channels", "0",
        "--learning_rate", str(learning_rate),
        "--train_epochs", str(epochs),
        "--patience", str(patience),
        "--seed", str(seed),
        "--loader_seed", str(seed),
        "--checkpoint_selection", "vali",
        "--stream_metrics",
        "--report_horizons", "96", "192", "336", "720",
        "--skip_epoch_test",
        "--skip_final_test",
        "--lradj", "cosine",
        "--use_amp",
        "--data_scale",
        "--des", "autotimes_atd_ds1_pilot",
        "--itr", "1",
    ]


def command(
        variant, llm_checkpoint, learning_rate=None, epochs=6, patience=2,
        seed=SEED):
    if variant == "q1":
        flags = common_flags(
            llm_checkpoint, model_id("q1", seed=seed), 0.002,
            epochs, patience, seed)
        flags.extend([
            "--external_autotimes_mode", "q1",
            "--distilled_multi_commit_patches", "1",
            "--distilled_multi_commit_eval_patches", "1",
            "--direct_patch_validation",
        ])
        return flags
    parent = checkpoint("q1", seed=seed)
    if parent is None:
        raise FileNotFoundError("the ds1 AutoTimes Q1 checkpoint is missing")
    flags = common_flags(
        llm_checkpoint,
        model_id("generated_k8", learning_rate, seed),
        learning_rate,
        epochs,
        patience,
        seed,
    )
    flags.extend([
        "--external_autotimes_mode", "generated",
        "--external_autotimes_pretrained", str(parent),
        "--distilled_multi_commit_patches", str(WIDTH),
        "--distilled_multi_commit_eval_patches", str(WIDTH),
    ])
    # Deliberately omit --include_initial_checkpoint: the pilot asks whether a
    # trained far branch can be validation-selected, not whether repetition of
    # the first patch is a strong untrained fallback.
    return flags


def run(
        variant, llm_checkpoint, learning_rate, epochs, patience, force,
        seed=SEED):
    existing = checkpoint(variant, learning_rate, seed)
    if existing is not None and not force:
        print(f"reuse {existing.relative_to(ROOT)}", flush=True)
        return
    path = log_path(variant, learning_rate, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command(
                variant, llm_checkpoint, learning_rate,
                epochs, patience, seed),
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"training failed; inspect {path}")
    print(
        f"completed {model_id(variant, learning_rate, seed)} in "
        f"{(time.time() - started) / 60.0:.2f} min",
        flush=True,
    )


def validation_history(path):
    pattern = re.compile(
        r"Epoch:\s*(\d+), Steps:\s*(\d+).*Vali Loss:\s*"
        r"([-+0-9.eE]+)")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({
                "epoch": int(match.group(1)),
                "steps": int(match.group(2)),
                "validation_h720_loss": float(match.group(3)),
            })
    return rows


def summarize(seed=SEED, learning_rates=LEARNING_RATES):
    rows = []
    for learning_rate in learning_rates:
        path = log_path("generated_k8", learning_rate, seed)
        if not path.is_file():
            continue
        history = validation_history(path)
        if not history:
            continue
        best = min(history, key=lambda row: row["validation_h720_loss"])
        rows.append({
            "learning_rate": learning_rate,
            "trained_epochs": len(history),
            "best_epoch": best["epoch"],
            "best_validation_h720_loss": best["validation_h720_loss"],
            "checkpoint": str(
                checkpoint(
                    "generated_k8", learning_rate, seed).relative_to(ROOT)),
        })
    for row in rows:
        print(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("q1", "screen", "all", "summarize"))
    parser.add_argument("--llm-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+",
        default=list(LEARNING_RATES),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    llm_checkpoint = args.llm_checkpoint.expanduser().resolve()
    if not (llm_checkpoint / "config.json").is_file():
        raise FileNotFoundError(llm_checkpoint)
    if args.stage == "summarize":
        summarize(args.seed, args.learning_rates)
        return 0
    stages = ("q1", "screen") if args.stage == "all" else (args.stage,)
    for stage in stages:
        jobs = (
            [("q1", None)] if stage == "q1"
            else [("generated_k8", value) for value in args.learning_rates]
        )
        for variant, learning_rate in jobs:
            flags = command(
                variant, llm_checkpoint, learning_rate,
                args.epochs, args.patience, args.seed)
            if args.dry_run:
                print(" ".join(flags))
            else:
                run(
                    variant, llm_checkpoint, learning_rate,
                    args.epochs, args.patience, args.force, args.seed)
    if not args.dry_run:
        summarize(args.seed, args.learning_rates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
