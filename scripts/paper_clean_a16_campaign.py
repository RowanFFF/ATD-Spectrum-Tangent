#!/usr/bin/env python3
"""Run the isolated A16 parent and matched decoder paper campaign.

This campaign never writes a canonical research ledger.  Every identifier,
log, protocol artifact, and summary is scoped to ``paper_clean_a16_v1``.
Training is validation-only; ``test`` is a separate explicit stage that opens
the full-origin test split once for each frozen checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_four_dataset_parent_campaign import (  # noqa: E402
    CHECKPOINT_ROOT,
    DATASETS as LEGACY_DATASETS,
    RESULT_ROOT,
    command as legacy_parent_command,
    find_artifact,
    read_metrics,
)
from paper_iclr_causal_k4_campaign import replace  # noqa: E402


CAMPAIGN = "paper_clean_a16_v1_20260814"
OUTPUT_ROOT = ROOT / "analysis_outputs" / "exploratory" / CAMPAIGN
LOG_ROOT = ROOT / "experiment_logs" / CAMPAIGN
RUN_LOG = OUTPUT_ROOT / "run_status.csv"
METRIC_OUTPUT = OUTPUT_ROOT / "test_metrics.csv"
VALIDATION_PATTERN = re.compile(
    r"Epoch: (\d+), Steps: \d+ \| Train Loss: [^ ]+ "
    r"Vali Loss: ([0-9.eE+-]+)"
)
SEEDS = (2021, 2022, 2023)
WIDTHS = (4, 8)


# P and D were frozen before this clean rerun.  A=16 is global; K=P/12 only
# determines the concat width entering the dataset-specific D-dimensional
# Transformer interface.
DATASETS = {
    "ETTh1": {"atoms": 2, "d_model": 64, "layers": 1},
    "ETTh2": {"atoms": 4, "d_model": 64, "layers": 1},
    "ETTm1": {"atoms": 8, "d_model": 64, "layers": 1},
    "ETTm2": {"atoms": 2, "d_model": 64, "layers": 2},
    "Weather": {"atoms": 4, "d_model": 64, "layers": 2},
    "ECL": {"atoms": 8, "d_model": 256, "layers": 2},
    "Traffic": {"atoms": 8, "d_model": 256, "layers": 2},
}
PERIODS = {
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
    "Weather": 144,
    "ECL": 168,
    "Traffic": 168,
}
METHODS = {
    "direct_mtp_joint": {
        "model": "DirectMTPAR",
        "target": "clean future at causal placeholder positions",
        "trunk": "joint",
    },
    "direct_mtp_frozen": {
        "model": "FrozenDirectMTPAR",
        "target": "clean future at causal placeholder positions",
        "trunk": "frozen; placeholders only",
    },
    "frozen_direct_exit": {
        "model": "MatchedFrozenMultiExitAR",
        "target": "clean future at Generated-matched far exits",
        "trunk": "frozen",
    },
    "generated": {
        "model": "MatchedFrozenMultiExitAR",
        "target": "deployed recursive Q1 trajectory",
        "trunk": "frozen",
    },
    "structured_tangent": {
        "model": "FarOnlyTemplateTangentAR",
        "target": "Generated plus frozen far-only template tangent gamma=3",
        "trunk": "zero-additional-parameter deterministic wrapper",
    },
}
TRAINED_METHODS = tuple(METHODS)[:4]
# The public release trains the neural methods only.  The superseded fixed-gamma
# structured tangent remains in the historical source for checkpoint lineage,
# but the local paper correction is now exposed independently through
# ``local_spectrum_tangent.py`` and ``models/PatchARSpectrumTangent.py``.
ALL_METHODS = TRAINED_METHODS


def _remove_flag(flags, name, takes_value=False):
    while name in flags:
        index = flags.index(name)
        del flags[index:index + (2 if takes_value else 1)]


def _append_or_replace(flags, name, value=None):
    if name in flags:
        if value is not None:
            replace(flags, name, value)
        return
    flags.append(name)
    if value is not None:
        flags.append(str(value))


def parent_model_id(dataset, seed):
    spec = DATASETS[dataset]
    patch = 12 * spec["atoms"]
    return (
        f"CA16V1_Q1_{dataset}_P{patch}A16D{spec['d_model']}"
        f"L{spec['layers']}_s{int(seed)}"
    )


def method_model_id(dataset, seed, method, width):
    spec = DATASETS[dataset]
    tags = {
        "direct_mtp_joint": "DMTPJ",
        "direct_mtp_frozen": "DMTPF",
        "frozen_direct_exit": "FDEX",
        "generated": "GEN",
        "structured_tangent": "TAN3",
    }
    return (
        f"CA16V1_{tags[method]}K{int(width)}_{dataset}_"
        f"P{12 * spec['atoms']}D{spec['d_model']}_s{int(seed)}"
    )


def _artifact(root, identifier, filename):
    return find_artifact(root, identifier, filename)


def checkpoint_for_id(identifier):
    return _artifact(CHECKPOINT_ROOT, identifier, "checkpoint.pth")


def result_for_id(identifier):
    metrics = _artifact(RESULT_ROOT, identifier, "metrics.npy")
    return None if metrics is None else metrics.parent


def parent_checkpoint(dataset, seed):
    return checkpoint_for_id(parent_model_id(dataset, seed))


def method_checkpoint(dataset, seed, method, width):
    return checkpoint_for_id(method_model_id(dataset, seed, method, width))


def parent_command(dataset, seed, epochs, patience):
    selected = DATASETS[dataset]
    atoms = int(selected["atoms"])
    flags = legacy_parent_command(
        "confirm", dataset, "atomic", atoms, int(seed),
        int(epochs), int(patience),
    )
    replace(flags, "--model_id", parent_model_id(dataset, seed))
    replace(flags, "--des", "paper_clean_a16_v1")
    replace(flags, "--d_model", selected["d_model"])
    replace(flags, "--d_ff", 4 * selected["d_model"])
    replace(flags, "--e_layers", selected["layers"])
    _append_or_replace(
        flags, "--ablation_patch_assembly_atom_dim", 16
    )
    _append_or_replace(
        flags,
        "--ablation_patch_assembly_interface_dim",
        atoms * 16,
    )
    # Test is opened only by the explicit test stage.
    _append_or_replace(flags, "--skip_final_test")
    return flags


def decoder_command(dataset, seed, method, width, epochs, patience):
    if method not in TRAINED_METHODS:
        raise ValueError(f"not a trained decoder method: {method}")
    parent = parent_checkpoint(dataset, seed)
    if parent is None:
        raise FileNotFoundError(
            f"missing clean parent for {dataset} seed={seed}"
        )
    flags = parent_command(dataset, seed, epochs, patience)
    _remove_flag(flags, "--direct_patch_validation")
    replace(
        flags, "--model_id",
        method_model_id(dataset, seed, method, width),
    )
    replace(flags, "--model", METHODS[method]["model"])
    replace(flags, "--des", "paper_clean_a16_v1")
    flags.extend([
        "--distilled_multi_commit_pretrained", str(parent),
        "--distilled_multi_commit_hidden",
        str(4 * DATASETS[dataset]["d_model"]),
        "--distilled_multi_commit_patches", str(int(width)),
        "--distilled_multi_commit_eval_patches", str(int(width)),
        "--distilled_multi_commit_truth_weight",
        "0" if method == "generated" else "1",
        "--distilled_multi_commit_truth_loss", "mse",
    ])
    if method in {"generated", "frozen_direct_exit"}:
        flags.extend([
            "--direct_patch_cached_training",
            "--direct_patch_cache_fraction", "1.0",
            "--direct_patch_cache_shuffle_unit", "channel",
        ])
    return flags


def tangent_command(dataset, seed, width, patience):
    parent = parent_checkpoint(dataset, seed)
    generated = method_checkpoint(dataset, seed, "generated", width)
    if parent is None or generated is None:
        raise FileNotFoundError(
            f"tangent requires parent and Generated K{width}: "
            f"{dataset} seed={seed}"
        )
    flags = decoder_command(
        dataset, seed, "generated", width, epochs=0, patience=patience
    )
    replace(
        flags, "--model_id",
        method_model_id(dataset, seed, "structured_tangent", width),
    )
    replace(flags, "--model", METHODS["structured_tangent"]["model"])
    replace(flags, "--train_epochs", 0)
    _remove_flag(flags, "--direct_patch_cached_training")
    _remove_flag(flags, "--direct_patch_cache_fraction", takes_value=True)
    _remove_flag(
        flags, "--direct_patch_cache_shuffle_unit", takes_value=True
    )
    flags.extend([
        "--geometric_coordinate_base_checkpoint", str(generated),
        "--geometric_coordinate_period", str(PERIODS[dataset]),
        "--closure_anchor_gamma", "3",
        "--closure_anchor_preserve_first_patch",
        "--include_initial_checkpoint",
    ])
    return flags


def training_command(dataset, seed, method, width, epochs, patience):
    if method == "q1":
        return parent_command(dataset, seed, epochs, patience)
    if method == "structured_tangent":
        return tangent_command(dataset, seed, width, patience)
    return decoder_command(dataset, seed, method, width, epochs, patience)


def test_command(training_flags):
    flags = list(training_flags)
    replace(flags, "--is_training", 0)
    _remove_flag(flags, "--skip_final_test")
    return flags


def log_path(dataset, seed, method, width=None, test=False):
    identifier = (
        parent_model_id(dataset, seed)
        if method == "q1"
        else method_model_id(dataset, seed, method, width)
    )
    prefix = "test_" if test else "train_"
    return LOG_ROOT / f"{prefix}{identifier}.log"


def best_validation(path):
    if not path.is_file():
        return float("nan"), -1
    observations = [
        (int(match.group(1)), float(match.group(2)))
        for match in VALIDATION_PATTERN.finditer(
            path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    return min(observations, key=lambda item: item[1])[::-1] \
        if observations else (float("nan"), -1)


def _write_csv_atomic(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _run_job(job, epochs, patience, force, testing=False):
    dataset, seed, method, width = job
    identifier = (
        parent_model_id(dataset, seed)
        if method == "q1"
        else method_model_id(dataset, seed, method, width)
    )
    finished = (
        result_for_id(identifier) is not None
        if testing else checkpoint_for_id(identifier) is not None
    )
    if finished and not force:
        return {
            "unix_time": time.time(), "stage": "test" if testing else "train",
            "dataset": dataset, "seed": seed, "method": method,
            "width": width or 1, "identifier": identifier,
            "status": "skipped", "minutes": 0.0, "returncode": 0,
        }
    flags = training_command(
        dataset, seed, method, width or 1, epochs, patience
    )
    if testing:
        if checkpoint_for_id(identifier) is None:
            raise FileNotFoundError(f"missing checkpoint: {identifier}")
        flags = test_command(flags)
    path = log_path(dataset, seed, method, width, test=testing)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            flags, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
    return {
        "unix_time": time.time(), "stage": "test" if testing else "train",
        "dataset": dataset, "seed": seed, "method": method,
        "width": width or 1, "identifier": identifier,
        "status": "completed" if completed.returncode == 0 else "failed",
        "minutes": (time.time() - started) / 60.0,
        "returncode": int(completed.returncode),
    }


def run_jobs(jobs, epochs, patience, workers, force, testing=False):
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(
                _run_job, job, epochs, patience, force, testing
            ): job for job in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['status']} {row['identifier']} "
                f"in {row['minutes']:.1f} min",
                flush=True,
            )
            if row["returncode"]:
                failures.append(row["identifier"])
    if rows:
        existing = []
        if RUN_LOG.is_file():
            with RUN_LOG.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        _write_csv_atomic(RUN_LOG, [*existing, *rows])
    if failures:
        raise RuntimeError(f"campaign jobs failed: {failures}")
    return rows


def jobs_for(stage, datasets, seeds):
    if stage == "parent":
        return [
            (dataset, seed, "q1", None)
            for dataset in datasets for seed in seeds
        ]
    if stage in ALL_METHODS:
        return [
            (dataset, seed, stage, width)
            for dataset in datasets for seed in seeds for width in WIDTHS
        ]
    if stage == "decoders":
        return [
            (dataset, seed, method, width)
            for method in TRAINED_METHODS
            for dataset in datasets for seed in seeds for width in WIDTHS
        ]
    if stage == "train":
        return [
            *jobs_for("parent", datasets, seeds),
            *jobs_for("decoders", datasets, seeds),
        ]
    if stage == "test":
        return [
            *jobs_for("parent", datasets, seeds),
            *[
                (dataset, seed, method, width)
                for method in ALL_METHODS
                for dataset in datasets for seed in seeds
                for width in WIDTHS
            ],
        ]
    raise ValueError(f"unsupported job stage: {stage}")


def summarize(datasets, seeds, require_complete=False, widths=WIDTHS):
    rows = []
    expected = set()
    for dataset in datasets:
        for seed in seeds:
            candidates = [("q1", 1)] + [
                (method, width)
                for method in ALL_METHODS for width in widths
            ]
            for method, width in candidates:
                identifier = (
                    parent_model_id(dataset, seed)
                    if method == "q1"
                    else method_model_id(dataset, seed, method, width)
                )
                expected.add((dataset, seed, method, width))
                path = result_for_id(identifier)
                if path is None:
                    continue
                train_log = log_path(
                    dataset, seed, method,
                    None if method == "q1" else width,
                )
                validation, epoch = best_validation(train_log)
                metrics = read_metrics(path)
                rows.append({
                    "campaign": CAMPAIGN,
                    "dataset": dataset,
                    "seed": int(seed),
                    "method": method,
                    "commit_patches": int(width),
                    "parent_patch": 12 * DATASETS[dataset]["atoms"],
                    "atom_dim": 16,
                    "concat_width": 16 * DATASETS[dataset]["atoms"],
                    "d_model": DATASETS[dataset]["d_model"],
                    "layers": DATASETS[dataset]["layers"],
                    "selection": (
                        "next_patch_validation" if method == "q1"
                        else "closed_loop_h720_validation"
                    ),
                    "selected_validation": validation,
                    "selected_epoch": epoch,
                    **metrics,
                    "identifier": identifier,
                    "result": str(path.relative_to(ROOT)),
                })
    observed = {
        (row["dataset"], row["seed"], row["method"],
         row["commit_patches"])
        for row in rows
    }
    if require_complete and observed != expected:
        raise RuntimeError(
            f"incomplete clean test grid: missing={sorted(expected-observed)}"
        )
    _write_csv_atomic(METRIC_OUTPUT, rows)
    return rows


def print_status(datasets, seeds):
    jobs = jobs_for("test", datasets, seeds)
    counts = {"trained": 0, "tested": 0, "total": len(jobs)}
    for dataset, seed, method, width in jobs:
        identifier = (
            parent_model_id(dataset, seed)
            if method == "q1"
            else method_model_id(dataset, seed, method, width)
        )
        counts["trained"] += int(checkpoint_for_id(identifier) is not None)
        counts["tested"] += int(result_for_id(identifier) is not None)
    print(json.dumps(counts, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "parent", *ALL_METHODS, "decoders", "train", "test",
            "summarize", "status", "all",
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--widths", nargs="+", type=int, choices=WIDTHS, default=list(WIDTHS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if args.epochs < 1 or args.patience < 1 or args.workers < 1:
        raise ValueError("epochs, patience, and workers must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "status":
        print_status(datasets, seeds)
        return 0
    if args.stage == "summarize":
        summarize(datasets, seeds, args.require_complete, args.widths)
        return 0

    stages = (
        ("parent", *TRAINED_METHODS, "test", "summarize")
        if args.stage == "all" else (args.stage,)
    )
    for stage in stages:
        if stage == "summarize":
            summarize(datasets, seeds, args.require_complete, args.widths)
            continue
        jobs = jobs_for(stage, datasets, seeds)
        jobs = [job for job in jobs if job[3] is None or job[3] in args.widths]
        if args.dry_run:
            for dataset, seed, method, width in jobs:
                print(" ".join(training_command(
                    dataset, seed, method, width or 1,
                    args.epochs, args.patience,
                )))
            continue
        run_jobs(
            jobs, args.epochs, args.patience, args.workers,
            args.force, testing=stage == "test",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
