#!/usr/bin/env python3
"""Causally matched K4 controls for the ICLR DPOD study.

All K4 candidates end with the same frozen Q1 backbone and three independent
residual far exits.  ``clean`` and ``generated`` differ only in whether those
exits target clean future patches or an exact four-call Q1 rollout.  The
``staged_k2`` and ``staged_k4`` jobs reconstruct Q1->K2->K4 under the same
closed-loop H720 validation selector used by the matched one-shot controls.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_four_dataset_parent_campaign import (  # noqa: E402
    DATASETS,
    best_validation,
    checkpoint as parent_checkpoint,
    command as parent_command,
    read_metrics,
    result as parent_result,
    selected_atoms,
    summary_log_path as parent_summary_log_path,
)


CHECKPOINT_ROOT = ROOT / "checkpoints"
RESULT_ROOT = ROOT / "results"
LOG_ROOT = ROOT / "experiment_logs" / "paper_iclr_causal_k4"
RAW_OUTPUT = ROOT / "analysis_outputs" / "paper_iclr_causal_k4.csv"
MEAN_OUTPUT = ROOT / "analysis_outputs" / "paper_iclr_causal_k4_mean.csv"
VARIANTS = ("clean", "generated", "staged_k2", "staged_k4")
K4_VARIANTS = ("clean", "generated", "staged_k4")
FINAL_SEEDS = (2021, 2022, 2023)
FAR_LOSS_TYPE = "mse"
# Loader-level global scaling and the accepted parent's training coordinate are
# separate from the far-exit regression coordinate.  Every far target is
# re-expressed in the statistics of the clean origin before the exit loss is
# evaluated (see DistilledMultiCommitAR.direct_patch_loss).
FAR_LOSS_SPACE = "origin_instance_normalized"
# The original complete paper grid was trained with global loader scaling and
# MSE far-exit losses.  Its pre-fingerprint artifacts are reusable only under
# that exact final protocol.
LEGACY_FINAL_DATASETS = frozenset(("Weather", "ECL", "Traffic"))
EPOCH_VALIDATION_PATTERN = re.compile(
    r"Epoch: (\d+), Steps: \d+ \| Train Loss: [^ ]+ "
    r"Vali Loss: ([0-9.eE+-]+)"
)


def replace(flags, name, value):
    flags[flags.index(name) + 1] = str(value)


def assert_final_data_scale(datasets):
    """Keep the matched K4 study on the frozen globally scaled protocol."""
    disabled = [
        dataset for dataset in datasets if not DATASETS[dataset]["data_scale"]
    ]
    if disabled:
        raise RuntimeError(
            "final causal K4 protocol requires data_scale=True; disabled for "
            + ", ".join(disabled)
        )


def model_id(dataset, atoms, variant, seed):
    if variant not in VARIANTS:
        raise ValueError(f"unknown causal K4 variant: {variant}")
    patch = 12 * atoms
    scale_tag = int(DATASETS[dataset]["data_scale"])
    loss_tag = f"_l{DATASETS[dataset]['loss_space']}"
    return (
        f"PICLR_CAUSAL_{variant}_{dataset}_P{patch}C12"
        f"_ds{scale_tag}_s{seed}{loss_tag}_f{FAR_LOSS_TYPE}"
    )


def legacy_model_id(dataset, atoms, variant, seed):
    patch = 12 * atoms
    return f"PICLR_CAUSAL_{variant}_{dataset}_P{patch}C12_s{seed}"


def allow_legacy_final_artifact(dataset):
    return (
        dataset in LEGACY_FINAL_DATASETS
        and DATASETS[dataset]["data_scale"]
        and FAR_LOSS_TYPE == "mse"
    )


def log_path(dataset, atoms, variant, seed):
    return LOG_ROOT / f"{model_id(dataset, atoms, variant, seed)}.log"


def resolved_log_path(dataset, atoms, variant, seed):
    current = log_path(dataset, atoms, variant, seed)
    if current.is_file() or not allow_legacy_final_artifact(dataset):
        return current
    legacy = LOG_ROOT / f"{legacy_model_id(dataset, atoms, variant, seed)}.log"
    return legacy if legacy.is_file() else current


def find_artifact(root, identifier, filename):
    matches = [
        path / filename
        for path in root.glob(f"*{identifier}*")
        if (path / filename).is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple {filename} artifacts for {identifier}: {matches}"
        )
    return matches[0] if matches else None


def checkpoint(dataset, atoms, variant, seed):
    current = find_artifact(
        CHECKPOINT_ROOT,
        model_id(dataset, atoms, variant, seed),
        "checkpoint.pth",
    )
    if current is not None or not allow_legacy_final_artifact(dataset):
        return current
    return find_artifact(
        CHECKPOINT_ROOT,
        legacy_model_id(dataset, atoms, variant, seed),
        "checkpoint.pth",
    )


def result(dataset, atoms, variant, seed):
    metrics_path = find_artifact(
        RESULT_ROOT,
        model_id(dataset, atoms, variant, seed),
        "metrics.npy",
    )
    if metrics_path is None and allow_legacy_final_artifact(dataset):
        metrics_path = find_artifact(
            RESULT_ROOT,
            legacy_model_id(dataset, atoms, variant, seed),
            "metrics.npy",
        )
    return None if metrics_path is None else metrics_path.parent


def command(dataset, atoms, variant, seed, epochs, patience):
    parent = parent_checkpoint("confirm", dataset, "atomic", atoms, seed)
    if parent is None:
        raise FileNotFoundError(
            f"missing confirmed atomic parent for {dataset} seed={seed}"
        )
    flags = parent_command(
        "confirm", dataset, "atomic", atoms, seed, epochs, patience
    )
    # All causal controls select checkpoints by the same chronological H720
    # closed-loop validation loss.  Training still uses each named target.
    flags.remove("--direct_patch_validation")
    replace(flags, "--model_id", model_id(dataset, atoms, variant, seed))
    replace(flags, "--des", "paper_iclr_causal_k4")

    width = 2 if variant == "staged_k2" else 4
    if variant in {"clean", "generated"}:
        model = "MatchedFrozenMultiExitAR"
    elif variant == "staged_k2":
        model = "DistilledMultiCommitAR"
    else:
        model = "ProgressiveNestedCoarseGrainAR"
    replace(flags, "--model", model)
    flags.extend([
        "--distilled_multi_commit_pretrained",
        str(parent),
        "--distilled_multi_commit_hidden",
        str(DATASETS[dataset]["d_ff"]),
        "--distilled_multi_commit_patches",
        str(width),
        "--distilled_multi_commit_eval_patches",
        str(width),
        "--distilled_multi_commit_truth_weight",
        "1" if variant == "clean" else "0",
        "--distilled_multi_commit_truth_loss",
        FAR_LOSS_TYPE,
        "--include_initial_checkpoint",
    ])
    if variant == "staged_k4":
        k2 = checkpoint(dataset, atoms, "staged_k2", seed)
        if k2 is None:
            raise FileNotFoundError(
                f"missing causal staged K2 for {dataset} seed={seed}"
            )
        flags.extend([
            "--internal_coarse_composition_weight",
            "1",
            "--internal_coarse_truth_weight",
            "0",
            "--internal_coarse_gradient_mode",
            "sum",
            "--internal_coarse_backbone_lr_scale",
            "0.01",
            "--internal_coarse_reference_weight",
            "0",
            "--progressive_nested_pretrained",
            str(k2),
            "--progressive_nested_protected_patches",
            "2",
            "--progressive_nested_teacher",
            "protected_prefix",
        ])
    return flags


def run_one(flags, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            flags,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


def run_job(dataset, atoms, variant, seed, epochs, patience, force):
    identifier = model_id(dataset, atoms, variant, seed)
    if not force and result(dataset, atoms, variant, seed) is not None:
        return identifier, "skipped", 0.0, 0
    path = log_path(dataset, atoms, variant, seed)
    started = time.time()
    returncode = run_one(
        command(dataset, atoms, variant, seed, epochs, patience),
        path,
    )
    status = "completed" if returncode == 0 else "failed"
    return identifier, status, (time.time() - started) / 60, returncode


def run_group(jobs, epochs, patience, workers, force):
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_job,
                *job,
                epochs,
                patience,
                force,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            identifier, status, minutes, returncode = future.result()
            print(f"{status} {identifier} in {minutes:.1f} min", flush=True)
            if returncode:
                failures.append((identifier, returncode))
    return failures


def validation_selection(path):
    """Expose whether closed-loop selection retained the zero-head fallback."""
    if not path.is_file():
        return {
            "selected_validation_epoch": -1,
            "initial_validation_h720": float("nan"),
            "best_trained_validation_h720": float("nan"),
            "selected_initial_checkpoint": 0,
        }
    observations = [
        (int(match.group(1)), float(match.group(2)))
        for match in EPOCH_VALIDATION_PATTERN.finditer(
            path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    if not observations:
        return {
            "selected_validation_epoch": -1,
            "initial_validation_h720": float("nan"),
            "best_trained_validation_h720": float("nan"),
            "selected_initial_checkpoint": 0,
        }
    selected_epoch, _ = min(observations, key=lambda item: item[1])
    initial = [value for epoch, value in observations if epoch == 0]
    trained = [value for epoch, value in observations if epoch > 0]
    return {
        "selected_validation_epoch": selected_epoch,
        "initial_validation_h720": (
            initial[0] if initial else float("nan")
        ),
        "best_trained_validation_h720": (
            min(trained) if trained else float("nan")
        ),
        "selected_initial_checkpoint": int(selected_epoch == 0),
    }


def summarize_mean(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    summary = []
    for candidates in groups.values():
        first = candidates[0]
        metric_names = [
            "mae", "mse", "h96_mse", "h192_mse", "h336_mse",
            "h720_mse", "avg_horizon_mse",
        ]
        values = {
            name: np.asarray([row[name] for row in candidates], dtype=float)
            for name in metric_names
        }
        validations = np.asarray([
            row["best_validation"] for row in candidates
        ], dtype=float)
        paired = np.asarray([
            row["delta_vs_q1"] for row in candidates
        ], dtype=float)
        summary.append({
            "dataset": first["dataset"],
            "data_scale": first["data_scale"],
            "loss_space": first["loss_space"],
            "parent_loss_space": first["parent_loss_space"],
            "far_loss_space": first["far_loss_space"],
            "far_loss_type": first["far_loss_type"],
            "method": first["method"],
            "selector": first["selector"],
            "target": first["target"],
            "training_path": first["training_path"],
            "parent_patch": first["parent_patch"],
            "commit_patches": first["commit_patches"],
            "backbone_calls": first["backbone_calls"],
            "n_seeds": len(candidates),
            "mean_validation_h720": float(np.nanmean(validations)),
            **{name: float(array.mean()) for name, array in values.items()},
            **{
                f"{name}_std": float(array.std(ddof=1))
                if len(array) > 1 else 0.0
                for name, array in values.items()
            },
            "paired_delta_vs_q1": float(paired.mean()),
            "paired_delta_vs_q1_std": (
                float(paired.std(ddof=1)) if len(paired) > 1 else 0.0
            ),
            "wins_vs_q1": int(np.sum(paired < 0)),
            "initial_checkpoint_selections": int(sum(
                row["selected_initial_checkpoint"] for row in candidates
            )),
            "mean_selected_validation_epoch": float(np.mean([
                row["selected_validation_epoch"] for row in candidates
            ])),
        })
    if summary:
        MEAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with MEAN_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def _require_causal_grid(rows, datasets, seeds):
    methods = ("Q1", *VARIANTS)
    expected = {
        (dataset, seed, method)
        for dataset in datasets for seed in seeds for method in methods
    }
    observed = {
        (row["dataset"], row["seed"], row["method"])
        for row in rows
    }
    if len(rows) != len(observed) or observed != expected:
        raise RuntimeError(
            "incomplete causal K4 grid: "
            f"expected={len(expected)} actual={len(observed)} "
            f"missing={sorted(expected - observed)[:5]} "
            f"extra={sorted(observed - expected)[:5]}"
        )


def _canonical_summary_scope(datasets, seeds):
    return set(datasets) == set(DATASETS) and set(seeds) == set(FINAL_SEEDS)


def summarize(datasets, seeds, require_complete=False):
    assert_final_data_scale(datasets)
    canonical_scope = _canonical_summary_scope(datasets, seeds)
    if require_complete and not canonical_scope:
        raise RuntimeError(
            "incomplete causal K4 scope cannot replace canonical CSVs"
        )
    choices = selected_atoms(datasets)
    rows = []
    for dataset in datasets:
        atoms = choices[dataset]
        patch = 12 * atoms
        for seed in seeds:
            parent_path = parent_result(
                "confirm", dataset, "atomic", atoms, seed
            )
            if parent_path is not None:
                parent_log = parent_summary_log_path(
                    "confirm", dataset, "atomic", atoms, seed
                )
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "data_scale": int(DATASETS[dataset]["data_scale"]),
                    "loss_space": DATASETS[dataset]["loss_space"],
                    "parent_loss_space": DATASETS[dataset]["loss_space"],
                    "far_loss_space": "not_applicable",
                    "far_loss_type": "not_applicable",
                    "method": "Q1",
                    "selector": "accepted_parent_next_patch",
                    "target": "next_patch_truth",
                    "training_path": "accepted_parent",
                    "atoms": atoms,
                    "parent_patch": patch,
                    "commit_patches": 1,
                    "backbone_calls": math.ceil(720 / patch),
                    "best_validation": best_validation(parent_log),
                    **validation_selection(parent_log),
                    **read_metrics(parent_path),
                    "setting": parent_path.name,
                })
            for variant in VARIANTS:
                path = result(dataset, atoms, variant, seed)
                if path is None:
                    continue
                width = 2 if variant == "staged_k2" else 4
                target = (
                    "clean_future" if variant == "clean"
                    else "q1_generated_composition"
                    if variant == "generated"
                    else "accepted_generated_composition"
                )
                training_path = (
                    "one_shot" if variant in {"clean", "generated"}
                    else "dyadic"
                )
                variant_log = resolved_log_path(
                    dataset, atoms, variant, seed
                )
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "data_scale": int(DATASETS[dataset]["data_scale"]),
                    "loss_space": DATASETS[dataset]["loss_space"],
                    "parent_loss_space": DATASETS[dataset]["loss_space"],
                    "far_loss_space": FAR_LOSS_SPACE,
                    "far_loss_type": FAR_LOSS_TYPE,
                    "method": variant,
                    "selector": "common_h720_rollout",
                    "target": target,
                    "training_path": training_path,
                    "atoms": atoms,
                    "parent_patch": patch,
                    "commit_patches": width,
                    "backbone_calls": math.ceil(720 / (width * patch)),
                    "best_validation": best_validation(variant_log),
                    **validation_selection(variant_log),
                    **read_metrics(path),
                    "setting": path.name,
                })
    complete = False
    if canonical_scope:
        try:
            _require_causal_grid(rows, DATASETS, FINAL_SEEDS)
            complete = True
        except RuntimeError:
            if require_complete:
                raise
    q1 = {
        (row["dataset"], row["seed"]): row
        for row in rows if row["method"] == "Q1"
    }
    for row in rows:
        parent = q1.get((row["dataset"], row["seed"]))
        row["delta_vs_q1"] = (
            row["mse"] - parent["mse"]
            if parent is not None else float("nan")
        )
    if complete:
        RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with RAW_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summarize_mean(rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[*VARIANTS, "matched", "all", "summarize"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2021, 2022, 2023]
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="leave shared CSV aggregation to the coordinating host",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    seeds = list(dict.fromkeys(args.seeds))
    assert_final_data_scale(datasets)
    if args.stage == "summarize":
        try:
            rows = summarize(datasets, seeds, require_complete=True)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
        print(
            f"rows={len(rows)} raw={RAW_OUTPUT.relative_to(ROOT)} "
            f"mean={MEAN_OUTPUT.relative_to(ROOT)}"
        )
        return 0
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    choices = selected_atoms(datasets)
    if args.stage == "all":
        groups = [("clean", "generated"), ("staged_k2",), ("staged_k4",)]
    elif args.stage == "matched":
        groups = [("clean", "generated")]
    else:
        groups = [(args.stage,)]
    failures = []
    for variants in groups:
        jobs = [
            (dataset, choices[dataset], variant, seed)
            for dataset in datasets
            for seed in seeds
            for variant in variants
        ]
        if args.dry_run:
            for job in jobs:
                print(" ".join(command(
                    *job,
                    args.epochs,
                    args.patience,
                )))
            continue
        group_failures = run_group(
            jobs,
            args.epochs,
            args.patience,
            args.workers,
            args.force,
        )
        failures.extend(group_failures)
        if group_failures:
            break
    if args.dry_run:
        return 0
    if args.skip_summary:
        for identifier, returncode in failures:
            print(
                f"failed {identifier} returncode={returncode}",
                file=sys.stderr,
            )
        return 1 if failures else 0
    if failures:
        for identifier, returncode in failures:
            print(
                f"failed {identifier} returncode={returncode}",
                file=sys.stderr,
            )
        return 1
    rows = summarize(datasets, seeds)
    expected_methods = {"Q1"}
    if args.stage in {"all", "summarize"}:
        expected_methods.update(VARIANTS)
    elif args.stage == "matched":
        expected_methods.update({"clean", "generated"})
    else:
        expected_methods.add(args.stage)
    selected = [row for row in rows if row["method"] in expected_methods]
    expected = len(datasets) * len(seeds) * len(expected_methods)
    if len(selected) != expected:
        print(
            "incomplete causal K4 rows: "
            f"expected={expected} actual={len(selected)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
