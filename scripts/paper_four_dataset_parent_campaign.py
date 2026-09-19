#!/usr/bin/env python3
"""Build frozen-scale atomic parents for the ICLR DPOD benchmark.

The paper campaign uses patch scales already established by the accumulated
development experiments.  It therefore does not repeat an expensive four-way
closed-loop scale sweep.  ``confirm`` trains direct-P and P12-assembly parents
for three seeds at the frozen scale, selects checkpoints with next-patch
validation (the Timer-style training protocol), and evaluates H720 once after
training.  The raw three runs are retained, while the paper-facing summary
reports means and sample standard deviations over the frozen configuration.

The legacy ``screen`` entry point remains available only as an optional
diagnostic and is not part of the main paper campaign.
"""

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CHECKPOINT_ROOT = ROOT / "checkpoints"
RESULT_ROOT = ROOT / "results"
LOG_ROOT = ROOT / "experiment_logs" / "paper_four_dataset_parent"
SCREEN_OUTPUT = (
    ROOT / "analysis_outputs" / "paper_iclr_patch_selection.csv"
)
CONFIRM_OUTPUT = (
    ROOT / "analysis_outputs" / "paper_iclr_parent_confirmation.csv"
)
CONFIRM_MEAN_OUTPUT = (
    ROOT / "analysis_outputs" / "paper_iclr_parent_mean.csv"
)
FROZEN_POLICY_OUTPUT = (
    ROOT / "analysis_outputs" / "paper_iclr_frozen_patch_policy.csv"
)

ATOMS = (1, 2, 4, 8)
MODES = ("direct", "atomic")
FINAL_SEEDS = (2021, 2022, 2023)
LEGACY_FINAL_DATASETS = frozenset(("Weather", "ECL", "Traffic"))
VALIDATION_PATTERN = re.compile(r"Vali Loss: ([0-9.eE+-]+)")

# Frozen from the accumulated development evidence before the final three-seed
# paper runs.  Patch selection is not a new claim and is not repeated here.
FROZEN_ATOMS = {
    "ETTh1": 2,   # P24: established low-dimensional DPOD regime.
    "ETTh2": 4,   # P48: established low-dimensional DPOD regime.
    "ETTm1": 8,   # P96: established minute-level DPOD regime.
    "ETTm2": 2,   # P24: established minute-level DPOD regime.
    "Weather": 4, # P48: established Weather parent/DPOD regime.
    "ECL": 8,     # P96: strong Timer-style capacity regime.
    "Traffic": 8, # P96: strong Timer-style capacity regime.
    "Exchange": 8,  # P96: daily, no calendar periodicity; large patch only.
}
FROZEN_EVIDENCE = {
    "ETTh1": "existing multi-seed P24 DPOD/PPMC development regime",
    "ETTh2": "existing multi-seed P48 DPOD development regime",
    "ETTm1": "existing multi-seed P96 DPOD development regime",
    "ETTm2": "existing multi-seed P24 DPOD development regime",
    "Weather": "existing multi-seed P48 parent/DPOD development regime",
    "ECL": "P96 Q1=0.198827; matched P12x8 Q1=0.199079 (seed 2021)",
    "Traffic": "P96 Q1=0.425147; matched P12x8 Q1=0.425346 (seed 2021)",
    "Exchange": "daily exchange rates, no calendar periodicity — cross-family control",
}


DATASETS = {
    "ETTh1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "freq": "h",
        "channels": 7,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 64,
        "layers": 1,
        "d_ff": 256,
        "batch_size": 32,
        "eval_batch_size": 128,
        "train_channels": 0,
        "learning_rate": 0.001,
        "lradj": "type1",
    },
    "ETTh2": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "freq": "h",
        "channels": 7,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 64,
        "layers": 1,
        "d_ff": 256,
        "batch_size": 32,
        "eval_batch_size": 128,
        "train_channels": 0,
        "learning_rate": 0.001,
        "lradj": "type1",
    },
    "ETTm1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "freq": "t",
        "channels": 7,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 64,
        "layers": 1,
        "d_ff": 256,
        "batch_size": 32,
        "eval_batch_size": 128,
        "train_channels": 0,
        "learning_rate": 0.001,
        "lradj": "type1",
    },
    "ETTm2": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "data": "ETTm2",
        "freq": "t",
        "channels": 7,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 64,
        "layers": 2,
        "d_ff": 256,
        "batch_size": 32,
        "eval_batch_size": 128,
        "train_channels": 0,
        "learning_rate": 0.001,
        "lradj": "type1",
    },
    "Weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "freq": "t",
        "channels": 21,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 64,
        "layers": 2,
        "d_ff": 256,
        "batch_size": 32,
        "eval_batch_size": 128,
        "train_channels": 0,
        "learning_rate": 0.001,
        "lradj": "type1",
    },
    "ECL": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "freq": "h",
        "channels": 321,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 256,
        "layers": 5,
        "d_ff": 1024,
        "batch_size": 4,
        "eval_batch_size": 32,
        "train_channels": 64,
        "learning_rate": 0.0005,
        "lradj": "cosine",
    },
    "Traffic": {
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "data": "custom",
        "freq": "h",
        "channels": 862,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 256,
        "layers": 4,
        "d_ff": 1024,
        "batch_size": 4,
        "eval_batch_size": 8,
        "train_channels": 64,
        "learning_rate": 0.0005,
        "lradj": "cosine",
    },
    "Exchange": {
        "root_path": "./dataset/exchange_rate/",
        "data_path": "exchange_rate.csv",
        "data": "custom",
        "freq": "d",
        "channels": 8,
        "data_scale": True,
        "loss_space": "point",
        "d_model": 64,
        "layers": 1,
        "d_ff": 256,
        "batch_size": 32,
        "eval_batch_size": 128,
        "train_channels": 0,
        "learning_rate": 0.001,
        "lradj": "type1",
    },
}


def assert_final_data_scale(datasets):
    """Keep every paper parent on the frozen globally scaled protocol."""
    disabled = [
        dataset for dataset in datasets if not DATASETS[dataset]["data_scale"]
    ]
    if disabled:
        raise RuntimeError(
            "final paper protocol requires data_scale=True; disabled for "
            + ", ".join(disabled)
        )


def model_id(stage, dataset, mode, atoms, seed):
    patch = 12 * atoms
    mode_tag = "C12" if mode == "atomic" else "RAW"
    scale_tag = int(DATASETS[dataset]["data_scale"])
    loss_tag = f"_l{DATASETS[dataset]['loss_space']}"
    return (
        f"P4D_{stage}_{dataset}_P{patch}{mode_tag}_s{seed}"
        f"_ds{scale_tag}{loss_tag}"
    )


def log_path(stage, dataset, mode, atoms, seed):
    return LOG_ROOT / f"{model_id(stage, dataset, mode, atoms, seed)}.log"


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


def checkpoint(stage, dataset, mode, atoms, seed):
    scale_tag = int(DATASETS[dataset]["data_scale"])
    identifier = model_id(stage, dataset, mode, atoms, seed)
    found = find_artifact(
        CHECKPOINT_ROOT, identifier, "checkpoint.pth"
    )
    if found is not None:
        return found
    # Legacy checkpoints (pre-ds-tag) carry the scale only inside the run.py
    # filename, as ..._ds{scale}_s{seed} right before the terminal seed.
    if dataset not in LEGACY_FINAL_DATASETS:
        return None
    loss_tag = f"_l{DATASETS[dataset]['loss_space']}"
    legacy = identifier.replace(f"_ds{scale_tag}{loss_tag}", "")
    matches = [
        path / "checkpoint.pth"
        for path in CHECKPOINT_ROOT.glob(f"*{legacy}*")
        if (path / "checkpoint.pth").is_file()
        and f"_ds{scale_tag}_s{seed}" in path.name
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple legacy checkpoint artifacts for {identifier}"
        )
    return matches[0] if matches else None


def result(stage, dataset, mode, atoms, seed):
    metrics_path = find_artifact(
        RESULT_ROOT,
        model_id(stage, dataset, mode, atoms, seed),
        "metrics.npy",
    )
    if metrics_path is None:
        if dataset not in LEGACY_FINAL_DATASETS:
            return None
        scale_tag = int(DATASETS[dataset]["data_scale"])
        identifier = model_id(stage, dataset, mode, atoms, seed)
        loss_tag = f"_l{DATASETS[dataset]['loss_space']}"
        legacy = identifier.replace(f"_ds{scale_tag}{loss_tag}", "")
        matches = [
            path / "metrics.npy"
            for path in RESULT_ROOT.glob(f"*{legacy}*")
            if (path / "metrics.npy").is_file()
            and f"_ds{scale_tag}_s{seed}" in path.name
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"multiple legacy result artifacts for {identifier}: {matches}"
            )
        metrics_path = matches[0] if matches else None
    return None if metrics_path is None else metrics_path.parent


def summary_log_path(stage, dataset, mode, atoms, seed):
    """Resolve both current ds-tagged and archived pre-tag paper logs."""
    current = log_path(stage, dataset, mode, atoms, seed)
    if current.is_file():
        return current
    scale_tag = int(DATASETS[dataset]["data_scale"])
    loss_tag = f"_l{DATASETS[dataset]['loss_space']}"
    legacy_name = current.name.replace(
        f"_ds{scale_tag}{loss_tag}", ""
    )
    legacy = current.with_name(legacy_name)
    return legacy


def command(stage, dataset, mode, atoms, seed, epochs, patience):
    if stage not in {"screen", "confirm"}:
        raise ValueError(f"unknown stage: {stage}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if atoms not in ATOMS:
        raise ValueError(f"unsupported atom count: {atoms}")
    spec = DATASETS[dataset]
    patch = 12 * atoms
    assembly = "chronological" if mode == "atomic" else "none"
    flags = [
        str(PYTHON),
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        spec["root_path"],
        "--data_path",
        spec["data_path"],
        "--model_id",
        model_id(stage, dataset, mode, atoms, seed),
        "--model",
        "TransformerAblationAR",
        "--data",
        spec["data"],
        "--features",
        "M",
        "--seq_len",
        "672",
        "--label_len",
        "0",
        "--pred_len",
        "720",
        "--freq",
        spec["freq"],
        "--ar_patch_len",
        str(patch),
        "--dense_ar_roll_patches",
        "1",
        "--dense_ar_loss_space",
        spec["loss_space"],
        "--dense_ar_loss_type",
        "mse",
        "--ar_norm_eps",
        "0.001",
        "--ar_transformer_norm",
        "post",
        "--enc_in",
        str(spec["channels"]),
        "--dec_in",
        str(spec["channels"]),
        "--c_out",
        str(spec["channels"]),
        "--d_model",
        str(spec["d_model"]),
        "--n_heads",
        "8",
        "--e_layers",
        str(spec["layers"]),
        "--d_layers",
        "1",
        "--d_ff",
        str(spec["d_ff"]),
        "--dropout",
        "0.1",
        "--activation",
        "gelu",
        "--batch_size",
        str(spec["batch_size"]),
        "--eval_batch_size",
        str(spec["eval_batch_size"]),
        "--num_workers",
        "0",
        "--direct_patch_train_channels",
        str(spec["train_channels"]),
        "--direct_patch_validation_channels",
        "0",
        "--learning_rate",
        str(spec["learning_rate"]),
        "--train_epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--loader_seed",
        str(seed),
        "--checkpoint_selection",
        "vali",
        "--stream_metrics",
        "--report_horizons",
        "96",
        "192",
        "336",
        "720",
        "--skip_epoch_test",
        "--lradj",
        spec["lradj"],
        "--use_amp",
        "--ablation_attention",
        "softmax",
        "--ablation_ffn",
        "mlp",
        "--ablation_norm",
        "post",
        "--ablation_position_encoding",
        "rope",
        "--ablation_attention_residual",
        "--ablation_ffn_residual",
        "--ablation_qk_norm",
        "dot",
        "--ablation_attention_temperature",
        "1.0",
        "--ablation_norm_kind",
        "layer",
        "--ablation_patch_assembly",
        assembly,
        "--ablation_patch_assembly_fine_len",
        "12",
        "--ablation_patch_assembly_stem",
        "linear",
        "--ablation_output_head",
        "full",
        "--des",
        "paper_four_dataset_parent",
        "--itr",
        "1",
    ]
    flags.append("--data_scale" if spec["data_scale"] else "--no_data_scale")
    if stage == "screen":
        flags.append("--skip_final_test")
    else:
        # Match the established Timer-style parents: checkpoint on next-token
        # validation, then run the closed-loop H720 test once after training.
        flags.append("--direct_patch_validation")
    return flags


def best_validation(path):
    if not path.is_file():
        return float("nan")
    values = [
        float(match.group(1))
        for match in VALIDATION_PATTERN.finditer(
            path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    return min(values) if values else float("nan")


def read_metrics(path):
    values = np.load(path / "metrics.npy").reshape(-1)
    horizons = {
        int(row[0]): float(row[2])
        for row in np.load(path / "horizon_metrics.npy")
    }
    metrics = {
        "mae": float(values[0]),
        "mse": float(values[1]),
        **{
            f"h{horizon}_mse": horizons[horizon]
            for horizon in (96, 192, 336, 720)
        },
    }
    metrics["avg_horizon_mse"] = float(np.mean([
        metrics[f"h{horizon}_mse"] for horizon in (96, 192, 336, 720)
    ]))
    return metrics


def summarize_screen(datasets, atoms, seed):
    rows = []
    for dataset in datasets:
        for count in atoms:
            current_checkpoint = checkpoint(
                "screen", dataset, "atomic", count, seed
            )
            validation = best_validation(
                log_path("screen", dataset, "atomic", count, seed)
            )
            if current_checkpoint is None or not np.isfinite(validation):
                continue
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "data_scale": int(DATASETS[dataset]["data_scale"]),
                "loss_space": DATASETS[dataset]["loss_space"],
                "atoms": count,
                "parent_patch": 12 * count,
                "best_validation": validation,
                "selected": False,
                "checkpoint": str(current_checkpoint.relative_to(ROOT)),
            })
    for dataset in datasets:
        candidates = [row for row in rows if row["dataset"] == dataset]
        if len(candidates) != len(atoms):
            continue
        selected = min(candidates, key=lambda row: row["best_validation"])
        selected["selected"] = True
    if rows:
        SCREEN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with SCREEN_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def selected_atoms(datasets):
    missing = [dataset for dataset in datasets if dataset not in FROZEN_ATOMS]
    if missing:
        raise RuntimeError("no frozen patch policy for: " + ", ".join(missing))
    return {dataset: FROZEN_ATOMS[dataset] for dataset in datasets}


def write_frozen_policy():
    rows = [
        {
            "dataset": dataset,
            "data_scale": int(DATASETS[dataset]["data_scale"]),
            "loss_space": DATASETS[dataset]["loss_space"],
            "atoms": atoms,
            "parent_patch": 12 * atoms,
            "selection": "frozen_from_prior_development",
            "evidence": FROZEN_EVIDENCE[dataset],
        }
        for dataset, atoms in FROZEN_ATOMS.items()
    ]
    FROZEN_POLICY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FROZEN_POLICY_OUTPUT.open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_confirm_mean(rows):
    summary = []
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["mode"]), []).append(row)
    for key, candidates in groups.items():
        first = candidates[0]
        metric_names = [
            "mae", "mse", "h96_mse", "h192_mse", "h336_mse",
            "h720_mse", "avg_horizon_mse",
        ]
        values = {
            name: np.asarray([row[name] for row in candidates], dtype=float)
            for name in metric_names
        }
        item = {
            "dataset": first["dataset"],
            "data_scale": first["data_scale"],
            "loss_space": first["loss_space"],
            "mode": first["mode"],
            "atoms": first["atoms"],
            "parent_patch": first["parent_patch"],
            "n_seeds": len(candidates),
            "mean_validation": float(np.mean([
                row["best_validation"] for row in candidates
            ])),
            **{name: float(array.mean()) for name, array in values.items()},
            **{
                f"{name}_std": float(array.std(ddof=1))
                if len(array) > 1 else 0.0
                for name, array in values.items()
            },
            "paired_delta_vs_direct": float("nan"),
            "paired_delta_vs_direct_std": float("nan"),
            "paired_avg_delta_vs_direct": float("nan"),
            "paired_avg_delta_vs_direct_std": float("nan"),
            "wins_vs_direct": 0,
            "avg_wins_vs_direct": 0,
        }
        summary.append(item)
    for item in summary:
        paired = [
            row["delta_vs_direct"] for row in rows
            if row["dataset"] == item["dataset"]
            and row["mode"] == item["mode"]
        ]
        if paired:
            item["paired_delta_vs_direct"] = float(np.mean(paired))
            item["paired_delta_vs_direct_std"] = (
                float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
            )
            item["wins_vs_direct"] = int(sum(value < 0 for value in paired))
        paired_avg = [
            row["avg_delta_vs_direct"] for row in rows
            if row["dataset"] == item["dataset"]
            and row["mode"] == item["mode"]
        ]
        if paired_avg:
            item["paired_avg_delta_vs_direct"] = float(np.mean(paired_avg))
            item["paired_avg_delta_vs_direct_std"] = (
                float(np.std(paired_avg, ddof=1))
                if len(paired_avg) > 1 else 0.0
            )
            item["avg_wins_vs_direct"] = int(
                sum(value < 0 for value in paired_avg)
            )
    if summary:
        with CONFIRM_MEAN_OUTPUT.open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def _require_confirm_grid(rows, datasets, modes, seeds):
    expected = {
        (dataset, seed, mode)
        for dataset in datasets for seed in seeds for mode in modes
    }
    observed = {
        (row["dataset"], row["seed"], row["mode"])
        for row in rows
    }
    if len(rows) != len(observed) or observed != expected:
        raise RuntimeError(
            "incomplete parent confirmation grid: "
            f"expected={len(expected)} actual={len(observed)} "
            f"missing={sorted(expected - observed)[:5]} "
            f"extra={sorted(observed - expected)[:5]}"
        )


def _canonical_confirm_scope(datasets, modes, seeds):
    return (
        set(datasets) == set(DATASETS)
        and set(modes) == set(MODES)
        and set(seeds) == set(FINAL_SEEDS)
    )


def summarize_confirm(datasets, modes, seeds, require_complete=False):
    canonical_scope = _canonical_confirm_scope(datasets, modes, seeds)
    if require_complete and not canonical_scope:
        raise RuntimeError(
            "incomplete parent confirmation scope cannot replace canonical CSVs"
        )
    choices = selected_atoms(datasets)
    rows = []
    for dataset in datasets:
        atoms = choices[dataset]
        for seed in seeds:
            for mode in modes:
                path = result("confirm", dataset, mode, atoms, seed)
                if path is None:
                    continue
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "data_scale": int(DATASETS[dataset]["data_scale"]),
                    "loss_space": DATASETS[dataset]["loss_space"],
                    "mode": mode,
                    "atoms": atoms,
                    "parent_patch": 12 * atoms,
                    "best_validation": best_validation(
                        summary_log_path(
                            "confirm", dataset, mode, atoms, seed
                        )
                    ),
                    **read_metrics(path),
                    "setting": path.name,
                })
    complete = False
    if canonical_scope:
        try:
            _require_confirm_grid(rows, DATASETS, MODES, FINAL_SEEDS)
            complete = True
        except RuntimeError:
            if require_complete:
                raise
    by_pair = {
        (row["dataset"], row["seed"], row["mode"]): row
        for row in rows
    }
    for row in rows:
        direct = by_pair.get((row["dataset"], row["seed"], "direct"))
        row["delta_vs_direct"] = (
            row["mse"] - direct["mse"]
            if direct is not None else float("nan")
        )
        row["avg_delta_vs_direct"] = (
            row["avg_horizon_mse"] - direct["avg_horizon_mse"]
            if direct is not None else float("nan")
        )
    if complete:
        CONFIRM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with CONFIRM_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summarize_confirm_mean(rows)
    return rows


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "confirm", "summarize"])
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--atoms", nargs="+", type=int, choices=ATOMS, default=ATOMS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--screen-seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    atoms = list(dict.fromkeys(args.atoms))
    modes = list(dict.fromkeys(args.modes))
    seeds = list(dict.fromkeys(args.seeds))
    assert_final_data_scale(datasets)
    write_frozen_policy()

    if args.stage == "summarize":
        print(
            f"frozen_policy={FROZEN_POLICY_OUTPUT.relative_to(ROOT)}"
        )
        try:
            confirm_rows = summarize_confirm(
                datasets, modes, seeds, require_complete=True
            )
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
        print(
            f"confirm_rows={len(confirm_rows)} "
            f"output={CONFIRM_OUTPUT.relative_to(ROOT)}"
        )
        return 0

    if args.stage == "screen":
        jobs = [
            (dataset, "atomic", count, args.screen_seed)
            for dataset in datasets
            for count in atoms
        ]
    else:
        choices = selected_atoms(datasets)
        jobs = [
            (dataset, mode, choices[dataset], seed)
            for dataset in datasets
            for seed in seeds
            for mode in modes
        ]

    if args.dry_run:
        for dataset, mode, count, seed in jobs:
            print(" ".join(command(
                args.stage,
                dataset,
                mode,
                count,
                seed,
                args.epochs,
                args.patience,
            )))
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset, mode, count, seed in jobs:
        identifier = model_id(args.stage, dataset, mode, count, seed)
        complete = (
            checkpoint(args.stage, dataset, mode, count, seed) is not None
            if args.stage == "screen"
            else result(args.stage, dataset, mode, count, seed) is not None
        )
        if complete:
            print(f"skip completed {identifier}", flush=True)
            continue
        path = log_path(args.stage, dataset, mode, count, seed)
        print(f"run {identifier}; log={path.relative_to(ROOT)}", flush=True)
        started = time.time()
        returncode = run_one(
            command(
                args.stage,
                dataset,
                mode,
                count,
                seed,
                args.epochs,
                args.patience,
            ),
            path,
        )
        if returncode:
            print(f"failed {identifier}; inspect {path}", file=sys.stderr)
            return returncode
        print(
            f"completed {identifier} in {(time.time() - started) / 60:.1f} min",
            flush=True,
        )
        if args.stage == "screen":
            summarize_screen(datasets, atoms, args.screen_seed)
        else:
            summarize_confirm(datasets, modes, seeds)

    if args.stage == "screen":
        rows = summarize_screen(datasets, atoms, args.screen_seed)
        for dataset in datasets:
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["selected"]
            ]
            if selected:
                row = selected[0]
                print(
                    f"selected {dataset}: m={row['atoms']} "
                    f"P={row['parent_patch']} val={row['best_validation']:.7f}"
                )
    else:
        rows = summarize_confirm(datasets, modes, seeds)
        print(f"confirmation rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
