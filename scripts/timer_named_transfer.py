#!/usr/bin/env python3
"""Frozen four-data-set transfer campaign for public Timer-base-84m.

The official Timer checkpoint is an immutable Q1 parent.  This campaign trains
only Generated-K8 far exits on ETTh1, ETTh2, ETTm1, and Weather, audits the
K1/K2/K4/K8 prefixes on validation, freezes train-only Spectrum Tangent
selection, and opens test only through explicit gated subcommands.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CHECKPOINT_ROOT = ROOT / "checkpoints"
LOG_ROOT = ROOT / "experiment_logs" / "timer_named_transfer"
CAMPAIGN = "timer_named_transfer_v1_20260815"
OUTPUT_ROOT = ROOT / "analysis_outputs" / "exploratory" / CAMPAIGN
DEFAULT_TIMER_CHECKPOINT = ROOT / "foundation_models" / "timer-base-84m"
DATASET_NAMES = ("ETTh1", "ETTh2", "ETTm1", "Weather")
SEEDS = (2021, 2022, 2023)
WIDTHS = (2, 4, 8)
PATCH_LEN = 96
WIDTH = 8
FAR_LR = 3e-5
TIMER_HF_REVISION = "70077a71acce1b4c00d98332fcaabc694255d8e5"
TIMER_TRANSFORMERS_VERSION = "4.40.1"
CONSTANT_WINDOW_POLICY = (
    "continuous ReVIN limit: zero normalized input and constant de-normalized "
    "output when the public parent has std=0"
)
TRAIN_BATCH = {
    "ETTh1": 128,
    "ETTh2": 128,
    "ETTm1": 128,
    "Weather": 64,
}
EVAL_BATCH = {
    "ETTh1": 64,
    "ETTh2": 64,
    "ETTm1": 64,
    "Weather": 32,
}


def measurement_code_paths():
    """Sources whose bytes define the frozen Timer measurement protocol."""
    return (
        ROOT / "run.py",
        ROOT / "exp" / "exp_long_term_forecasting.py",
        ROOT / "models" / "TimerOperatorAR.py",
        ROOT / "models" / "AutoTimesSpectrumTangent.py",
        ROOT / "scripts" / "autotimes_atd_ds1_validation.py",
        ROOT / "scripts" / "autotimes_spectrum_tangent_compatibility.py",
        ROOT / "scripts" / "paper_four_dataset_parent_campaign.py",
        Path(__file__).resolve(),
    )

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import autotimes_atd_ds1_validation as validation  # noqa: E402
import autotimes_spectrum_tangent_compatibility as tangent  # noqa: E402
from models.TimerOperatorAR import Model  # noqa: E402
from models.AutoTimesSpectrumTangent import (  # noqa: E402
    build_phase_template,
    from_channel_batch,
    normalized_tangent_direction,
)
from paper_four_dataset_parent_campaign import DATASETS  # noqa: E402


_ORIGINAL_SELECT_AND_SUMMARIZE = tangent.select_and_summarize
_ORIGINAL_METRIC_ROWS = tangent.metric_rows
_ORIGINAL_SUMMARIZE_TEST = tangent.summarize_test


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path, rows):
    if not rows:
        raise ValueError("cannot write an empty Timer metric table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def model_id(dataset, seed):
    return (
        f"PATD_NAMED_TIMER_generated_k8_{dataset}_P96_"
        f"lr3em05_s{int(seed)}"
    )


def checkpoint(dataset, seed):
    identifier = model_id(dataset, seed)
    matches = [
        directory / "checkpoint.pth"
        for directory in CHECKPOINT_ROOT.glob(f"*{identifier}*")
        if (directory / "checkpoint.pth").is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple Timer checkpoints for {identifier}: {matches}")
    return matches[0] if matches else None


def model_config(timer_checkpoint, mode):
    return SimpleNamespace(
        seq_len=672,
        pred_len=720,
        external_timer_checkpoint=str(timer_checkpoint),
        external_timer_mode=mode,
        distilled_multi_commit_hidden=512,
        distilled_multi_commit_patches=1 if mode == "q1" else WIDTH,
        distilled_multi_commit_eval_patches=1 if mode == "q1" else WIDTH,
    )


def load_model(path, timer_checkpoint, mode, device):
    model = Model(model_config(timer_checkpoint, mode))
    if path is not None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def data_config(dataset, eval_batch_size, seed):
    spec = DATASETS[dataset]
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
        batch_size=int(eval_batch_size),
        eval_batch_size=int(eval_batch_size),
        num_workers=0,
        loader_seed=int(seed),
        augmentation_ratio=0,
        disable_autocast=True,
    )


def training_command(dataset, timer_checkpoint, seed, epochs, patience):
    spec = DATASETS[dataset]
    return [
        str(PYTHON), "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", spec["root_path"],
        "--data_path", spec["data_path"],
        "--model_id", model_id(dataset, seed),
        "--model", "TimerOperatorAR",
        "--data", spec["data"],
        "--features", "M",
        "--seq_len", "672",
        "--label_len", "576",
        "--pred_len", "720",
        "--freq", spec["freq"],
        "--enc_in", str(spec["channels"]),
        "--dec_in", str(spec["channels"]),
        "--c_out", str(spec["channels"]),
        "--d_model", "1024",
        "--dropout", "0.0",
        "--external_timer_checkpoint", str(timer_checkpoint),
        "--external_timer_mode", "generated",
        "--distilled_multi_commit_hidden", "512",
        "--distilled_multi_commit_patches", str(WIDTH),
        "--distilled_multi_commit_eval_patches", str(WIDTH),
        "--batch_size", str(TRAIN_BATCH[dataset]),
        "--eval_batch_size", str(EVAL_BATCH[dataset]),
        "--num_workers", "0",
        "--direct_patch_train_channels", "0",
        "--direct_patch_validation_channels", "0",
        "--learning_rate", str(FAR_LR),
        "--train_epochs", str(int(epochs)),
        "--patience", str(int(patience)),
        "--seed", str(int(seed)),
        "--loader_seed", str(int(seed)),
        "--checkpoint_selection", "vali",
        "--stream_metrics",
        "--report_horizons", "96", "192", "336", "720",
        "--skip_epoch_test",
        "--skip_final_test",
        "--lradj", "cosine",
        "--use_amp",
        "--data_scale",
        "--des", "timer_named_transfer",
        "--itr", "1",
    ]


def required_timer_files(timer_checkpoint):
    timer_checkpoint = Path(timer_checkpoint)
    names = (
        "config.json",
        "configuration_timer.py",
        "generation_config.json",
        "modeling_timer.py",
        "model.safetensors",
        "ts_generation_mixin.py",
    )
    paths = [timer_checkpoint / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def require_timer_runtime():
    import transformers

    if transformers.__version__ != TIMER_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Timer campaign requires transformers=="
            f"{TIMER_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    return transformers.__version__


def freeze_protocol(timer_checkpoint):
    require_timer_runtime()
    protocol_path = OUTPUT_ROOT / "protocol_frozen.json"
    code_paths = measurement_code_paths()
    payload = {
        "campaign": CAMPAIGN,
        "status": "frozen_before_new_measurement",
        "frozen_at_utc": utc_now(),
        "parent": "thuml/timer-base-84m",
        "parent_huggingface_revision": TIMER_HF_REVISION,
        "transformers_version": TIMER_TRANSFORMERS_VERSION,
        "parent_training": "none; immutable public pretrained Q1",
        "constant_window_policy": CONSTANT_WINDOW_POLICY,
        "datasets": list(DATASET_NAMES),
        "seeds": list(SEEDS),
        "seed_semantics": "far-exit initialization and loader only",
        "lookback": 672,
        "horizon": 720,
        "native_patch": PATCH_LEN,
        "variable_policy": "channel-independent, matching public Timer deployment",
        "trained_artifact": "Generated-K8 far exits only",
        "evaluated_prefixes": [1, 2, 4, 8],
        "far_learning_rate": FAR_LR,
        "train_epochs": 6,
        "patience": 2,
        "selector": "closed-loop validation H720",
        "test_during_training": False,
        "tangent_selection": "train-only blocks 1-3; block 4 confirmation",
        "benchmark_test_previously_exposed": True,
        "evidence_scope": "post-test named-parent compatibility",
        "timer_artifacts": {
            relative_path(path): sha256_file(path)
            for path in required_timer_files(timer_checkpoint)
        },
        "code_hashes": {
            relative_path(path): sha256_file(path) for path in code_paths
        },
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("frozen_at_utc", None)
        expected = dict(payload)
        expected.pop("frozen_at_utc", None)
        if comparable != expected:
            raise RuntimeError("existing Timer protocol does not match current sources")
        return protocol_path
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(protocol_path, payload)
    return protocol_path


def verify_frozen_protocol(timer_checkpoint):
    require_timer_runtime()
    protocol_path = OUTPUT_ROOT / "protocol_frozen.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("freeze the Timer protocol before measurement")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    errors = []
    if protocol.get("transformers_version") != TIMER_TRANSFORMERS_VERSION:
        errors.append("Timer transformers runtime changed")
    for recorded, expected in protocol["timer_artifacts"].items():
        path = ROOT / recorded if not Path(recorded).is_absolute() else Path(recorded)
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            errors.append(f"Timer artifact mismatch: {recorded}")
    for recorded, expected in protocol["code_hashes"].items():
        path = ROOT / recorded if not Path(recorded).is_absolute() else Path(recorded)
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            errors.append(f"code mismatch: {recorded}")
    current_files = required_timer_files(timer_checkpoint)
    if {relative_path(path) for path in current_files} != set(protocol["timer_artifacts"]):
        errors.append("Timer checkpoint file set changed")
    if errors:
        raise RuntimeError("frozen Timer protocol verification failed: " + "; ".join(errors))
    return protocol


def train_one(dataset, timer_checkpoint, seed, epochs, patience, dry_run):
    existing = checkpoint(dataset, seed)
    if existing is not None:
        print({"stage": "reuse", "checkpoint": relative_path(existing)}, flush=True)
        return
    command = training_command(dataset, timer_checkpoint, seed, epochs, patience)
    if dry_run:
        print(" ".join(command), flush=True)
        return
    log_path = LOG_ROOT / dataset / f"{model_id(dataset, seed)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        raise RuntimeError(f"Timer training failed; inspect {log_path}")
    resolved = checkpoint(dataset, seed)
    if resolved is None:
        raise FileNotFoundError(f"Timer training produced no checkpoint: {dataset} s{seed}")
    print({
        "stage": "trained",
        "dataset": dataset,
        "seed": int(seed),
        "minutes": (time.time() - started) / 60.0,
        "checkpoint": relative_path(resolved),
    }, flush=True)


def run_train(datasets, timer_checkpoint, epochs, patience, dry_run):
    if not dry_run:
        verify_frozen_protocol(timer_checkpoint)
    for dataset in datasets:
        for seed in SEEDS:
            train_one(
                dataset, timer_checkpoint, seed, epochs, patience, dry_run
            )


def evaluate_dataset(dataset, timer_checkpoint, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation.DATASET = dataset
    validation.TOKEN_LEN = PATCH_LEN
    rows = []
    for seed in SEEDS:
        path = checkpoint(dataset, seed)
        if path is None:
            raise FileNotFoundError(f"missing Timer Generated-K8: {dataset} s{seed}")
        q1 = load_model(None, timer_checkpoint, "q1", device)
        generated = load_model(path, timer_checkpoint, "generated", device)
        rows.extend(validation.evaluate(
            q1,
            generated,
            data_config(dataset, EVAL_BATCH[dataset], seed),
            "val" if split == "validation" else "test",
            seed,
            device,
        ))
        del q1, generated
        if device.type == "cuda":
            torch.cuda.empty_cache()
    k1 = [
        row for row in rows
        if row["artifact"] == "atd_k8" and int(row["commit_patches"]) == 1
    ]
    if len(k1) != len(SEEDS) or any(float(row["q1_max_abs_delta"]) != 0.0 for row in k1):
        raise RuntimeError(f"Timer K1 exactness failed on {dataset} {split}")
    return rows


def run_validation(datasets, timer_checkpoint):
    verify_frozen_protocol(timer_checkpoint)
    for dataset in datasets:
        rows = evaluate_dataset(dataset, timer_checkpoint, "validation")
        path = OUTPUT_ROOT / dataset / "validation_metrics.csv"
        write_rows(path, rows)
        print({
            "stage": "validation",
            "dataset": dataset,
            "rows": len(rows),
            "k1_exact": True,
        }, flush=True)


def selection_output(dataset):
    return OUTPUT_ROOT / dataset / "tangent"


def source_artifacts_for(dataset, timer_checkpoint, seeds):
    candidates = [
        ("timer_checkpoint", "", path)
        for path in required_timer_files(timer_checkpoint)
    ]
    validation_path = OUTPUT_ROOT / dataset / "validation_metrics.csv"
    candidates.append(("validation_metrics", "", validation_path))
    for seed in seeds:
        path = checkpoint(dataset, seed)
        if path is None:
            raise FileNotFoundError(f"missing Timer checkpoint: {dataset} s{seed}")
        candidates.append(("generated_k8_checkpoint", int(seed), path))
    rows = []
    for role, seed, path in candidates:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "role": role,
            "seed": seed,
            "path": relative_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


@torch.inference_mode()
def collect_origin_moments(
    model, dataset, indices, batch_size, width, periods, device,
):
    output = []
    total_patches = (model.pred_len + model.patch_len - 1) // model.patch_len
    for batch_x, batch_y, _, _ in tangent.loader_for_indices(
        dataset, indices, batch_size
    ):
        observed = batch_x.float().to(device)
        truth = batch_y[:, -model.pred_len:].float().to(device)
        batch_origins = observed.size(0)
        history, _, channels = model._to_channel_batch(observed[:, -model.seq_len:])
        origin_mean, origin_std = model._instance_stats(history)
        templates = {
            int(period): build_phase_template(model, history.clone(), int(period))
            for period in periods
        }
        moments = torch.zeros(
            batch_origins, len(periods), 3,
            dtype=torch.float64, device=device,
        )
        state = model._initial_rollout(history.squeeze(-1))
        committed = 0
        age = 0
        while committed < total_patches:
            local_width = min(int(width), total_patches - committed)
            patches = model._propose_patches(state, local_width)
            normalized = (patches - origin_mean) / origin_std
            raw = from_channel_batch(
                patches.reshape(history.size(0), -1), batch_origins, channels
            )
            valid = min(raw.size(1), model.pred_len - age)
            error = (truth[:, age:age + valid] - raw[:, :valid]).double()
            error_energy = error.square().sum(dim=(1, 2))
            for period_index, period in enumerate(periods):
                direction = normalized_tangent_direction(
                    normalized,
                    templates[int(period)],
                    committed,
                    model.patch_len,
                )
                direction_points = direction * origin_std
                direction_batch = from_channel_batch(
                    direction_points.reshape(history.size(0), -1),
                    batch_origins,
                    channels,
                )[:, :valid].double()
                moments[:, period_index, 0].add_(error_energy)
                moments[:, period_index, 1].add_(
                    direction_batch.square().sum(dim=(1, 2))
                )
                moments[:, period_index, 2].add_(
                    (direction_batch * error).sum(dim=(1, 2))
                )
            committed += local_width
            age += raw.size(1)
            if committed < total_patches:
                state = model._consume_patches(state, patches)
        output.append((moments / float(model.pred_len * channels)).cpu().numpy())
    return np.concatenate(output, axis=0)


def configure_tangent(dataset, timer_checkpoint):
    tangent.DATASET = dataset
    tangent.CAMPAIGN = f"{CAMPAIGN}_{dataset}"
    tangent.LEARNING_RATE = FAR_LR
    tangent.CODE_PATHS = (
        *measurement_code_paths(),
    )
    tangent.data_config = lambda batch_size, seed: data_config(
        dataset, batch_size, seed
    )
    tangent.collect_origin_moments = collect_origin_moments
    tangent.forecast_with_explicit_tangent = (
        lambda model, values, width, period, gamma:
        model.forecast_with_explicit_tangent(values, width, period, gamma)
    )

    def local_load_generated(seed, ignored_checkpoint, device):
        del ignored_checkpoint
        path = checkpoint(dataset, int(seed))
        if path is None:
            raise FileNotFoundError(f"missing Timer checkpoint: {dataset} s{seed}")
        model = load_model(path, timer_checkpoint, "generated", device)
        return model, Path(timer_checkpoint) / "model.safetensors", path

    def local_source_artifacts(ignored_checkpoint, seeds):
        del ignored_checkpoint
        return source_artifacts_for(dataset, timer_checkpoint, seeds)

    def local_protocol(args, artifacts, widths, seeds, periods):
        return {
            "campaign": f"{CAMPAIGN}_{dataset}",
            "created_at_utc": utc_now(),
            "status": "frozen_before_train_measurement",
            "method": "spectrum-selected explicit tangent",
            "parent": "thuml/timer-base-84m Generated-K8",
            "dataset": dataset,
            "evidence_scope": "post-test named-parent compatibility",
            "benchmark_test_previously_exposed": True,
            "selection_split": "train_only",
            "validation_opened_for_selection": False,
            "test_opened_for_selection": False,
            "train_origins_per_seed": int(args.train_origins),
            "seeds": list(map(int, seeds)),
            "commit_widths": list(map(int, widths)),
            "candidate_periods": list(map(int, periods)),
            "chronology_blocks": tangent.CHRONOLOGY_BLOCKS,
            "selection_blocks": [1, 2, 3],
            "forward_confirmation_block": 4,
            "selection_score": "max(B,0)^2/(A*raw_error_energy)",
            "coefficient": "gamma=max(B/A,0)",
            "base_beta": tangent.DEFAULT_BETA,
            "band_edges": list(tangent.DEFAULT_BAND_EDGES),
            "band_multipliers": list(tangent.DEFAULT_BANDS),
            "protected_prefix": "slot 1 of every macro is hard-preserved",
            "template": "immutable; initial observed history only",
            "rollout": "corrected far commits are written back",
            "source_learning_rate": FAR_LR,
            "source_artifact_manifest": "artifact_manifest.csv",
            "source_artifact_count": len(artifacts),
            "canonical_ledger_writes": False,
            "checkpoint_writes": False,
        }

    def local_select_and_summarize(rows, widths, seeds, periods):
        summaries = _ORIGINAL_SELECT_AND_SUMMARIZE(
            rows, widths, seeds, periods
        )
        for row in summaries:
            row["parent"] = "Timer-base-84m"
        return summaries

    def local_metric_rows(seed, width, period, gamma, values, baseline):
        rows, chronology = _ORIGINAL_METRIC_ROWS(
            seed, width, period, gamma, values, baseline
        )
        for row in rows + chronology:
            row["parent"] = "Timer-base-84m"
        return rows, chronology

    def local_summarize_test(rows, chronology, widths, seeds):
        summaries = _ORIGINAL_SUMMARIZE_TEST(
            rows, chronology, widths, seeds
        )
        for row in summaries:
            row["parent"] = "Timer-base-84m"
        return summaries

    def local_baseline_lookup():
        path = OUTPUT_ROOT / dataset / "test_baseline.csv"
        rows = read_rows(path)
        return {
            (int(row["seed"]), int(row["commit_patches"])): row
            for row in rows
            if row["split"] == "test" and row["artifact"] == "atd_k8"
        }

    tangent.load_generated = local_load_generated
    tangent.source_artifacts = local_source_artifacts
    tangent.preregistered_protocol = local_protocol
    tangent.select_and_summarize = local_select_and_summarize
    tangent.metric_rows = local_metric_rows
    tangent.summarize_test = local_summarize_test
    tangent.source_baseline_lookup = local_baseline_lookup


def tangent_args(dataset, timer_checkpoint, open_test=False):
    return SimpleNamespace(
        llm_checkpoint=Path(timer_checkpoint),
        output_dir=selection_output(dataset),
        train_origins=512,
        train_batch_size=EVAL_BATCH[dataset],
        test_batch_size=EVAL_BATCH[dataset],
        test_amp=True,
        open_test=bool(open_test),
    )


def run_select(datasets, timer_checkpoint):
    verify_frozen_protocol(timer_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        configure_tangent(dataset, timer_checkpoint)
        output = selection_output(dataset)
        if (output / "selection_lock.json").is_file():
            tangent.verify_selection_lock(output)
            print({"stage": "selection_reuse", "dataset": dataset}, flush=True)
            continue
        tangent.run_select(
            tangent_args(dataset, timer_checkpoint),
            WIDTHS,
            SEEDS,
            tangent.PERIODS,
            device,
        )


def verify_all_selection_locks(datasets, timer_checkpoint):
    verify_frozen_protocol(timer_checkpoint)
    for dataset in datasets:
        configure_tangent(dataset, timer_checkpoint)
        tangent.verify_selection_lock(selection_output(dataset))


def run_baseline_test(datasets, timer_checkpoint, open_test):
    if not open_test:
        raise RuntimeError("baseline test requires explicit --open-test")
    verify_all_selection_locks(datasets, timer_checkpoint)
    for dataset in datasets:
        path = OUTPUT_ROOT / dataset / "test_baseline.csv"
        if path.exists():
            raise RuntimeError(f"Timer test baseline already exists: {path}")
        rows = evaluate_dataset(dataset, timer_checkpoint, "test")
        write_rows(path, rows)
        print({
            "stage": "baseline_test",
            "dataset": dataset,
            "rows": len(rows),
            "k1_exact": True,
        }, flush=True)


def run_tangent_test(datasets, timer_checkpoint, open_test):
    if not open_test:
        raise RuntimeError("tangent test requires explicit --open-test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        baseline = OUTPUT_ROOT / dataset / "test_baseline.csv"
        if not baseline.is_file():
            raise FileNotFoundError(
                f"fresh Timer baseline must precede tangent test: {baseline}"
            )
        configure_tangent(dataset, timer_checkpoint)
        tangent.run_test(
            tangent_args(dataset, timer_checkpoint, open_test=True),
            WIDTHS,
            SEEDS,
            device,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "freeze",
            "train",
            "validation",
            "select",
            "all-pretest",
            "baseline-test",
            "tangent-test",
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument(
        "--timer-checkpoint", type=Path,
        default=DEFAULT_TIMER_CHECKPOINT,
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--open-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = tuple(args.datasets)
    if args.epochs != 6 or args.patience != 2:
        raise ValueError("Timer protocol fixes epochs=6 and patience=2")
    if args.stage == "freeze":
        path = freeze_protocol(args.timer_checkpoint)
        print({"stage": "frozen", "protocol": relative_path(path)}, flush=True)
    elif args.stage == "train":
        run_train(
            datasets, args.timer_checkpoint, args.epochs,
            args.patience, args.dry_run,
        )
    elif args.stage == "validation":
        run_validation(datasets, args.timer_checkpoint)
    elif args.stage == "select":
        run_select(datasets, args.timer_checkpoint)
    elif args.stage == "all-pretest":
        if not args.dry_run:
            freeze_protocol(args.timer_checkpoint)
        else:
            required_timer_files(args.timer_checkpoint)
        run_train(
            datasets, args.timer_checkpoint, args.epochs,
            args.patience, args.dry_run,
        )
        if not args.dry_run:
            run_validation(datasets, args.timer_checkpoint)
            run_select(datasets, args.timer_checkpoint)
    elif args.stage == "baseline-test":
        run_baseline_test(datasets, args.timer_checkpoint, args.open_test)
    else:
        run_tangent_test(datasets, args.timer_checkpoint, args.open_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
