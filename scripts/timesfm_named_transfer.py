#!/usr/bin/env python3
"""Frozen four-data-set transfer campaign for public TimesFM 2.5 200M.

The public cached median-writeback decoder is the immutable Q1 parent.  A
single deterministic train/validation trajectory cache is shared by three
far-exit initializations; test remains inaccessible until train-only Tangent
selection locks have been written and verified.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT / "analysis_outputs" / "exploratory" /
    "timesfm_named_transfer_v1_20260815"
)
DEFAULT_SOURCE = ROOT / "external_backbones" / "timesfm" / "src"
DEFAULT_CHECKPOINT = ROOT / "foundation_models" / "timesfm-2.5-200m-pytorch"
DATASET_NAMES = ("ETTh1", "ETTh2", "ETTm1", "Weather")
SEEDS = (2021, 2022, 2023)
WIDTHS = (2, 4, 8)
WIDTH = 8
BLOCK_LEN = 128
TRAIN_ORIGINS = 2048
VALIDATION_ORIGINS = 512
CACHE_BATCH = {"ETTh1": 16, "ETTh2": 16, "ETTm1": 16, "Weather": 4}
EVAL_BATCH = {"ETTh1": 16, "ETTh2": 16, "ETTm1": 16, "Weather": 4}
HEAD_BATCH = 2048
HEAD_LR = 3e-4
HEAD_EPOCHS = 30
HEAD_PATIENCE = 4

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import autotimes_atd_ds1_validation as validation  # noqa: E402
import autotimes_spectrum_tangent_compatibility as tangent  # noqa: E402
from data_provider.data_factory import data_provider  # noqa: E402
from models.AutoTimesSpectrumTangent import (  # noqa: E402
    build_phase_template,
    from_channel_batch,
    normalized_tangent_direction,
)
from models.TimesFMOperatorAR import Model  # noqa: E402
from paper_four_dataset_parent_campaign import DATASETS  # noqa: E402


CAMPAIGN = "timesfm_named_transfer_v1_20260815"
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


def sha256_tree(root, pattern="*.py"):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no {pattern} files below {root}")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
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
        raise ValueError("cannot write an empty TimesFM table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_commit(source):
    completed = subprocess.run(
        ["git", "-C", str(Path(source).parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def required_parent_files(checkpoint):
    checkpoint = Path(checkpoint)
    paths = [checkpoint / "config.json", checkpoint / "model.safetensors"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def measurement_code_paths():
    return (
        ROOT / "models" / "TimesFMOperatorAR.py",
        ROOT / "models" / "AutoTimesSpectrumTangent.py",
        ROOT / "scripts" / "autotimes_atd_ds1_validation.py",
        ROOT / "scripts" / "autotimes_spectrum_tangent_compatibility.py",
        ROOT / "scripts" / "paper_four_dataset_parent_campaign.py",
        Path(__file__).resolve(),
    )


def model_config(source, checkpoint, mode):
    return SimpleNamespace(
        seq_len=672,
        pred_len=720,
        external_timesfm_source=str(source),
        external_timesfm_checkpoint=str(checkpoint),
        external_timesfm_mode=mode,
        distilled_multi_commit_hidden=512,
        distilled_multi_commit_patches=1 if mode == "q1" else WIDTH,
        distilled_multi_commit_eval_patches=1 if mode == "q1" else WIDTH,
    )


def generated_checkpoint(dataset, seed):
    path = OUTPUT_ROOT / dataset / "checkpoints" / f"seed_{int(seed)}.pth"
    return path if path.is_file() else None


def load_model(source, parent, learned, device):
    model = Model(model_config(source, parent, "generated"))
    if learned is not None:
        state = torch.load(learned, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def data_config(dataset, batch_size, seed):
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
        batch_size=int(batch_size),
        eval_batch_size=int(batch_size),
        num_workers=0,
        loader_seed=int(seed),
        augmentation_ratio=0,
        disable_autocast=True,
    )


def freeze_protocol(source, checkpoint):
    path = OUTPUT_ROOT / "protocol_frozen.json"
    payload = {
        "campaign": CAMPAIGN,
        "status": "frozen_before_new_measurement",
        "frozen_at_utc": utc_now(),
        "parent": "google/timesfm-2.5-200m-pytorch",
        "parent_source_commit": source_commit(source),
        "parent_source_tree_sha256": sha256_tree(source),
        "parent_training": "none; immutable public pretrained Q1",
        "datasets": list(DATASET_NAMES),
        "seeds": list(SEEDS),
        "seed_semantics": "far-exit initialization and cache shuffling only",
        "lookback": 672,
        "horizon": 720,
        "native_input_patch": 32,
        "native_output_block": BLOCK_LEN,
        "native_writeback": "cached median block (quantile index 5)",
        "variable_policy": "channel-independent native TimesFM deployment",
        "train_cache_origins": TRAIN_ORIGINS,
        "validation_cache_origins": VALIDATION_ORIGINS,
        "cache_indices": "uniformly spaced, deterministic, shared across seeds",
        "trained_artifact": "Generated-K8 far exits only",
        "evaluated_prefixes": [1, 2, 4, 8],
        "head_learning_rate": HEAD_LR,
        "head_epochs": HEAD_EPOCHS,
        "head_patience": HEAD_PATIENCE,
        "head_batch": HEAD_BATCH,
        "checkpoint_selector": "recursive-target validation-cache MSE",
        "test_during_training": False,
        "tangent_selection": "train-only blocks 1-3; block 4 confirmation",
        "benchmark_test_previously_exposed": True,
        "evidence_scope": "post-test named-parent compatibility",
        "parent_artifacts": {
            relative_path(item): sha256_file(item)
            for item in required_parent_files(checkpoint)
        },
        "code_hashes": {
            relative_path(item): sha256_file(item)
            for item in measurement_code_paths()
        },
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("frozen_at_utc", None)
        expected = dict(payload)
        expected.pop("frozen_at_utc", None)
        if comparable != expected:
            raise RuntimeError("existing TimesFM protocol differs from sources")
        return path
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path


def verify_protocol(source, checkpoint):
    path = OUTPUT_ROOT / "protocol_frozen.json"
    if not path.is_file():
        raise FileNotFoundError("freeze TimesFM protocol before measurement")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if source_commit(source) != protocol["parent_source_commit"]:
        errors.append("TimesFM source commit changed")
    if sha256_tree(source) != protocol["parent_source_tree_sha256"]:
        errors.append("TimesFM source tree changed")
    for recorded, expected in {
        **protocol["parent_artifacts"], **protocol["code_hashes"]
    }.items():
        item = Path(recorded)
        if not item.is_absolute():
            item = ROOT / item
        actual = sha256_file(item) if item.is_file() else "missing"
        if actual != expected:
            errors.append(f"artifact mismatch: {recorded}")
    current = {relative_path(item) for item in required_parent_files(checkpoint)}
    if current != set(protocol["parent_artifacts"]):
        errors.append("TimesFM parent file set changed")
    if errors:
        raise RuntimeError("TimesFM protocol verification failed: " + "; ".join(errors))
    return protocol


def cache_path(dataset, split):
    return OUTPUT_ROOT / dataset / "trajectory_cache" / f"{split}.pth"


@torch.inference_mode()
def build_cache(dataset, split, origins, source, parent, device):
    output = cache_path(dataset, split)
    if output.is_file():
        print({"stage": "cache_reuse", "path": relative_path(output)}, flush=True)
        return
    model = load_model(source, parent, None, device)
    dataset_object, _ = data_provider(
        data_config(dataset, CACHE_BATCH[dataset], SEEDS[0]),
        "train" if split == "train" else "val",
    )
    indices = tangent.uniformly_spaced_indices(len(dataset_object), int(origins))
    features = []
    targets = []
    for batch_x, _, _, _ in tangent.loader_for_indices(
        dataset_object, indices, CACHE_BATCH[dataset]
    ):
        history, _, _ = model._to_channel_batch(
            batch_x.float().to(device)[:, -model.seq_len:]
        )
        local_features, local_targets = model.trajectory_training_pair(
            history.squeeze(-1)
        )
        features.append(local_features.float().cpu())
        targets.append(local_targets.float().cpu())
    payload = {
        "dataset": dataset,
        "split": split,
        "origin_indices": torch.as_tensor(indices, dtype=torch.int64),
        "features": torch.cat(features),
        "targets": torch.cat(targets),
        "parent_sha256": sha256_file(Path(parent) / "model.safetensors"),
        "source_commit": source_commit(source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print({
        "stage": "cache",
        "dataset": dataset,
        "split": split,
        "origins": len(indices),
        "channel_rows": int(payload["features"].size(0)),
        "path": relative_path(output),
    }, flush=True)
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_cache(datasets, source, parent):
    verify_protocol(source, parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        build_cache(dataset, "train", TRAIN_ORIGINS, source, parent, device)
        build_cache(
            dataset, "validation", VALIDATION_ORIGINS, source, parent, device
        )


def load_cache(dataset, split, parent):
    payload = torch.load(cache_path(dataset, split), map_location="cpu", weights_only=True)
    if payload["parent_sha256"] != sha256_file(Path(parent) / "model.safetensors"):
        raise RuntimeError("trajectory cache parent hash mismatch")
    protocol = json.loads(
        (OUTPUT_ROOT / "protocol_frozen.json").read_text(encoding="utf-8")
    )
    if payload["source_commit"] != protocol["parent_source_commit"]:
        raise RuntimeError("trajectory cache source commit mismatch")
    return payload["features"], payload["targets"]


def far_prediction(model, features):
    return torch.stack([head(features) for head in model.far_heads], dim=1)


@torch.inference_mode()
def cache_mse(model, features, targets, device):
    square = 0.0
    count = 0
    for start in range(0, features.size(0), HEAD_BATCH):
        local_features = features[start:start + HEAD_BATCH].to(device)
        local_targets = targets[start:start + HEAD_BATCH].to(device)
        error = far_prediction(model, local_features) - local_targets
        square += float(error.double().square().sum().cpu())
        count += error.numel()
    return square / count


def train_one(dataset, seed, source, parent, device):
    output = OUTPUT_ROOT / dataset / "checkpoints" / f"seed_{int(seed)}.pth"
    if output.is_file():
        print({"stage": "train_reuse", "path": relative_path(output)}, flush=True)
        return
    train_features, train_targets = load_cache(dataset, "train", parent)
    val_features, val_targets = load_cache(dataset, "validation", parent)
    torch.manual_seed(int(seed))
    model = load_model(source, parent, None, device).train()
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(HEAD_LR), lr=HEAD_LR
    )
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=HEAD_BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
    )
    best = math.inf
    best_state = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        train_square = 0.0
        train_count = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = far_prediction(model, features)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            loss.backward()
            optimizer.step()
            train_square += float(loss.detach()) * targets.numel()
            train_count += targets.numel()
        model.eval()
        validation_mse = cache_mse(
            model, val_features, val_targets, device
        )
        history.append({
            "epoch": epoch,
            "train_target_mse": train_square / train_count,
            "validation_target_mse": validation_mse,
        })
        if validation_mse < best:
            best = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= HEAD_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("TimesFM far-head training produced no checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    write_json(output.with_suffix(".json"), {
        "dataset": dataset,
        "seed": int(seed),
        "best_validation_target_mse": best,
        "epochs_ran": len(history),
        "minutes": (time.time() - started) / 60.0,
        "history": history,
    })
    print({
        "stage": "trained",
        "dataset": dataset,
        "seed": int(seed),
        "best_validation_target_mse": best,
        "epochs": len(history),
        "path": relative_path(output),
    }, flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_train(datasets, source, parent):
    verify_protocol(source, parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        for seed in SEEDS:
            train_one(dataset, seed, source, parent, device)


def evaluate_dataset(dataset, source, parent, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation.DATASET = dataset
    validation.TOKEN_LEN = BLOCK_LEN
    rows = []
    for seed in SEEDS:
        learned = generated_checkpoint(dataset, seed)
        if learned is None:
            raise FileNotFoundError(f"missing TimesFM far exits: {dataset} s{seed}")
        generated = load_model(source, parent, learned, device)
        rows.extend(validation.evaluate(
            generated,
            generated,
            data_config(dataset, EVAL_BATCH[dataset], seed),
            "val" if split == "validation" else "test",
            seed,
            device,
        ))
        del generated
        if device.type == "cuda":
            torch.cuda.empty_cache()
    k1 = [
        row for row in rows
        if row["artifact"] == "atd_k8" and int(row["commit_patches"]) == 1
    ]
    if len(k1) != len(SEEDS) or any(float(row["q1_max_abs_delta"]) != 0.0 for row in k1):
        raise RuntimeError(f"TimesFM K1 exactness failed on {dataset} {split}")
    return rows


def run_validation(datasets, source, parent):
    verify_protocol(source, parent)
    for dataset in datasets:
        rows = evaluate_dataset(dataset, source, parent, "validation")
        write_rows(OUTPUT_ROOT / dataset / "validation_metrics.csv", rows)
        print({"stage": "validation", "dataset": dataset, "rows": len(rows)}, flush=True)


def selection_output(dataset):
    return OUTPUT_ROOT / dataset / "tangent"


def source_artifacts_for(dataset, source, parent, seeds):
    candidates = [
        ("timesfm_parent", "", item) for item in required_parent_files(parent)
    ]
    candidates.extend(
        ("timesfm_source", "", item)
        for item in sorted(Path(source).rglob("*.py"))
        if item.is_file()
    )
    candidates.append((
        "validation_metrics", "", OUTPUT_ROOT / dataset / "validation_metrics.csv"
    ))
    for split in ("train", "validation"):
        candidates.append(("trajectory_cache", split, cache_path(dataset, split)))
    for seed in seeds:
        learned = generated_checkpoint(dataset, int(seed))
        if learned is None:
            raise FileNotFoundError(f"missing TimesFM far exits: {dataset} s{seed}")
        candidates.append(("generated_k8_checkpoint", int(seed), learned))
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
    total_blocks = (model.pred_len + model.patch_len - 1) // model.patch_len
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
        state = model._initial_rollout(history.squeeze(-1), total_blocks)
        committed = 0
        age = 0
        while committed < total_blocks:
            local_width = min(int(width), total_blocks - committed)
            blocks = model._propose_blocks(state, local_width)
            normalized = (blocks - origin_mean) / origin_std
            raw = from_channel_batch(
                blocks.reshape(history.size(0), -1), batch_origins, channels
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
            if committed < total_blocks:
                state = model._consume_blocks(state, blocks)
        output.append((moments / float(model.pred_len * channels)).cpu().numpy())
    return np.concatenate(output, axis=0)


def configure_tangent(dataset, source, parent):
    tangent.DATASET = dataset
    tangent.CAMPAIGN = f"{CAMPAIGN}_{dataset}"
    tangent.LEARNING_RATE = HEAD_LR
    tangent.CODE_PATHS = measurement_code_paths()
    tangent.data_config = lambda batch_size, seed: data_config(
        dataset, batch_size, seed
    )
    tangent.collect_origin_moments = collect_origin_moments
    tangent.forecast_with_explicit_tangent = (
        lambda model, values, width, period, gamma:
        model.forecast_with_explicit_tangent(values, width, period, gamma)
    )

    def local_load(seed, ignored_checkpoint, device):
        del ignored_checkpoint
        learned = generated_checkpoint(dataset, int(seed))
        if learned is None:
            raise FileNotFoundError(f"missing TimesFM far exits: {dataset} s{seed}")
        return (
            load_model(source, parent, learned, device),
            Path(parent) / "model.safetensors",
            learned,
        )

    def local_artifacts(ignored_checkpoint, seeds):
        del ignored_checkpoint
        return source_artifacts_for(dataset, source, parent, seeds)

    def local_protocol(args, artifacts, widths, seeds, periods):
        return {
            "campaign": f"{CAMPAIGN}_{dataset}",
            "created_at_utc": utc_now(),
            "status": "frozen_before_train_measurement",
            "method": "spectrum-selected explicit tangent",
            "parent": "TimesFM-2.5-200M Generated-K8",
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
            "protected_prefix": "first native 128-point block of every macro",
            "template": "immutable; initial observed history only",
            "rollout": "corrected far blocks enter the native cached writeback",
            "source_learning_rate": HEAD_LR,
            "source_artifact_manifest": "artifact_manifest.csv",
            "source_artifact_count": len(artifacts),
            "canonical_ledger_writes": False,
            "checkpoint_writes": False,
        }

    def local_select(rows, widths, seeds, periods):
        summaries = _ORIGINAL_SELECT_AND_SUMMARIZE(rows, widths, seeds, periods)
        for row in summaries:
            row["parent"] = "TimesFM-2.5-200M"
        return summaries

    def local_metrics(seed, width, period, gamma, values, baseline):
        rows, chronology = _ORIGINAL_METRIC_ROWS(
            seed, width, period, gamma, values, baseline
        )
        for row in rows + chronology:
            row["parent"] = "TimesFM-2.5-200M"
        return rows, chronology

    def local_summary(rows, chronology, widths, seeds):
        summaries = _ORIGINAL_SUMMARIZE_TEST(rows, chronology, widths, seeds)
        for row in summaries:
            row["parent"] = "TimesFM-2.5-200M"
        return summaries

    def local_baseline():
        rows = read_rows(OUTPUT_ROOT / dataset / "test_baseline.csv")
        return {
            (int(row["seed"]), int(row["commit_patches"])): row
            for row in rows
            if row["split"] == "test" and row["artifact"] == "atd_k8"
        }

    tangent.load_generated = local_load
    tangent.source_artifacts = local_artifacts
    tangent.preregistered_protocol = local_protocol
    tangent.select_and_summarize = local_select
    tangent.metric_rows = local_metrics
    tangent.summarize_test = local_summary
    tangent.source_baseline_lookup = local_baseline


def tangent_args(dataset, parent, open_test=False):
    return SimpleNamespace(
        llm_checkpoint=Path(parent),
        output_dir=selection_output(dataset),
        train_origins=512,
        train_batch_size=EVAL_BATCH[dataset],
        test_batch_size=EVAL_BATCH[dataset],
        test_amp=False,
        open_test=bool(open_test),
    )


def run_select(datasets, source, parent):
    verify_protocol(source, parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        configure_tangent(dataset, source, parent)
        output = selection_output(dataset)
        if (output / "selection_lock.json").is_file():
            tangent.verify_selection_lock(output)
            print({"stage": "selection_reuse", "dataset": dataset}, flush=True)
            continue
        tangent.run_select(
            tangent_args(dataset, parent), WIDTHS, SEEDS, tangent.PERIODS, device
        )


def verify_selection(datasets, source, parent):
    verify_protocol(source, parent)
    for dataset in datasets:
        configure_tangent(dataset, source, parent)
        tangent.verify_selection_lock(selection_output(dataset))


def run_baseline_test(datasets, source, parent, open_test):
    if not open_test:
        raise RuntimeError("baseline test requires explicit --open-test")
    verify_selection(datasets, source, parent)
    for dataset in datasets:
        output = OUTPUT_ROOT / dataset / "test_baseline.csv"
        if output.exists():
            raise RuntimeError(f"TimesFM test baseline already exists: {output}")
        rows = evaluate_dataset(dataset, source, parent, "test")
        write_rows(output, rows)
        print({"stage": "baseline_test", "dataset": dataset, "rows": len(rows)}, flush=True)


def run_tangent_test(datasets, source, parent, open_test):
    if not open_test:
        raise RuntimeError("tangent test requires explicit --open-test")
    verify_selection(datasets, source, parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        baseline = OUTPUT_ROOT / dataset / "test_baseline.csv"
        if not baseline.is_file():
            raise FileNotFoundError(f"fresh TimesFM baseline required: {baseline}")
        configure_tangent(dataset, source, parent)
        tangent.run_test(
            tangent_args(dataset, parent, open_test=True),
            WIDTHS,
            SEEDS,
            device,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "freeze", "cache", "train", "validation", "select",
            "all-pretest", "baseline-test", "tangent-test",
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument("--timesfm-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--timesfm-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = tuple(args.datasets)
    source = args.timesfm_source
    parent = args.timesfm_checkpoint
    if args.dry_run:
        required_parent_files(parent)
        source_commit(source)
        print({
            "stage": args.stage,
            "datasets": datasets,
            "source": str(source),
            "parent": str(parent),
            "test_open": bool(args.open_test),
        })
        return 0
    if args.stage == "freeze":
        print({"stage": "frozen", "path": relative_path(freeze_protocol(source, parent))})
    elif args.stage == "cache":
        run_cache(datasets, source, parent)
    elif args.stage == "train":
        run_train(datasets, source, parent)
    elif args.stage == "validation":
        run_validation(datasets, source, parent)
    elif args.stage == "select":
        run_select(datasets, source, parent)
    elif args.stage == "all-pretest":
        freeze_protocol(source, parent)
        run_cache(datasets, source, parent)
        run_train(datasets, source, parent)
        run_validation(datasets, source, parent)
        run_select(datasets, source, parent)
    elif args.stage == "baseline-test":
        run_baseline_test(datasets, source, parent, args.open_test)
    else:
        run_tangent_test(datasets, source, parent, args.open_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
