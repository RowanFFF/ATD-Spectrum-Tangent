#!/usr/bin/env python3
"""Frozen AutoTimes compatibility audit for the spectrum-selected tangent.

This campaign reuses the accepted three-seed AutoTimes Generated-K8 artifacts.
It has two deliberately separate stages:

``select``
    Fit one period and one non-negative scalar per commit width using only 512
    uniformly spaced train origins.  Chronology blocks 1--3 select; block 4 is
    forward confirmation.  The exact choices, source hashes, and code hashes
    are frozen in ``selection_lock.json``.

``test``
    Refuse to run unless ``--open-test`` is supplied and the selection lock
    still verifies.  Evaluate the raw Generated and corrected rollouts on the
    already-exposed ETTh1 benchmark test split.  These results are therefore
    post-test architecture-compatibility evidence, never prospective evidence.

The script writes only its isolated exploratory output directory and never
touches canonical ledgers or model checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from autotimes_atd_ds1_pilot import checkpoint  # noqa: E402
from autotimes_atd_ds1_validation import (  # noqa: E402
    data_config,
    load_model,
    model_config,
)
from data_provider.data_factory import data_provider  # noqa: E402
from models.AutoTimesSpectrumTangent import (  # noqa: E402
    DEFAULT_BAND_EDGES,
    DEFAULT_BANDS,
    DEFAULT_BETA,
    build_phase_template,
    forecast_with_explicit_tangent,
    from_channel_batch,
    normalized_tangent_direction,
)


CAMPAIGN = "autotimes_spectrum_tangent_compatibility_v1_20260815"
OUTPUT_DIR = ROOT / "analysis_outputs" / "exploratory" / CAMPAIGN
DEFAULT_LLM_CHECKPOINT = ROOT / "external_backbones" / "weights" / "gpt2-11c5a3d"
DATASET = "ETTh1"
SEEDS = (2021, 2022, 2023)
WIDTHS = (2, 4, 8)
PERIODS = tuple(range(12, 217, 12))
HORIZONS = (96, 192, 336, 720)
LEARNING_RATE = 3e-5
CHRONOLOGY_BLOCKS = 4
SOURCE_BASELINE = ROOT / "analysis_outputs" / "autotimes_atd_ds1_three_seed.csv"
CODE_PATHS = (
    ROOT / "models" / "AutoTimesOperatorAR.py",
    ROOT / "models" / "AutoTimesSpectrumTangent.py",
    Path(__file__).resolve(),
)


def uniformly_spaced_indices(length, count):
    """Choose the frozen, uniformly spaced train origins used by selection."""
    if int(length) < 1 or int(count) < 1:
        raise ValueError("length and count must be positive")
    count = min(int(length), int(count))
    return (
        torch.linspace(0, int(length) - 1, steps=count)
        .round().long().unique().tolist()
    )


def loader_for_indices(dataset, indices, batch_size):
    """Build a deterministic loader over an explicit list of origins."""
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def write_csv_atomic(path, rows):
    """Write a non-empty CSV without exposing a partially written file."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_values(text, cast):
    return tuple(
        cast(value.strip()) for value in str(text).split(",")
        if value.strip()
    )


def require_empty_output(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"selection output already exists and will not be overwritten: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def source_artifacts(llm_checkpoint, seeds):
    llm_checkpoint = Path(llm_checkpoint)
    candidates = [
        ("llm_config", "", llm_checkpoint / "config.json"),
        ("llm_weights", "", llm_checkpoint / "model.safetensors"),
        ("source_baseline", "", SOURCE_BASELINE),
    ]
    for seed in seeds:
        q1_path = checkpoint("q1", seed=int(seed))
        generated_path = checkpoint(
            "generated_k8", LEARNING_RATE, seed=int(seed)
        )
        if q1_path is None or generated_path is None:
            raise FileNotFoundError(
                f"missing accepted AutoTimes lineage for seed={seed}"
            )
        candidates.extend([
            ("q1_checkpoint", int(seed), q1_path),
            ("generated_k8_checkpoint", int(seed), generated_path),
        ])
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


def resolve_recorded_path(recorded):
    path = Path(recorded)
    return path if path.is_absolute() else ROOT / path


def load_generated(seed, llm_checkpoint, device):
    q1_path = checkpoint("q1", seed=int(seed))
    generated_path = checkpoint(
        "generated_k8", LEARNING_RATE, seed=int(seed)
    )
    if q1_path is None or generated_path is None:
        raise FileNotFoundError(
            f"missing accepted AutoTimes lineage for seed={seed}"
        )
    model = load_model(
        generated_path,
        model_config(llm_checkpoint, "generated", q1_path),
        device,
    )
    return model, q1_path, generated_path


def normalized_to_batch(normalized, scale, batch_size, channels):
    values = (normalized * scale).reshape(
        normalized.size(0), normalized.size(1) * normalized.size(2)
    )
    return from_channel_batch(values, batch_size, channels)


@torch.inference_mode()
def collect_origin_moments(
    model, dataset, indices, batch_size, width, periods, device,
):
    """Collect [raw error energy, direction energy, alignment] per origin."""
    macro_points = int(width) * int(model.patch_len)
    output = []
    for batch_x, batch_y, _, _ in loader_for_indices(
        dataset, indices, batch_size
    ):
        observed = batch_x.float().to(device)
        truth = batch_y[:, -int(model.pred_len):].float().to(device)
        batch_origins = int(observed.size(0))
        initial_history, _, initial_channels = model._to_channel_batch(
            observed[:, -int(model.seq_len):]
        )
        templates = {
            int(period): build_phase_template(
                model, initial_history.clone(), int(period)
            )
            for period in periods
        }
        moments = torch.zeros(
            batch_origins, len(periods), 3,
            device=device, dtype=torch.float64,
        )
        working = observed
        committed_patches = 0
        for age in range(0, int(model.pred_len), macro_points):
            history, _, channels = model._to_channel_batch(
                working[:, -int(model.seq_len):]
            )
            if channels != initial_channels:
                raise RuntimeError("channel count changed during rollout")
            commits, _, mean, std = model._protected_commits_normalized(
                history, int(width)
            )
            raw_batch = from_channel_batch(
                (commits * std + mean).reshape(
                    history.size(0), macro_points
                ),
                batch_origins,
                channels,
            )
            valid = min(macro_points, int(model.pred_len) - age)
            error = (
                truth[:, age:age + valid] - raw_batch[:, :valid]
            ).double()
            error_energy = error.square().sum(dim=(1, 2))
            for period_index, period in enumerate(periods):
                direction = normalized_tangent_direction(
                    commits,
                    templates[int(period)],
                    committed_patches,
                    int(model.patch_len),
                )
                direction_batch = normalized_to_batch(
                    direction, std, batch_origins, channels
                )[:, :valid].double()
                moments[:, period_index, 0].add_(error_energy)
                moments[:, period_index, 1].add_(
                    direction_batch.square().sum(dim=(1, 2))
                )
                moments[:, period_index, 2].add_(
                    (direction_batch * error).sum(dim=(1, 2))
                )
            working = torch.cat([working, raw_batch], dim=1)[
                :, -int(model.seq_len):
            ]
            committed_patches += int(width)
        denominator = float(int(model.pred_len) * initial_channels)
        output.append((moments / denominator).cpu().numpy())
    return np.concatenate(output, axis=0)


def fit_from_stats(error_energy, direction_energy, alignment):
    direction_energy = max(float(direction_energy), 1e-30)
    alignment = float(alignment)
    gamma = max(alignment / direction_energy, 0.0)
    gain = (
        2.0 * gamma * alignment
        - gamma * gamma * direction_energy
    )
    r2 = gain / max(float(error_energy), 1e-30)
    return gamma, gain, r2


def block_rows(width, seed, moments, periods, blocks=CHRONOLOGY_BLOCKS):
    rows = []
    positions = np.arange(len(moments))
    for block_index, local_positions in enumerate(
        np.array_split(positions, int(blocks)), start=1
    ):
        for period_index, period in enumerate(periods):
            values = np.asarray(
                moments[local_positions, period_index], dtype=np.float64
            ).mean(axis=0)
            gamma, gain, r2 = fit_from_stats(*values)
            rows.append({
                "dataset": DATASET,
                "commit_patches": int(width),
                "seed": int(seed),
                "chronology_block": int(block_index),
                "origins": int(len(local_positions)),
                "period": int(period),
                "history_occurrences_floor": int(672 // int(period)),
                "error_energy": float(values[0]),
                "direction_energy": float(values[1]),
                "alignment": float(values[2]),
                "positive_gamma_oracle": gamma,
                "one_direction_gain_oracle": gain,
                "directional_r2_oracle": r2,
            })
    return rows


def pooled_stats(rows, width, period, blocks, seeds):
    blocks = set(map(int, blocks))
    seeds = set(map(int, seeds))
    local = [
        row for row in rows
        if int(row["commit_patches"]) == int(width)
        and int(row["period"]) == int(period)
        and int(row["chronology_block"]) in blocks
        and int(row["seed"]) in seeds
    ]
    if not local:
        raise ValueError("empty pooled AutoTimes spectrum cell")
    return tuple(float(np.mean([
        float(row[key]) for row in local
    ])) for key in ("error_energy", "direction_energy", "alignment"))


def transported_gain(stats, gamma):
    _, direction_energy, alignment = stats
    return (
        2.0 * float(gamma) * float(alignment)
        - float(gamma) ** 2 * float(direction_energy)
    )


def select_and_summarize(rows, widths, seeds, periods):
    summaries = []
    for width in widths:
        candidates = []
        for period in periods:
            stats = pooled_stats(
                rows, width, period, (1, 2, 3), seeds
            )
            gamma, gain, r2 = fit_from_stats(*stats)
            candidates.append((-r2, int(period), stats, gamma, gain))
        selected = min(candidates)
        period = int(selected[1])
        fit_stats = selected[2]
        gamma = float(selected[3])
        fit_gain = float(selected[4])
        seed_periods = []
        confirmation_gains = []
        confirmation_r2 = []
        for seed in seeds:
            seed_candidates = []
            for candidate_period in periods:
                stats = pooled_stats(
                    rows, width, candidate_period, (1, 2, 3), (seed,)
                )
                _, _, local_r2 = fit_from_stats(*stats)
                seed_candidates.append((-local_r2, int(candidate_period)))
            seed_periods.append(min(seed_candidates)[1])
            heldout = pooled_stats(rows, width, period, (4,), (seed,))
            gain = transported_gain(heldout, gamma)
            confirmation_gains.append(gain)
            confirmation_r2.append(
                gain / max(float(heldout[0]), 1e-30)
            )
        summaries.append({
            "dataset": DATASET,
            "parent": "AutoTimes-GPT2",
            "commit_patches": int(width),
            "selected_period": period,
            "seed_selected_periods": "|".join(map(str, seed_periods)),
            "seed_period_agreement": int(sum(
                value == period for value in seed_periods
            )),
            "fit_positive_gamma": gamma,
            "fit_effective_beta": gamma * DEFAULT_BETA,
            "fit_directional_r2": (
                fit_gain / max(float(fit_stats[0]), 1e-30)
            ),
            "confirmation_r2_mean": float(np.mean(confirmation_r2)),
            "confirmation_r2_min": float(np.min(confirmation_r2)),
            "confirmation_r2_by_seed": "|".join(
                f"{value:.10g}" for value in confirmation_r2
            ),
            "confirmation_gain_mean": float(np.mean(confirmation_gains)),
            "confirmation_gain_min": float(np.min(confirmation_gains)),
            "confirmation_seed_wins": int(sum(
                value > 0.0 for value in confirmation_gains
            )),
            "selection_rule": (
                "max pooled blocks1-3 positive directional R2; "
                "smallest period tie-break"
            ),
            "confirmation_rule": (
                "frozen period and gamma on train chronology block4"
            ),
        })
    return summaries


def preregistered_protocol(args, artifacts, widths, seeds, periods):
    return {
        "campaign": CAMPAIGN,
        "created_at_utc": utc_now(),
        "status": "frozen_before_train_measurement",
        "method": "spectrum-selected explicit tangent",
        "parent": "AutoTimes-GPT2 Generated-K8",
        "dataset": DATASET,
        "evidence_scope": (
            "post-test cross-architecture compatibility; not prospective"
        ),
        "benchmark_test_previously_exposed": True,
        "selection_split": "train_only",
        "validation_opened_for_selection": False,
        "test_opened_for_selection": False,
        "train_origins_per_seed": int(args.train_origins),
        "seeds": list(map(int, seeds)),
        "commit_widths": list(map(int, widths)),
        "candidate_periods": list(map(int, periods)),
        "chronology_blocks": CHRONOLOGY_BLOCKS,
        "selection_blocks": [1, 2, 3],
        "forward_confirmation_block": 4,
        "selection_score": "max(B,0)^2/(A*raw_error_energy)",
        "coefficient": "gamma=max(B/A,0)",
        "base_beta": DEFAULT_BETA,
        "band_edges": list(DEFAULT_BAND_EDGES),
        "band_multipliers": list(DEFAULT_BANDS),
        "protected_prefix": "slot 1 of every macro is hard-preserved",
        "template": "immutable; initial observed history only",
        "rollout": "corrected far commits are written back",
        "source_learning_rate": LEARNING_RATE,
        "source_artifact_manifest": "artifact_manifest.csv",
        "source_artifact_count": len(artifacts),
        "canonical_ledger_writes": False,
        "checkpoint_writes": False,
    }


def run_select(args, widths, seeds, periods, device):
    require_empty_output(args.output_dir)
    artifacts = source_artifacts(args.llm_checkpoint, seeds)
    write_csv_atomic(args.output_dir / "artifact_manifest.csv", artifacts)
    protocol = preregistered_protocol(
        args, artifacts, widths, seeds, periods
    )
    write_json_atomic(
        args.output_dir / "protocol_preregistered.json", protocol
    )
    rows = []
    for seed in seeds:
        model, _, _ = load_generated(
            seed, args.llm_checkpoint, device
        )
        train_dataset, _ = data_provider(
            data_config(args.train_batch_size, seed), "train"
        )
        indices = uniformly_spaced_indices(
            len(train_dataset), args.train_origins
        )
        for width in widths:
            moments = collect_origin_moments(
                model,
                train_dataset,
                indices,
                args.train_batch_size,
                width,
                periods,
                device,
            )
            rows.extend(block_rows(width, seed, moments, periods))
            print({
                "stage": "select",
                "seed": int(seed),
                "K": int(width),
                "origins": len(indices),
            }, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summaries = select_and_summarize(rows, widths, seeds, periods)
    spectrum_path = args.output_dir / "train_period_spectrum.csv"
    summary_path = args.output_dir / "train_selection.csv"
    write_csv_atomic(spectrum_path, rows)
    write_csv_atomic(summary_path, summaries)
    code_hashes = {
        relative_path(path): sha256_file(path) for path in CODE_PATHS
    }
    lock = {
        "campaign": CAMPAIGN,
        "frozen_at_utc": utc_now(),
        "test_opened": False,
        "selection": [{
            "dataset": DATASET,
            "commit_patches": int(row["commit_patches"]),
            "period": int(row["selected_period"]),
            "gamma": float(row["fit_positive_gamma"]),
        } for row in summaries],
        "locked_files": {
            "protocol_preregistered.json": sha256_file(
                args.output_dir / "protocol_preregistered.json"
            ),
            "artifact_manifest.csv": sha256_file(
                args.output_dir / "artifact_manifest.csv"
            ),
            "train_period_spectrum.csv": sha256_file(spectrum_path),
            "train_selection.csv": sha256_file(summary_path),
        },
        "code_hashes": code_hashes,
    }
    write_json_atomic(args.output_dir / "selection_lock.json", lock)
    write_json_atomic(args.output_dir / "run_state.json", {
        "status": "selection_frozen",
        "updated_at_utc": utc_now(),
        "test_opened": False,
        "selection_lock_sha256": sha256_file(
            args.output_dir / "selection_lock.json"
        ),
    })
    for row in summaries:
        print(row, flush=True)
    return summaries


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_selection_lock(output_dir):
    output_dir = Path(output_dir)
    lock_path = output_dir / "selection_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("selection_lock.json is required before test")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors = []
    for name, expected in lock["locked_files"].items():
        path = output_dir / name
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            errors.append(f"locked file mismatch: {name}")
    for recorded, expected in lock["code_hashes"].items():
        path = resolve_recorded_path(recorded)
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            errors.append(f"code mismatch after selection: {recorded}")
    artifacts = read_csv(output_dir / "artifact_manifest.csv")
    for row in artifacts:
        path = resolve_recorded_path(row["path"])
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != row["sha256"]:
            errors.append(f"source artifact mismatch: {row['path']}")
    if errors:
        raise RuntimeError("selection lock verification failed: " + "; ".join(errors))
    return lock


def per_origin_metrics(prediction, truth):
    output = {}
    for horizon in HORIZONS:
        error = (prediction[:, :horizon] - truth[:, :horizon]).double()
        output[horizon] = {
            "mse": error.square().mean(dim=(1, 2)).cpu().numpy(),
            "mae": error.abs().mean(dim=(1, 2)).cpu().numpy(),
        }
    return output


def append_origin_metrics(store, values):
    for horizon in HORIZONS:
        for metric in ("mse", "mae"):
            store[horizon][metric].append(values[horizon][metric])


def new_origin_store():
    return {
        horizon: {"mse": [], "mae": []} for horizon in HORIZONS
    }


def concatenate_origin_store(store):
    return {
        horizon: {
            metric: np.concatenate(store[horizon][metric])
            for metric in ("mse", "mae")
        }
        for horizon in HORIZONS
    }


def source_baseline_lookup():
    rows = read_csv(SOURCE_BASELINE)
    return {
        (int(row["seed"]), int(row["commit_patches"])): row
        for row in rows
        if row["split"] == "test" and row["artifact"] == "atd_k8"
    }


@torch.inference_mode()
def evaluate_cell(
    model, seed, width, period, gamma, args, device,
):
    _, loader = data_provider(
        data_config(args.test_batch_size, seed), "test"
    )
    stores = {
        "generated": new_origin_store(),
        "spectrum_tangent": new_origin_store(),
    }
    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float().to(device)
        truth = batch_y[:, -int(model.pred_len):].float().to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=bool(args.test_amp and device.type == "cuda"),
        ):
            generated = model.forecast_with_commit_patches(
                batch_x, int(width)
            )
            corrected = forecast_with_explicit_tangent(
                model,
                batch_x,
                int(width),
                int(period),
                float(gamma),
            )
        append_origin_metrics(
            stores["generated"], per_origin_metrics(generated.float(), truth)
        )
        append_origin_metrics(
            stores["spectrum_tangent"],
            per_origin_metrics(corrected.float(), truth),
        )
    return {
        name: concatenate_origin_store(store)
        for name, store in stores.items()
    }


def metric_rows(seed, width, period, gamma, values, baseline_reference):
    rows = []
    chronology = []
    origins = len(values["generated"][720]["mse"])
    blocks = np.array_split(np.arange(origins), CHRONOLOGY_BLOCKS)
    for artifact in ("generated", "spectrum_tangent"):
        row = {
            "dataset": DATASET,
            "parent": "AutoTimes-GPT2",
            "seed": int(seed),
            "split": "test",
            "artifact": artifact,
            "commit_patches": int(width),
            "backbone_calls": math.ceil(720 / (int(width) * 96)),
            "selected_period": int(period) if artifact != "generated" else "",
            "selected_gamma": float(gamma) if artifact != "generated" else "",
            "origins": origins,
        }
        for horizon in HORIZONS:
            row[f"h{horizon}_mse"] = float(np.mean(
                values[artifact][horizon]["mse"]
            ))
            row[f"h{horizon}_mae"] = float(np.mean(
                values[artifact][horizon]["mae"]
            ))
        reference = baseline_reference.get((int(seed), int(width)))
        row["source_baseline_h720_mse"] = (
            float(reference["h720_mse"])
            if artifact == "generated" and reference is not None else ""
        )
        row["source_baseline_h720_abs_delta"] = (
            abs(row["h720_mse"] - float(reference["h720_mse"]))
            if artifact == "generated" and reference is not None else ""
        )
        rows.append(row)
        for block_index, positions in enumerate(blocks, start=1):
            for horizon in HORIZONS:
                chronology.append({
                    "dataset": DATASET,
                    "parent": "AutoTimes-GPT2",
                    "seed": int(seed),
                    "split": "test",
                    "artifact": artifact,
                    "commit_patches": int(width),
                    "chronology_block": int(block_index),
                    "origins": len(positions),
                    "horizon": int(horizon),
                    "mse": float(np.mean(
                        values[artifact][horizon]["mse"][positions]
                    )),
                    "mae": float(np.mean(
                        values[artifact][horizon]["mae"][positions]
                    )),
                })
    return rows, chronology


def summarize_test(rows, chronology, widths, seeds):
    summaries = []
    for width in widths:
        for horizon in HORIZONS:
            baseline = [
                row for row in rows
                if int(row["commit_patches"]) == int(width)
                and row["artifact"] == "generated"
            ]
            corrected = [
                row for row in rows
                if int(row["commit_patches"]) == int(width)
                and row["artifact"] == "spectrum_tangent"
            ]
            baseline_by_seed = {int(row["seed"]): row for row in baseline}
            corrected_by_seed = {int(row["seed"]): row for row in corrected}
            deltas = np.asarray([
                float(corrected_by_seed[seed][f"h{horizon}_mse"])
                - float(baseline_by_seed[seed][f"h{horizon}_mse"])
                for seed in seeds
            ])
            base_values = np.asarray([
                float(baseline_by_seed[seed][f"h{horizon}_mse"])
                for seed in seeds
            ])
            chronology_deltas = []
            for seed in seeds:
                for block in range(1, CHRONOLOGY_BLOCKS + 1):
                    lookup = {
                        row["artifact"]: row for row in chronology
                        if int(row["commit_patches"]) == int(width)
                        and int(row["horizon"]) == int(horizon)
                        and int(row["seed"]) == int(seed)
                        and int(row["chronology_block"]) == int(block)
                    }
                    chronology_deltas.append(
                        float(lookup["spectrum_tangent"]["mse"])
                        - float(lookup["generated"]["mse"])
                    )
            chronology_deltas = np.asarray(chronology_deltas)
            summaries.append({
                "dataset": DATASET,
                "parent": "AutoTimes-GPT2",
                "commit_patches": int(width),
                "horizon": int(horizon),
                "n_seeds": len(seeds),
                "generated_mse_mean": float(np.mean(base_values)),
                "spectrum_tangent_mse_mean": float(np.mean(
                    base_values + deltas
                )),
                "delta_mse_mean": float(np.mean(deltas)),
                "relative_delta_mse": float(
                    np.mean(deltas) / np.mean(base_values)
                ),
                "seed_wins": int(np.sum(deltas < 0.0)),
                "chronology_block_wins": int(np.sum(chronology_deltas < 0.0)),
                "chronology_blocks": int(len(chronology_deltas)),
                "worst_seed_delta": float(np.max(deltas)),
                "worst_chronology_delta": float(np.max(chronology_deltas)),
            })
    return summaries


def run_test(args, widths, seeds, device):
    if not args.open_test:
        raise RuntimeError(
            "test stage requires explicit --open-test after selection freeze"
        )
    lock = verify_selection_lock(args.output_dir)
    if (args.output_dir / "test_metrics.csv").exists():
        raise RuntimeError("test output already exists and will not be overwritten")
    selection = {
        int(row["commit_patches"]): (
            int(row["period"]), float(row["gamma"])
        )
        for row in lock["selection"]
    }
    if set(selection) != set(map(int, widths)):
        raise RuntimeError(
            "requested test widths do not equal the frozen selection widths"
        )
    write_json_atomic(args.output_dir / "run_state.json", {
        "status": "test_opened",
        "updated_at_utc": utc_now(),
        "test_opened": True,
        "selection_lock_sha256": sha256_file(
            args.output_dir / "selection_lock.json"
        ),
    })
    baseline_reference = source_baseline_lookup()
    rows = []
    chronology = []
    for seed in seeds:
        model, _, _ = load_generated(seed, args.llm_checkpoint, device)
        for width in widths:
            period, gamma = selection[int(width)]
            values = evaluate_cell(
                model, seed, width, period, gamma, args, device
            )
            cell_rows, cell_chronology = metric_rows(
                seed, width, period, gamma, values, baseline_reference
            )
            rows.extend(cell_rows)
            chronology.extend(cell_chronology)
            print({
                "stage": "test",
                "seed": int(seed),
                "K": int(width),
                "period": int(period),
                "gamma": float(gamma),
                "delta_h720": (
                    cell_rows[1]["h720_mse"]
                    - cell_rows[0]["h720_mse"]
                ),
            }, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summaries = summarize_test(rows, chronology, widths, seeds)
    write_csv_atomic(args.output_dir / "test_metrics.csv", rows)
    write_csv_atomic(args.output_dir / "test_chronology.csv", chronology)
    write_csv_atomic(args.output_dir / "test_summary.csv", summaries)
    baseline_max_delta = max(
        float(row["source_baseline_h720_abs_delta"])
        for row in rows if row["artifact"] == "generated"
    )
    audit = {
        "campaign": CAMPAIGN,
        "completed_at_utc": utc_now(),
        "selection_lock_verified": True,
        "test_was_previously_exposed": True,
        "evidence_scope": "post-test cross-architecture compatibility",
        "canonical_ledger_writes": False,
        "checkpoint_writes": False,
        "baseline_source_h720_max_abs_delta": baseline_max_delta,
        "baseline_source_match_tolerance": 5e-6,
        "baseline_source_match": baseline_max_delta <= 5e-6,
        "result_hashes": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "test_metrics.csv", "test_chronology.csv", "test_summary.csv"
            )
        },
    }
    write_json_atomic(args.output_dir / "test_audit.json", audit)
    write_json_atomic(args.output_dir / "run_state.json", {
        "status": "test_completed",
        "updated_at_utc": utc_now(),
        "test_opened": True,
        "selection_lock_sha256": sha256_file(
            args.output_dir / "selection_lock.json"
        ),
        "test_audit_sha256": sha256_file(
            args.output_dir / "test_audit.json"
        ),
    })
    for row in summaries:
        if int(row["horizon"]) == 720:
            print(row, flush=True)
    if not audit["baseline_source_match"]:
        raise RuntimeError(
            "fresh Generated baseline does not reproduce the source ledger"
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "test", "audit"), required=True)
    parser.add_argument("--llm-checkpoint", type=Path, default=DEFAULT_LLM_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    parser.add_argument("--periods", default=",".join(map(str, PERIODS)))
    parser.add_argument("--train-origins", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-test", action="store_true")
    args = parser.parse_args()

    seeds = parse_values(args.seeds, int)
    widths = parse_values(args.widths, int)
    periods = parse_values(args.periods, int)
    if seeds != SEEDS:
        raise ValueError(f"frozen protocol requires seeds={SEEDS}")
    if widths != WIDTHS:
        raise ValueError(f"frozen protocol requires widths={WIDTHS}")
    if periods != PERIODS:
        raise ValueError("frozen protocol requires the P12 period grid [12,216]")
    if args.train_origins != 512:
        raise ValueError("frozen protocol requires 512 train origins per seed")
    if min(args.train_batch_size, args.test_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.stage == "select":
        run_select(args, widths, seeds, periods, device)
    elif args.stage == "test":
        run_test(args, widths, seeds, device)
    else:
        lock = verify_selection_lock(args.output_dir)
        print({
            "selection_lock_verified": True,
            "selection": lock["selection"],
        }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
