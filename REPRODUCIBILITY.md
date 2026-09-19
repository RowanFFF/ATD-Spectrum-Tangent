# Reproducibility scope

This is a core-method release, not the full historical research workspace.
It provides local parent/ATD/direct-control training and the local Spectrum
Tangent selection/evaluation path. Complete ablations, diagnostic analyses,
timing, plotting, and paper-table generation are intentionally outside scope.

## Local protocol

- Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic.
- Seeds: 2021--2023; W=672; trained execution widths K=4 and K=8.
- Parent checkpoints: next-patch validation. ATD checkpoints: closed-loop validation.
- Global standardization: training split only. Instance normalization:
  per-channel rolling-window mean and std with variance epsilon 0.001.
- Atomic input: 12-point atoms, 16-dimensional lift, chronological concatenation;
  parent P, width D, and depth are dataset-specific.
- Shared weights across channels; no cross-variable mixing in this grid.
- Local exit input: last observed token after the final Transformer block's
  post-normalization and before output projection.

The JSON grid documents the configuration; runtime architecture values are
checked against it by a unit test. Its selected periods are paper reference
values, not a replacement for train-only selection on newly trained checkpoints.
The legacy campaign contains a superseded fixed-gamma route that is not exposed
by its public CLI. Use `local_spectrum_tangent.py` for the released correction.

## Tangent selection

For each dataset and K, sample 512 evenly spaced, chronological training
origins per seed. Divide into four equal blocks; pool seeds and blocks 1--3
to select a period from 12, 24, ..., 216. Block 4 only confirms the frozen
choice and can report negative gain; it never selects, rejects, or retunes it.

The normalized direction is
`d = beta * b(endpoint) * (T_period - G)`, with beta=0.0625 and each call's
first slot exactly zero. The endpoint ramp is (.25, .5, 1, 2, 3), changing
bands at (96, 192, 336, 672), with equality entering the higher band.
The template is built once from initial history, excluding the oldest
occurrence of each absolute-time phase. Current-call normalization maps
directions back into data coordinates.

On the uncorrected ATD trajectory, collect V=mean(error^2), A=mean(d^2),
B=mean(d*error), after mapping d into data coordinates. Fit
`gamma=max(B/A,0)` and select the highest `(2*gamma*B-gamma^2*A)/V`;
ties choose the smaller period. The paper's effective alpha is `gamma*beta`,
not gamma itself. The script records both.

Inference uses corrected writebacks, so later histories can differ from ATD.
Selection is FP32; test uses CUDA FP16 autocast or FP32 on CPU. CPU/different
GPU environments need not reproduce historical floating-point values exactly.
Test metrics use all origins and original channel/point weighting, in globally
standardized benchmark coordinates (not the parent's instance-normalized frame).

## Output and integrity

`select` writes training moments and a selection lock with data, checkpoint,
code, and configuration hashes. `test --open-test` verifies those inputs
before reading the test split and uses the scope stored in the lock.
It writes per-seed MSE/MAE at H=96/192/336/720 and a mean/sample-std summary.
Single-seed standard deviations are left empty, not presented as zero uncertainty.

Use a new output directory for another experiment. Existing outputs are not
overwritten. The lock is an integrity/lineage record, not a claim that code
prevents all possible manual access to benchmark labels.
A single-seed quickstart or reduced origin count is not the paper protocol.

Generated directories are ignored: `dataset/`, `checkpoints/`, `results/`,
`experiment_logs/`, `analysis_outputs/`, `external_backbones/`,
`foundation_models/`. The release scanner examines Git-publishable files;
this does not replace a pre-publication Git-history review.

## Method-to-code map

| Paper component | Implementation |
| --- | --- |
| Atomic Patch-AR parent | `TransformerAblationAR.py`, `layers/PatchARAblation.py` |
| ATD / matched Frozen Direct | `MatchedFrozenMultiExitAR.py`, target weight 0 / 1 |
| Joint Direct | `DirectMTPAR.py` |
| Additional frozen-placeholder control | `FrozenDirectMTPAR.py` |
| Local Spectrum Tangent | `PatchARSpectrumTangent.py`, `scripts/local_spectrum_tangent.py` |
| AutoTimes / Timer / TimesFM | Corresponding `*OperatorAR.py`; see FOUNDATION_MODELS.md |

Named-parent readouts differ: AutoTimes uses GPT2's final 768-dimensional
token after ln_f; Timer uses the final 1024-dimensional token after stack
LayerNorm; TimesFM uses the final 1280-dimensional block output before the
output projections, with no extra stack-final normalization. TimesFM's
32-point input tokens and 128-point output blocks are different units.
All are call-local readouts, not states held fixed through the whole horizon.

## Verification boundary

The release has unit tests and a synthetic select-to-test integration test.
These check implementation contracts, not benchmark accuracy. No full
seven-dataset retraining or fresh-environment dependency installation was
performed as part of this release cleanup. No checkpoint or historical
result bundle is redistributed. Larger-batch throughput and deployment
break-even points are not inferred from batch-one paper timings.

A subsequent real ETTh1 validation replayed three-seed ATD-4/8 checkpoints
and ran the seed-2021 local training pipeline from scratch. See
[LOCAL_VALIDATION.md](LOCAL_VALIDATION.md) for numerical agreement, fresh-run
differences, exact commands, hardware, and untested boundaries.
