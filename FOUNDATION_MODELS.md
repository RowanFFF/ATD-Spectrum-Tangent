# Optional foundation parents

These are optional extensions, not prerequisites for the local Patch-AR
quickstart. Checkpoints and external source code are not redistributed.
Full fresh-environment compatibility with the real downloaded models has not
been revalidated in this release pass; adapter tests use small test backbones.

## Sources and required local files

| Parent | Official source | Local requirements |
| --- | --- | --- |
| AutoTimes-GPT2 | [GPT2](https://huggingface.co/openai-community/gpt2), [AutoTimes](https://github.com/thuml/AutoTimes) | GPT2 config.json + model.safetensors; trained local AutoTimes parent and ATD checkpoints |
| Timer-base-84m | [Timer checkpoint](https://huggingface.co/thuml/timer-base-84m) | Entire model snapshot, including configuration_timer.py and modeling_timer.py |
| TimesFM-2.5-200M | [Weights](https://huggingface.co/google/timesfm-2.5-200m-pytorch), [source](https://github.com/google-research/timesfm) | Source checkout plus config.json + model.safetensors |

Install adapter dependencies:

```bash
python -m pip install -r requirements-foundation.txt
```

Download model snapshots using the official model pages or Hugging Face tools.
For reproducible experiments, choose a fixed model revision and record it;
do not assume that a moving main branch is the historical paper snapshot.
The campaigns hash their actual input files. They do not certify that an
arbitrary newly downloaded snapshot matches the paper checkpoint.

Timer loads local custom model code with `trust_remote_code=True`; review
that official code before use. Its model card recommends an older Transformers
version than this repository's adapter environment, so do not silently replace
one setup with the other if reproducing a named-parent result.

For TimesFM, an old `timesfm==1.3.0` wheel is not the 2.5 implementation.
The inspected local source revision is
`3dae50b20d7a724981e8ea36cda75578f80dd2dc`. To use that source:

```bash
git clone https://github.com/google-research/timesfm.git external_backbones/timesfm
git -C external_backbones/timesfm checkout 3dae50b20d7a724981e8ea36cda75578f80dd2dc
python -m pip install -e 'external_backbones/timesfm[torch]'
```

Use compatible weights from the official TimesFM 2.5 PyTorch model repository.
The adapter passes the source path explicitly and uses the native cached
median-writeback path, not the high-level API's optional forecast heuristics.

## Timer / TimesFM campaign commands

Run from the repository root after placing the requested snapshots.
These commands train exits; they are not smoke tests.

```bash
python scripts/timer_named_transfer.py all-pretest \
  --timer-checkpoint foundation_models/timer-base-84m
python scripts/timer_named_transfer.py baseline-test --open-test \
  --timer-checkpoint foundation_models/timer-base-84m
python scripts/timer_named_transfer.py tangent-test --open-test \
  --timer-checkpoint foundation_models/timer-base-84m

python scripts/timesfm_named_transfer.py all-pretest \
  --timesfm-source external_backbones/timesfm/src \
  --timesfm-checkpoint foundation_models/timesfm-2.5-200m-pytorch
python scripts/timesfm_named_transfer.py baseline-test --open-test \
  --timesfm-source external_backbones/timesfm/src \
  --timesfm-checkpoint foundation_models/timesfm-2.5-200m-pytorch
python scripts/timesfm_named_transfer.py tangent-test --open-test \
  --timesfm-source external_backbones/timesfm/src \
  --timesfm-checkpoint foundation_models/timesfm-2.5-200m-pytorch
```

Default datasets are ETTh1, ETTh2, ETTm1, Weather. Use `--help` to inspect
stages; keep freeze/cache/train/validation/select before any test stage.

## AutoTimes scope

The model adapter is included, but the retained `autotimes_atd_ds1_*`
scripts and `autotimes_spectrum_tangent_compatibility.py` cover the historical
ETTh1 path only. They expect three-seed trained parent/ATD checkpoints and
`analysis_outputs/autotimes_atd_ds1_three_seed.csv`.
Those historical artifacts and their full aggregation pipeline are not
included. Merely downloading GPT2 is insufficient to run the retained
compatibility audit. The four-dataset AutoTimes campaign is not claimed as
an end-to-end reproduction entry point in this release.
