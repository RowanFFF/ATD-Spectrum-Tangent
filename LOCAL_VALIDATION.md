# ETTh1 local validation — 2026-09-19

The local core workflow was exercised on real ETTh1 data, in two isolated
copies of this release. Original research datasets, checkpoints, and result
records were not modified. This is not a full seven-dataset or foundation-model
validation, and no fresh dependency installation was performed.

Environment: NVIDIA RTX 5060 Ti 16 GB, Python 3.13.13, PyTorch 2.13.0+cu130.
Selection used FP32; test inference used CUDA FP16 autocast. CPU thread counts
were limited to two. The existing Python environment supplied dependencies,
but the executed model/runner code came from the isolated release copies.

ETTh1 CSV SHA256:
`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`.
Both copies matched this source hash. Each full H720 test evaluation contained
2,161 origins; shorter horizons were prefixes of the same forecast origins.

## 1. Historical checkpoint replay: numerical agreement

Copied the paper's three parent checkpoints and six ATD checkpoints
(seeds 2021/2022/2023, K=4/8), then ran the release's selection and test CLI:

```bash
python scripts/local_spectrum_tangent.py select --datasets ETTh1 \
  --seeds 2021 2022 2023 --widths 4 8 --batch-size 8 \
  --output-dir analysis_outputs/replay
python scripts/local_spectrum_tangent.py test --open-test \
  --output-dir analysis_outputs/replay
```

Reference: the paper's frozen `paper_clean_a16_spectrum_tangent_v1_20260815`
selection manifest and full-origin test metrics, not rounded PDF entries.

- Both widths selected period 24 again.
- K4 alpha: 0.481454235492; reference 0.481454231650.
- K8 alpha: 0.399659800939; reference 0.399659804603.
- Across 48 dataset/seed/width/method/horizon cells, maximum absolute MSE
  difference was 9.82963e-7; maximum absolute MAE difference was 7.77761e-7.
- All 48 cells matched to three decimal places for both metrics.
- The maximum first-patch difference between ATD and Tangent was exactly zero.

H720, means over the three seeds:

| Method | Release MSE | Paper-source MSE | Release MAE | Paper-source MAE |
| --- | ---: | ---: | ---: | ---: |
| ATD-4 | 0.405389 | 0.405389 | 0.439831 | 0.439831 |
| ATD-4 + Tangent | 0.396956 | 0.396957 | 0.432891 | 0.432891 |
| ATD-8 | 0.405204 | 0.405204 | 0.439411 | 0.439411 |
| ATD-8 + Tangent | 0.397986 | 0.397987 | 0.433503 | 0.433503 |

## 2. Training from scratch: runnable, not bitwise reproduction

In another copy without historical checkpoints, ran these stages with the
default ten-epoch limit, patience three, and one worker:

```bash
python scripts/paper_clean_a16_campaign.py parent --datasets ETTh1 --seeds 2021
python scripts/paper_clean_a16_campaign.py decoders --datasets ETTh1 --seeds 2021 --widths 4
python scripts/paper_clean_a16_campaign.py test --datasets ETTh1 --seeds 2021 --widths 4
python scripts/paper_clean_a16_campaign.py summarize --datasets ETTh1 --seeds 2021 --widths 4 --require-complete
python scripts/local_spectrum_tangent.py select --datasets ETTh1 --seeds 2021 --widths 4 --output-dir analysis_outputs/fresh_tangent
python scripts/local_spectrum_tangent.py test --open-test --output-dir analysis_outputs/fresh_tangent
```

All five neural training jobs, five test jobs, the complete summary, and
Tangent selection/test finished successfully. The following are single-seed
H720 values, not the paper's three-seed mean:

| Method | Fresh MSE | Historical seed-2021 MSE | Difference |
| --- | ---: | ---: | ---: |
| Parent | 0.407384 | 0.407124 | +0.000260 |
| Joint Direct-4 | 0.471763 | 0.474846 | -0.003083 |
| Frozen-placeholder-4 | 0.543701 | 0.542946 | +0.000755 |
| Frozen Direct-4 | 0.405265 | 0.406317 | -0.001052 |
| ATD-4 | 0.402223 | 0.402974 | -0.000751 |

The newly trained parent's best validation epoch was 10, versus 8 in the
historical record. These results establish a working training path, not
bitwise reproducibility across runs/environments; the cause of every training
difference was not isolated by this validation.

Fresh Tangent selection used only seed 2021, unlike the paper's pooled
three-seed selection. It chose period 24 and alpha=0.501226756466.
H720 MSE changed from 0.402223 to 0.397012 (1.2955% improvement), and MAE
from 0.437778 to 0.432666. The initial protected patch difference was zero.

## Evidence and limits

Local validation workspace basename: `atd-local-validation-CVRfGp`.
`checkpoint_replay/` contains copied historical checkpoints, the selection
lock, training moments, full-origin metrics, and summary. `fresh_training/`
contains new checkpoints, training/test logs, campaign metrics, and the new
Tangent selection/test outputs. These large/local artifacts are not published
with the code repository.

This check does not establish fresh-install portability, all-data-set
retraining, named-foundation-model runtime compatibility, or paper timing
reproduction on this different GPU. No new benchmark result replaces a
frozen paper result.
