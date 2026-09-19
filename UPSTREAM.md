# Upstream provenance

This repository is a focused derivative of THUML's
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib).

- Upstream repository: `https://github.com/thuml/Time-Series-Library.git`
- Pinned base commit: `4e938a1767106324dd753b2a44832bf870a0252e`
- Upstream license: MIT
- Local release branch: `paper-release`

The upstream `LICENSE` file is preserved unchanged. Paper-specific model,
training, metric, and orchestration code is distributed under that same license.

## What is retained

- ETT and custom long-horizon data loaders;
- the long-term forecasting experiment runtime;
- Patch-AR, ATD, matched direct controls, and external-parent adapters;
- the explicit Spectrum Tangent implementation and frozen selection protocol;
- only the tests and launchers needed by those components.

## What is not redistributed

- TSLib models and task families unrelated to the paper;
- downloaded data;
- checkpoints or foundation-model weights;
- vendored copies of Transformers, Timer, TimesFM, or other third-party code;
- private logs, exploratory analyses, machine-specific paths, or manuscript
  archives.

External code and checkpoints remain subject to their own licenses and should
be downloaded from their official sources by the user.

