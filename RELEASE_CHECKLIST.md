# Public-release checklist

This directory is prepared as a core-method code release. No repository has
been created or pushed by this cleanup, and no paper submission was made.

## Local checks (2026-09-19)

- Unit tests, synthetic CSV/checkpoint integration, and command previews pass.
- The local Tangent path uses the same seven-dataset architecture/normalization
  configuration as the research loader. A direction-formula comparison against
  the research implementation differed by at most 5.96e-8 in FP32 random probes
  (arithmetic ordering), not a benchmark accuracy comparison.
- No formal training or real benchmark test split was run in this cleanup.
- Current checks reuse an existing Python 3.13 environment. Fresh installation
  from requirements and real foundation-model integration remain unchecked.
- The obsolete export-only script depending on unreleased historical ledgers
  was removed from this release tree. Its research-repository copy is retained.

## Before publishing

Follow-up: real ETTh1 checkpoint replay and single-seed training/evaluation
were completed after the initial cleanup above; see
[LOCAL_VALIDATION.md](LOCAL_VALIDATION.md). This does not complete the fresh
installation or real foundation-model checks below.

- [ ] Independently install the dependencies and run `make audit`.
- [ ] Review the pending changes and all files that will enter the commit;
  inspect Git history separately. The scanner does not inspect commit history.
- [ ] Fill author names, final repository URL, and the current paper title in
  `CITATION.cff.template`; rename to `CITATION.cff` once metadata is ready.
- [ ] Create your own GitHub repository and configure a separate `origin`.
  Verify `git remote -v`; the existing `upstream` points to THUML/TSLib.
- [ ] Commit the prepared working tree before publishing. The upstream base
  commit alone does not contain this release implementation.
- [ ] Tag a reviewed version (for example `v0.1-arxiv`) and put its code link
  in the arXiv manuscript. Keep the anonymous submission manuscript separate.
- [ ] After the arXiv identifier is assigned, add it to the README and citation.

Missing author/repository/arXiv fields are intentional placeholders, not
invented metadata. The release does not include all paper ablations/analyses.
