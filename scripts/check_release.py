#!/usr/bin/env python3
"""Fail on common accidental disclosures in the public release tree."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".bin", ".ckpt", ".npy", ".npz", ".pth", ".pt", ".safetensors"
}
REQUIRED = {
    "LICENSE",
    "README.md",
    "UPSTREAM.md",
    "models/AutoTimesSpectrumTangent.py",
    "models/DistilledMultiCommitAR.py",
    "models/MatchedFrozenMultiExitAR.py",
}
PRIVATE_PATTERNS = {
    "absolute user path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
}
TEXT_SUFFIXES = {
    "", ".cff", ".template", ".csv", ".json", ".md", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"
}


def release_files():
    # Inspect the publishable tree, not downloaded data or an installed venv.
    # Tracked files are always included, even if a later ignore rule matches.
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        paths = [ROOT / name for name in sorted(set(result.stdout.split("\0"))) if name]
    else:
        excluded = {".git", ".venv", "venv", "__pycache__", "checkpoints", "results",
                    "analysis_outputs", "experiment_logs", "external_backbones",
                    "foundation_models", "dataset", "data"}
        paths = []
        for directory, folders, files in os.walk(ROOT):
            folders[:] = [name for name in folders if name not in excluded]
            paths.extend(Path(directory) / name for name in files)
    yield from (path for path in paths if path.is_file() and path.suffix != ".pyc")


def main():
    failures = []
    observed = {
        str(path.relative_to(ROOT)): path for path in release_files()
    }
    for relative in sorted(REQUIRED - set(observed)):
        failures.append(f"missing required file: {relative}")
    for relative, path in sorted(observed.items()):
        if path.is_symlink():
            failures.append(f"symlink must not be published: {relative}")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"binary artifact is not allowed: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected binary-looking file: {relative}")
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label} in {relative}")
    if failures:
        raise SystemExit("release check failed:\n- " + "\n- ".join(failures))
    print(f"release check passed: {len(observed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
