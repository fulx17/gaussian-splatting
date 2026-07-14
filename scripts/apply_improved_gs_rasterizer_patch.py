"""Apply the tracked Improved-GS rasterizer patch to the Git submodule.

The CUDA rasterizer is an upstream Git submodule, so edits inside it are not
stored by commits in this parent repository.  This helper makes a fresh
``git clone --recursive`` reproducible by applying the parent-owned patch before
the extension is built.  It is intentionally idempotent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RASTERIZER = ROOT / "submodules" / "diff-gaussian-rasterization"
PATCH = ROOT / "patches" / "improved-gs-rasterizer.patch"


def _git_apply(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *arguments, str(PATCH)],
        cwd=RASTERIZER,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply or verify the Improved-GS CUDA rasterizer patch."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="fail unless the patch is already applied; do not change files",
    )
    args = parser.parse_args()

    if not PATCH.is_file():
        raise FileNotFoundError("Rasterizer patch not found: {}".format(PATCH))
    if not RASTERIZER.is_dir():
        raise FileNotFoundError(
            "Rasterizer submodule not found. Run `git submodule update --init "
            "--recursive` first."
        )

    already_applied = _git_apply("--reverse", "--check")
    if already_applied.returncode == 0:
        print("Improved-GS rasterizer patch is already applied.")
        return 0

    if args.check_only:
        print(
            "Improved-GS rasterizer patch is not applied. Run this script "
            "without --check-only before building the extension.",
            file=sys.stderr,
        )
        return 1

    can_apply = _git_apply("--check")
    if can_apply.returncode != 0:
        details = (can_apply.stderr or can_apply.stdout).strip()
        print(
            "The rasterizer submodule does not match the expected base commit; "
            "refusing a partial patch.\n{}".format(details),
            file=sys.stderr,
        )
        return 2

    applied = _git_apply()
    if applied.returncode != 0:
        details = (applied.stderr or applied.stdout).strip()
        print("Failed to apply rasterizer patch:\n{}".format(details), file=sys.stderr)
        return applied.returncode

    print("Applied Improved-GS AbsGrad/EAS rasterizer patch.")
    print("Rebuild submodules/diff-gaussian-rasterization before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
