#!/usr/bin/env python3
"""Download checkpoint files from the 'cache' folder on Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

os_env = __import__("os").environ
os_env["HF_ENDPOINT"] = "https://huggingface.co"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download checkpoint files from the 'cache' folder of a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default="onandon/SEAL",
        help="Hugging Face dataset repo id. Default: onandon/SEAL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Root directory where the 'cache' folder will be created. Default: current directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Not required for a public repo.",
    )
    return parser.parse_args()


def download_checkpoints(repo_id: str, output_dir: Path, token: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["cache/**"],
        token=token,
    )
    # Remove .cache directory created by huggingface_hub
    hf_cache = output_dir / ".cache"
    if hf_cache.is_dir():
        shutil.rmtree(hf_cache)

    print(f"checkpoints downloaded to {output_dir / 'cache'}")


if __name__ == "__main__":
    args = parse_args()
    download_checkpoints(
        repo_id=args.repo_id,
        output_dir=args.output_dir.resolve(),
        token=args.token,
    )
