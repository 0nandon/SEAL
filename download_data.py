#!/usr/bin/env python3
"""Download sharded dataset archives from Hugging Face and restore the original tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


os.environ["HF_ENDPOINT"] = "https://huggingface.co"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download .tar/.tar.zst dataset shards from Hugging Face and extract them into a local folder."
    )
    parser.add_argument(
        "--repo-id",
        default="onandon/SEAL",
        help="Hugging Face dataset repo id. Default: onandon/SEAL.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("dataset_shards"),
        help="Where shard archives are downloaded. Default: ./dataset_shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Where archives are extracted. Default: ./dataset.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Not required for a public repo.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep downloaded shard archives after extraction.",
    )
    return parser.parse_args()


def find_manifest(download_dir: Path) -> Path | None:
    manifests = sorted(download_dir.rglob("shard-manifest.json"))
    if not manifests:
        return None
    return manifests[0]


def load_archive_paths(download_dir: Path) -> list[Path]:
    archives = sorted(
        path for path in download_dir.rglob("*") if path.is_file() and path.suffix in {".tar", ".zst"}
    )
    tar_archives: list[Path] = []
    for path in archives:
        if path.name.endswith(".tar") or path.name.endswith(".tar.zst"):
            tar_archives.append(path)
    return sorted(tar_archives, key=lambda path: path.name)


def load_archives_from_manifest(download_dir: Path, manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    archives: list[Path] = []
    for row in payload.get("shards", []):
        archive_name = row.get("archive")
        if not archive_name:
            continue
        matches = list(download_dir.rglob(archive_name))
        if not matches:
            raise FileNotFoundError(f"archive listed in manifest was not downloaded: {archive_name}")
        archives.append(matches[0])
    return archives


def download_shards(repo_id: str, download_dir: Path, token: str | None) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(download_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.tar",
            "*.tar.zst",
            "**/*.tar",
            "**/*.tar.zst",
            "shard-manifest.json",
            "**/shard-manifest.json",
        ],
        token=token,
    )
    return download_dir


def extract_archive(archive_path: Path, output_dir: Path, index: int, total: int) -> None:
    if archive_path.name.endswith(".tar.zst"):
        cmd = ["tar", "--zstd", "-xf", str(archive_path), "-C", str(output_dir)]
    elif archive_path.name.endswith(".tar"):
        cmd = ["tar", "-xf", str(archive_path), "-C", str(output_dir)]
    else:
        raise ValueError(f"unsupported archive type: {archive_path}")

    print(f"[{index}/{total}] extracting {archive_path.name}")
    subprocess.run(cmd, check=True)


def restore_dataset(
    repo_id: str,
    download_dir: Path,
    output_dir: Path,
    token: str | None,
    keep_shards: bool,
) -> None:
    if shutil.which("tar") is None:
        raise SystemExit("tar is required but was not found in PATH")

    download_dir = download_shards(repo_id=repo_id, download_dir=download_dir, token=token)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = find_manifest(download_dir)
    if manifest_path is not None:
        archives = load_archives_from_manifest(download_dir, manifest_path)
    else:
        archives = load_archive_paths(download_dir)

    if not archives:
        raise SystemExit(f"no .tar or .tar.zst shard archives were found under {download_dir}")

    for index, archive_path in enumerate(archives, start=1):
        extract_archive(archive_path=archive_path, output_dir=output_dir, index=index, total=len(archives))

    if not keep_shards:
        print(f"removing downloaded shards under {download_dir}")
        shutil.rmtree(download_dir)

    print(f"dataset restored under {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    restore_dataset(
        repo_id=args.repo_id,
        download_dir=args.download_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        token=args.token,
        keep_shards=args.keep_shards,
    )
