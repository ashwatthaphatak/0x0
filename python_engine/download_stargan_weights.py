#!/usr/bin/env python3
"""
Download or import StarGAN pretrained weights for the desktop app.

Default output:
  ~/.deepfake-defense-models/stargan_celeba_128/models/200000-G.ckpt

Usage examples:
  python3 python_engine/download_stargan_weights.py
  python3 python_engine/download_stargan_weights.py --from-file /path/to/celeba-128x128-5attrs.zip
  python3 python_engine/download_stargan_weights.py --from-file /path/to/200000-G.ckpt
  python3 python_engine/download_stargan_weights.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1"
CHECKPOINT_REL_PATH = Path("stargan_celeba_128/models/200000-G.ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/import StarGAN pretrained weights.")
    parser.add_argument(
        "--from-url",
        default=DEFAULT_URL,
        help=f"Remote URL to download (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--from-file",
        help="Local .zip or .ckpt file to import instead of downloading",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path.home() / ".deepfake-defense-models"),
        help="Base output directory for model cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoint if present",
    )
    return parser.parse_args()


def ensure_from_zip(zip_path: Path, output_root: Path, target_ckpt: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_root)

    if target_ckpt.exists():
        return target_ckpt

    candidates = list(output_root.rglob(target_ckpt.name))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {target_ckpt.name} after extracting {zip_path}"
        )

    target_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], target_ckpt)
    return target_ckpt


def download_to_temp(url: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix="stargan-", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)

    req = Request(url, headers={"User-Agent": "deepfake-defense/1.0"})
    try:
        with urlopen(req, timeout=180) as response, open(tmp_path, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (HTTPError, URLError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download StarGAN weights: {exc}") from exc

    return tmp_path


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    target_ckpt = output_root / CHECKPOINT_REL_PATH

    if target_ckpt.exists() and not args.force:
        print(f"Checkpoint already exists: {target_ckpt}")
        print("Use --force to re-import/re-download.")
        return 0

    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        source = Path(args.from_file).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Input file does not exist: {source}")

        print(f"Importing weights from local file: {source}")
        if source.suffix.lower() == ".zip":
            resolved = ensure_from_zip(source, output_root, target_ckpt)
        else:
            shutil.copy2(source, target_ckpt)
            resolved = target_ckpt
    else:
        print(f"Downloading StarGAN weights from: {args.from_url}")
        tmp_zip = download_to_temp(args.from_url)
        try:
            resolved = ensure_from_zip(tmp_zip, output_root, target_ckpt)
        finally:
            tmp_zip.unlink(missing_ok=True)

    print(f"Ready: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
