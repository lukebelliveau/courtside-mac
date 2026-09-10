#!/usr/bin/env python3
"""Download or verify the public, checksum-pinned basketball demo assets.

Uses Python's standard library and unauthenticated HTTPS. Model execution is
separate. Existing files with unexpected contents are never replaced.

  python download_assets.py
  python download_assets.py --verify-only
  python download_assets.py --models-only --assets /path/to/assets

The default destination is this workspace's work/assets directory. A failed
download leaves its .download.partial file for inspection; no files are deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path(__file__).with_name("assets.json")


def fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def verify(path: Path, item: dict) -> None:
    size, digest = fingerprint(path)
    if size != item["bytes"] or digest != item["sha256"]:
        raise ValueError(
            f"Checksum mismatch for {path}: got {size} bytes / {digest}; "
            f"expected {item['bytes']} bytes / {item['sha256']}. "
            "The file was preserved. Move it with trash before retrying."
        )


def obtain(item: dict, directory: Path, verify_only: bool) -> str:
    name = item["filename"]
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Manifest filenames must be plain basenames")
    url = urllib.parse.urlsplit(item["url"])
    if url.scheme != "https" or url.username or url.password:
        raise ValueError(f"Expected a public HTTPS URL for {name}")
    target = directory / name
    if target.is_file():
        verify(target, item)
        return f"Verified {name}"
    if verify_only:
        raise FileNotFoundError(f"Missing asset: {target}")

    directory.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download.partial")
    if temporary.exists():
        # Resume only a fully downloaded, checksum-valid file. Partial files
        # remain untouched so a failure never silently discards evidence.
        verify(temporary, item)
    else:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "courtside-mac/1.0"})
        received = 0
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("xb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                received += len(block)
                if received > item["bytes"]:
                    raise ValueError(f"Download exceeded the expected size for {name}; partial file preserved")
                handle.write(block)
        verify(temporary, item)
    if target.exists():
        raise FileExistsError(f"Asset appeared during download; preserving both files: {target}")
    temporary.rename(target)
    return f"Downloaded and verified {name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--assets", type=Path, default=ROOT / "work" / "assets")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported asset manifest version")
    items = manifest["models"] + ([] if args.models_only else manifest["clips"])
    errors = []
    for item in items:
        try:
            print(obtain(item, args.assets, args.verify_only), flush=True)
        except (OSError, ValueError, urllib.error.URLError) as error:
            errors.append(f"{item.get('filename', '?')}: {error}")
            print(errors[-1], file=sys.stderr, flush=True)
    if errors:
        print(f"{len(errors)} asset(s) require attention; all existing files were preserved", file=sys.stderr)
        return 1
    print(f"All {len(items)} assets match the pinned checksums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
