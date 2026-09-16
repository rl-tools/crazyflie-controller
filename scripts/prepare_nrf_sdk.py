#!/usr/bin/env python3
"""Prepare a verified, patched Nordic SDK in the ignored build directory."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


SDK_SHA256 = "9eb8b18c140135ab05066ad7394260f95413cb6cb403c623d389d131ceb5e381"
SDK_URL = (
    "https://nsscprodmedia.blob.core.windows.net/prod/"
    "software-and-other-downloads/sdks/nrf5/binaries/nrf5sdk1230.zip"
)


def verify_archive(archive):
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SDK_SHA256:
        raise ValueError(f"Nordic SDK checksum mismatch; remove {archive} and retry")


def prepare(source, cache):
    patch = source / "tools/nrf5sdk.patch"
    # A changed submodule patch gets its own SDK, without reusing stale patches
    # or modifying a developer's SDK in external/nrf-firmware/vendor.
    key = hashlib.sha256(SDK_SHA256.encode() + patch.read_bytes()).hexdigest()[:16]
    destination = cache / key / "nrf5sdk"
    if (destination / ".prepared").is_file():
        return destination
    if destination.exists():
        raise ValueError(f"Incomplete SDK setup; remove {destination} and retry")

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "nrf5sdk1230.zip"
    legacy_archive = source / "vendor/nrf5sdk1230.zip"
    if not archive.exists() and legacy_archive.exists():
        archive = legacy_archive
    if not archive.exists():
        print(f"Downloading Nordic SDK from {SDK_URL}", file=sys.stderr)
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
            download = Path(temporary.name)
        try:
            with urllib.request.urlopen(SDK_URL, timeout=60) as response, download.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify_archive(download)
            download.replace(archive)
        finally:
            download.unlink(missing_ok=True)
    else:
        verify_archive(archive)

    print("Preparing Nordic SDK", file=sys.stderr)
    with tempfile.TemporaryDirectory(dir=cache, prefix=".prepare-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            package.extractall(staging)
        sdk = staging / "nrf5sdk"
        (staging / "nRF5_SDK_12.3.0_d7731ad").rename(sdk)
        for relative in (
            "components/toolchain/gcc/Makefile.posix",
            "components/libraries/mailbox/app_mailbox.c",
        ):
            path = sdk / relative
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        # Fail immediately on a rejected patch. Only publish a complete SDK.
        with patch.open("rb") as patch_input:
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "-p0"],
                cwd=staging, stdin=patch_input, stdout=sys.stderr, check=True,
            )
        (sdk / ".prepared").write_text(key + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sdk.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    try:
        print(prepare(args.source.resolve(), args.cache.resolve()))
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SDK setup failed: {error}\n")


if __name__ == "__main__":
    main()
