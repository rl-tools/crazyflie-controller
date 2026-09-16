#!/usr/bin/env python3
"""Package local STM32 and nRF application firmware for cfclient/cfloader."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
PLATFORMS = ("cf2", "cf21bl")


def local_version(repository):
    return "local-" + subprocess.check_output(
        ["git", "-C", str(repository), "describe", "--always", "--dirty"],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        text=True,
    ).strip()


def package(platform, stm32, nrf, output):
    if platform not in PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")
    binaries = {
        "stm32": (stm32, f"{platform}.bin", local_version(ROOT),
                  "https://github.com/rl-tools/crazyflie-controller"),
        "nrf51": (nrf, f"{platform}_nrf.bin",
                  local_version(ROOT / "external/nrf-firmware"),
                  "https://github.com/rl-tools/crazyflie2-nrf-bootloader"),
    }
    # Keep 'platform: cf2' even for cf21bl: cfloader uses it for main-board
    # targets. fw_platform identifies the actual hardware variant.
    files = {}
    payloads = {}
    for target, (binary, name, release, repository) in binaries.items():
        data = binary.read_bytes()
        if not data:
            raise ValueError(f"Empty firmware binary: {binary}")
        entry = {
            "platform": "cf2",
            "target": target,
            "type": "fw",
            "release": release,
            "repository": repository,
        }
        if target == "nrf51":
            # The installed SoftDevice must match our nRF build. Omitting this
            # requirement makes cfloader infer legacy S110 and add its updater.
            entry["requires"] = ["sd-s130"]
        files[name] = entry
        payloads[name] = data
    manifest = {
        "version": 2,
        "subversion": 1,
        "fw_platform": platform,
        "release": binaries["stm32"][2],
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as destination:
            destination.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            for name, data in payloads.items():
                destination.writestr(name, data)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Created {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=PLATFORMS)
    parser.add_argument("--stm32", required=True, type=Path)
    parser.add_argument("--nrf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        package(args.platform, args.stm32, args.nrf, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
