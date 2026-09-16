#!/usr/bin/env bash
# Build STM32 + nRF application packages from the current local sources.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./build_firmware.sh [all|cf2|cf21bl]

Build STM32 + nRF application firmware in build/firmware-<platform>.zip.
The ZIP contains no bootloader, SoftDevice, or deck firmware.
Requires S130 already installed on the Crazyflie.
Defaults to both platforms. Set JOBS to control make parallelism.
The first build downloads the Nordic SDK.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# > 1 )); then
    usage >&2
    exit 1
fi
case "${1:-all}" in
    all) platforms=(cf2 cf21bl) ;;
    cf2|cf21bl) platforms=("$1") ;;
    *) usage >&2; exit 1 ;;
esac

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root_dir"

for tool in make python3 git cc patch arm-none-eabi-gcc arm-none-eabi-g++ arm-none-eabi-objcopy; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing dependency: $tool (see README.md)." >&2
        exit 1
    fi
done
python3 -c 'import sys; sys.exit("Python 3.8+ is required") if sys.version_info < (3, 8) else None'
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}
if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer." >&2
    exit 1
fi

for source in external/firmware/Makefile external/nrf-firmware/Makefile \
    external/rl_tools/include/rl_tools/operations/arm.h external/blob/actor.h \
    external/firmware/vendor/CMSIS/CMSIS/Core/Include/core_cm4.h \
    external/firmware/vendor/FreeRTOS/include/FreeRTOS.h \
    external/firmware/vendor/libdw1000/inc/libdw1000.h; do
    if [[ ! -f $source ]]; then
        echo "Missing $source. Initialize the submodules listed in README.md." >&2
        exit 1
    fi
done

# The upstream STM32 build shares out-of-tree object files across platforms.
mkdir -p "$root_dir/build"
lock_dir="$root_dir/build/.firmware-build.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "Another firmware build is running. Run one build at a time." >&2
    echo "If a previous build was killed, remove $lock_dir after confirming it has stopped." >&2
    exit 1
fi
cleanup() {
    local build_status=$?
    rmdir "$lock_dir"
    if (( build_status != 0 )); then
        echo "Build failed. Existing ZIPs may be from an earlier build." >&2
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nrf_dir="$root_dir/external/nrf-firmware"
sdk_dir=$(python3 "$root_dir/scripts/prepare_nrf_sdk.py" "$nrf_dir" "$root_dir/build/nrf-sdk")

# Upstream version generators refresh the Git index. Give each build a
# disposable copy so they cannot change the user's index, including staging.
make_with_private_index() (
    local repository=$1
    shift
    local index_copy original_index
    index_copy=$(mktemp)
    trap 'rm -f "$index_copy" "$index_copy.lock"' EXIT
    original_index=$(git -C "$repository" rev-parse --path-format=absolute --git-path index)
    cp "$original_index" "$index_copy"
    export GIT_INDEX_FILE=$index_copy
    export GIT_OPTIONAL_LOCKS=0
    make "$@"
)

# The Nordic SDK defaults to /usr; also support toolchains elsewhere on PATH.
compiler=$(command -v arm-none-eabi-gcc)
toolchain_dir=$(cd -- "$(dirname -- "$compiler")/.." && pwd)

for platform in "${platforms[@]}"; do
    echo "Building STM32 controller for $platform"
    make_with_private_index "$root_dir/external/firmware" "${platform}_defconfig"
    make_with_private_index "$root_dir/external/firmware" -j"$jobs" all

    echo "Building nRF firmware for $platform"
    # The SDK does not track compiler flag changes; rebuild to honor BLE=1.
    # Suppress make's directory messages: its linker file list uses a sub-make.
    make_with_private_index "$nrf_dir" --no-print-directory -C "$nrf_dir" -B -j"$jobs" \
        "PLATFORM=$platform" "SDK_ROOT=$sdk_dir" "GNU_INSTALL_ROOT=$toolchain_dir" \
        "MK=mkdir -p" BLE=1

    python3 "$root_dir/scripts/package_firmware.py" "$platform" \
        --stm32 "$root_dir/build/$platform.bin" \
        --nrf "$nrf_dir/_build/${platform}_nrf.bin" \
        --output "$root_dir/build/firmware-$platform.zip"
done
