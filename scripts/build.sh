#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Build Hidden Valley Mastering Compressor on Linux for this machine's architecture:
#   1. fetch DPF (the plugin framework) at the pinned commit into third_party/DPF, if it is not there yet;
#   2. build and run the DSP unit tests (a failing test stops the build);
#   3. build the VST3 and the CLAP.
# Needs g++ with C++17, make, pkg-config and git. On Debian 12: apt-get install build-essential pkg-config git ca-certificates
# Output: build/bin/HiddenValleyMC.vst3/Contents/<arch>-linux/HiddenValleyMC.so, and build/tests/ (the unit-test log and
# cpp_reference.json for tests/crosscheck.py). Building on a second architecture adds its folder to the same bundle.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPF_URL=https://github.com/DISTRHO/DPF.git
DPF_COMMIT=4238e1c7f0351bbe488d79f0899c540543ac7583
DPF="$ROOT/third_party/DPF"
ARCH="$(uname -m)"
JOBS="$(nproc 2>/dev/null || echo 2)"

git_dpf() { git -c safe.directory="$DPF" -C "$DPF" "$@"; }
if [ ! -e "$DPF/Makefile.plugins.mk" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --quiet "$DPF_URL" "$DPF"
fi
if [ "$(git_dpf rev-parse HEAD)" != "$DPF_COMMIT" ]; then
  git_dpf -c advice.detachedHead=false checkout --quiet "$DPF_COMMIT"
fi
echo "== DPF $(git_dpf rev-parse HEAD)"
echo "== $(g++ --version | head -1), $ARCH"

# unit tests of the DSP core
mkdir -p "$ROOT/build/tests"
g++ -std=gnu++17 -O2 -Wall -Wextra -I"$ROOT/src" -o "$ROOT/build/tests/test_dsp" "$ROOT/tests/test_dsp.cpp"
"$ROOT/build/tests/test_dsp" "$ROOT/build/tests" | tee "$ROOT/build/tests/test_dsp_$ARCH.log"

# the plugin (objects per architecture, one bundle for all)
make -C "$ROOT/src" -j"$JOBS" DPF_BUILD_DIR="../build/obj-$ARCH"
SO="$ROOT/build/bin/HiddenValleyMC.vst3/Contents/$ARCH-linux/HiddenValleyMC.so"
test -f "$SO"
echo "== built $(sha256sum "$SO" | sed "s|$ROOT/||")"
