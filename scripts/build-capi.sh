#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Build the C interface to the engine (src/capi/hvmc_capi.cpp) as a shared library for the Python fitting and tests:
# build/capi/libhvmc.dylib (macOS) or build/capi/libhvmc.so (Linux). Not part of the plugin.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/build/capi"
CXX="${CXX:-c++}"
case "$(uname -s)" in Darwin) EXT=dylib ;; *) EXT=so ;; esac
"$CXX" -std=gnu++17 -O2 -fno-fast-math -Wall -Wextra -shared -fPIC -I"$ROOT/src" -o "$ROOT/build/capi/libhvmc.$EXT" "$ROOT/src/capi/hvmc_capi.cpp"
echo "== built build/capi/libhvmc.$EXT"
