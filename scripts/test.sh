#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Run the tests on a built bundle, stopping at the first failure: build the engine's C interface (build/capi, used by the Python
# tests), rerun the C++ unit tests if the build left their binary, then the Pedalboard tests: the cross-check of plugin, C interface and
# unit tests, the parameter table and host contract, the whole measurement protocol against the reference features, control coverage,
# and the README example. Needs python3 with the packages in tests/requirements.txt (a venv is fine; PYTHON=<interpreter> picks one).
# usage: scripts/test.sh [bundle]   (default build/macos/bin/HiddenValleyMC.vst3 on macOS, build/bin/HiddenValleyMC.vst3 elsewhere; on
# macOS the Audio Unit, build/macos/bin/HiddenValleyMC.component, is tested the same way and is the one whose meters can be read)
# HVMC_QUICK=1 runs the protocol subset instead of the whole protocol. Results go to build/test-results/<platform-arch-format>/.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
if [ "$(uname -s)" = Darwin ]; then
  BUNDLE="${1:-$ROOT/build/macos/bin/HiddenValleyMC.vst3}"
  arch="$("$PY" -c 'import platform; print(platform.machine())')"
  case "${BUNDLE%/}" in *.component) format=au ;; *) format=vst3 ;; esac
  OUT="$ROOT/build/test-results/macos-$arch-$format"
  REF="$ROOT/build/macos/tests/$arch/cpp_reference.json"
  TESTBIN="$ROOT/build/macos/tests/test_dsp"
else
  BUNDLE="${1:-$ROOT/build/bin/HiddenValleyMC.vst3}"
  OUT="$ROOT/build/test-results/$(uname -m)"
  REF="$ROOT/build/tests/cpp_reference.json"
  TESTBIN="$ROOT/build/tests/test_dsp"
fi
mkdir -p "$OUT"
run() { echo "== $*"; "$@" || { echo "!! failed: $*"; echo "== FAILED"; exit 1; }; }
run bash "$ROOT/scripts/build-capi.sh"
if [ -x "$TESTBIN" ]; then
  mkdir -p "$(dirname "$REF")"
  run "$TESTBIN" "$(dirname "$REF")"
fi
run "$PY" "$ROOT/tests/crosscheck.py" "$REF" "$BUNDLE"
run "$PY" "$ROOT/tests/pb_load.py" "$BUNDLE" "$OUT/parameters.json"
if [ -n "${HVMC_QUICK:-}" ]; then
  run "$PY" "$ROOT/tests/pb_reference.py" "$BUNDLE" "$OUT/pb_reference.json" --quick
else
  run "$PY" "$ROOT/tests/pb_reference.py" "$BUNDLE" "$OUT/pb_reference.json"
fi
run "$PY" "$ROOT/tests/pb_coverage.py" "$BUNDLE" "$OUT/pb_coverage.json"
run "$PY" "$ROOT/tests/readme_examples.py" "$BUNDLE"
echo "== ALL PASSED"
