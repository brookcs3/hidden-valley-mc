#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Build Hidden Valley Mastering Compressor on macOS as universal binaries (arm64 and x86_64), as a VST3, a CLAP and an Audio Unit (v2):
#   1. fetch DPF (the plugin framework) at the commit pinned in scripts/build.sh into third_party/DPF, if it is not there yet;
#   2. build the DSP unit tests for both architectures and run them (a failing test stops the build; the x86_64 run needs Rosetta 2,
#      and without it that run is skipped with a note);
#   3. build the VST3 and the AU from a copy of DPF in build/macos/DPF with scripts/dpf-au-parameter-strings.patch applied (the AU
#      then shows the panel legends, as the VST3 does; the patch changes only DPF's AU wrapper), put LICENSE and
#      THIRD_PARTY_NOTICES.md into each bundle's Contents/Resources, and sign both ad hoc.
# Needs the Xcode command line tools (clang, make, git, lipo, codesign): xcode-select --install
# Output: build/macos/bin/HiddenValleyMC.vst3 (Contents/MacOS/HiddenValleyMC) and build/macos/bin/HiddenValleyMC.component
# (type aufx, subtype HVmc, manufacturer Cbrk), plus build/macos/tests/ (the unit-test logs, and <arch>/cpp_reference.json for
# tests/crosscheck.py, which scripts/test.sh runs). Minimum macOS 11.0; MACOS_MIN=<version> changes it. (scripts/build.sh is the Linux build.)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "$(uname -s)" = Darwin ] || { echo "scripts/build-macos.sh runs on macOS; on Linux use scripts/build.sh" >&2; exit 2; }
DPF_URL=https://github.com/DISTRHO/DPF.git
DPF_COMMIT="$(sed -n 's/^DPF_COMMIT=//p' "$ROOT/scripts/build.sh")"
[ -n "$DPF_COMMIT" ] || { echo "no DPF_COMMIT in scripts/build.sh" >&2; exit 2; }
DPF="$ROOT/third_party/DPF"
MACOS_MIN="${MACOS_MIN:-11.0}"
ARCHS=(arm64 x86_64)
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 2)"
OUT="$ROOT/build/macos"
NAME=HiddenValleyMC

git_dpf() { git -C "$DPF" "$@"; }
if [ ! -e "$DPF/Makefile.plugins.mk" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --quiet "$DPF_URL" "$DPF"
fi
if [ "$(git_dpf rev-parse HEAD)" != "$DPF_COMMIT" ]; then
  git_dpf -c advice.detachedHead=false checkout --quiet "$DPF_COMMIT"
fi
echo "== DPF $(git_dpf rev-parse HEAD)"
echo "== $(clang++ --version | head -1), macOS $(sw_vers -productVersion) on $(uname -m), SDK $(xcrun --show-sdk-version), target ${ARCHS[*]}, minimum macOS $MACOS_MIN"

ARCH_FLAGS=()
for a in "${ARCHS[@]}"; do ARCH_FLAGS+=(-arch "$a"); done
FLAGS="${ARCH_FLAGS[*]} -mmacosx-version-min=$MACOS_MIN"

# unit tests of the DSP core, one universal test binary run once per architecture
mkdir -p "$OUT/tests"
clang++ -std=gnu++17 -O2 -Wall -Wextra $FLAGS -I"$ROOT/src" -o "$OUT/tests/test_dsp" "$ROOT/tests/test_dsp.cpp"
for a in "${ARCHS[@]}"; do
  if arch "-$a" /usr/bin/true 2>/dev/null; then
    mkdir -p "$OUT/tests/$a"
    arch "-$a" "$OUT/tests/test_dsp" "$OUT/tests/$a" | tee "$OUT/tests/test_dsp_$a.log"
  else
    echo "== cannot run $a code on this Mac (for x86_64, install Rosetta 2: softwareupdate --install-rosetta); unit tests not run for $a"
  fi
done

# DPF for the AU: a copy of third_party/DPF at the same commit, with the value-string patch
PDPF="$OUT/DPF"
rm -rf "$PDPF"
git -c init.defaultBranch=main clone --quiet --shared --no-checkout "$DPF" "$PDPF"
git -C "$PDPF" -c advice.detachedHead=false checkout --quiet "$DPF_COMMIT"
git -C "$PDPF" apply "$ROOT/scripts/dpf-au-parameter-strings.patch"
echo "== DPF copy for the AU: $(git -C "$PDPF" rev-parse HEAD) + scripts/dpf-au-parameter-strings.patch"

# the plugins: DPF's own macOS rules make the bundles, the AU's Info.plist comes from DPF's export tool. The flags go in through the
# environment, not the make command line, so that src/Makefile still appends its own (-fno-fast-math and -std=gnu++17).
rm -rf "$OUT/obj" "$OUT/bin/$NAME.vst3" "$OUT/bin/$NAME.component" "$OUT/bin/$NAME.clap"
CFLAGS="$FLAGS" CXXFLAGS="$FLAGS" LDFLAGS="$FLAGS" \
  make -C "$ROOT/src" -j"$JOBS" DPF_PATH=../build/macos/DPF DPF_BUILD_DIR=../build/macos/obj DPF_TARGET_DIR=../build/macos/bin \
  PKG_CONFIG=/usr/bin/false
VST3="$OUT/bin/$NAME.vst3"; AU="$OUT/bin/$NAME.component"; CLAP="$OUT/bin/$NAME.clap"

# DPF's generic VST3 Info.plist carries DPF's own bundle identifier and version 1.0: give it this plugin's
version="$(plutil -extract CFBundleShortVersionString raw "$AU/Contents/Info.plist")"
plutil -replace CFBundleIdentifier -string io.github.brookcs3.hidden-valley-mc.vst3 "$VST3/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$version" "$VST3/Contents/Info.plist"
plutil -insert CFBundleShortVersionString -string "$version" "$VST3/Contents/Info.plist" 2>/dev/null ||
  plutil -replace CFBundleShortVersionString -string "$version" "$VST3/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string io.github.brookcs3.hidden-valley-mc.clap "$CLAP/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$version" "$CLAP/Contents/Info.plist"
plutil -insert CFBundleShortVersionString -string "$version" "$CLAP/Contents/Info.plist" 2>/dev/null ||
  plutil -replace CFBundleShortVersionString -string "$version" "$CLAP/Contents/Info.plist"
plutil -lint "$VST3/Contents/Info.plist" "$AU/Contents/Info.plist" "$CLAP/Contents/Info.plist" >/dev/null

for b in "$VST3" "$AU" "$CLAP"; do
  mkdir -p "$b/Contents/Resources"
  cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$b/Contents/Resources/"
  bin="$b/Contents/MacOS/$NAME"
  test -f "$bin"
  have="$(lipo -archs "$bin")"
  for a in "${ARCHS[@]}"; do
    case " $have " in *" $a "*) ;; *) echo "!! $bin lacks $a (has: $have)" >&2; exit 1 ;; esac
  done
  codesign --force --deep --sign - "$b"
  codesign --verify --deep --strict "$b"
  echo "== built ${b#"$ROOT"/} ($have, ad-hoc signed)"
  for a in "${ARCHS[@]}"; do
    lipo "$bin" -thin "$a" -output "$OUT/slice.tmp"
    echo "   $a slice: $(shasum -a 256 "$OUT/slice.tmp" | cut -d' ' -f1)"
  done
  rm -f "$OUT/slice.tmp"
done
