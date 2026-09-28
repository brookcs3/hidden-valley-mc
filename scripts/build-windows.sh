#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Build Hidden Valley Mastering Compressor for 64-bit Windows with MinGW-w64 (GCC, POSIX thread model), cross-compiling on Linux or natively in an
# MSYS2 MINGW64 shell:
#   1. fetch DPF (the plugin framework) at the commit pinned in scripts/build.sh into third_party/DPF, if it is not there yet;
#   2. build the DSP unit tests as a Windows program and run it where Windows programs run (on Windows, or under Wine when wine is on
#      the PATH; a failing run stops the build); on a plain Linux host it is built and not run, with a note;
#   3. build the VST3 and check that it imports no MinGW runtime DLL.
# libgcc, libstdc++ and winpthread are linked in statically (DPF links Windows builds with -static), so the DLL needs only DLLs that
# ship with Windows. The PE time stamp is left out (--no-insert-timestamp), so the same sources give the same DLL.
# Needs, on Debian 12: apt-get install make git ca-certificates g++-mingw-w64-x86-64-posix binutils-mingw-w64-x86-64
#   (scripts/docker-build.sh windows-x64 runs this in a throwaway debian:12 container; the release binary is built that way);
#   in MSYS2: pacman -S make git mingw-w64-x86_64-gcc, then run this from the MINGW64 shell. CXX=<compiler> picks another compiler.
# Output: build/windows/bin/HiddenValleyMC.vst3/Contents/x86_64-win/HiddenValleyMC.vst3 (the plugin DLL), and build/windows/tests/
# (test_dsp.exe; when it ran, its log and cpp_reference.json for tests/crosscheck.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPF_URL=https://github.com/DISTRHO/DPF.git
DPF_COMMIT="$(sed -n 's/^DPF_COMMIT=//p' "$ROOT/scripts/build.sh")"
[ -n "$DPF_COMMIT" ] || { echo "no DPF_COMMIT in scripts/build.sh" >&2; exit 2; }
DPF="$ROOT/third_party/DPF"
JOBS="$(nproc 2>/dev/null || echo 2)"
OUT="$ROOT/build/windows"
NAME=HiddenValleyMC

# the compiler: Debian's POSIX-thread MinGW-w64 GCC, else any GCC that targets x86_64 MinGW (MSYS2's is plain g++). DPF reads the
# target from $(CC) -dumpmachine, so CC has to be the matching MinGW gcc.
if [ -z "${CXX:-}" ]; then
  for c in x86_64-w64-mingw32-g++-posix x86_64-w64-mingw32-g++ g++; do
    if command -v "$c" >/dev/null 2>&1 && [ "$("$c" -dumpmachine)" = x86_64-w64-mingw32 ]; then CXX="$c"; break; fi
  done
fi
[ -n "${CXX:-}" ] || { echo "no MinGW-w64 C++ compiler for x86_64 found (on Debian: apt-get install g++-mingw-w64-x86-64-posix)" >&2; exit 2; }
if [ -z "${CC:-}" ]; then
  case "$CXX" in
    *g++-posix) CC="${CXX%g++-posix}gcc-posix" ;;
    *g++)       CC="${CXX%g++}gcc" ;;
    *)          CC="$CXX" ;;
  esac
fi
if [ "$("$CXX" -v 2>&1 | sed -n 's/^Thread model: //p')" != posix ]; then
  echo "$CXX does not use the POSIX thread model (DPF uses pthreads on Windows); on Debian use x86_64-w64-mingw32-g++-posix" >&2; exit 2
fi
OBJDUMP="${OBJDUMP:-$(command -v x86_64-w64-mingw32-objdump || command -v objdump || true)}"

git_dpf() { git -c safe.directory="$DPF" -C "$DPF" "$@"; }
if [ ! -e "$DPF/Makefile.plugins.mk" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --quiet "$DPF_URL" "$DPF"
fi
if [ "$(git_dpf rev-parse HEAD)" != "$DPF_COMMIT" ]; then
  git_dpf -c advice.detachedHead=false checkout --quiet "$DPF_COMMIT"
fi
echo "== DPF $(git_dpf rev-parse HEAD)"
echo "== $("$CXX" --version | head -1) ($CXX), target $("$CXX" -dumpmachine), host $(uname -s) $(uname -m)"

# unit tests of the DSP core, as a Windows program that needs no MinGW runtime DLL
mkdir -p "$OUT/tests"
"$CXX" -std=gnu++17 -O2 -Wall -Wextra -I"$ROOT/src" -static -static-libgcc -static-libstdc++ -Wl,--no-insert-timestamp \
  -o "$OUT/tests/test_dsp.exe" "$ROOT/tests/test_dsp.cpp"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) runner="" ;;
  *) if command -v wine >/dev/null 2>&1; then runner=wine; else runner=none; fi ;;
esac
if [ "$runner" = none ]; then
  echo "== built ${OUT#"$ROOT"/}/tests/test_dsp.exe, not run here (no Windows, no Wine); on Windows: test_dsp.exe <out dir>"
else
  $runner "$OUT/tests/test_dsp.exe" "$OUT/tests" | tr -d '\r' | tee "$OUT/tests/test_dsp_windows-x64.log"
fi

# the plugin. LDFLAGS goes in through the environment so that DPF's own link flags (-static among them) are kept.
LDFLAGS="-Wl,--no-insert-timestamp" \
  make -C "$ROOT/src" -j"$JOBS" CC="$CC" CXX="$CXX" PKG_CONFIG=false DPF_BUILD_DIR=../build/windows/obj DPF_TARGET_DIR=../build/windows/bin
DLL="$OUT/bin/$NAME.vst3/Contents/x86_64-win/$NAME.vst3"
test -f "$DLL"
if [ -n "$OBJDUMP" ]; then
  imports="$("$OBJDUMP" -p "$DLL" | sed -n 's/^[[:space:]]*DLL Name: //p' | tr -d '\r' | sort -f | tr '\n' ' ')"
  echo "== imports: $imports"
  case " $imports" in
    *" lib"*|*" LIB"*) echo "!! the DLL imports a MinGW runtime DLL; it has to be linked statically" >&2; exit 1 ;;
  esac
fi
echo "== built $(sha256sum "$DLL" | sed "s|$ROOT/||")"
