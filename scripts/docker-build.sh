#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Run the build inside a throwaway Debian 12 container, once per target. The release binaries for Linux and Windows are built this way.
# usage: scripts/docker-build.sh [aarch64|x86_64|windows-x64 ...]   (default: this machine's architecture)
#   aarch64, x86_64: scripts/build.sh for that Linux architecture (another architecture than this machine's runs under emulation);
#   windows-x64:     scripts/build-windows.sh, the 64-bit Windows VST3 cross-compiled with MinGW-w64, in a container of this machine's
#                    own architecture (the DLL comes out the same from an aarch64 or an x86_64 container).
# The container mounts this repository at /src, so the outputs land in build/ exactly as with the scripts themselves. IMAGE=<image>
# picks another base image (default debian:12).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-debian:12}"
[ $# -gt 0 ] || set -- "$(uname -m)"
case "$(uname -m)" in
  aarch64|arm64) native=linux/arm64 ;;
  *)             native=linux/amd64 ;;
esac
for target in "$@"; do
  script=build.sh
  packages="build-essential pkg-config git ca-certificates"
  case "$target" in
    aarch64|arm64) platform=linux/arm64 ;;
    x86_64|amd64)  platform=linux/amd64 ;;
    windows-x64|win64)
      platform="$native"
      script=build-windows.sh
      packages="make git ca-certificates g++-mingw-w64-x86-64-posix binutils-mingw-w64-x86-64" ;;
    *) echo "unknown target: $target (aarch64, x86_64 or windows-x64)" >&2; exit 2 ;;
  esac
  docker run --rm --platform "$platform" -v "$ROOT":/src -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -e SCRIPT="$script" -e PACKAGES="$packages" "$IMAGE" bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends $PACKAGES >/dev/null
    rc=0; bash "/src/scripts/$SCRIPT" || rc=$?
    chown -R "$HOST_UID:$HOST_GID" /src/build /src/third_party 2>/dev/null || true
    exit $rc'
done
