#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Run scripts/test.sh inside a throwaway Debian 12 container, with Pedalboard, numpy and scipy from PyPI (tests/requirements.txt).
# usage: scripts/docker-test.sh [aarch64|x86_64] [bundle, as a path inside this repository]
#        (defaults: this machine's architecture, build/bin/HiddenValleyMC.vst3). IMAGE=<image> picks another base image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-debian:12}"
arch="${1:-$(uname -m)}"
bundle="${2:-build/bin/HiddenValleyMC.vst3}"
case "$arch" in
  aarch64|arm64) platform=linux/arm64 ;;
  x86_64|amd64)  platform=linux/amd64 ;;
  *) echo "unknown architecture: $arch" >&2; exit 2 ;;
esac
docker run --rm --platform "$platform" -v "$ROOT":/src -e BUNDLE="$bundle" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends python3 python3-venv libatomic1 build-essential >/dev/null   # Pedalboard'"'"'s wheel needs libatomic; the C interface needs a compiler
  python3 -m venv /opt/venv
  /opt/venv/bin/pip install -q --disable-pip-version-check -r /src/tests/requirements.txt
  rc=0; PYTHON=/opt/venv/bin/python bash /src/scripts/test.sh "/src/$BUNDLE" || rc=$?
  chown -R "$HOST_UID:$HOST_GID" /src/build 2>/dev/null || true
  exit $rc'
