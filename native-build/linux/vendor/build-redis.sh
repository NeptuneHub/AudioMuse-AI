#!/usr/bin/env bash
#
# Build a self-contained ``redis-server`` for the native Linux bundle and drop it
# at ``native-build/linux/vendor/redis/<arch>/redis-server`` (where the PyInstaller spec
# expects it). Run from the repo root on the target architecture (the CI
# workflow runs it on an x86_64 and an aarch64 runner).
#
# Unlike the macOS build (which commits the binary for byte-for-byte
# reproducibility), the Linux binary is built fresh in CI: committing per-distro
# ELF binaries to git is heavy, and a from-source build on the oldest supported
# runner (Ubuntu 22.04) gives broad glibc compatibility.
#
# The embedded instance only ever uses a unix socket (no TLS -- see
# taskqueue.build_embedded_redis_argv), so a no-TLS build is functionally
# complete and avoids depending on a host OpenSSL.
set -euo pipefail

REDIS_VERSION="${REDIS_VERSION:-7.4.2}"
ARCH="$(uname -m)"   # x86_64 | aarch64
DEST="native-build/linux/vendor/redis/${ARCH}"

echo "==> Building redis-server ${REDIS_VERSION} for ${ARCH}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -fsSL "https://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz" \
  | tar xz -C "$work"
src="$work/redis-${REDIS_VERSION}"

make -C "$src" -j"$(nproc)" BUILD_TLS=no MALLOC=libc redis-server

mkdir -p "$DEST"
cp "$src/src/redis-server" "$DEST/redis-server"
chmod +x "$DEST/redis-server"

echo "==> Built $DEST/redis-server"
file "$DEST/redis-server"
echo "==> Shared library dependencies (should be base system libs only):"
ldd "$DEST/redis-server" || true
