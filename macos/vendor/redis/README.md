# Vendored `redis-server`

The standalone macOS app embeds Redis (RQ broker + pub/sub). The binary is
committed here per architecture and copied into the bundle as-is by
the shared `AudioMuse-AI.spec` — the build never downloads or compiles Redis, so the
bundled binary is byte-identical to what was tested.

```
redis/arm64/redis-server   # Apple Silicon (M1/M2/M3/M4...)
```

Only `arm64` is provided — the macOS build is Apple-Silicon-only.

## Pinned version

- **Redis v8.8.0**, arm64, built **from source without TLS** (`make BUILD_TLS=no`).

## Why built from source (not copied from Homebrew)

Homebrew's `redis-server` links Homebrew's `openssl@3`
(`/opt/homebrew/opt/openssl@3/lib/lib{ssl,crypto}.3.dylib`), which does not exist
on an end-user Mac without Homebrew, so the bundled app would fail to start
Redis. The embedded instance only ever uses a unix socket (no TLS — see
`taskqueue.build_embedded_redis_argv`), so a no-TLS build is functionally
complete and depends only on `/usr/lib/libSystem.B.dylib` (verify with
`otool -L`).

## How to regenerate (on an Apple Silicon Mac)

```bash
curl -fsSL https://download.redis.io/redis-stable.tar.gz | tar xz
cd redis-stable
make -j"$(sysctl -n hw.ncpu)" BUILD_TLS=no
otool -L src/redis-server          # must show ONLY /usr/lib/libSystem.B.dylib
cp src/redis-server ../macos/vendor/redis/arm64/redis-server
chmod +x ../macos/vendor/redis/arm64/redis-server
```

`scripts/standalone/platforms/macos.py` ad-hoc signs the binary along with every other nested executable.
