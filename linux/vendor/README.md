# Vendored native build inputs (Linux)

The native Linux bundle embeds Redis and two PostgreSQL contrib extensions that
`pgserver`'s minimal PostgreSQL does not ship. Both are **built fresh in CI**
(not committed to git) by the helper scripts here, then copied into the bundle
by the shared `AudioMuse-AI.spec`.

```
linux/vendor/build-redis.sh                 # -> redis/<arch>/redis-server
linux/vendor/pg-contrib/build-pg-contrib.sh # -> pg-contrib/<arch>/{lib,extension,tsearch_data}/...
```

`<arch>` is `x86_64` or `aarch64` (the output of `uname -m`).

## Why built in CI rather than committed

The macOS build commits its `redis-server` / pg-contrib artifacts for
byte-for-byte reproducibility. On Linux we build them per-arch in the workflow
instead: committing per-distro ELF binaries to git is heavy, and building from
source on the oldest supported runner (Ubuntu 22.04) gives broad glibc
compatibility. The pinned versions (`REDIS_VERSION`, and PostgreSQL = whatever
`pgserver` bundles) make the result deterministic enough for our purposes.

See `pg-contrib/README.md` for why the contrib modules must be compiled against
the *exact* PostgreSQL version `pgserver` ships.
