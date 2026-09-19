# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Pinned third-party executables the end-to-end stack downloads once into test/.cache.

Navidrome (the real media server) and fpcalc (the Chromaprint binary the app
shells out to, config.FPCALC_BINARY) come from their GitHub release tarballs,
are verified against a pinned sha256 and extracted into a versioned cache
directory. CI and the developer's WSL box therefore execute the same bytes, and
neither needs a package manager, sudo or a container daemon.

Main Features:
* Asset describes one release tarball per CPU architecture: url, sha256 and
  the executable member to extract
* ensure(asset) returns the cached executable, downloading only on a cache
  miss; a checksum mismatch deletes the archive and raises StackError
* AUDIOMUSE_E2E_<NAME>_BIN points the stack at a local executable instead
"""

import hashlib
import os
import platform
import shutil
import tarfile
import urllib.request

from .errors import StackError
from .paths import CACHE_DIR

DOWNLOAD_TIMEOUT_SECONDS = 300
USER_AGENT = 'AudioMuse-AI-e2e'


class Asset:
    def __init__(self, name, version, member, urls, sha256):
        self.name = name
        self.version = version
        self.member = member
        self.urls = urls
        self.sha256 = sha256

    @property
    def arch(self):
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            return 'x86_64'
        if machine in ('aarch64', 'arm64'):
            return 'aarch64'
        raise StackError(f'{self.name}: no release asset for CPU architecture {machine}')

    @property
    def url(self):
        return self.urls[self.arch]

    @property
    def expected_sha256(self):
        return self.sha256[self.arch]

    @property
    def cache_dir(self):
        return os.path.join(CACHE_DIR, self.name, self.version, self.arch)

    @property
    def executable(self):
        return os.path.join(self.cache_dir, self.member)

    @property
    def override_env(self):
        return f'AUDIOMUSE_E2E_{self.name.upper()}_BIN'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, dest):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    partial = dest + '.part'
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # nosec B310 - pinned https release URLs
            with open(partial, 'wb') as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        if os.path.exists(partial):
            os.remove(partial)
        raise StackError(f'download failed for {url}: {exc}') from exc
    os.replace(partial, dest)


def _extract_member(archive, member_name, dest):
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and os.path.basename(member.name) == member_name:
                with tar.extractfile(member) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                os.chmod(dest, 0o755)
                return
    raise StackError(f'{archive}: no member named {member_name}')


def ensure(asset):
    override = os.environ.get(asset.override_env, '').strip()
    if override:
        if not os.access(override, os.X_OK):
            raise StackError(f'{asset.override_env}={override} is not an executable file')
        return override
    executable = asset.executable
    if os.access(executable, os.X_OK):
        return executable
    os.makedirs(asset.cache_dir, exist_ok=True)
    archive = os.path.join(asset.cache_dir, os.path.basename(asset.url))
    if not os.path.isfile(archive):
        _download(asset.url, archive)
    digest = _sha256(archive)
    if digest != asset.expected_sha256:
        os.remove(archive)
        raise StackError(
            f'{asset.name} {asset.version}: sha256 mismatch for {asset.url}: '
            f'got {digest}, expected {asset.expected_sha256}'
        )
    _extract_member(archive, asset.member, executable)
    return executable
