# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The pinned Chromaprint fpcalc binary the analysis stage shells out to.

The app finds fpcalc through the FPCALC environment variable (config.FPCALC_BINARY);
the container image installs it from apt, the stack fetches the static release
build instead so CI and a developer box without root run the same binary and
the chromaprint table is populated in both.

Main Features:
* FPCALC_ASSET pins the release, per-architecture URL and sha256
"""

from .binaries import Asset

FPCALC_VERSION = '1.6.1'
FPCALC_ASSET = Asset(
    name='fpcalc',
    version=FPCALC_VERSION,
    member='fpcalc',
    urls={
        'x86_64': (
            'https://github.com/acoustid/chromaprint/releases/download/'
            f'v{FPCALC_VERSION}/chromaprint-fpcalc-{FPCALC_VERSION}-linux-x86_64.tar.gz'
        ),
        'aarch64': (
            'https://github.com/acoustid/chromaprint/releases/download/'
            f'v{FPCALC_VERSION}/chromaprint-fpcalc-{FPCALC_VERSION}-linux-arm64.tar.gz'
        ),
    },
    sha256={
        'x86_64': 'fc16cd37a70168040bc9ceb45f1d4d1216f5a75bc4c9cf8564bea70ac6a45733',
        'aarch64': '7eaf5d655c4aa172ab28e3c870b8bb61dd2c327ac94de145676f88842cf6215a',
    },
)
