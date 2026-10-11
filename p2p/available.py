# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Whether this build can run P2P Sharing at all.

P2P Sharing needs the Python libraries of the node and the Tor program, the
official prebuilt binary (the Ubuntu "tor" package in the image, the Tor Expert
Bundle or a system package on native installs). Both the setup wizard and the
node ask this module, which imports nothing heavy, so Flask never loads the
node's libraries.

Main Features:
* REQUIRED_MODULES and missing_modules: stem, trio, trio_websocket, coincurve
* tor_binary: the Tor executable named by TOR_BINARY, found on the PATH or as
  a path, or None
* unavailable_reason: None when P2P Sharing can run, else a short sentence
"""

import importlib.util
import os
import shutil

import config

REQUIRED_MODULES = ('stem', 'trio', 'trio_websocket', 'coincurve')


def missing_modules():
    missing = []
    for name in REQUIRED_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def tor_binary():
    name = str(config.TOR_BINARY or '').strip()
    if not name:
        return None
    if os.path.isfile(name) and os.access(name, os.X_OK):
        return name
    return shutil.which(name)


def unavailable_reason():
    missing = missing_modules()
    if missing:
        return f"this build lacks the Python libraries {', '.join(missing)}"
    if tor_binary() is None:
        return f"the Tor program ({config.TOR_BINARY}) is not installed"
    return None
