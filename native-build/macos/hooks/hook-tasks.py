# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""PyInstaller hook that bundles the ``tasks`` package for the macOS build.

The RQ workers import task modules by name at runtime, so PyInstaller's static
analysis misses them; this hook collects every ``tasks`` submodule as a hidden
import so the frozen macOS app can enqueue and run all job types.

Main Features:
* Adds all ``tasks`` submodules to ``hiddenimports`` for the PyInstaller graph.
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("tasks")
