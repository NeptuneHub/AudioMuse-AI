# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Plugin subsystem package for AudioMuse-AI.

Houses the Jellyfin-style plugin system: the sanctioned author-facing API, the
loader/registry that materializes plugins from the ``plugins`` DB table, and the
admin blueprint. Installed plugin code is materialized under ``installed/`` and
imported through the runtime ``audiomuse_plugins`` namespace package.

Main Features:
* ``api`` exposes the stable surface plugins import (context + DB/config/settings helpers).
* ``manager`` discovers, materializes, and loads plugins with per-plugin failure isolation.
"""
