# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""P2P sharing of analysis results between AudioMuse-AI servers, over Tor.

The package is split so that only the node process ever imports stem, trio and
the directory client: the analysis worker, Flask and the noavx2 image use the
light modules.

Main Features:
* errors, ipc, available and compare are pure Python (plus numpy) and shared
  by the worker, Flask and the node
* client is the analysis worker's side of the worker-to-node requests
* node is the supervised p2p-node entry point; runtime, tor, directory, peers,
  wire, serve, fetch, limits, spaces and store make up the running node
"""
