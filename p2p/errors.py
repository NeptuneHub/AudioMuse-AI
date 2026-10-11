# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The exception the P2P node raises when it cannot run.

It deliberately does not derive from the builtin TimeoutError or from requests
errors, which error_manager classifies as media-server failures: a broken P2P
node is not a broken media server.

Main Features:
* P2PError ends the node's run; p2p/node.py records it as 7005 and exits
"""


class P2PError(Exception):
    pass
