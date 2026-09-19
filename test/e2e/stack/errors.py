# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one exception the stack modules raise when the live stack cannot be provided.

The stack modules never import pytest: they raise StackError with a message that
already carries the log tail or the missing resource, and the conftest turns it
into a fixture failure. A failure here is never a skip, because every resource
the stack needs is either declared by an environment variable or committed.

Main Features:
* StackError carries a complete, human-readable reason for the fixture failure
"""


class StackError(RuntimeError):
    pass
