# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Selects the Whisper ASR backend for the lyrics pipeline.

Default is onnx (whisper_onnx). LYRICS_WHISPER_BACKEND=faster switches to
faster-whisper/CTranslate2, used on the AMD/ROCm image since MIGraphX can't
run the ONNX Whisper decoder. Both modules share the same public surface.
"""

from __future__ import annotations

import os

_FASTER = {"faster", "faster-whisper", "faster_whisper", "ct2", "ctranslate2"}


def get_asr_backend():
    choice = os.environ.get("LYRICS_WHISPER_BACKEND", "onnx").strip().lower()
    if choice in _FASTER:
        from . import whisper_faster

        return whisper_faster
    from . import whisper_onnx

    return whisper_onnx
