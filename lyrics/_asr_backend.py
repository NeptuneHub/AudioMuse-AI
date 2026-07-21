# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Resolves the Whisper/ASR backend for the lyrics pipeline.

The default is the built-in ONNX backend (whisper_onnx). A plugin can replace it
by registering an alternative with ``ctx.register_analysis_provider('asr', ...)``
- used, for example, by an AMD/ROCm plugin that swaps in faster-whisper because
MIGraphX cannot run the ONNX Whisper decoder. The replacement must expose the
same public surface: ``load_whisper_model()``, ``transcribe(wav, sr, language=None)``,
``is_loaded()`` and ``unload()``.

Main Features:
* Consults the loaded plugins for an 'asr' analysis provider before the built-in
* Falls back to the built-in whisper_onnx backend when no plugin registered one
* Swallows any plugin-resolution error so a broken plugin never breaks lyrics
"""

from __future__ import annotations


def get_asr_backend():
    override = _plugin_asr_backend()
    if override is not None:
        return override
    from . import whisper_onnx

    return whisper_onnx


def _plugin_asr_backend():
    try:
        from plugin.manager import plugin_manager

        return plugin_manager.get_analysis_provider('asr')
    except Exception:
        return None
