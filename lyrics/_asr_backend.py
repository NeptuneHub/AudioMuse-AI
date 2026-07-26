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
* Falls back to the built-in whisper_onnx backend when no plugin registered one,
  or when the replacement is missing part of the required surface
* Swallows any plugin-resolution error so a broken plugin never breaks lyrics
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

REQUIRED_METHODS = ('load_whisper_model', 'transcribe', 'is_loaded', 'unload')


def get_asr_backend():
    override = _plugin_asr_backend()
    if override is not None and _has_required_surface(override):
        return override
    from . import whisper_onnx

    return whisper_onnx


def _plugin_asr_backend():
    try:
        from plugin.manager import plugin_manager

        return plugin_manager.get_analysis_provider('asr')
    except Exception:
        return None


# Rejects a backend that cannot stand in for whisper_onnx. A missing method would
# surface far from here: a broken is_loaded makes is_lyrics_loaded() report True
# forever, so every album pays a full memory cleanup for nothing, and a broken
# transcribe fails every song with no way back to the built-in.
def _has_required_surface(backend):
    missing = [
        name for name in REQUIRED_METHODS if not callable(getattr(backend, name, None))
    ]
    if not missing:
        return True
    logger.warning(
        'Plugin ASR backend %r is missing %s; using the built-in whisper instead',
        backend, ', '.join(missing),
    )
    return False
