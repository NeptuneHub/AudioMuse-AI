# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Where the ONNX models and the HuggingFace tokenizer cache come from for the stack.

The app reads one environment variable per model file (config.py), so the
harness resolves a single model directory and derives every path from it. That
directory is AUDIOMUSE_E2E_MODEL_DIR when set, else test/models (what CI's
download-models action fills), else the developer's git-ignored model/. Ambient
*_MODEL_PATH exports are deliberately ignored so the two app processes can never
be pointed at different models by a stray shell export.

Main Features:
* resolve() returns {ENV_KEY: absolute path} for every model variable the app
  reads, plus LYRICS_MODEL_DIR and HF_HOME
* every missing file is listed in one StackError, never skipped
"""

import os

from .errors import StackError
from .paths import REPO_ROOT

MODEL_DIR_ENV = 'AUDIOMUSE_E2E_MODEL_DIR'

MODEL_FILES = {
    'EMBEDDING_MODEL_PATH': 'musicnn_embedding.onnx',
    'PREDICTION_MODEL_PATH': 'musicnn_prediction.onnx',
    'CLAP_AUDIO_MODEL_PATH': 'model_epoch_36.onnx',
    'CLAP_TEXT_MODEL_PATH': 'clap_text_model.onnx',
    'CLAP_SAE_ENCODER_PATH': 'dclap_sae_k20_d1024_best_encoder.onnx',
    'CLAP_SAE_MODEL_PATH': 'dclap_sae_k20_d1024_best_decoder.onnx',
    'NEURAL_FINGERPRINT_MODEL_PATH': 'neural_fingerprint.onnx',
    'NEURAL_FINGERPRINT_CODEBOOK_PATH': 'neural_fingerprint_pq.npz',
    'LYRICS_GTE_ONNX_PATH': 'gte-multilingual-base-int8.onnx',
    'SILERO_VAD_ONNX_PATH': 'silero_vad.onnx',
}

MODEL_DIRS = {
    'LYRICS_GTE_TOKENIZER_DIR': 'gte-multilingual-base',
    'LYRICS_WHISPER_MODEL_DIR': 'whisper-small-onnx',
}

COMPANION_FILES = ('model_epoch_36.onnx.data',)

ANCHOR_FILE = 'musicnn_embedding.onnx'


def _candidates():
    override = os.environ.get(MODEL_DIR_ENV, '').strip()
    if override:
        return [os.path.abspath(override)]
    return [
        os.path.join(REPO_ROOT, 'test', 'models'),
        os.path.join(REPO_ROOT, 'model'),
    ]


def _hf_home(model_dir):
    for candidate in (
        os.path.join(model_dir, 'huggingface'),
        os.path.join(REPO_ROOT, 'test', '.hf_cache'),
    ):
        if os.path.isdir(os.path.join(candidate, 'hub')):
            return candidate
    return None


def resolve():
    candidates = _candidates()
    chosen = next(
        (d for d in candidates if os.path.isfile(os.path.join(d, ANCHOR_FILE))), None
    )
    if chosen is None:
        raise StackError(
            'no model directory found (looked for '
            f'{ANCHOR_FILE} in {candidates}); set {MODEL_DIR_ENV} or download the models'
        )
    missing = []
    env = {}
    for key, name in MODEL_FILES.items():
        path = os.path.join(chosen, name)
        if not os.path.isfile(path):
            missing.append(name)
        env[key] = path
    for key, name in MODEL_DIRS.items():
        path = os.path.join(chosen, name)
        if not os.path.isdir(path):
            missing.append(name + os.sep)
        env[key] = path
    for name in COMPANION_FILES:
        if not os.path.isfile(os.path.join(chosen, name)):
            missing.append(name)
    hf_home = _hf_home(chosen)
    if hf_home is None:
        missing.append('huggingface/hub (HF_HOME with the roberta-base tokenizer)')
    if missing:
        raise StackError(f'model directory {chosen} is incomplete, missing: {missing}')
    env['LYRICS_MODEL_DIR'] = chosen
    env['HF_HOME'] = hf_home
    return env
