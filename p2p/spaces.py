# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Model space ids: which installs compute comparable results for each model.

Two servers may swap a model's result only when they run the same model with
the same parameters. A space id is the sha256 of the model files' own sha256
digests plus the parameters that shape the result, so it changes with any
weight file or setting that would change the output. The node never imports
the analysis stack and never reads a .py source (a frozen build has none):
the lyrics axes and the GTE default paths come from tasks.lyrics_constants,
the import-free module the lyrics package itself takes them from.

Main Features:
* compute_spaces: {'m', 'c', 'l', 'n'} -> 32-byte id, or None where a model
  file is missing (that model is then neither asked for nor served)
* MusiCNN (embedding model), CLAP (audio model + its .data file + CLAP_AUDIO_*),
  lyrics (GTE model + tokenizer + LYRICS_GTE_MAX_TOKENS + the axis labels and
  descriptions), neural fingerprint (model + codebook id + NFP2 layout)
* local_codebook_crc: the codebook id a peer's neural blob must carry
* file digests are cached per path, size and mtime for the process lifetime
"""

import hashlib
import json
import logging
import os

import config
from tasks import lyrics_constants, nfp_format

logger = logging.getLogger(__name__)

SPACE_SCHEME = 'audiomuse-ai/space/1'
_DIGEST_CHUNK = 1024 * 1024
_DIGESTS = {}


def file_digest(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    cached = _DIGESTS.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(_DIGEST_CHUNK), b''):
                digest.update(chunk)
    except OSError:
        logger.exception("Could not read the model file %s to derive its P2P space id", path)
        return None
    _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def space_id(name, files, params):
    digests = []
    for path in files:
        digest = file_digest(path)
        if digest is None:
            logger.warning(
                "P2P sharing of %s results is off on this server: the model file %s is missing",
                name, path,
            )
            return None
        digests.append(digest)
    body = json.dumps(
        {'scheme': SPACE_SCHEME, 'name': name, 'files': digests, 'params': params},
        sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    )
    return hashlib.sha256(body.encode('ascii')).digest()


def local_codebook_crc():
    try:
        return nfp_format.codebook_id(nfp_format.load_codebook(config.NEURAL_FINGERPRINT_CODEBOOK_PATH))
    except (OSError, ValueError, KeyError):
        logger.warning(
            "P2P sharing of neural fingerprints is off on this server: the codebook %s cannot be read",
            config.NEURAL_FINGERPRINT_CODEBOOK_PATH,
        )
        return None


def musicnn_space():
    return space_id(
        'musicnn', [config.EMBEDDING_MODEL_PATH], {'dim': config.EMBEDDING_DIMENSION},
    )


def clap_space():
    files = [config.CLAP_AUDIO_MODEL_PATH]
    data_file = config.CLAP_AUDIO_MODEL_PATH + '.data'
    if os.path.exists(data_file):
        files.append(data_file)
    params = {
        'dim': config.CLAP_EMBEDDING_DIMENSION,
        'n_mels': config.CLAP_AUDIO_N_MELS,
        'n_fft': config.CLAP_AUDIO_N_FFT,
        'hop_length': config.CLAP_AUDIO_HOP_LENGTH,
        'fmin': config.CLAP_AUDIO_FMIN,
        'fmax': config.CLAP_AUDIO_FMAX,
        'mel_transpose': config.CLAP_AUDIO_MEL_TRANSPOSE,
    }
    return space_id('clap', files, params)


def lyrics_space():
    onnx_path = os.environ.get(lyrics_constants.GTE_ONNX_PATH_ENV, lyrics_constants.GTE_DEFAULT_ONNX_PATH)
    tokenizer_dir = os.environ.get(
        lyrics_constants.GTE_TOKENIZER_DIR_ENV, lyrics_constants.GTE_DEFAULT_TOKENIZER_DIR,
    )
    params = {
        'dim': config.LYRICS_EMBEDDING_DIMENSION,
        'max_tokens': config.LYRICS_GTE_MAX_TOKENS,
        'axes': lyrics_constants.MUSIC_ANALYSIS_AXES,
    }
    tokenizer = os.path.join(tokenizer_dir, lyrics_constants.GTE_TOKENIZER_FILE)
    return space_id('lyrics', [onnx_path, tokenizer], params)


def neural_space(codebook_crc):
    if codebook_crc is None:
        return None
    params = {
        'codebook': int(codebook_crc),
        'dim': nfp_format.DIM,
        'subspaces': nfp_format.PQ_SUBSPACES,
        'centroids': nfp_format.PQ_CENTROIDS,
    }
    return space_id('neural', [config.NEURAL_FINGERPRINT_MODEL_PATH], params)


def compute_spaces():
    codebook_crc = local_codebook_crc()
    spaces = {
        'm': musicnn_space(),
        'c': clap_space(),
        'l': lyrics_space(),
        'n': neural_space(codebook_crc),
    }
    return spaces, codebook_crc
