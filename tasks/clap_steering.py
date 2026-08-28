# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/master/LICENSE>
#
# Method from "Steering dense music retrieval with open-vocabulary concept discovery"
# by Julien Guinot, Alain Riou, Elio Quinton and Gyorgy Fazekas
# <https://arxiv.org/abs/2608.08757>, used under CC BY 4.0
# <https://creativecommons.org/licenses/by/4.0/>

"""Concept steering for DCLAP text search, following arXiv:2608.08757.

A sparse autoencoder trained on the DCLAP audio space gives a dictionary of
concept latents. Equations 6 and 7 of the paper steer by editing the query's own
sparse activations rather than by displacing the query vector: the query is
encoded, the concept's coordinates are raised (amplification) or lowered and
clipped at zero (suppression) by alpha times each latent's mean activation, and
and the edited code is decoded back. Only the difference between the edited and
the unedited reconstruction is applied to the query, because the autoencoder does
not round trip a text embedding exactly and returning the raw reconstruction
would change half the results before any concept was touched. Editing coordinates
rather than displacing the vector is what lets several concepts stack without one
erasing the others. The IVF index is never rebuilt or re-quantised: only the
query moves.

Only concepts that survived SAE/validate_concepts.py are offered. That script
keeps a term when the tracks its latents fire on agree with the tracks CLAP
itself ranks highest for the same words, which is the guard the paper omits and
the reason a term absent from the library cannot silently return noise.

Main Features:
* Lazy warmup and idle unload of the decoder session, matching how the CLAP and
  GTE models are handled, so an unused feature costs no resident memory.
* rank_candidates scores retrieved tracks on every requested concept and
  returns the WEAKEST score per track, so a refinement is a conjunction rather
  than a nudge: missing one concept cannot be offset by excelling at another.
* Scores are percentiles of the concept's share of a track's total activation,
  which stops globally loud tracks from winning every concept at once.
* The shipped catalogue carries no track names: a concept is a latent support
  and a grounding score, never an example out of somebody's library.
* rank_candidates returns None when no concept is requested, so a search with no
  refinement never loads a graph and never leaves the legacy path.
"""

import json
import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_STATE = {
    'encoder': None,
    'encoder_input': None,
    'decoder': None,
    'decoder_input': None,
    'd_sae': 0,
    'concepts': {},
    'catalogue': None,
    'last_used': 0.0,
    'unavailable_reason': None,
}
_LOCK = threading.RLock()


def _load_concepts(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _warmup_locked():
    from config import (
        CLAP_SAE_CONCEPTS_PATH,
        CLAP_SAE_ENCODER_PATH,
        CLAP_SAE_MODEL_PATH,
        CLAP_SAE_STEERING_ENABLED,
    )

    if not CLAP_SAE_STEERING_ENABLED:
        _STATE['unavailable_reason'] = 'CLAP_SAE_STEERING_ENABLED is false'
        return False
    if _STATE['encoder'] is not None and _STATE['decoder'] is not None:
        return True

    for label, path in (
        ('encoder', CLAP_SAE_ENCODER_PATH),
        ('decoder', CLAP_SAE_MODEL_PATH),
        ('concepts', CLAP_SAE_CONCEPTS_PATH),
    ):
        if not os.path.exists(path):
            _STATE['unavailable_reason'] = f'SAE {label} missing at {path}'
            logger.warning("CLAP steering disabled: %s", _STATE['unavailable_reason'])
            return False

    try:
        import onnxruntime as ort

        catalogue = _load_concepts(CLAP_SAE_CONCEPTS_PATH)
        concepts = catalogue.get('concepts') or []
        if not concepts:
            _STATE['unavailable_reason'] = 'SAE concept catalogue is empty'
            logger.warning("CLAP steering disabled: %s", _STATE['unavailable_reason'])
            return False

        options = ort.SessionOptions()
        options.log_severity_level = 3
        encoder = ort.InferenceSession(
            CLAP_SAE_ENCODER_PATH, sess_options=options, providers=['CPUExecutionProvider']
        )
        decoder = ort.InferenceSession(
            CLAP_SAE_MODEL_PATH, sess_options=options, providers=['CPUExecutionProvider']
        )

        _STATE['encoder'] = encoder
        _STATE['encoder_input'] = encoder.get_inputs()[0].name
        _STATE['decoder'] = decoder
        _STATE['decoder_input'] = decoder.get_inputs()[0].name
        _STATE['d_sae'] = int(catalogue.get('d_sae') or decoder.get_inputs()[0].shape[1])
        _STATE['concepts'] = {entry['term']: entry for entry in concepts}
        _STATE['catalogue'] = catalogue
        _STATE['unavailable_reason'] = None
        logger.info(
            "CLAP steering ready: %d concepts, d_sae %d",
            len(_STATE['concepts']),
            _STATE['d_sae'],
        )
        return True
    except Exception:
        _STATE['unavailable_reason'] = 'failed to load the SAE graphs, check container logs'
        logger.exception("Failed to warm up CLAP concept steering")
        _STATE['encoder'] = None
        _STATE['decoder'] = None
        return False


def warmup():
    with _LOCK:
        ready = _warmup_locked()
        if ready:
            _STATE['last_used'] = time.time()
        return ready


def unload_if_idle():
    from config import CLAP_SAE_IDLE_UNLOAD_SECONDS

    with _LOCK:
        if _STATE['encoder'] is None and _STATE['decoder'] is None:
            return False
        if CLAP_SAE_IDLE_UNLOAD_SECONDS <= 0:
            return False
        if time.time() - _STATE['last_used'] < CLAP_SAE_IDLE_UNLOAD_SECONDS:
            return False
        _STATE['encoder'] = None
        _STATE['encoder_input'] = None
        _STATE['decoder'] = None
        _STATE['decoder_input'] = None
        _STATE['concepts'] = {}
        logger.info("CLAP steering graphs unloaded after being idle")
        return True


def is_available():
    with _LOCK:
        return warmup()


def get_catalogue():
    from config import CLAP_SAE_ALPHA_STEPS, CLAP_SAE_DEFAULT_ALPHA, CLAP_SAE_MAX_TERMS

    with _LOCK:
        if not _warmup_locked():
            return {
                'available': False,
                'reason': _STATE['unavailable_reason'],
                'categories': [],
                'alpha_steps': CLAP_SAE_ALPHA_STEPS,
                'default_alpha': CLAP_SAE_DEFAULT_ALPHA,
                'max_terms': CLAP_SAE_MAX_TERMS,
            }
        _STATE['last_used'] = time.time()
        catalogue = _STATE['catalogue']
        grouped = {}
        for entry in catalogue.get('concepts') or []:
            bucket = grouped.setdefault(
                entry['category'], {'category': entry['category'], 'label': entry['label'], 'terms': []}
            )
            bucket['terms'].append(
                {'term': entry['term'], 'grounding': entry['grounding']}
            )
        order = catalogue.get('category_order') or sorted(grouped)
        return {
            'available': True,
            'reason': None,
            'categories': [grouped[name] for name in order if name in grouped],
            'alpha_steps': CLAP_SAE_ALPHA_STEPS,
            'default_alpha': CLAP_SAE_DEFAULT_ALPHA,
            'max_terms': CLAP_SAE_MAX_TERMS,
        }


def normalize_terms(raw_terms):
    from config import CLAP_SAE_ALPHA_STEPS, CLAP_SAE_DEFAULT_ALPHA, CLAP_SAE_MAX_TERMS

    if not raw_terms:
        return [], []
    if not isinstance(raw_terms, list):
        return [], ['steering must be a list of {term, weight, direction} objects']

    with _LOCK:
        if not _warmup_locked():
            return [], [_STATE['unavailable_reason'] or 'concept steering is unavailable']
        known = set(_STATE['concepts'])

    cleaned = []
    problems = []
    seen = set()
    for item in raw_terms[: CLAP_SAE_MAX_TERMS + 1]:
        if len(cleaned) >= CLAP_SAE_MAX_TERMS:
            problems.append(f'at most {CLAP_SAE_MAX_TERMS} concepts can be combined')
            break
        if isinstance(item, str):
            item = {'term': item}
        if not isinstance(item, dict):
            problems.append('each steering entry must be a string or an object')
            continue
        term = (item.get('term') or '').strip()
        if not term:
            problems.append('a steering entry has no term')
            continue
        if term not in known:
            problems.append(f'"{term}" is not a validated concept for this library')
            continue
        if term in seen:
            problems.append(f'"{term}" was passed more than once')
            continue

        try:
            weight = float(item.get('weight', CLAP_SAE_DEFAULT_ALPHA))
        except (TypeError, ValueError):
            problems.append(f'"{term}" has a non numeric weight')
            continue
        weight = min(CLAP_SAE_ALPHA_STEPS, key=lambda step: abs(step - abs(weight)))

        direction = str(item.get('direction') or 'more').lower()
        if direction not in ('more', 'less'):
            problems.append(f'"{term}" has direction "{direction}", expected more or less')
            continue

        seen.add(term)
        cleaned.append({'term': term, 'weight': weight, 'direction': direction})
    return cleaned, problems


def rank_candidates(vectors, terms):
    if vectors is None or not terms or len(vectors) == 0:
        return None, []

    with _LOCK:
        if not _warmup_locked():
            return None, []
        _STATE['last_used'] = time.time()
        encoder, encoder_input = _STATE['encoder'], _STATE['encoder_input']
        entries = [(t, _STATE['concepts'].get(t['term'])) for t in terms]

    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return None, []

    try:
        acts = encoder.run(None, {encoder_input: rows})[0].astype(np.float32)
    except Exception:
        logger.exception("CLAP concept encoder failed, leaving the ranking untouched")
        return None, []

    total = np.maximum(acts.sum(axis=1), 1e-6)
    n = acts.shape[0]
    applied = []
    scores = []
    for term, entry in entries:
        if entry is None:
            continue
        share = acts[:, np.asarray(entry['support'], dtype=np.int64)].sum(axis=1) / total
        percentile = share.argsort().argsort().astype(np.float32) / max(1, n - 1)
        if term['direction'] == 'less':
            percentile = 1.0 - percentile
        strength = min(1.0, float(term['weight']))
        scores.append(1.0 - strength * (1.0 - percentile))
        applied.append(term)

    if not scores:
        return None, []
    return np.min(np.stack(scores), axis=0), applied
