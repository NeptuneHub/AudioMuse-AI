# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Search by recording (ALGORITHM.md section 17) with a real clip of a fixture track.

Ten seconds cut from the middle of a fixture clip are uploaded exactly as the
page does; the neural fingerprint index must identify the source track and the
alignment offset. Searching by a stored track finds its other copy in the
catalogue, and unusable clips are refused.

Main Features:
* an uploaded slice identifies its source track with the right offset
* by_track finds the padded copy of the same recording
* silent, missing and unknown inputs are 400
* warmup reports the neural and encoder models loaded
"""

import io

import librosa
import numpy as np
import pytest
import soundfile

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e


def _wav_bytes(samples, rate):
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, rate, format='WAV', subtype='PCM_16')
    return buffer.getvalue()


@pytest.fixture(scope='module')
def probe_clip(stack, library, analyzed_library):
    probe = library.manifest['recording_probe']
    audio, rate = librosa.load(library.path(probe['key']), sr=None, mono=True)
    start = int(probe['start_s'] * rate)
    end = start + int(probe['length_s'] * rate)
    return probe, _wav_bytes(audio[start:end], rate), rate


def test_clip_identifies_its_track(stack, api, library, probe_clip):
    probe, wav, _rate = probe_clip
    response = api.post(
        '/api/recording_search/search',
        files={'clip': ('clip.wav', wav, 'audio/wav')},
        data={'n_results': '5'},
        timeout=180,
    )
    assert response.status_code == 200, response.text[:400]
    body = response.json()
    assert_no_fp_ids(body)
    assert abs(body['clip_seconds'] - probe['length_s']) < 0.5, body['clip_seconds']
    results = body['results']
    assert results and body['count'] == len(results)
    best = results[0]
    assert best['item_id'] == library.pid(probe['key']), results
    assert best['identified'] is True, best
    assert abs(best['offset_seconds'] - probe['start_s']) <= 1.5, best


def test_by_track_finds_the_padded_copy(stack, api, library, analyzed_library):
    body = api.json('POST', '/api/recording_search/by_track', json={'item_id': library.pid('A02'), 'n_results': 5})
    assert_no_fp_ids(body)
    assert body['item_id'] == library.pid('A02')
    assert body['results'], body
    assert body['results'][0]['item_id'] == library.pid('F02'), body['results']
    assert body['results'][0]['identified'] is True


def test_unusable_inputs(stack, api, probe_clip):
    _probe, _wav, rate = probe_clip
    silent = _wav_bytes(np.zeros(rate * 10, dtype=np.float32), rate)
    response = api.post('/api/recording_search/search', files={'clip': ('silent.wav', silent, 'audio/wav')}, data={'n_results': '5'})
    assert response.status_code == 400, response.text[:300]
    assert api.post('/api/recording_search/search', data={'n_results': '5'}).status_code == 400
    assert api.post('/api/recording_search/by_track', json={'item_id': 'nope', 'n_results': 5}).status_code == 400


def test_warmup_reports_models(stack, api, analyzed_library):
    body = api.json('POST', '/api/recording_search/warmup', timeout=300)
    assert body['loaded'] is True, body
    assert body['models']['neural'] is True and body['models']['encoder'] is True, body
