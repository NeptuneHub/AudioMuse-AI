#!/usr/bin/env python3
# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Listening demo: the DJtransGAN CNN against the smart fade transition API.

A standalone experiment, not used by the app. It renders the same hand-over of
two songs twice so they can be compared by ear: once with the curves the
pretrained DJtransGAN generator predicts (ICASSP 2022, MIT licensed code and
weights by its authors, downloaded at run time), once with the curves the
AudioMuse-AI transition API answers. Both versions use the same songs, the same
start, length and beat alignment (the API's speed lists) and the same
-14 LUFS gains; only the fade and EQ curves differ.

Main Features:
* export: downloads the DJtransGAN code (pinned commit) and the minmax
  checkpoint, keeps only the generator network (encoder plus the two
  parameter heads; no spectrogram frontend, no mixer, no discriminator) and
  writes it as djtransgan_generator.onnx: two 128-band mel spectrograms of
  60 seconds in, 14 numbers per song out (3 band crossovers, 4 band fades).
  Audio libraries the package imports but the network never uses are stubbed.
* mix: logs in to an AudioMuse-AI server, takes the seed id and its most
  similar song (or --next), asks smart_fade and transition for that pair,
  downloads both songs and writes song1_transition_cnn_song2.wav and
  song1_transition_audiomuse_song2.wav (whole songs) plus _short.wav cuts.
  The CNN side reproduces the DJtransGAN mixer in numpy: STFT masks of 4
  bands times 4 time fades over a 60 s window centred on the transition,
  inputs loudness-normalised to -12 LUFS as in its training.
* Everything lands in scripts/smart_fade_cnn_demo/, which gets its own
  .gitignore so nothing downloaded or rendered can be committed.

Usage (repo virtualenv, which has torch, onnx and onnxruntime):
  python scripts/smart_fade_cnn_demo.py export
  python scripts/smart_fade_cnn_demo.py mix --server https://host:8000
      --user USER --password PASS --seed ITEM_ID [--next ITEM_ID] [--insecure]
"""

import argparse
import io
import json
import os
import re
import sys
import types
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, '..'))
WORK_DIR = os.path.join(HERE, 'smart_fade_cnn_demo')
CODE_SHA = '64228931f3b4514f289fbbbc0e5675adb57aeb88'
CODE_URL = f'https://codeload.github.com/ChenPaulYu/DJtransGAN/zip/{CODE_SHA}'
WEIGHTS_ID = '1L1TXWY7YOJEaozrFr4MqOZ5aXwJ0_o31'
WEIGHTS_URL = 'https://drive.usercontent.google.com/download'
WEIGHTS_FILE = 'djtransgan_minmax.pt'
ONNX_FILE = 'djtransgan_generator.onnx'
RATE = 44100
WINDOW_SECONDS = 60
N_FFT = 2048
HOP = 512
N_MELS = 128
FRAMES = 1 + RATE * WINDOW_SECONDS // HOP
BAND_FREQS = (20.0, 300.0, 5000.0, 20000.0)
MODEL_LUFS = -12.0
SHORT_SECONDS = 20.0
STUBBED = ('torchaudio', 'torchaudio.transforms', 'torchlibrosa', 'nnAudio', 'openunmix',
           'openunmix.transforms', 'madmom', 'madmom.features', 'madmom.features.downbeats',
           'pyloudnorm', 'pyrubberband', 'acoustics', 'acoustics.generator', 'matplotlib',
           'matplotlib.offsetbox', 'matplotlib.pyplot', 'IPython', 'IPython.display', 'auraloss',
           'gdown', 'tqdm')


def prepare(work):
    os.makedirs(work, exist_ok=True)
    ignore = os.path.join(work, '.gitignore')
    if not os.path.exists(ignore):
        with open(ignore, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write('*\n')


def download(url, path, params=None):
    import requests

    with requests.get(url, params=params, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(path, 'wb') as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    return path


def fetch_code(work):
    target = os.path.join(work, 'code')
    if os.path.isdir(os.path.join(target, 'djtransgan')):
        return target
    import requests

    response = requests.get(CODE_URL, timeout=300)
    response.raise_for_status()
    prefix = f'DJtransGAN-{CODE_SHA}/djtransgan/'
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or '..' in name or name.endswith('/'):
                continue
            path = os.path.join(target, 'djtransgan', name[len(prefix):])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as handle:
                handle.write(archive.read(name))
    return target


def fetch_weights(work):
    path = os.path.join(work, WEIGHTS_FILE)
    if os.path.exists(path) and os.path.getsize(path) > 1 << 20:
        return path
    params = {'id': WEIGHTS_ID, 'export': 'download', 'confirm': 't'}
    download(WEIGHTS_URL, path, params)
    if os.path.getsize(path) < 1 << 20:
        page = open(path, encoding='utf-8', errors='ignore').read()
        fields = dict(re.findall(r'name="(\w+)" value="([^"]*)"', page))
        download(WEIGHTS_URL, path, {**params, **fields})
    if os.path.getsize(path) < 1 << 20:
        raise RuntimeError('Google Drive did not hand over the checkpoint; download it by hand into ' + path)
    return path


class _Anything:
    def __init__(self, *args, **kwargs):
        self.args = args


def _stub_attribute(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return _Anything


def _stub_unused():
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = _stub_attribute
        module.__path__ = []
        sys.modules[name] = module
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(sys.modules[parent], child, module)


def export(args):
    import torch
    from torch import nn

    work = args.work
    prepare(work)
    code = fetch_code(work)
    weights = fetch_weights(work)
    print(f'checkpoint {weights}: {os.path.getsize(weights) / 1e6:.1f} MB')
    _stub_unused()
    sys.path.insert(0, code)
    from djtransgan.model.component import Encoder, PostProcessor
    from djtransgan.model.utils import estimate_postprocessor_in_dim, estimate_postprocessor_out_dims

    encoder_args = {'in_dim': 1, 'out_dims': [4, 8, 16], 'last_dim': None, 'cnn_type': 'res_cnn',
                    'activate': 'relu', 'norm_type': 'batch2d'}
    processor_args = {'out_dims': [1024, 512], 'activate': 'lrelu', 'norm_type': 'batch1d', 'pool_type': 'mlps'}
    context_args = {'activate': 'lrelu', 'norm_type': 'batch2d'}
    unzipper_args = {'n_band': 4, 'n_fader': 1, 'n_param': 2}
    processor_args['in_dim'] = estimate_postprocessor_in_dim(encoder_args)
    processor_args['out_dims'] = estimate_postprocessor_out_dims(processor_args, unzipper_args)
    processor_args['last_activate'] = 'sigmoid'

    class GeneratorCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(encoder_args)
            self.post_processors = nn.ModuleList([PostProcessor(processor_args, context_args) for _ in range(2)])

        def forward(self, mel_from, mel_to):
            vec_from, vec_to = self.encoder(mel_from), self.encoder(mel_to)
            return self.post_processors[0](vec_from, vec_to), self.post_processors[1](vec_from, vec_to)

    state = torch.load(weights, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    groups = {}
    for key, tensor in state.items():
        groups.setdefault(key.split('.')[0], [0, 0])
        groups[key.split('.')[0]][0] += 1
        groups[key.split('.')[0]][1] += tensor.numel()
    for group, (count, numbers) in sorted(groups.items()):
        print(f'  checkpoint part {group}: {count} tensors, {numbers:,} numbers ({numbers * 4 / 1e6:.1f} MB as float32)')
    core = GeneratorCore()
    kept = {key: value for key, value in state.items() if key.startswith(('encoder.', 'post_processors.'))}
    core.load_state_dict(kept, strict=True)
    core.eval()
    print(f'generator network kept: {sum(p.numel() for p in core.parameters()):,} parameters')
    rng = np.random.default_rng(0)
    mels = [torch.from_numpy((rng.standard_normal((1, 1, N_MELS, FRAMES)) * 20 - 40).astype(np.float32)) for _ in range(2)]
    path = os.path.join(work, ONNX_FILE)
    torch.onnx.export(core, tuple(mels), path, input_names=['mel_from', 'mel_to'],
                      output_names=['params_from', 'params_to'], opset_version=17, dynamo=False)
    import onnxruntime

    session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {'mel_from': mels[0].numpy(), 'mel_to': mels[1].numpy()})
    with torch.no_grad():
        torch_out = [t.numpy() for t in core(*mels)]
    gap = max(float(np.abs(a - b).max()) for a, b in zip(onnx_out, torch_out))
    print(f'wrote {path}: {os.path.getsize(path) / 1e6:.1f} MB, onnx vs torch max difference {gap:.2e}')


def _login(args):
    import requests
    import urllib3

    session = requests.Session()
    session.verify = not args.insecure
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = session.post(args.server + '/auth', json={'user': args.user, 'password': args.password},
                            headers={'X-Requested-With': 'XMLHttpRequest'}, timeout=60)
    response.raise_for_status()
    return session


def _api(session, method, url, **kwargs):
    response = session.request(method, url, timeout=600, **kwargs)
    response.raise_for_status()
    return response


def _most_similar(session, server, seed):
    found = _api(session, 'GET', server + '/api/similar_tracks', params={'item_id': seed, 'n': 5}).json()
    rows = found.get('results', found) if isinstance(found, dict) else found
    for row in rows:
        item = row.get('item_id') if isinstance(row, dict) else None
        if item and item != seed:
            return item
    raise RuntimeError('The server returned no similar song for this seed.')


def _save_song(session, server, item, work, label):
    response = _api(session, 'GET', f'{server}/api/recording_search/play/{item}')
    kind = response.headers.get('content-type', '')
    extension = {'audio/mpeg': '.mp3', 'audio/flac': '.flac', 'audio/ogg': '.ogg', 'audio/mp4': '.m4a',
                 'audio/x-m4a': '.m4a', 'audio/wav': '.wav', 'audio/x-wav': '.wav'}.get(kind.split(';')[0], '.audio')
    path = os.path.join(work, label + extension)
    with open(path, 'wb') as handle:
        handle.write(response.content)
    return path


def _whole(levels, path):
    source = levels._source(path)
    _begin, audio = levels._window(source, 0.0, 24 * 3600.0)
    return source, levels._lanes(audio, 2)


def _mix_points(answer):
    return np.arange(len(answer['from']['volume'])) * answer['step']


def _positions(answer, side):
    speed = np.array(answer[side]['speed'])
    first = answer['start'] if side == 'from' else answer['to']['start']
    return first + np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2.0 * answer['step'])])


def _along(levels, audio, times, positions):
    return levels._along(audio, 0.0, np.interp(times, *positions))


def render_api(levels, answer, song_a, song_b):
    rate = RATE
    points = _mix_points(answer)
    seconds = answer['duration']
    times = np.arange(int(round(seconds * rate))) / rate
    part_a = _along(levels, song_a, times, (points, _positions(answer, 'from'))) * answer['from']['gain']
    part_b = _along(levels, song_b, times, (points, _positions(answer, 'to'))) * answer['to']['gain']

    def value(side, key, at):
        return float(np.interp(at, points, answer[side][key]))

    def filters_a(at):
        return [levels._low_shelf(rate, value('from', 'bass_db', at)), levels._mid_peak(rate, value('from', 'mid_db', at)),
                levels._high_shelf(rate, value('from', 'high_db', at))]

    def filters_b(at):
        if answer['to']['highpass_hz']:
            low = levels._high_pass(rate, value('to', 'highpass_hz', at))
        else:
            low = levels._low_shelf(rate, value('to', 'bass_db', at))
        return [low, levels._mid_peak(rate, value('to', 'mid_db', at)), levels._high_shelf(rate, value('to', 'high_db', at))]

    part_a = levels._filtered(part_a, times, filters_a) * np.interp(times, points, answer['from']['volume'])
    part_b = levels._filtered(part_b, times, filters_b) * np.interp(times, points, answer['to']['volume'])
    head = song_a[:, :int(round(answer['start'] * rate))] * answer['from']['gain']
    tail = song_b[:, int(round(answer['to']['continue_from'] * rate)):] * answer['to']['gain']
    return np.concatenate([head, part_a + part_b, tail], axis=1), head.shape[1], head.shape[1] + part_a.shape[1]


def _ramp(x, low, high, param):
    start = (high - low) * param[0] + low
    slope = 6.0 / max(high - start, 1e-9) / max(param[1], 1e-9)
    return np.clip(np.clip(x - start, 0.0, 6.0) * slope, 0.0, 6.0) / 6.0


def band_masks(band):
    x = np.linspace(0.0, 6.0, N_FFT // 2 + 1)
    nyquist = RATE / 2.0
    curves = [1.0 - _ramp(x, BAND_FREQS[k] / nyquist * 6.0, BAND_FREQS[k + 1] / nyquist * 6.0, band[k]) for k in range(3)]
    return np.diff(np.stack([np.zeros_like(x)] + curves + [np.ones_like(x)]), axis=0)


def band_fades(fader, cue, frames, fade_in):
    x = np.linspace(0.0, 6.0, frames)
    fades = []
    for band in fader:
        ramp = np.mean([_ramp(x, cue[0] * 6.0, cue[1] * 6.0, param) for param in band], axis=0)
        fades.append(ramp if fade_in else 1.0 - ramp)
    return np.stack(fades)


def _mel_db(mono):
    import librosa

    magnitude = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP, window='hann', center=True, pad_mode='reflect'))
    bank = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=RATE / 2.0, htk=True, norm=None)
    return (20.0 * np.log10(np.maximum(bank @ magnitude, 1e-10)))[None, None].astype(np.float32)


def _masked(audio, masks, fades):
    import librosa

    out = []
    for channel in audio:
        spectrum = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP, window='hann', center=True, pad_mode='reflect')
        weight = np.einsum('bf,bt->ft', masks, fades[:, :spectrum.shape[1]])
        out.append(librosa.istft(spectrum * weight, hop_length=HOP, window='hann', center=True, length=channel.size))
    return np.stack(out)


def render_cnn(levels, answer, song_a, song_b, loud_a, loud_b, model_path):
    import onnxruntime

    rate = RATE
    points = _mix_points(answer)
    seconds = answer['duration']
    first = seconds / 2.0 - WINDOW_SECONDS / 2.0
    times = first + np.arange(RATE * WINDOW_SECONDS) / rate
    inside = (times >= 0.0) & (times <= seconds)
    pos_a = np.where(times < 0.0, answer['start'] + times, np.interp(times, points, _positions(answer, 'from')))
    pos_b = np.where(times > seconds, answer['to']['continue_from'] + times - seconds, np.interp(times, points, _positions(answer, 'to')))
    win_a = levels._along(song_a, 0.0, pos_a) * (times <= seconds)
    win_b = levels._along(song_b, 0.0, pos_b) * (times >= 0.0)
    to_model = [10.0 ** ((MODEL_LUFS - loud) / 20.0) for loud in (loud_a, loud_b)]
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    params_a, params_b = session.run(None, {'mel_from': _mel_db(win_a.mean(axis=0) * to_model[0]),
                                            'mel_to': _mel_db(win_b.mean(axis=0) * to_model[1])})
    cue = ((0.0 - first) / WINDOW_SECONDS, (seconds - first) / WINDOW_SECONDS)
    frames = FRAMES
    curves = {}
    mixed = np.zeros_like(win_a)
    for side, window, params, gain, fade_in in (('from', win_a, params_a[0], answer['from']['gain'], False),
                                                ('to', win_b, params_b[0], answer['to']['gain'], True)):
        masks = band_masks(params[:6].reshape(3, 2))
        fades = band_fades(params[6:].reshape(4, 1, 2), cue, frames, fade_in)
        mixed += _masked(window * gain, masks, fades)
        frame_times = first + np.arange(frames) * HOP / rate
        crossovers = [float(np.fft.rfftfreq(N_FFT, 1.0 / rate)[np.argmax(masks[k + 1] > masks[k])]) for k in range(3)]
        moves = []
        for fade in fades:
            moving = np.flatnonzero((fade > 0.01) & (fade < 0.99))
            moves.append([round(float(frame_times[moving[0]]), 2), round(float(frame_times[moving[-1]]), 2)] if moving.size else None)
        curves[side] = {'crossovers_hz': [round(c) for c in crossovers], 'band_fades_seconds': moves, 'raw': params.tolist()}
    head = song_a[:, :int(round((answer['start'] + first) * rate))] * answer['from']['gain']
    tail = song_b[:, int(round((answer['to']['continue_from'] + first + WINDOW_SECONDS - seconds) * rate)):] * answer['to']['gain']
    start = head.shape[1] + int(np.argmax(inside))
    return np.concatenate([head, mixed, tail], axis=1), start, start + int(round(seconds * rate)), curves


def _write(work, name, audio, start, end):
    import soundfile

    path = os.path.join(work, name + '.wav')
    soundfile.write(path, audio.T, RATE, subtype='FLOAT')
    short = audio[:, max(0, start - int(SHORT_SECONDS * RATE)):end + int(SHORT_SECONDS * RATE)]
    soundfile.write(os.path.join(work, name + '_short.wav'), short.T, RATE, subtype='FLOAT')
    peak = float(np.abs(audio).max())
    print(f'wrote {path} ({audio.shape[1] / RATE:.1f} s, peak {peak:.2f}{", above 1.0: it may clip" if peak > 1.0 else ""}) and its _short cut')


def mix(args):
    work = args.work
    prepare(work)
    model_path = os.path.join(work, ONNX_FILE)
    if not os.path.exists(model_path):
        raise SystemExit('Run the export step first: python scripts/smart_fade_cnn_demo.py export')
    sys.path.insert(0, REPO_ROOT)
    from tasks import smart_fade_levels as levels

    server = args.server.rstrip('/')
    session = _login(args)
    seed = args.seed
    nxt = args.next or _most_similar(session, server, seed)
    order = _api(session, 'POST', server + '/api/recording_search/smart_fade', json={'item_ids': [seed, nxt], 'keep_order': True}).json()
    rows = {row['item_id']: row for row in order['results']}
    query = {'from_id': seed, 'to_id': nxt, 'out': rows[seed]['fade_out_seconds'], 'in': rows[nxt]['fade_in_seconds']}
    answer = _api(session, 'GET', server + '/api/recording_search/transition', params=query).json()
    print(f'pair {seed} -> {nxt}: {answer["type"]}, starts at {answer["start"]} s of song 1, lasts {answer["duration"]} s')
    path_a = _save_song(session, server, seed, work, 'song1')
    path_b = _save_song(session, server, nxt, work, 'song2')
    source_a, song_a = _whole(levels, path_a)
    source_b, song_b = _whole(levels, path_b)
    loud_a, loud_b = levels.song_level(source_a)['loudness'], levels.song_level(source_b)['loudness']
    ours, start, end = render_api(levels, answer, song_a, song_b)
    _write(work, 'song1_transition_audiomuse_song2', ours, start, end)
    cnn, start, end, curves = render_cnn(levels, answer, song_a, song_b, loud_a, loud_b, model_path)
    _write(work, 'song1_transition_cnn_song2', cnn, start, end)
    with open(os.path.join(work, 'transition_curves.json'), 'w', encoding='utf-8', newline='\n') as handle:
        json.dump({'query': query, 'audiomuse': answer, 'cnn': curves}, handle)
    for side, label in (('from', 'song 1'), ('to', 'song 2')):
        print(f'cnn {label}: band crossovers {curves[side]["crossovers_hz"]} Hz, band fades (s after start) {curves[side]["band_fades_seconds"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--work', default=WORK_DIR)
    steps = parser.add_subparsers(dest='step', required=True)
    steps.add_parser('export')
    mixing = steps.add_parser('mix')
    mixing.add_argument('--server', required=True)
    mixing.add_argument('--user', required=True)
    mixing.add_argument('--password', required=True)
    mixing.add_argument('--seed', required=True)
    mixing.add_argument('--next')
    mixing.add_argument('--insecure', action='store_true')
    args = parser.parse_args()
    {'export': export, 'mix': mix}[args.step](args)


if __name__ == '__main__':
    main()
