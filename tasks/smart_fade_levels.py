# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What a player executes for a smart fade transition, one value per point.

tasks.smart_fade_mix plans where two songs mix; transition turns that plan
into the transition endpoint's answer, numbers only, every STEP_SECONDS of
the mix, so any frontend just schedules them: type, start, duration,
middle, step, filters, then from (from_id) and to (to_id): gain, position,
volume, high_db, mid_db, bass_db, plus to's highpass_hz and continue_from;
then info.

Main Features:
* song_level: one streaming pass at the song's own rate, K-weighted and
  gated (BS.1770) loudness plus sample peak; the gain brings it to
  TARGET_LUFS and raises a quiet song only until its peak hits
  PEAK_CEILING. AAC (no libsndfile) is read as mono, two equal channels.
* time_maps: the second of each song at each point. A fade plays both as
  they are; a blend or a rise glides the tempo beat by beat from the first
  song's to the next song's, so the mix lasts the mean of the two parts.
* mix_shapes, a DJ's 3-band EQ mix, every move an S-curve (ease): the next
  song's highs (shelf at HIGH_HZ) come in first, its mids (peak at MID_HZ)
  next, the bass lines swap in one equal-power move over two beats (shelf
  at BASS_HZ), then the first song's mids and its highs go out; the faders
  only fade the two ends in and out over EDGE_BEATS. A rise sweeps a
  high-pass from RISE_FROM_HZ to RISE_TO_HZ instead of the next song's
  bass. A crossfade keeps equal-power faders.
* swap_point fits the swap to the songs: the 4-beat bar fewest beats from
  the middle beat (a bar of the next song, as the mix starts on its section
  start in whole 8-beat steps) where the next song's own bass plays
  (SWAP_PRESENT of its loudest bar), inside SWAP_RANGE, the later on a
  tie; none plays there, the last bar of SWAP_RANGE.
* transition plays the mix in advance at RATE: both songs at their gains,
  along the time maps, through the filters and faders, K-weighted in
  400 ms blocks. fader_scale moves both faders together so the mix follows
  the line from the first song's level over the ANCHOR_SECONDS before the
  mix to the next song's after it (DAFx 2022: real DJ mixes keep a steady
  energy), at most SHAPE_LIMIT_DB, smoothed over SHAPE_SMOOTH_SECONDS, none
  at the two ends. Then the faders drop only where the sum would peak above
  the louder song (MIX_HEADROOM).
* crossfade_plan: the default crossfade at the two fade points, for a pair
  without a steady beat.
"""

import math

import numpy as np

from tasks import smart_fade_mix as mix

TARGET_LUFS = -14.0
PEAK_CEILING = 0.891
GUESSED_PEAK_MARGIN = math.sqrt(2.0)
MIX_HEADROOM = 1.12
BASS_HZ = 180.0
BASS_CUT_DB = -40.0
MID_HZ = 1000.0
MID_Q = 0.5
MID_CUT_DB = -27.0
HIGH_HZ = 4000.0
HIGH_CUT_DB = -40.0
EDGE_BEATS = 2
RISE_FROM_HZ = 800.0
RISE_TO_HZ = 10.0
HIGHPASS_Q_DB = 1.0
ANCHOR_SECONDS = 3.0
SHAPE_LIMIT_DB = 6.0
SHAPE_SMOOTH_SECONDS = 1.0
SHAPE_EDGE = 0.1
SILENT_DB = -60.0
WARM_SECONDS = 0.5
FILTER_BLOCK = 512
READ_BLOCK = 65536
PEAK_FLOOR = 1e-6
TYPES = {'blend': 'beat_mix', 'rise': 'beat_rise', 'fade': 'crossfade'}
SWAP_RANGE = (0.3, 0.7)
SWAP_PRESENT = 0.25


def _source(path):
    import soundfile

    try:
        soundfile.info(path)
        return {'path': path, 'mono': None}
    except soundfile.SoundFileRuntimeError:
        return {'path': path, 'mono': mix._load(path, mix.RATE)}


def song_level(source):
    import soundfile
    from scipy.signal import lfilter

    if source['mono'] is not None:
        handle, rate, guessed, mono = None, mix.RATE, True, source['mono']
        blocks = (np.stack([mono[i:i + READ_BLOCK]] * 2, axis=1) for i in range(0, mono.size, READ_BLOCK))
    else:
        handle = soundfile.SoundFile(source['path'])
        rate, guessed = handle.samplerate, False
        blocks = handle.blocks(blocksize=READ_BLOCK, dtype='float32', always_2d=True)
    filters = mix.k_filters(rate)
    hop = int(round(mix.STEP_SECONDS * rate))
    hops, carry, peak, states, frames = [], np.zeros(0), 0.0, None, 0
    try:
        for block in blocks:
            frames += block.shape[0]
            peak = max(peak, float(np.abs(block).max(initial=0.0)))
            if states is None:
                states = [np.zeros((2, block.shape[1])) for _ in filters]
            for n, (b, a) in enumerate(filters):
                block, states[n] = lfilter(b, a, block, axis=0, zi=states[n])
            energy = np.concatenate([carry, np.square(block, dtype=np.float64).sum(axis=1)])
            whole = energy.size // hop
            hops.append(energy[:whole * hop].reshape(whole, hop).sum(axis=1) / hop)
            carry = energy[whole * hop:]
    finally:
        if handle is not None:
            handle.close()
    power = np.concatenate(hops) if hops else np.zeros(0)
    blocks4 = (power[:-3] + power[1:-2] + power[2:-1] + power[3:]) / 4.0 if power.size >= 4 else power
    loudness = mix._integrated(blocks4) if blocks4.size else mix.ABSOLUTE_GATE_LUFS
    peak *= GUESSED_PEAK_MARGIN if guessed else 1.0
    gain = 1.0
    if loudness > mix.ABSOLUTE_GATE_LUFS:
        gain = 10.0 ** ((TARGET_LUFS - loudness) / 20.0)
        if gain > 1.0:
            gain = max(1.0, min(gain, PEAK_CEILING / max(peak, PEAK_FLOOR)))
    return {'loudness': float(loudness), 'peak': peak, 'gain': float(gain), 'seconds': frames / rate}


def time_maps(plan):
    mix_at, in_at = plan['mix_at'], plan['in_at']
    if plan['style'] == 'fade' or not plan.get('beat_seconds'):
        return (plan['mix_seconds'], plan['swap_at'] - mix_at,
                lambda t: mix_at + np.asarray(t, dtype=np.float64), lambda t: in_at + np.asarray(t, dtype=np.float64), None)
    beats = max(1, int(round(plan['mix_seconds'] / plan['beat_seconds'])))
    first = plan['mix_seconds'] / beats
    second = (plan['resume_at'] - in_at) / beats
    times = np.concatenate([[0.0], np.cumsum(first + (second - first) * (np.arange(beats) + 0.5) / beats)])
    steps = np.arange(beats + 1, dtype=np.float64)

    def along(start, period):
        def at(t):
            t = np.asarray(t, dtype=np.float64)
            inside = start + np.interp(t, times, steps) * period
            return np.where(t < 0.0, start + t, np.where(t > times[-1], start + beats * period + t - times[-1], inside))
        return at

    return float(times[-1]), float(times[beats >> 1]), along(mix_at, first), along(in_at, second), times


def ease(times, start, end):
    x = np.clip((np.asarray(times, dtype=np.float64) - start) / max(end - start, 1e-6), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _cut_db(wave, floor):
    return np.maximum(floor, 20.0 * np.log10(np.maximum(wave, 1e-6)))


def _bar_level(points, values, start, end):
    inside = values[(points >= start) & (points < end)]
    return float(inside.mean()) if inside.size else 0.0


def swap_point(beat_times, points, low_in):
    seconds = float(beat_times[-1])
    last = len(beat_times) - 1
    bars = list(range(0, last - 3, 4))
    if not bars:
        return float(beat_times[last >> 1])
    level = {k: _bar_level(points, low_in, beat_times[k], beat_times[k + 4]) for k in bars}
    middle = [k for k in bars if SWAP_RANGE[0] <= beat_times[k] / seconds <= SWAP_RANGE[1]]
    if not middle:
        return float(beat_times[last >> 1])
    full = max(level.values())
    present = [k for k in middle if level[k] >= SWAP_PRESENT * full]
    if not present:
        return float(beat_times[middle[-1]])
    return float(beat_times[min(present, key=lambda k: (abs(2 * k - last), -k))])


def mix_shapes(style, seconds, swap, beat, times):
    if style == 'fade':
        share = np.asarray(times, dtype=np.float64) / seconds
        return {'from_volume': np.cos(np.pi / 2 * share), 'to_volume': np.sin(np.pi / 2 * share), 'eq': None}
    rise = style == 'rise'
    lead_in = swap / 3.0
    rest = seconds - swap - beat
    edge = min(EDGE_BEATS * beat, lead_in)
    swapping = ease(times, swap - beat, swap + beat)
    return {
        'from_volume': 1.0 - ease(times, seconds - edge, seconds),
        'to_volume': ease(times, 0.0, edge),
        'eq': {
            'from_high_db': _cut_db(1.0 - ease(times, swap + beat + rest / 3.0, seconds), HIGH_CUT_DB),
            'from_mid_db': MID_CUT_DB * ease(times, swap + beat, swap + beat + 2.0 * rest / 3.0),
            'from_bass_db': _cut_db(np.cos(np.pi / 2 * swapping), BASS_CUT_DB),
            'to_high_db': _cut_db(ease(times, 0.0, lead_in), HIGH_CUT_DB),
            'to_mid_db': MID_CUT_DB * (1.0 - ease(times, lead_in, swap - beat)),
            'to_bass_db': None if rise else _cut_db(np.sin(np.pi / 2 * swapping), BASS_CUT_DB),
            'to_highpass_hz': RISE_FROM_HZ * (RISE_TO_HZ / RISE_FROM_HZ) ** ease(times, lead_in, swap + beat) if rise else None,
        },
    }


def _shelf(rate, hz, db, side):
    gain = 10.0 ** (db / 40.0)
    angle = 2.0 * math.pi * hz / rate
    root = math.sqrt(2.0 * gain) * math.sin(angle)
    cos = side * math.cos(angle)
    b = [gain * ((gain + 1) - (gain - 1) * cos + root), 2 * side * gain * ((gain - 1) - (gain + 1) * cos), gain * ((gain + 1) - (gain - 1) * cos - root)]
    a = [(gain + 1) + (gain - 1) * cos + root, -2 * side * ((gain - 1) + (gain + 1) * cos), (gain + 1) + (gain - 1) * cos - root]
    return np.array(b) / a[0], np.array(a) / a[0]


def _low_shelf(rate, db):
    return _shelf(rate, BASS_HZ, db, 1.0)


def _high_shelf(rate, db):
    return _shelf(rate, HIGH_HZ, db, -1.0)


def _mid_peak(rate, db):
    gain = 10.0 ** (db / 40.0)
    angle = 2.0 * math.pi * MID_HZ / rate
    alpha = math.sin(angle) / (2.0 * MID_Q)
    a = [1 + alpha / gain, -2 * math.cos(angle), 1 - alpha / gain]
    return np.array([1 + alpha * gain, -2 * math.cos(angle), 1 - alpha * gain]) / a[0], np.array(a) / a[0]


def _high_pass(rate, hz):
    angle = 2.0 * math.pi * hz / rate
    alpha = math.sin(angle) / (2.0 * 10.0 ** (HIGHPASS_Q_DB / 20.0))
    cos = math.cos(angle)
    a = [1 + alpha, -2 * cos, 1 - alpha]
    return np.array([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]) / a[0], np.array(a) / a[0]


def _filtered(signal, t, filters):
    from scipy.signal import lfilter

    out = np.empty_like(signal)
    states = None
    for start in range(0, signal.shape[1], FILTER_BLOCK):
        stop = min(signal.shape[1], start + FILTER_BLOCK)
        chunk = signal[:, start:stop]
        coefficients = filters(float(t[start]))
        if states is None:
            states = [np.zeros((signal.shape[0], 2)) for _ in coefficients]
        for n, (b, a) in enumerate(coefficients):
            chunk, states[n] = lfilter(b, a, chunk, axis=1, zi=states[n])
        out[:, start:stop] = chunk
    return out


def _window(source, start, seconds):
    import librosa

    begin = max(0.0, start)
    length = max(0.0, start + seconds - begin)
    if source['mono'] is None:
        audio, _sr = librosa.load(source['path'], sr=mix.RATE, mono=False, offset=begin, duration=length)
        return begin, np.atleast_2d(np.asarray(audio, dtype=np.float32))
    first = int(round(begin * mix.RATE))
    mono = source['mono'][first:first + int(round(length * mix.RATE))]
    return begin, np.stack([mono, mono])


def _lanes(audio, lanes):
    return audio[np.minimum(np.arange(lanes), audio.shape[0] - 1)]


def _along(audio, begin, positions):
    grid = np.arange(audio.shape[1], dtype=np.float64)
    index = (positions - begin) * mix.RATE
    return np.stack([np.interp(index, grid, channel, left=0.0, right=0.0).astype(np.float32) for channel in audio])


def _block_energy(signal, zero, centers):
    return _block_power(mix._k_weighted(signal, mix.RATE), zero, centers)


def _low_energy(signal, zero, centers):
    from scipy.signal import butter, sosfilt

    return _block_power(sosfilt(butter(2, BASS_HZ, btype='lowpass', fs=mix.RATE, output='sos'), signal, axis=1), zero, centers)


def _block_power(filtered, zero, centers):
    power = np.concatenate([[0.0], np.cumsum(np.square(filtered, dtype=np.float64).sum(axis=0))])
    half = int(round(mix.BLOCK_SECONDS / 2 * mix.RATE))
    middle = zero + np.round(np.asarray(centers) * mix.RATE).astype(np.int64)
    low = np.clip(middle - half, 0, power.size - 1)
    high = np.clip(middle + half, 0, power.size - 1)
    return (power[high] - power[low]) / np.maximum(high - low, 1)


def _span_energy(audio, begin, start, seconds):
    warm = int(round(WARM_SECONDS * mix.RATE))
    low = max(0, int(round((start - begin) * mix.RATE)))
    high = min(audio.shape[1], int(round((start + seconds - begin) * mix.RATE)))
    if high - low < int(mix.BLOCK_SECONDS * mix.RATE):
        return 0.0
    head = max(0, low - warm)
    weighted = mix._k_weighted(audio[:, head:high], mix.RATE)[:, low - head:]
    return float(np.square(weighted, dtype=np.float64).sum(axis=0).mean())


def _segment_max(values, edges):
    edges = np.clip(edges, 0, values.size - 1)
    out = np.maximum.reduceat(values, edges)
    return np.where(np.diff(np.append(edges, values.size)) > 0, out, 0.0)


def fader_scale(mixed, before, after, step):
    mixed = np.asarray(mixed, dtype=np.float64)
    share = np.linspace(0.0, 1.0, mixed.size) if mixed.size > 1 else np.zeros(1)
    if min(before, after) <= 10.0 ** (SILENT_DB / 10.0):
        return np.ones(mixed.size)
    start, end = 10.0 * math.log10(before), 10.0 * math.log10(after)
    wanted = np.clip(start + (end - start) * share - 10.0 * np.log10(np.maximum(mixed, 1e-12)), -SHAPE_LIMIT_DB, SHAPE_LIMIT_DB)
    reach = max(1, int(round(SHAPE_SMOOTH_SECONDS / 2 / step)))
    smooth = np.array([wanted[max(0, j - reach):j + reach + 1].mean() for j in range(wanted.size)])
    return 10.0 ** (np.minimum(1.0, np.minimum(share, 1.0 - share) / SHAPE_EDGE) * smooth / 20.0)


def _rounded(values, digits):
    return None if values is None else (np.round(np.asarray(values, dtype=np.float64), digits) + 0.0).tolist()


def transition(path_out, path_in, plan):
    rate = mix.RATE
    sources = [_source(path_out), _source(path_in)]
    songs = [song_level(source) for source in sources]
    gain_a, gain_b = songs[0]['gain'], songs[1]['gain']
    seconds, swap, lead, follow, beat_times = time_maps(plan)
    count = max(1, int(round(seconds / mix.STEP_SECONDS)))
    step = seconds / count
    points = np.arange(count + 1) * step
    zero = int(round(WARM_SECONDS * rate))
    t = np.arange(-zero, int(round(seconds * rate))) / rate
    a_begin, a_audio = _window(sources[0], plan['mix_at'] - ANCHOR_SECONDS - WARM_SECONDS, ANCHOR_SECONDS + WARM_SECONDS + float(lead(seconds)) - plan['mix_at'] + 0.5)
    b_from = float(follow(-WARM_SECONDS))
    b_begin, b_audio = _window(sources[1], b_from, float(follow(seconds)) + ANCHOR_SECONDS + 0.5 - b_from)
    lanes = max(a_audio.shape[0], b_audio.shape[0])
    a_audio, b_audio = _lanes(a_audio, lanes), _lanes(b_audio, lanes)
    raw_a, raw_b = _along(a_audio, a_begin, lead(t)), _along(b_audio, b_begin, follow(t))
    beat = plan.get('beat_seconds') or 0.0
    if beat_times is not None:
        swap = swap_point(beat_times, points, _low_energy(raw_b, zero, points))
        beat = float(np.median(np.diff(beat_times)))
    shapes = mix_shapes(plan['style'], seconds, swap, beat, points)
    eq = shapes['eq']
    sa, sb = raw_a * gain_a, raw_b * gain_b
    if eq is not None:
        value = lambda key, at: float(np.interp(at, points, eq[key]))
        sa = _filtered(sa, t, lambda at: [_low_shelf(rate, value('from_bass_db', at)), _mid_peak(rate, value('from_mid_db', at)),
                                          _high_shelf(rate, value('from_high_db', at))])
        if eq['to_highpass_hz'] is not None:
            low = lambda at: _high_pass(rate, value('to_highpass_hz', at))
        else:
            low = lambda at: _low_shelf(rate, value('to_bass_db', at))
        sb = _filtered(sb, t, lambda at: [low(at), _mid_peak(rate, value('to_mid_db', at)), _high_shelf(rate, value('to_high_db', at))])
    fall, rise = shapes['from_volume'], shapes['to_volume']
    mixed = _block_energy(sa * np.interp(t, points, fall) + sb * np.interp(t, points, rise), zero, points)
    before = _span_energy(a_audio, a_begin, plan['mix_at'] - ANCHOR_SECONDS, ANCHOR_SECONDS)
    after = _span_energy(b_audio, b_begin, float(follow(seconds)), ANCHOR_SECONDS)
    scale = fader_scale(mixed, before * gain_a ** 2, after * gain_b ** 2, step)
    fade_from, fade_to = fall * scale, rise * scale
    edges = np.round(points[:-1] * rate).astype(np.int64)
    total = np.abs(sa[:, zero:] * np.interp(t[zero:], points, fade_from) + sb[:, zero:] * np.interp(t[zero:], points, fade_to)).max(axis=0)
    ceiling = max(1.0, float(np.abs(sa[:, zero:]).max(initial=0.0)), float(np.abs(sb[:, zero:]).max(initial=0.0)))
    trim = np.minimum(1.0, ceiling / np.maximum(_segment_max(total, edges) * MIX_HEADROOM, 1e-9))
    inner = np.minimum(trim[:-1], trim[1:])
    fade_from[1:-1] *= inner
    fade_to[1:-1] *= inner
    flat = [0.0] * points.size

    def curve(key):
        return flat if eq is None or eq[key] is None else _rounded(eq[key], 2)

    return {
        'type': TYPES[plan['style']],
        'start': round(plan['mix_at'], 4),
        'duration': round(seconds, 4),
        'middle': round(swap, 4),
        'step': round(step, 6),
        'filters': {'high_hz': HIGH_HZ, 'mid_hz': MID_HZ, 'mid_q': MID_Q, 'bass_hz': BASS_HZ},
        'from': {
            'gain': round(gain_a, 4),
            'position': _rounded(lead(points), 5),
            'volume': _rounded(fade_from, 4),
            'high_db': curve('from_high_db'),
            'mid_db': curve('from_mid_db'),
            'bass_db': curve('from_bass_db'),
        },
        'to': {
            'gain': round(gain_b, 4),
            'position': _rounded(follow(points), 5),
            'volume': _rounded(fade_to, 4),
            'high_db': curve('to_high_db'),
            'mid_db': curve('to_mid_db'),
            'bass_db': curve('to_bass_db'),
            'highpass_hz': None if eq is None or eq['to_highpass_hz'] is None else _rounded(eq['to_highpass_hz'], 2),
            'continue_from': round(float(follow(seconds)), 4),
        },
        'info': {
            'tempo_from': plan['tempo_from'],
            'tempo_to': plan['tempo_to'],
            'confidence': plan['confidence'],
            'loudness_from': round(songs[0]['loudness'], 2),
            'loudness_to': round(songs[1]['loudness'], 2),
        },
    }


def crossfade_plan(path_out, out_seconds, path_in, in_seconds):
    ending, starting = mix._levels(path_out), mix._levels(path_in)
    out_seconds = min(out_seconds, ending['seconds'])
    overlap = min(mix.MIN_MIX_SECONDS, out_seconds, starting['seconds'] - in_seconds)
    if overlap <= mix.BLOCK_SECONDS:
        raise ValueError('No room for a crossfade at the fade points.')
    mix_at = out_seconds - overlap
    return {
        'mix_at': round(mix_at, 4),
        'mix_seconds': round(overlap, 4),
        'in_at': round(in_seconds, 4),
        'resume_at': round(in_seconds + overlap, 4),
        'stretch': 1.0,
        'tempo_from': None,
        'tempo_to': None,
        'style': 'fade',
        'swap_at': round(mix_at + overlap / 2, 4),
        'beat_seconds': None,
        'confidence': mix._confidence(ending, mix_at, starting, in_seconds, overlap, overlap, False),
    }
