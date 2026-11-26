# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Choose audible crossfade windows without making vocal clearance a hard rule.

Main Features:
* Detect the union of MusiCNN singing patches and Silero speech spans, while
  measuring the outgoing energy curve over a bounded cue-search window.
* Jointly rank nearby incoming/outgoing cues by energy, simultaneous voices,
  and distance from the original musical cues. Bound incoming trimming and
  leave a small audible tail, rather than waiting for terminal silence.
* Preserve a sustained overlap. Dense vocals can shorten a long mix to a
  few bars, but never turn it into a sequential handoff or delayed entrance.
* Keep beat locks when cue shifts preserve their grids and the onsets still
  line up. Moved cues that lose the lock give back the verified original
  cues when no voices clash there, else the pair's crossfade (the plan's
  fade_seconds) ending at the same second. Pass the amount of vocal competition
  to smart_fade_levels for a modest, gradual EQ exchange.
* A percussion mix alternative wins when no voice overlaps another there.
* Recover a few skipped seconds with a local pass, preserving fade length,
  audible energy, vocal balance and each source's half-bar grid.
* Restore early endings toward the requested cue with a separate bounded pass.
  Keep the incoming cue and approved vocal taper; reject fading energy and
  increased competition in the later half, retaining a small audible tail.
* Detector failures preserve normal crossfading; energy checks still apply.
  The ending song is read only up to TAIL_READ_SECONDS past the planned end.
* Skip only singing patches contained in an already proven vocal union;
  refine uncovered patches and their neighbours at the original resolution.
"""

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
from tasks import smart_fade_mix as mix

logger = logging.getLogger(__name__)

VAD_RATE = 16000
VOCAL_TAGS = ('female vocalists', 'male vocalists', 'female vocalist')
VOCAL_SCORE = 0.15
VOCAL_RUN = 2
MEL_HOP = 256
FFT_SIZE = 512
PATCH_FRAMES = 187
PATCH_HOP_FRAMES = 31
PATCH_SECONDS = ((PATCH_FRAMES - 1) * MEL_HOP + FFT_SIZE) / VAD_RATE
SILERO_THRESHOLD = 0.5
SILERO_NEG_THRESHOLD = 0.35
MIN_VOICE_MS = 300
MIN_GAP_MS = 500
VOICE_PAD_MS = 200
BAR_BEATS = 4
FREE_STEP_SECONDS = 0.5
SILENT_DBFS = -50.0
FRAME_SECONDS = 0.1
RECOVERY_ENERGY_TOLERANCE = 0.25
RECOVERY_VOICE_TOLERANCE = 0.05
RECOVERY_TIME_TOLERANCE = 0.25
ENDING_ENERGY_TOLERANCE = 0.5
RECOVERY_MIN_TAIL_SECONDS = 2.0
TAIL_READ_SECONDS = 60.0
EMBEDDING_INPUT = 'model/Placeholder:0'
EMBEDDING_OUTPUT = 'model/dense/BiasAdd:0'
PREDICTION_INPUT = 'serving_default_model_Placeholder:0'
PREDICTION_OUTPUT = 'PartitionedCall:0'

_sessions = None
_sessions_lock = threading.Lock()


def _musicnn():
    global _sessions
    with _sessions_lock:
        if _sessions is None:
            from config import EMBEDDING_MODEL_PATH, MOOD_LABELS, PREDICTION_MODEL_PATH
            from tasks.onnx_utils import _default_sess_options, create_onnx_session, resolve_providers

            providers = resolve_providers(label='smart_fade_voice', cpu_only_default=True)
            options = _default_sess_options()
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            _sessions = (
                create_onnx_session(EMBEDDING_MODEL_PATH, providers, label='smart_fade_voice', sess_options=options),
                create_onnx_session(PREDICTION_MODEL_PATH, providers, label='smart_fade_voice', sess_options=options),
                [MOOD_LABELS.index(tag) for tag in VOCAL_TAGS],
            )
        return _sessions


def _patches(audio):
    import librosa

    if not audio.size:
        return np.zeros(0), np.zeros((0, PATCH_FRAMES, 96), dtype=np.float32)
    samples = (PATCH_FRAMES - 1) * MEL_HOP + FFT_SIZE
    audio = np.pad(audio, (0, max(0, samples - audio.size)))
    mel = librosa.feature.melspectrogram(y=audio, sr=VAD_RATE, n_fft=FFT_SIZE, hop_length=MEL_HOP, n_mels=96, window='hann',
                                         center=False, power=2.0, norm='slaney', htk=False)
    log_mel = np.log10(1 + 10000 * np.maximum(mel, 0.0))
    last = log_mel.shape[1] - PATCH_FRAMES
    starts = np.unique(np.append(np.arange(0, last + 1, PATCH_HOP_FRAMES), last))
    return starts * MEL_HOP / VAD_RATE, np.array([log_mel[:, k:k + PATCH_FRAMES] for k in starts]).transpose(0, 2, 1).astype(np.float32)


def _predict(patches):
    if not len(patches):
        return np.zeros(0)
    embedding, prediction, tags = _musicnn()
    features = embedding.run([EMBEDDING_OUTPUT], {EMBEDDING_INPUT: patches})[0]
    logits = prediction.run([PREDICTION_OUTPUT], {PREDICTION_INPUT: features})[0]
    return (1.0 / (1.0 + np.exp(-logits[:, tags]))).max(axis=1)


def vocal_scores(audio):
    """The MusiCNN singing score of each patch and the second each patch starts."""
    starts, patches = _patches(audio)
    return starts, _predict(patches)


def _patch_spans(starts, scores):
    spans, run = [], []
    for k, sung in enumerate(np.append(scores >= VOCAL_SCORE, False)):
        if sung:
            run.append(k)
            continue
        if run and (len(run) >= VOCAL_RUN or run[0] == 0 or run[-1] == len(scores) - 1):
            spans.append((float(starts[run[0]]), float(starts[run[-1]]) + PATCH_SECONDS))
        run = []
    return spans


def _sung_patches(starts, predict):
    """Omit only patches whose classification cannot change the vocal union."""
    count = len(starts)
    if not count:
        return []
    stride = max(VOCAL_RUN, VOCAL_RUN - 1 + int(PATCH_SECONDS * VAD_RATE / (PATCH_HOP_FRAMES * MEL_HOP)))
    anchors = np.arange(0, count, stride)
    probes = np.unique(np.r_[np.minimum(anchors[:, None] + np.arange(VOCAL_RUN), count - 1).ravel(), count - 1])
    scores = np.full(count, np.nan)
    scores[probes] = predict(probes)
    merged = []
    for begin, end in _patch_spans(starts, scores):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    covered = np.zeros(count, dtype=bool)
    for begin, end in merged:
        covered |= (starts >= begin) & (starts + PATCH_SECONDS <= end)
    outside = np.flatnonzero(~covered)
    needed = np.unique(np.r_[outside, np.maximum(0, outside - 1), np.minimum(count - 1, outside + 1)])
    needed = needed[np.isnan(scores[needed])]
    if needed.size:
        scores[needed] = predict(needed)
    return _patch_spans(starts, scores)


def sung_spans(audio):
    starts, patches = _patches(audio)
    return _sung_patches(starts, lambda indices: _predict(patches[indices]))


def warmup():
    """Initialize numerical runtimes before the web worker accepts transitions."""
    try:
        from tasks.paged_ivf import _pin_blas_single_thread

        voiced_spans(np.zeros((PATCH_FRAMES - 1) * MEL_HOP + FFT_SIZE, dtype=np.float32))
        audio = np.zeros(12 * mix.RATE, dtype=np.float32)
        burst = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(int(0.03 * mix.RATE)) / mix.RATE)
        for at in np.arange(0, audio.size, mix.RATE // 2):
            audio[at:at + burst.size] = burst
        mix._beats(audio)
        _pin_blas_single_thread()
        mix._phrase_entry(audio, 4.0)
        mix._agreement(audio)
        logger.info('Smart fade models and musical analysis warmed up')
    except Exception:
        logger.warning('Smart fade warmup failed; the next transition will retry', exc_info=True)


def spoken_spans(audio):
    from lyrics.silero_onnx import analyze_audio

    result = analyze_audio(audio, sample_rate=VAD_RATE, threshold=SILERO_THRESHOLD, neg_threshold=SILERO_NEG_THRESHOLD,
                           min_speech_duration_ms=MIN_VOICE_MS, min_silence_duration_ms=MIN_GAP_MS,
                           speech_pad_ms=VOICE_PAD_MS)
    return [(t['start'] / VAD_RATE, t['end'] / VAD_RATE) for t in result.get('segments') or []]


def voiced_spans(audio):
    """Every voice in audio, as (start, end) seconds: what MusiCNN or Silero hears."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    spans, failures = [], []
    for detector in (sung_spans, spoken_spans):
        try:
            spans += detector(audio)
        except Exception as exc:
            failures.append(exc)
            logger.warning("The %s voice detector failed for a smart fade transition: %s", detector.__name__, exc)
    if len(failures) == 2:
        raise failures[0]
    return sorted(spans)


def _sound_bounds(audio):
    frame = int(FRAME_SECONDS * VAD_RATE)
    count = audio.size // frame
    if not count:
        return None
    rms = np.sqrt(np.mean(np.square(audio[:count * frame].reshape(count, frame), dtype=np.float64), axis=1))
    loud = np.flatnonzero(rms > 10.0 ** (SILENT_DBFS / 20.0))
    return (loud[0] * FRAME_SECONDS, (loud[-1] + 1) * FRAME_SECONDS) if loud.size else None


def _steps(plan):
    beat = plan.get('beat_seconds')
    if not beat or plan['style'] == 'fade':
        return FREE_STEP_SECONDS, FREE_STEP_SECONDS
    beats = max(1, int(round(plan['mix_seconds'] / beat)))
    return BAR_BEATS * plan['mix_seconds'] / beats, BAR_BEATS * (plan['resume_at'] - plan['in_at']) / beats


def _heard(path, start, seconds):
    audio = mix._load(path, VAD_RATE, start, seconds)
    try:
        spans = voiced_spans(audio)
    except Exception:
        logger.warning("No voice detector available; retaining the normal crossfade", exc_info=True)
        spans = None
    duration = audio.size / VAD_RATE
    if spans is not None:
        spans = [(max(0.0, begin), min(duration, end)) for begin, end in spans if begin < duration and end > 0.0]
    size = max(1, int(round(mix.BLOCK_SECONDS * VAD_RATE)))
    hops = np.arange(0, max(1, audio.size - size + 1), int(round(mix.STEP_SECONDS * VAD_RATE)))
    power = np.concatenate([[0.0], np.cumsum(np.square(mix._k_weighted(audio, VAD_RATE), dtype=np.float64))])
    ends = np.minimum(hops + size, audio.size)
    energy = (power[ends] - power[hops]) / np.maximum(1, ends - hops)
    return {'spans': spans, 'bounds': _sound_bounds(audio), 'seconds': duration, 'samples': audio,
            'loudness': mix._lufs(energy), 'at': start}


def _activity(heard, points):
    """A soft cue preference: uncertain or absent detections impose no veto."""
    active = np.zeros(np.asarray(points).shape, dtype=bool)
    for begin, end in heard['spans'] or []:
        active |= (points >= heard['at'] + begin) & (points < heard['at'] + end)
    return active


def _candidates(anchor, lower, upper, step):
    if upper < lower:
        raise ValueError('No audible cue window near the proposed transition.')
    first = math.ceil((lower - anchor) / step - 1e-6)
    last = math.floor((upper - anchor) / step + 1e-6)
    return anchor + np.arange(first, last + 1) * step if last >= first else np.array([lower])


def _energy_cost(heard, points, weights):
    grid = heard['at'] + np.arange(heard['loudness'].size) * mix.STEP_SECONDS
    level = np.interp(points, grid, heard['loudness'])
    reference = float(np.percentile(heard['loudness'], 80))
    weak = np.maximum(0.0, reference - level - config.SMART_FADE_ENERGY_DROP_DB)
    silent = level < max(SILENT_DBFS, reference - 18.0)
    return float(np.average(weak + 12.0 * silent, weights=weights))


def _recover(ending, starting, plan):
    """Keep more source audio without spending the fade's duration or energy."""
    budget = max(0.0, config.SMART_FADE_RECOVER_SECONDS)
    if not budget or ending['spans'] is None or starting['spans'] is None:
        return plan
    seconds = plan['mix_seconds']
    source_seconds = plan['resume_at'] - plan['in_at']
    step_out, step_in = _steps(plan)
    if plan['style'] != 'fade' and plan.get('beat_seconds'):
        step_out /= 2.0
        step_in /= 2.0
    audible_end = ending['at'] + ending['bounds'][1]
    margin = min(max(0.0, config.SMART_FADE_END_MARGIN_SECONDS), audible_end / 10.0)
    room_out = max(0.0, min(budget, audible_end - margin - seconds - plan['mix_at']))
    earliest = min(plan['in_at'], max(starting['at'] + starting['bounds'][0], 1.0))
    room_in = max(0.0, min(budget, plan['in_at'] - earliest))
    outgoing = np.arange(math.floor(room_out / step_out + 1e-6) + 1) * step_out
    incoming = np.arange(math.floor(room_in / step_in + 1e-6) + 1) * step_in
    share = np.linspace(0.0, 1.0, 81)
    out_weights = np.maximum(0.05, np.cos(np.pi / 2 * share) ** 2)
    in_weights = np.maximum(0.05, np.sin(np.pi / 2 * share) ** 2)
    central = np.sin(np.pi * share) ** 2
    out_cost = {float(d): _energy_cost(ending, plan['mix_at'] + d + seconds * share, out_weights) for d in outgoing}
    in_cost = {float(d): _energy_cost(starting, plan['in_at'] - d + source_seconds * share, in_weights) for d in incoming}

    def clash(a, b):
        active = _activity(ending, plan['mix_at'] + a + seconds * share) & _activity(starting, plan['in_at'] - b + source_seconds * share)
        return float(np.average(active, weights=central))

    original_clash = clash(0.0, 0.0)
    safe = []
    for a in outgoing:
        for b in incoming:
            if (a + b > budget + 1e-6 or out_cost[a] > out_cost[0.0] + RECOVERY_ENERGY_TOLERANCE
                    or in_cost[b] > in_cost[0.0] + RECOVERY_ENERGY_TOLERANCE):
                continue
            competition = clash(a, b)
            if competition > original_clash + RECOVERY_VOICE_TOLERANCE:
                continue
            loss = max(0.0, out_cost[a] - out_cost[0.0]) + max(0.0, in_cost[b] - in_cost[0.0]) + 3.0 * max(0.0, competition - original_clash)
            safe.append((float(a), float(b), competition, loss))
    most = max(a + b for a, b, _competition, _loss in safe)
    near = [row for row in safe if row[0] + row[1] >= most - RECOVERY_TIME_TOLERANCE]
    a, b, competition, _loss = min(near, key=lambda row: (row[3], max(row[0], row[1]), -row[0]))
    return dict(plan, mix_at=round(plan['mix_at'] + a, 4), in_at=round(plan['in_at'] - b, 4),
                resume_at=round(plan['resume_at'] - b, 4), swap_at=round(plan['swap_at'] + a, 4),
                voice=dict(plan['voice'], clash=competition > 0.1, overlap=round(competition, 4),
                           end_margin=round(audible_end - plan['mix_at'] - a - seconds, 4),
                           recovered_seconds=round(a + b, 4)))


def _later_ending(ending, starting, plan):
    """Restore a few ending seconds without altering the approved vocal envelope."""
    budget = max(0.0, config.SMART_FADE_END_RECOVER_SECONDS)
    requested = plan.get('requested_end')
    seconds = plan['mix_seconds']
    finish = plan['mix_at'] + seconds
    if not budget or requested is None or requested <= finish or ending['spans'] is None or starting['spans'] is None:
        return plan
    audible_end = ending['at'] + ending['bounds'][1]
    margin = min(max(0.0, config.SMART_FADE_END_MARGIN_SECONDS), audible_end / 10.0)
    tail = max(min(margin, RECOVERY_MIN_TAIL_SECONDS), margin - budget)
    room = max(0.0, min(budget, requested - finish, audible_end - tail - finish))
    step = _steps(plan)[0]
    if plan['style'] != 'fade' and plan.get('beat_seconds'):
        step /= 2.0
    share = np.linspace(0.0, 1.0, 81)
    weights = np.maximum(0.05, np.cos(np.pi / 2 * share) ** 2)
    central = np.sin(np.pi * share) ** 2
    late = central * (share >= 0.5)
    incoming = _activity(starting, plan['in_at'] + (plan['resume_at'] - plan['in_at']) * share)

    def quality(delta):
        points = plan['mix_at'] + delta + seconds * share
        simultaneous = _activity(ending, points) & incoming
        return (_energy_cost(ending, points, weights), float(np.average(simultaneous, weights=central)),
                float(np.average(simultaneous, weights=late)))

    energy, clash, late_clash = quality(0.0)
    for delta in np.arange(1, math.floor(room / step + 1e-6) + 1)[::-1] * step:
        candidate_energy, candidate_clash, candidate_late = quality(delta)
        if (candidate_energy > energy + ENDING_ENERGY_TOLERANCE or candidate_clash > clash + RECOVERY_VOICE_TOLERANCE
                or candidate_late > late_clash + RECOVERY_VOICE_TOLERANCE
                or (not plan['voice']['clash'] and candidate_clash > 0.1)):
            continue
        return dict(plan, mix_at=round(plan['mix_at'] + delta, 4), swap_at=round(plan['swap_at'] + delta, 4),
                    voice=dict(plan['voice'], end_margin=round(audible_end - finish - delta, 4),
                               ending_recovered_seconds=round(float(delta), 4)))
    return plan


def _moved(path_out, path_in, plan):
    step_out, step_in = _steps(plan)
    look = max(0.0, plan['mix_at'] - config.SMART_FADE_SEARCH_SECONDS)
    incoming_limit = max(0.0, config.SMART_FADE_ENTRY_LIMIT_SECONDS)
    alternatives = plan.get('alternatives', ())
    look = min([look] + [option['mix_at'] for option in alternatives])
    incoming_end = max([plan['resume_at'], incoming_limit + plan['mix_seconds']]
                       + [option['resume_at'] for option in alternatives])
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='smart-fade-voice') as pool:
        ending = pool.submit(_heard, path_out, look, plan['mix_at'] + plan['mix_seconds'] + TAIL_READ_SECONDS - look)
        starting = pool.submit(_heard, path_in, 0.0, incoming_end + 3.0)
        ending, starting = ending.result(), starting.result()
    if ending['bounds'] is None or starting['bounds'] is None:
        raise ValueError('No audible room for a crossfade.')
    for option in alternatives:
        if _clear_handoff(ending, starting, option):
            chosen = dict(plan, **option, confidence='high' if option['beat_seconds'] else 'medium',
                          selection_reason=option['style'])
            return chosen
    audible_end = look + ending['bounds'][1]
    incoming_end = starting['bounds'][1]
    incoming_limit += starting['bounds'][0]
    locked = plan['style'] != 'fade' and bool(plan.get('beat_seconds'))
    ratio = (plan['resume_at'] - plan['in_at']) / plan['mix_seconds'] if locked else 1.0
    margin = min(max(0.0, config.SMART_FADE_END_MARGIN_SECONDS), audible_end / 10.0)
    seconds = min(plan['mix_seconds'], audible_end - look - ending['bounds'][0] - margin,
                  (incoming_end - starting['bounds'][0]) / ratio)
    probe = np.linspace(0.0, seconds, 81)
    dense = np.mean(_activity(ending, plan['mix_at'] + probe) & _activity(starting, plan['in_at'] + probe))
    if dense > 0.5 and seconds > config.SMART_FADE_VOCAL_MIX_SECONDS:
        minimum = mix.MIN_BEAT_MIX_SECONDS if locked else mix.MIN_MIX_SECONDS
        cap = max(minimum, config.SMART_FADE_VOCAL_MIX_SECONDS)
        bars = max(math.ceil(minimum / step_out), math.floor(cap / step_out))
        seconds = min(seconds, bars * step_out if locked else cap)
    if seconds <= mix.BLOCK_SECONDS:
        raise ValueError('No audible room for a crossfade.')
    if locked:
        whole_bars = math.floor((seconds + 1e-6) / step_out)
        if seconds + 0.001 >= plan['mix_seconds']:
            seconds = plan['mix_seconds']
        elif whole_bars >= 2:
            seconds = whole_bars * step_out
        else:
            locked = False
            ratio = 1.0
            step_out = step_in = FREE_STEP_SECONDS
            seconds = min(seconds, incoming_end - starting['bounds'][0])
    latest = max(0.0, audible_end - margin - seconds)
    original_end = plan['mix_at'] + plan['mix_seconds']
    out_anchor = original_end - seconds
    outgoing = _candidates(plan['mix_at'], max(look + ending['bounds'][0], min(out_anchor, latest) - config.SMART_FADE_SEARCH_SECONDS),
                           min(latest, out_anchor + 4.0), step_out)
    source_seconds = seconds * ratio
    entry_max = min(incoming_limit, incoming_end - source_seconds, max(plan['in_at'] + 4.0, starting['bounds'][0]))
    entry_min = max(starting['bounds'][0], min(1.0, entry_max))
    incoming = _candidates(plan['in_at'], entry_min, entry_max, step_in)
    share = np.linspace(0.0, 1.0, 81)
    out_weights = np.maximum(0.05, np.cos(np.pi / 2 * share) ** 2)
    in_weights = np.maximum(0.05, np.sin(np.pi / 2 * share) ** 2)
    central = np.sin(np.pi * share) ** 2
    outgoing_cost = {float(at): _energy_cost(ending, at + seconds * share, out_weights) for at in outgoing}
    incoming_cost = {float(at): _energy_cost(starting, at + source_seconds * share, in_weights) for at in incoming}

    def score(pair):
        at, entry = pair
        simultaneous = _activity(ending, at + seconds * share) & _activity(starting, entry + source_seconds * share)
        clash = float(np.average(simultaneous, weights=central))
        displacement = abs(at + seconds - original_end) + abs(entry - min(plan['in_at'], incoming_limit))
        trim = max(0.0, original_end - at - seconds) + entry
        return 3.0 * (outgoing_cost[at] + incoming_cost[entry]) + 3.0 * clash + 0.12 * displacement + 0.025 * trim

    at, entry = min(((float(a), float(b)) for a in outgoing for b in incoming), key=score)
    if locked and any(abs(value - anchor - round((value - anchor) / step) * step) > 0.001
                      for value, anchor, step in ((at, plan['mix_at'], step_out), (entry, plan['in_at'], step_in))):
        locked = False
        shorter = min(seconds, plan.get('fade_seconds', mix.MIN_MIX_SECONDS), incoming_end - entry)
        at, seconds = at + seconds - shorter, shorter
        source_seconds = seconds
    simultaneous = _activity(ending, at + seconds * share) & _activity(starting, entry + source_seconds * share)
    clash = float(np.average(simultaneous, weights=central))
    moved = dict(plan, mix_at=round(at, 4), in_at=round(entry, 4), mix_seconds=round(seconds, 4),
                 resume_at=round(entry + source_seconds, 4), swap_at=round(at + seconds / 2.0, 4),
                 style=plan['style'] if locked else 'fade', beat_seconds=plan.get('beat_seconds') if locked else None,
                 stretch=plan['stretch'] if locked else 1.0, confidence=plan['confidence'] if locked else 'medium',
                 voice={'clash': clash > 0.1, 'overlap': round(clash, 4),
                        'end_margin': round(audible_end - at - seconds, 4)})
    moved = _recover(ending, starting, moved)
    moved = _later_ending(ending, starting, moved)
    if (moved['style'] != 'fade' and 'samples' in ending and 'samples' in starting
            and (abs(moved['mix_at'] - plan['mix_at']) > 0.001 or abs(moved['in_at'] - plan['in_at']) > 0.001)):
        duration = moved['mix_seconds']
        rate = (moved['resume_at'] - moved['in_at']) / duration
        if not mix._in_step(ending['samples'], moved['mix_at'] - ending['at'], starting['samples'],
                            moved['in_at'] - starting['at'], duration, rate, moved['beat_seconds'], VAD_RATE):
            shorter = min(duration, moved.get('fade_seconds', mix.MIN_MIX_SECONDS), incoming_end - moved['in_at'])
            moved.update(mix._handoff('fade', moved['mix_at'] + duration - shorter, moved['in_at'], shorter))
            moved['confidence'] = 'medium'
    if moved['style'] == 'fade' and plan['style'] != 'fade':
        moved = _kept_lock(ending, starting, plan, moved)
    if moved['style'] == 'percussion_mix' and moved['voice']['clash']:
        moved['style'] = 'blend'
    logger.info("Smart crossfade at %.2f s for %.2f s, incoming cue %.2f s, audible tail %.2f s, vocal competition %.2f",
                moved['mix_at'], seconds, moved['in_at'], moved['voice']['end_margin'], moved['voice']['overlap'])
    return moved


def _kept_lock(ending, starting, plan, moved):
    seconds, source = plan['mix_seconds'], plan['resume_at'] - plan['in_at']
    if (not plan.get('beat_seconds') or plan['mix_at'] < ending['at'] + ending['bounds'][0]
            or plan['mix_at'] + seconds > ending['at'] + ending['bounds'][1]
            or plan['in_at'] < starting['bounds'][0] or plan['resume_at'] > starting['bounds'][1]):
        return dict(moved, beat_rejection='audible_cue')
    share = np.linspace(0.0, 1.0, 81)
    both = _activity(ending, plan['mix_at'] + seconds * share) & _activity(starting, plan['in_at'] + source * share)
    clash = float(np.average(both, weights=np.sin(np.pi * share) ** 2))
    if clash > 0.1:
        return dict(moved, beat_rejection='audible_cue')
    return dict(plan, voice={'clash': False, 'overlap': round(clash, 4),
                             'end_margin': round(ending['at'] + ending['bounds'][1] - plan['mix_at'] - seconds, 4)})


def _clear_handoff(ending, starting, option):
    if ending['spans'] is None or starting['spans'] is None:
        return False
    at, entry, seconds = option['mix_at'], option['in_at'], option['mix_seconds']
    end = ending['at'] + ending['bounds'][1]
    if (at < ending['at'] + ending['bounds'][0] or at + seconds > end
            or entry < starting['bounds'][0] or option['resume_at'] > starting['bounds'][1]):
        return False
    outgoing = _activity(ending, at + np.linspace(0, seconds, 21))
    incoming = _activity(starting, np.linspace(entry, option['resume_at'], 21))
    return not (outgoing & incoming).any()


def clear_voice(path_out, path_in, plan):
    try:
        return _moved(path_out, path_in, plan)
    except Exception as exc:
        raise ValueError('The songs could not be checked for an audible crossfade.') from exc
