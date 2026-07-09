# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CLAP audio segmentation tests.

Covers the window slicing used before ONNX CLAP audio embedding so exact
hop-aligned durations do not duplicate the final segment.

Main Features:
* Short audio pads to one full segment.
* Hop-aligned tails are emitted once.
* Non-aligned tails still keep the final window.
"""

import numpy as np

from tasks.clap_analyzer import _split_audio_segments


SEGMENT_LENGTH = 480000
HOP_LENGTH = 240000


def _audio(length):
    return np.arange(length, dtype=np.float32)


def test_short_audio_pads_to_one_segment():
    segments = _split_audio_segments(_audio(4), segment_length=8, hop_length=4)

    assert len(segments) == 1
    assert segments[0].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]


def test_hop_aligned_tail_is_not_duplicated():
    audio = _audio(SEGMENT_LENGTH + HOP_LENGTH)

    segments = _split_audio_segments(audio, SEGMENT_LENGTH, HOP_LENGTH)

    assert len(segments) == 2
    assert segments[0][0] == 0
    assert segments[1][0] == HOP_LENGTH


def test_non_aligned_tail_keeps_final_window():
    audio = _audio(SEGMENT_LENGTH + HOP_LENGTH + 1)

    segments = _split_audio_segments(audio, SEGMENT_LENGTH, HOP_LENGTH)

    assert len(segments) == 3
    assert segments[-1][0] == HOP_LENGTH + 1
