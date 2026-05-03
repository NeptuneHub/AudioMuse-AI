"""Minimal unit tests for the lyrics analysis helpers.

Focus on the deterministic, dependency-free pieces:
- ``axis_columns()`` — canonical (axis, label) ordering and total size.
- ``_score_axes()`` — produced vector matches ``axis_columns()`` length and
  ordering (each axis chunk is a softmax that sums to 1.0).
- ``_sanitize_lyrics_text()`` — real cleanup behaviour on representative input.
- ``_softmax()`` — basic sanity (sums to 1, monotonic).

These tests do NOT load whisper, transformers, or torch heavyweights. The
embedding integration is covered separately by
``test/test_lyrics_analysis_integration.py``.
"""
from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope='module')
def lt():
    """Import lyrics.lyrics_transcriber once for the whole module."""
    return importlib.import_module('lyrics.lyrics_transcriber')


# ---------------------------------------------------------------------------
# axis_columns(): the most critical invariant for downstream BYTEA storage.
# ---------------------------------------------------------------------------

class TestAxisColumns:
    # Hard-coded expected size: axes 1-3 have 6 labels, axis 4 has 5, axis 5
    # has 4 -> 6+6+6+5+4 = 27. Bumping this requires a deliberate schema change
    # since the resulting BYTEA vector is persisted in the DB.
    EXPECTED_TOTAL_LABELS = 27

    # Canonical (axis, label) sequence. Pinned here so accidental reordering of
    # ``MUSIC_ANALYSIS_AXES`` (which would silently shuffle every persisted
    # vector dimension) fails this test loudly.
    EXPECTED_ORDER = (
        ('AXIS_1_SETTING', 'URBAN'),
        ('AXIS_1_SETTING', 'WILDERNESS'),
        ('AXIS_1_SETTING', 'INTERIOR'),
        ('AXIS_1_SETTING', 'TRANSIT'),
        ('AXIS_1_SETTING', 'EXTRATERRESTRIAL'),
        ('AXIS_1_SETTING', 'SURREAL_ABSTRACT'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'SOLITARY'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'ROMANTIC'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'KINSHIP'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'COLLECTIVE'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'ADVERSARIAL'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'DIVINE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'RADIANT'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'MELANCHOLIC'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'VOLATILE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'VULNERABLE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'SERENE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'NUMB'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'RETROSPECTIVE'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'CHRONICLE'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'EXISTENTIAL'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'STORYTELLING'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'DIRECT_PLEA'),
        ('AXIS_5_THEMATIC_WEIGHT', 'TRIVIAL'),
        ('AXIS_5_THEMATIC_WEIGHT', 'MORTAL'),
        ('AXIS_5_THEMATIC_WEIGHT', 'POLITICAL'),
        ('AXIS_5_THEMATIC_WEIGHT', 'SENSORIAL'),
    )

    def test_total_length_matches_sum_of_labels(self, lt):
        expected = sum(len(meta['labels']) for meta in lt.MUSIC_ANALYSIS_AXES.values())
        assert len(lt.axis_columns()) == expected

    def test_total_length_is_exactly_27(self, lt):
        assert len(lt.axis_columns()) == self.EXPECTED_TOTAL_LABELS

    def test_canonical_order_is_pinned(self, lt):
        """Hard-coded order guard: the persisted vector layout must NEVER
        change without a deliberate migration. If this fails, every BYTEA
        vector already in the DB would be misinterpreted."""
        assert tuple(lt.axis_columns()) == self.EXPECTED_ORDER

    def test_order_is_axes_then_labels_in_definition_order(self, lt):
        cols = lt.axis_columns()
        idx = 0
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            for label in meta['labels'].keys():
                assert cols[idx] == (axis_name, label), (
                    f'Position {idx}: expected {(axis_name, label)}, got {cols[idx]}'
                )
                idx += 1
        assert idx == len(cols)

    def test_returns_tuples_of_two_strings(self, lt):
        for axis_name, label in lt.axis_columns():
            assert isinstance(axis_name, str) and axis_name
            assert isinstance(label, str) and label

    def test_no_duplicates(self, lt):
        cols = lt.axis_columns()
        assert len(cols) == len(set(cols))

    def test_is_pure_function_stable_across_calls(self, lt):
        assert lt.axis_columns() == lt.axis_columns()


# ---------------------------------------------------------------------------
# _score_axes(): same length & order invariants on the produced vector.
# ---------------------------------------------------------------------------

class TestScoreAxes:
    @staticmethod
    def _fake_axis_state(lt):
        """Build a deterministic fake (label_map, axis_embeddings) keyed in
        the canonical order with random unit-norm row vectors. The actual
        numbers don't matter — we only assert structural invariants."""
        rng = np.random.default_rng(42)
        label_map = {}
        axis_embeddings = {}
        emb_dim = 8
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            labels = [(name, name.lower()) for name in meta['labels'].keys()]
            label_map[axis_name] = labels
            mat = rng.standard_normal((len(labels), emb_dim)).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
            axis_embeddings[axis_name] = mat
        return label_map, axis_embeddings, emb_dim

    def test_vector_length_matches_axis_columns(self, lt):
        label_map, axis_embeddings, emb_dim = self._fake_axis_state(lt)
        embedding = np.ones(emb_dim, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        with patch.object(lt, '_get_axis_embeddings',
                          return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(embedding)

        assert vec.dtype == np.float32
        assert vec.shape == (len(lt.axis_columns()),)
        # Hard-coded absolute size guard (see TestAxisColumns).
        assert vec.shape == (27,)

    def test_each_axis_chunk_is_softmax_summing_to_one(self, lt):
        label_map, axis_embeddings, emb_dim = self._fake_axis_state(lt)
        embedding = np.ones(emb_dim, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        with patch.object(lt, '_get_axis_embeddings',
                          return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(embedding)

        offset = 0
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            n = len(meta['labels'])
            chunk = vec[offset:offset + n]
            assert np.all(chunk >= 0.0), f'{axis_name}: negative softmax entry'
            assert np.all(chunk <= 1.0), f'{axis_name}: softmax entry > 1'
            assert chunk.sum() == pytest.approx(1.0, abs=1e-5), (
                f'{axis_name}: chunk sum {chunk.sum()} != 1.0'
            )
            offset += n
        assert offset == vec.shape[0]

    def test_chunk_layout_aligns_with_axis_columns(self, lt):
        """The i-th component of the vector must correspond to axis_columns()[i]."""
        label_map, axis_embeddings, emb_dim = self._fake_axis_state(lt)
        embedding = np.ones(emb_dim, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        with patch.object(lt, '_get_axis_embeddings',
                          return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(embedding)

        cols = lt.axis_columns()
        # Group columns by axis and verify each grouped sum == 1.0 in vec order.
        by_axis: dict[str, list[int]] = {}
        for i, (axis_name, _label) in enumerate(cols):
            by_axis.setdefault(axis_name, []).append(i)
        # axis_columns() iterates axes in MUSIC_ANALYSIS_AXES order, and within
        # each axis its indices are contiguous and ascending.
        prev_end = 0
        for axis_name in lt.MUSIC_ANALYSIS_AXES.keys():
            indices = by_axis[axis_name]
            assert indices == list(range(prev_end, prev_end + len(indices)))
            assert vec[indices].sum() == pytest.approx(1.0, abs=1e-5)
            prev_end += len(indices)

    def test_argmax_per_axis_points_to_targeted_label(self, lt):
        """End-to-end ordering proof: build axis embeddings where each axis
        has ONE specific label whose row matches the query embedding exactly,
        then assert that the argmax position inside that axis's chunk in the
        produced vector is the index of that targeted label.

        This catches any off-by-one or accidental shuffling between
        ``_score_axes``' internal iteration and ``axis_columns()`` ordering.
        """
        rng = np.random.default_rng(7)
        emb_dim = 8
        query = rng.standard_normal(emb_dim).astype(np.float32)
        query /= np.linalg.norm(query)

        # Per axis, hand-pick a winning label (varied positions across axes).
        winners = {
            'AXIS_1_SETTING': 'TRANSIT',           # idx 3 of 6
            'AXIS_2_SOCIAL_DYNAMIC': 'DIVINE',     # idx 5 of 6 (last)
            'AXIS_3_EMOTIONAL_VALENCE': 'RADIANT', # idx 0 of 6 (first)
            'AXIS_4_NARRATIVE_TEMPORALITY': 'EXISTENTIAL',  # idx 2 of 5
            'AXIS_5_THEMATIC_WEIGHT': 'POLITICAL', # idx 2 of 4
        }

        label_map = {}
        axis_embeddings = {}
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            labels = [(name, name.lower()) for name in meta['labels'].keys()]
            label_map[axis_name] = labels
            mat = rng.standard_normal((len(labels), emb_dim)).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
            # Force the targeted label's row to equal the query (max dot product).
            target_idx = [name for name, _ in labels].index(winners[axis_name])
            mat[target_idx] = query
            axis_embeddings[axis_name] = mat

        with patch.object(lt, '_get_axis_embeddings',
                          return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(query)

        cols = lt.axis_columns()
        offset = 0
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            n = len(meta['labels'])
            chunk = vec[offset:offset + n]
            argmax_global = offset + int(np.argmax(chunk))
            actual_axis, actual_label = cols[argmax_global]
            assert actual_axis == axis_name, (
                f'Argmax for {axis_name} landed in {actual_axis}'
            )
            assert actual_label == winners[axis_name], (
                f'{axis_name}: expected winning label {winners[axis_name]!r}, '
                f'got {actual_label!r} (vector ordering mismatch)'
            )
            offset += n


# ---------------------------------------------------------------------------
# _sanitize_lyrics_text(): real cleanup, no mocks.
# ---------------------------------------------------------------------------

class TestSanitizeLyricsText:
    def test_empty_input_returns_empty(self, lt):
        assert lt._sanitize_lyrics_text('') == ''
        assert lt._sanitize_lyrics_text(None) == ''

    def test_strips_control_chars_and_zero_width(self, lt):
        text = 'hello\u200bworld\ufefftest\x00\x07end'
        out = lt._sanitize_lyrics_text(text)
        assert '\u200b' not in out
        assert '\ufeff' not in out
        assert '\x00' not in out
        assert '\x07' not in out
        assert 'hello' in out and 'world' in out and 'end' in out

    def test_strips_html_tags(self, lt):
        text = 'line one <b>bold</b> line\n<script>alert(1)</script>after'
        out = lt._sanitize_lyrics_text(text)
        assert '<b>' not in out and '</b>' not in out
        assert '<script>' not in out
        assert 'alert(1)' not in out
        assert 'bold' in out
        assert 'line one' in out and 'after' in out

    def test_truncates_to_max_words(self, lt):
        text = ' '.join(['word'] * 500)
        out = lt._sanitize_lyrics_text(text, max_words=100)
        assert len(out.split()) == 100

    def test_collapses_blank_line_runs(self, lt):
        text = 'a\n\n\n\nb\n\n\n\n\nc'
        out = lt._sanitize_lyrics_text(text)
        # No more than one consecutive blank line.
        lines = out.split('\n')
        for i in range(len(lines) - 1):
            if lines[i] == '' and lines[i + 1] == '':
                pytest.fail(f'Multiple consecutive blank lines: {lines!r}')


# ---------------------------------------------------------------------------
# _strip_lrc_timestamps(): real behaviour.
# ---------------------------------------------------------------------------

class TestStripLrcTimestamps:
    def test_strips_leading_timestamps(self, lt):
        lrc = '[00:01.23]first line\n[00:05.67]second line'
        out = lt._strip_lrc_timestamps(lrc)
        assert out == 'first line\nsecond line'

    def test_drops_empty_lines_after_stripping(self, lt):
        lrc = '[00:01.00]\n[00:02.00]only content\n[00:03.00]'
        out = lt._strip_lrc_timestamps(lrc)
        assert out == 'only content'


# ---------------------------------------------------------------------------
# _softmax(): basic numeric sanity.
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self, lt):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = lt._softmax(v, temperature=0.5)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_monotonic_under_positive_temperature(self, lt):
        v = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        out = lt._softmax(v, temperature=0.1)
        assert out[0] < out[1] < out[2]

    def test_empty_input_returns_empty(self, lt):
        v = np.array([], dtype=np.float32)
        out = lt._softmax(v, temperature=0.1)
        assert out.size == 0
