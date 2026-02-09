"""Unit tests for playlist ordering module

Tests cover the composite distance calculation, Circle of Fifths key distance,
greedy nearest-neighbor ordering algorithm, energy arc shaping, and edge cases.
All tests run without external services using unittest.mock for database calls.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock


def _import_ordering():
    """Import playlist_ordering directly, bypassing tasks/__init__.py which
    pulls in heavyweight deps (pydub, librosa) not needed for these tests."""
    import importlib.util, os, sys
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tasks', 'playlist_ordering.py'
    )
    mod_path = os.path.normpath(mod_path)
    if 'tasks.playlist_ordering' not in sys.modules:
        spec = importlib.util.spec_from_file_location('tasks.playlist_ordering', mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['tasks.playlist_ordering'] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules['tasks.playlist_ordering']
    return (
        mod._key_distance,
        mod._composite_distance,
        mod.order_playlist,
        mod._apply_energy_arc,
        mod.CIRCLE_OF_FIFTHS,
    )


@pytest.mark.unit
class TestKeyDistance:
    def test_same_key_zero_distance(self):
        kd, *_ = _import_ordering()
        assert kd("C", "major", "C", "major") == 0.0

    def test_same_key_different_scale_zero(self):
        kd, *_ = _import_ordering()
        assert kd("G", "major", "G", "minor") == 0.0

    def test_adjacent_key_c_to_g(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "G", None) == pytest.approx(1.0/6.0)

    def test_adjacent_key_c_to_f(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "F", None) == pytest.approx(1.0/6.0)

    def test_opposite_key_c_to_fsharp(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "F#", None) == pytest.approx(1.0)

    def test_opposite_key_c_to_gb(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "Gb", None) == pytest.approx(1.0)

    def test_two_steps_c_to_d(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "D", None) == pytest.approx(2.0/6.0)

    def test_three_steps_c_to_a(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "A", None) == pytest.approx(0.5)

    def test_four_steps_c_to_e(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "E", None) == pytest.approx(4.0/6.0)

    def test_five_steps_c_to_b(self):
        kd, *_ = _import_ordering()
        assert kd("C", None, "B", None) == pytest.approx(5.0/6.0)

    def test_symmetry(self):
        kd, *_ = _import_ordering()
        assert kd("A", None, "E", None) == kd("E", None, "A", None)

    def test_same_scale_bonus_reduces_distance(self):
        kd, *_ = _import_ordering()
        d_no = kd("C", None, "D", None)
        d_same = kd("C", "major", "D", "major")
        assert d_same == pytest.approx(d_no * 0.8)

    def test_different_scale_no_bonus(self):
        kd, *_ = _import_ordering()
        d_diff = kd("C", "major", "D", "minor")
        d_no = kd("C", None, "D", None)
        assert d_diff == pytest.approx(d_no)

    def test_missing_key1_returns_neutral(self):
        kd, *_ = _import_ordering()
        assert kd(None, "major", "C", "major") == 0.5
        assert kd("", "major", "C", "major") == 0.5

    def test_missing_key2_returns_neutral(self):
        kd, *_ = _import_ordering()
        assert kd("C", "major", None, "major") == 0.5
        assert kd("C", "major", "", "major") == 0.5

    def test_both_keys_missing_returns_neutral(self):
        kd, *_ = _import_ordering()
        assert kd(None, None, None, None) == 0.5

    def test_unknown_key_name_returns_neutral(self):
        kd, *_ = _import_ordering()
        assert kd("X", None, "C", None) == 0.5
        assert kd("C", None, "Z", None) == 0.5

    def test_case_insensitive_keys(self):
        kd, *_ = _import_ordering()
        assert kd("c", None, "g", None) == kd("C", None, "G", None)

    def test_sharp_flat_enharmonic_equivalents(self):
        kd, *_ = _import_ordering()
        assert kd("C#", None, "Db", None) == 0.0
        assert kd("D#", None, "Eb", None) == 0.0
        assert kd("G#", None, "Ab", None) == 0.0
        assert kd("A#", None, "Bb", None) == 0.0

    def test_scale_comparison_case_insensitive(self):
        kd, *_ = _import_ordering()
        d1 = kd("C", "major", "D", "major")
        d2 = kd("C", "Major", "D", "MAJOR")
        assert d1 == pytest.approx(d2)


@pytest.mark.unit
class TestCompositeDistance:
    def test_identical_songs_zero_distance(self):
        _, cd, *_ = _import_ordering()
        s = {"tempo": 120, "energy": 0.08, "key": "C", "scale": "major"}
        assert cd(s, s) == 0.0

    def test_tempo_weight_35_percent(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 100, "energy": 0.08, "key": "C", "scale": "major"}
        b = {"tempo": 180, "energy": 0.08, "key": "C", "scale": "major"}
        assert cd(a, b) == pytest.approx(0.35)

    def test_energy_weight_35_percent(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 120, "energy": 0.01, "key": "C", "scale": "major"}
        b = {"tempo": 120, "energy": 0.15, "key": "C", "scale": "major"}
        assert cd(a, b) == pytest.approx(0.35)

    def test_key_weight_30_percent(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 120, "energy": 0.08, "key": "C", "scale": "major"}
        b = {"tempo": 120, "energy": 0.08, "key": "F#", "scale": "minor"}
        assert cd(a, b) == pytest.approx(0.30)

    def test_max_distance_all_features_differ(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 80, "energy": 0.01, "key": "C", "scale": "major"}
        b = {"tempo": 160, "energy": 0.15, "key": "F#", "scale": "minor"}
        assert cd(a, b) == pytest.approx(1.0)

    def test_tempo_normalised_by_80bpm(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 100, "energy": 0, "key": "", "scale": ""}
        b = {"tempo": 140, "energy": 0, "key": "", "scale": ""}
        assert cd(a, b) == pytest.approx(0.35*0.5 + 0.30*0.5)

    def test_tempo_diff_capped_at_one(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 60, "energy": 0.08, "key": "C", "scale": "major"}
        b = {"tempo": 200, "energy": 0.08, "key": "C", "scale": "major"}
        assert cd(a, b) == pytest.approx(0.35)

    def test_energy_diff_capped_at_one(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 120, "energy": 0.0, "key": "C", "scale": "major"}
        b = {"tempo": 120, "energy": 0.5, "key": "C", "scale": "major"}
        assert cd(a, b) == pytest.approx(0.35)

    def test_missing_tempo_treated_as_zero(self):
        _, cd, *_ = _import_ordering()
        a = {"energy": 0.08, "key": "C", "scale": "major"}
        b = {"tempo": 80, "energy": 0.08, "key": "C", "scale": "major"}
        assert cd(a, b) == pytest.approx(0.35)

    def test_missing_energy_treated_as_zero(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 120, "key": "C", "scale": "major"}
        b = {"tempo": 120, "energy": 0.07, "key": "C", "scale": "major"}
        assert cd(a, b) == pytest.approx(0.35*0.5)

    def test_missing_key_gives_neutral(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 120, "energy": 0.08}
        b = {"tempo": 120, "energy": 0.08}
        assert cd(a, b) == pytest.approx(0.30*0.5)

    def test_custom_weights(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 100, "energy": 0.01, "key": "C", "scale": "major"}
        b = {"tempo": 180, "energy": 0.15, "key": "F#", "scale": "minor"}
        assert cd(a, b, w_tempo=0.5, w_energy=0.3, w_key=0.2) == pytest.approx(1.0)

    def test_symmetry(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 90, "energy": 0.05, "key": "D", "scale": "minor"}
        b = {"tempo": 140, "energy": 0.12, "key": "Ab", "scale": "major"}
        assert cd(a, b) == cd(b, a)

    def test_partial_distance_contribution(self):
        _, cd, *_ = _import_ordering()
        a = {"tempo": 120, "energy": 0.08, "key": "C", "scale": "major"}
        b = {"tempo": 160, "energy": 0.08, "key": "G", "scale": "major"}
        expected = 0.35*0.5 + 0.30*(1.0/6.0*0.8)
        assert cd(a, b) == pytest.approx(expected)


def _make_mock_db_rows(sd):
    rows = []
    for iid, data in sd.items():
        row = dict(data)
        row["item_id"] = iid
        rows.append(row)
    return rows

def _patch_order_playlist(sd):
    """Patch DB calls for order_playlist.  Pre-register mock modules
    so the lazy imports inside order_playlist() never trigger the heavy
    tasks/__init__.py import chain."""
    import sys
    rows = _make_mock_db_rows(sd)
    mc = MagicMock()
    mc.fetchall.return_value = rows
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = Mock(return_value=mc)
    conn.cursor.return_value.__exit__ = Mock(return_value=None)
    # Pre-register lightweight mocks for modules imported inside order_playlist()
    if 'tasks.mcp_server' not in sys.modules:
        mock_mcp = MagicMock()
        sys.modules['tasks.mcp_server'] = mock_mcp
    if 'psycopg2' not in sys.modules:
        sys.modules['psycopg2'] = MagicMock()
    if 'psycopg2.extras' not in sys.modules:
        sys.modules['psycopg2.extras'] = MagicMock()
    sys.modules['tasks.mcp_server'].get_db_connection = Mock(return_value=conn)
    return patch.object(sys.modules['tasks.mcp_server'], 'get_db_connection', return_value=conn)


@pytest.mark.unit
class TestOrderPlaylist:
    def test_empty_list_returns_empty(self):
        _, _, op, *_ = _import_ordering()
        with _patch_order_playlist({}):
            assert op([]) == []

    def test_single_song_returns_unchanged(self):
        _, _, op, *_ = _import_ordering()
        assert op(["song1"]) == ["song1"]

    def test_two_songs_returns_both(self):
        _, _, op, *_ = _import_ordering()
        assert op(["song1", "song2"]) == ["song1", "song2"]

    def test_all_input_songs_in_output(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 80+i*10, "energy": 0.02+i*0.02, "key": "C", "scale": "major"} for i in range(10)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            result = op(ids)
        assert set(result) == set(ids) and len(result) == len(ids)

    def test_no_duplicates_in_output(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 100+i*5, "energy": 0.05+i*0.01, "key": "G", "scale": "minor"} for i in range(15)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            result = op(ids)
        assert len(result) == len(set(result))

    def test_starts_from_25th_percentile_energy(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.02+i*0.01, "key": "C", "scale": "major"} for i in range(8)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            result = op(ids)
        first_e = sd[result[0]]["energy"]
        sorted_e = sorted(sd[s]["energy"] for s in ids)
        expected_e = sorted_e[len(sorted_e) // 4]
        assert first_e == pytest.approx(expected_e)

    def test_adjacent_songs_have_small_distance(self):
        _, cd, op, *_ = _import_ordering()
        import random
        random.seed(42)
        kl = ["C","G","D","A","E","B","F#","Db","Ab","Eb"]
        sd = {f"s{i}": {"tempo": 80+i*8, "energy": 0.02+i*0.012, "key": kl[i%10], "scale": "major" if i%2==0 else "minor"} for i in range(12)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            ordered = op(ids)
        def td(seq):
            return sum(cd(sd[seq[i]], sd[seq[i+1]]) for i in range(len(seq)-1))
        od = td(ordered)
        rd = []
        for _ in range(20):
            s = list(ids)
            random.shuffle(s)
            rd.append(td(s))
        assert od <= sum(rd)/len(rd)

    def test_unorderable_songs_appended_at_end(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 100+i*10, "energy": 0.05+i*0.02, "key": "C", "scale": "major"} for i in range(3)}
        ids = ["s0", "s1", "s2", "s_missing"]
        with _patch_order_playlist(sd):
            result = op(ids)
        assert result[-1] == "s_missing"
        assert set(result) == set(ids)

    def test_no_db_rows_returns_original_order(self):
        _, _, op, *_ = _import_ordering()
        ids = ["a", "b", "c"]
        with _patch_order_playlist({}):
            assert op(ids) == ids

    def test_only_two_orderable_returns_original(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": 100, "energy": 0.05, "key": "C", "scale": "major"},
            "s1": {"tempo": 110, "energy": 0.06, "key": "G", "scale": "major"},
        }
        ids = ["s0", "s1", "s_no_data"]
        with _patch_order_playlist(sd):
            assert op(ids) == ids


@pytest.mark.unit
class TestEnergyArc:
    def test_energy_arc_false_deterministic(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.01+i*0.013, "key": "C", "scale": "major"} for i in range(12)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            r1 = op(ids, energy_arc=False)
        with _patch_order_playlist(sd):
            r2 = op(ids, energy_arc=False)
        assert r1 == r2

    def test_energy_arc_true_reshapes_10_plus(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.01+i*0.013, "key": "C", "scale": "major"} for i in range(12)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            r_arc = op(ids, energy_arc=True)
        with _patch_order_playlist(sd):
            r_no = op(ids, energy_arc=False)
        assert r_arc != r_no
        assert set(r_arc) == set(r_no)

    def test_energy_arc_skipped_under_10(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.01+i*0.02, "key": "C", "scale": "major"} for i in range(8)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            r1 = op(ids, energy_arc=True)
        with _patch_order_playlist(sd):
            r2 = op(ids, energy_arc=False)
        assert r1 == r2

    def test_apply_energy_arc_peak_in_middle(self):
        *_, ea, _ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.01+i*0.012, "key": "C", "scale": "major"} for i in range(12)}
        ids = [f"s{i}" for i in range(12)]
        arc = ea(ids, sd)
        assert set(arc) == set(ids) and len(arc) == 12
        energies = [sd[s]["energy"] for s in arc]
        n = len(energies)
        fq = sum(energies[:n//4]) / (n//4)
        mid = sum(energies[n//3:2*n//3]) / (2*n//3 - n//3)
        lq = sum(energies[3*n//4:]) / (n - 3*n//4)
        assert mid > fq and mid > lq

    def test_apply_energy_arc_preserves_all(self):
        *_, ea, _ = _import_ordering()
        sd = {f"s{i}": {"tempo": 100, "energy": 0.01+i*0.01, "key": "D", "scale": "minor"} for i in range(15)}
        ids = list(sd.keys())
        arc = ea(ids, sd)
        assert set(arc) == set(ids) and len(arc) == len(ids)

    def test_apply_energy_arc_exact_10(self):
        *_, ea, _ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.01+i*0.014, "key": "C", "scale": "major"} for i in range(10)}
        ids = list(sd.keys())
        arc = ea(ids, sd)
        assert set(arc) == set(ids) and len(arc) == 10


@pytest.mark.unit
class TestEdgeCases:
    def test_songs_with_missing_tempo(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": None, "energy": 0.05, "key": "C", "scale": "major"},
            "s1": {"tempo": 120, "energy": 0.06, "key": "G", "scale": "major"},
            "s2": {"tempo": None, "energy": 0.07, "key": "D", "scale": "minor"},
        }
        with _patch_order_playlist(sd):
            assert set(op(list(sd.keys()))) == set(sd.keys())

    def test_songs_with_missing_energy(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": 100, "energy": None, "key": "C", "scale": "major"},
            "s1": {"tempo": 110, "energy": 0.05, "key": "G", "scale": "major"},
            "s2": {"tempo": 120, "energy": None, "key": "D", "scale": "minor"},
        }
        with _patch_order_playlist(sd):
            assert set(op(list(sd.keys()))) == set(sd.keys())

    def test_songs_with_missing_key(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": 100, "energy": 0.05, "key": None, "scale": None},
            "s1": {"tempo": 110, "energy": 0.06, "key": "", "scale": ""},
            "s2": {"tempo": 120, "energy": 0.07, "key": "C", "scale": "major"},
        }
        with _patch_order_playlist(sd):
            assert set(op(list(sd.keys()))) == set(sd.keys())

    def test_songs_with_all_missing_attributes(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": None, "energy": None, "key": None, "scale": None},
            "s1": {"tempo": None, "energy": None, "key": None, "scale": None},
            "s2": {"tempo": None, "energy": None, "key": None, "scale": None},
        }
        with _patch_order_playlist(sd):
            assert set(op(list(sd.keys()))) == set(sd.keys())

    def test_all_songs_same_bpm_energy_key(self):
        _, _, op, *_ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.08, "key": "C", "scale": "major"} for i in range(6)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            result = op(ids)
        assert set(result) == set(ids) and len(result) == 6

    @pytest.mark.slow
    def test_large_playlist_completes(self):
        _, _, op, *_ = _import_ordering()
        n = 120
        keys = ["C","G","D","A","E","B","F#","Db","Ab","Eb","Bb","F"]
        sd = {f"s{i}": {"tempo": 70+(i*7)%130, "energy": 0.01+(i*0.0012)%0.14, "key": keys[i%12], "scale": "major" if i%2==0 else "minor"} for i in range(n)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            result = op(ids)
        assert len(result) == n and set(result) == set(ids)

    @pytest.mark.slow
    def test_large_playlist_with_energy_arc(self):
        _, _, op, *_ = _import_ordering()
        n = 100
        keys = ["C","G","D","A","E","B","F#","Db","Ab","Eb","Bb","F"]
        sd = {f"s{i}": {"tempo": 80+(i*5)%100, "energy": 0.01+(i*0.0014)%0.14, "key": keys[i%12], "scale": "major" if i%3!=0 else "minor"} for i in range(n)}
        ids = list(sd.keys())
        with _patch_order_playlist(sd):
            result = op(ids, energy_arc=True)
        assert len(result) == n and set(result) == set(ids)

    def test_duplicate_ids_in_input(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": 100, "energy": 0.05, "key": "C", "scale": "major"},
            "s1": {"tempo": 110, "energy": 0.06, "key": "G", "scale": "major"},
            "s2": {"tempo": 120, "energy": 0.07, "key": "D", "scale": "minor"},
        }
        with _patch_order_playlist(sd):
            result = op(["s0", "s1", "s2", "s0"])
        assert len(result) >= 3

    def test_zero_tempo_and_energy_songs(self):
        _, _, op, *_ = _import_ordering()
        sd = {
            "s0": {"tempo": 0, "energy": 0, "key": "C", "scale": "major"},
            "s1": {"tempo": 0, "energy": 0, "key": "G", "scale": "major"},
            "s2": {"tempo": 0, "energy": 0, "key": "D", "scale": "minor"},
        }
        with _patch_order_playlist(sd):
            assert set(op(list(sd.keys()))) == set(sd.keys())


@pytest.mark.unit
class TestCircleOfFifthsMap:
    def test_all_12_chromatic_notes_mapped(self):
        *_, cof = _import_ordering()
        assert set(cof.values()) == set(range(12))

    def test_enharmonic_pairs_same_position(self):
        *_, cof = _import_ordering()
        for a, b in [("F#","GB"),("C#","DB"),("G#","AB"),("D#","EB"),("A#","BB")]:
            assert cof[a] == cof[b], f"{a} and {b} should be equal"

    def test_c_is_position_zero(self):
        *_, cof = _import_ordering()
        assert cof["C"] == 0

    def test_g_is_position_one(self):
        *_, cof = _import_ordering()
        assert cof["G"] == 1

    def test_f_is_position_eleven(self):
        *_, cof = _import_ordering()
        assert cof["F"] == 11


@pytest.mark.unit
class TestApplyEnergyArcDirect:
    def test_low_energy_at_start_and_end(self):
        *_, ea, _ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": float(i)} for i in range(12)}
        ids = [f"s{i}" for i in range(12)]
        arc = ea(ids, sd)
        energies = [sd[s]["energy"] for s in arc]
        assert energies[0] < 4.0 and energies[-1] < 4.0

    def test_high_energy_in_middle(self):
        *_, ea, _ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": float(i)} for i in range(15)}
        ids = [f"s{i}" for i in range(15)]
        arc = ea(ids, sd)
        energies = [sd[s]["energy"] for s in arc]
        n = len(energies)
        mid_sec = energies[n//3:2*n//3]
        assert sum(mid_sec)/len(mid_sec) > sum(energies)/len(energies)

    def test_arc_with_identical_energies(self):
        *_, ea, _ = _import_ordering()
        sd = {f"s{i}": {"tempo": 120, "energy": 0.08} for i in range(12)}
        ids = [f"s{i}" for i in range(12)]
        arc = ea(ids, sd)
        assert set(arc) == set(ids) and len(arc) == 12
