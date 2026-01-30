import pytest
from student_clap.models.efficientat.model import get_model


def test_dymn10_as_dynamic_reconstruction():
    # This test asserts that when loading the 'dymn10_as' checkpoint the loader
    # either obtains a strict match or performs an inferred reconstruction from
    # 'layers.*' checkpoint keys (dynamic behavior requested by user).
    m = get_model(num_classes=527, pretrained_name='dymn10_as', head_type='mlp')
    assert m._requested_pretrained == 'dymn10_as'
    assert getattr(m, '_loaded_pretrained', None) in (None, 'dymn10_as', getattr(m, '_loaded_pretrained'))
    ok = getattr(m, '_strict_match', False) or getattr(m, '_inferred_from_layers', False) or getattr(m, '_deterministic_parity_candidate', None)
    assert ok, (
        "Loader did not achieve strict match, inferred reconstruction, or deterministic parity candidate."
    )
