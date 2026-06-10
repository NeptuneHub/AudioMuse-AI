import pytest

import config


def test_obsolete_fields_has_same_keys_as_fields_by_type():
    assert set(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE) == set(
        config.MEDIASERVER_FIELDS_BY_TYPE
    )


@pytest.mark.parametrize(
    'media_type', sorted(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE)
)
def test_obsolete_fields_are_union_of_other_types(media_type):
    all_fields = set()
    for fields in config.MEDIASERVER_FIELDS_BY_TYPE.values():
        all_fields.update(fields)
    own_fields = set(config.MEDIASERVER_FIELDS_BY_TYPE[media_type])
    obsolete = config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE[media_type]
    assert set(obsolete) == all_fields - own_fields


@pytest.mark.parametrize(
    'media_type', sorted(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE)
)
def test_obsolete_fields_never_include_own_fields(media_type):
    obsolete = set(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE[media_type])
    own_fields = set(config.MEDIASERVER_FIELDS_BY_TYPE[media_type])
    assert obsolete.isdisjoint(own_fields)
