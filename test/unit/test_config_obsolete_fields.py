# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Consistency of the mediaserver obsolete-fields config maps.

Asserts MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE is derived correctly from
MEDIASERVER_FIELDS_BY_TYPE so per-type field cleanup stays coherent.

Main Features:
* Both maps cover the same media types
* Each type's obsolete set equals the union of other types' fields minus its own
* A type's obsolete set never includes its own active fields
"""

import pytest

import config


def test_obsolete_fields_has_same_keys_as_fields_by_type():
    assert set(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE) == set(config.MEDIASERVER_FIELDS_BY_TYPE)


@pytest.mark.parametrize('media_type', sorted(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE))
def test_obsolete_fields_are_union_of_other_types(media_type):
    all_fields = set()
    for fields in config.MEDIASERVER_FIELDS_BY_TYPE.values():
        all_fields.update(fields)
    own_fields = set(config.MEDIASERVER_FIELDS_BY_TYPE[media_type])
    obsolete = config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE[media_type]
    assert set(obsolete) == all_fields - own_fields


@pytest.mark.parametrize('media_type', sorted(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE))
def test_obsolete_fields_never_include_own_fields(media_type):
    obsolete = set(config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE[media_type])
    own_fields = set(config.MEDIASERVER_FIELDS_BY_TYPE[media_type])
    assert obsolete.isdisjoint(own_fields)
