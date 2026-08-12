# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the Hyperbolic Explorer page route.

Verifies that GET /hyperbolic renders through the Flask blueprint with the
expected template context and answers 200, without touching the database or
the real template (render_template is patched), following the repo's
unit-test conventions.
"""

from unittest.mock import patch

from flask import Flask

import app_hyperbolic


def test_hyperbolic_page_renders_template():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(app_hyperbolic.hyperbolic_bp)
    with patch("app_hyperbolic.render_template", return_value="<html>ok</html>") as mock_rt:
        response = app.test_client().get("/hyperbolic")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "<html>ok</html>"
    mock_rt.assert_called_once()
    kwargs = mock_rt.call_args.kwargs
    assert kwargs["title"] == "AudioMuse-AI - Hyperbolic Explorer"
    assert kwargs["active"] == "hyperbolic"
    assert kwargs["app_version"] == "v3.2.0"
