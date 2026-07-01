# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-provider LLM API client subpackage.

Collects the concrete transport clients that ``tasks.ai.api`` dispatches to:
``gemini`` (google-genai), ``mistral`` (mistralai SDK), and ``openai`` (the
OpenAI-compatible HTTP path shared by OpenAI, OpenRouter, and Ollama). Each
module exposes ``generate_text`` and ``call_with_tools`` with a uniform shape.

Main Features:
* Namespace package only - callers import ``gemini``/``mistral``/``openai`` directly; nothing is re-exported here.
* Isolates vendor-specific request/response quirks so the routing layer stays provider-agnostic.
"""
