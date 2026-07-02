# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""LLM playlist tool-calling and brainstorm subpackage.

Houses the AI layer that turns a natural-language playlist request into a
plan of tool calls run against the real library. Submodules split by role:
``api`` (provider routing + text cleanup), ``providers`` (per-vendor
clients), ``planner`` (single-call plan/execute), ``prompts``,
``tools``/``tool_impl`` (schemas + grounded implementations), and ``vocab``.

Main Features:
* Namespace package only - each submodule is imported explicitly; nothing is re-exported here.
* Tool-calling is single-turn (no multi-turn context) - the plan is emitted in one response and executed locally.
"""
