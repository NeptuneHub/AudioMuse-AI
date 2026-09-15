# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One retry decision for every task, and the backoff between two attempts.

The worker used to retry exactly one thing: a worker death. A task that raised
was failed on the spot, with attempts and max_attempts never consulted, and a
FAIL row is unreachable afterwards because the claim reads NEW rows and reclaim
reads RUNNING ones. So a media-server 502 at 03:00 ended an overnight run.

Now every outcome passes through decide, which is the only place that knows
whether a failed attempt gets another one. The worker carries the distinction
between a retryable failure, a permanent one and a cooperative cancel in the
status slot of the same 3-tuple its job child already reports over the pipe,
using the three sentinels below, and row_status maps them back before anything
reaches the database. No sentinel is ever stored.

One counter, attempts, covers worker deaths AND application errors, which is
what every comparable queue does. The consequence is deliberate: two worker
deaths plus one transient error exhausts QUEUE_MAX_ATTEMPTS, and the wedged-main
nudge spends from the same budget because it ends the worker and reclaim then
charges the attempt.

backoff_seconds spaces the attempts out so a deterministic failure cannot burn
its whole budget in a few milliseconds. The clock and the random source are
injectable so a test can pin the schedule.

Main Features:
* decide: RETRY only for a retryable failure with an attempt left, else FINISH
* row_status: the sentinel-free status the terminal row is written with
* backoff_seconds: doubling from QUEUE_RETRY_BASE_SECONDS, capped at
  QUEUE_RETRY_MAX_SECONDS, with a small jitter so retries never line up
"""

import random

import config

FAIL_RETRYABLE = 'FAIL_RETRYABLE'
FAIL_PERMANENT = 'FAIL_PERMANENT'
REVOKED_BY_TASK = 'REVOKED_BY_TASK'

RETRY = 'retry'
FINISH = 'finish'

JITTER_FRACTION = 0.10


def decide(job, outcome):
    if outcome != FAIL_RETRYABLE:
        return FINISH
    max_attempts = int(job.get('max_attempts') or 0)
    if max_attempts <= 0:
        return FINISH
    if int(job.get('attempts') or 0) + 1 > max_attempts:
        return FINISH
    return RETRY


def row_status(outcome):
    if outcome in (FAIL_RETRYABLE, FAIL_PERMANENT):
        return config.TASK_STATUS_FAIL
    if outcome == REVOKED_BY_TASK:
        return config.TASK_STATUS_REVOKED
    return outcome


def backoff_seconds(attempt, rng=None):
    base = float(config.QUEUE_RETRY_BASE_SECONDS)
    cap = float(config.QUEUE_RETRY_MAX_SECONDS)
    if base <= 0:
        return 0.0
    raw = min(cap, base * (2 ** max(0, int(attempt) - 1)))
    spread = (rng or random).uniform(-JITTER_FRACTION, JITTER_FRACTION)
    return max(0.0, raw * (1.0 + spread))
