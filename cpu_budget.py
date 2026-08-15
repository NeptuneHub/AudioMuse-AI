# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Container-aware CPU count for the worker and ONNX thread caps.

Resolves how many CPUs this process may really use, preferring the cgroup quota
that Docker and Kubernetes write over the host-wide os.cpu_count(), which is
blind to a --cpus limit. Every caller keeps its own formula and its own floor;
this module only decides which number goes into that formula, and any doubt at
all resolves to the caller's existing fallback.

Imports nothing outside the standard library on purpose. The worker entrypoint
calls it before importing any numeric library, because a BLAS thread cap set
after numpy loads has no effect at all, and pulling in joblib here would load
numpy and silently defeat the very cap it was meant to size.

Main Features:
* Reads cgroup v2 cpu.max and the cgroup v1 cfs_quota_us/cfs_period_us pair,
  mirroring joblib.externals.loky.backend.context without importing it.
* Honours a CPU affinity mask where the platform exposes one, so --cpuset-cpus
  and taskset are respected too.
* Takes the smallest of quota, affinity and os.cpu_count(), raising a real but
  tiny reading up to the caller's minimum rather than discarding it.
* Reports at INFO and hands back the caller's fallback on any failure or on a
  value that makes no sense at all, so a broken probe can only ever reproduce
  today's behaviour.
* usable_cpu_count feeds the ONNX intra_op_num_threads, the only channel
  onnxruntime reads, since it ignores the BLAS environment variables, the
  affinity mask and the cgroup quota alike. It answers only when a real
  restriction was found below the host count; with no restriction it returns
  None and the caller leaves onnxruntime's own default untouched, so an
  unconstrained host keeps exactly the thread count it has today.
"""

import logging
import os

logger = logging.getLogger(__name__)

_CGROUP_V2_CPU_MAX = '/sys/fs/cgroup/cpu.max'
_CGROUP_V1_QUOTA = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'
_CGROUP_V1_PERIOD = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'


def _read_text(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def _cgroup_cpu_quota():
    raw = _read_text(_CGROUP_V2_CPU_MAX)
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != 'max':
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return -(-quota // period)
        return None

    quota_raw = _read_text(_CGROUP_V1_QUOTA)
    period_raw = _read_text(_CGROUP_V1_PERIOD)
    if quota_raw and period_raw:
        try:
            quota, period = int(quota_raw), int(period_raw)
        except ValueError:
            return None
        if quota > 0 and period > 0:
            return -(-quota // period)
    return None


def _affinity_cpu_count():
    getaffinity = getattr(os, 'sched_getaffinity', None)
    if getaffinity is None:
        return None
    return len(getaffinity(0))


def host_cpu_count():
    host = os.cpu_count()
    return host if isinstance(host, int) and host > 0 else None


def _probe_cpu_count():
    candidates = []
    for probe in (_cgroup_cpu_quota, _affinity_cpu_count, host_cpu_count):
        try:
            value = probe()
        except Exception:
            value = None
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    return min(candidates) if candidates else None


def usable_cpu_count():
    try:
        detected = _probe_cpu_count()
        host = host_cpu_count()
    except Exception:
        logger.info("CPU probe failed; leaving the ONNX thread count to the runtime")
        return None

    if not isinstance(detected, int) or detected <= 0:
        logger.info(
            "CPU probe returned %r, which is unusable; leaving the ONNX thread count to the runtime",
            detected,
        )
        return None

    if host is not None and detected >= host:
        return None

    logger.info("CPU probe found %s usable CPUs; capping ONNX to that many threads", detected)
    return detected


def detect_cpu_count(fallback, minimum, label=''):
    prefix = f"{label} " if label else ''
    try:
        detected = _probe_cpu_count()
    except Exception:
        logger.info("%sCPU probe failed; keeping the usual count of %s", prefix, fallback)
        return fallback, 'fallback (probe failed)'

    if not isinstance(detected, int) or detected <= 0:
        logger.info(
            "%sCPU probe returned %r, which is unusable; keeping the usual count of %s",
            prefix,
            detected,
            fallback,
        )
        return fallback, 'fallback (unusable value)'

    if detected < minimum:
        logger.info(
            "%sCPU probe found only %s usable CPUs; raising it to the minimum of %s",
            prefix,
            detected,
            minimum,
        )
        return minimum, 'detected (raised to minimum)'

    if detected == fallback:
        return detected, 'host'

    logger.info(
        "%sCPU probe found %s usable CPUs instead of the host's %s",
        prefix,
        detected,
        fallback,
    )
    return detected, 'detected'
