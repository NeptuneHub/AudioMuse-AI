from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_I8 = 2

_CODE_TO_NAME = {DTYPE_F32: "f32", DTYPE_F16: "f16", DTYPE_I8: "i8"}
_NAME_TO_CODE = {v: k for k, v in _CODE_TO_NAME.items()}
_CODE_TO_NP = {DTYPE_F32: np.float32, DTYPE_F16: np.float16, DTYPE_I8: np.int8}

I8_SCALE = np.float32(127.0)

try:
    import numkong as _nk

    HAVE_NUMKONG = True
except Exception:  # pragma: no cover - exercised only on builds without the wheel
    _nk = None
    HAVE_NUMKONG = False


def dtype_code(name) -> int:
    return _NAME_TO_CODE.get((name or "f32").lower(), DTYPE_F32)


def dtype_name(code) -> str:
    return _CODE_TO_NAME.get(int(code), "f32")


def np_dtype(code):
    return _CODE_TO_NP.get(int(code), np.float32)


def elem_size(code) -> int:
    return int(np.dtype(np_dtype(code)).itemsize)


def effective_code(requested_code, metric) -> int:
    code = int(requested_code)
    if code == DTYPE_I8 and (metric or "angular").lower() != "angular":
        return DTYPE_F16
    return code


def encode_vectors(vecs_f32, code) -> np.ndarray:
    if code == DTYPE_I8:
        scaled = np.rint(np.asarray(vecs_f32, dtype=np.float32) * I8_SCALE)
        return np.clip(scaled, -127, 127).astype(np.int8)
    if code == DTYPE_F16:
        return np.ascontiguousarray(vecs_f32, dtype=np.float16)
    return np.ascontiguousarray(vecs_f32, dtype=np.float32)


def decode_row(v, code) -> np.ndarray:
    if code == DTYPE_I8:
        return np.asarray(v, dtype=np.float32) / I8_SCALE
    return np.array(v, dtype=np.float32)


def prepare_query(q_f32, code, metric) -> np.ndarray:
    q = np.asarray(q_f32, dtype=np.float32).reshape(-1)
    if (metric or "angular").lower() == "angular":
        q = q / (float(np.linalg.norm(q)) + 1e-12)
    if code == DTYPE_I8:
        return np.clip(np.rint(q * I8_SCALE), -127, 127).astype(np.int8)
    if code == DTYPE_F16:
        return q.astype(np.float16)
    return q.astype(np.float32)


def cell_distances(metric, code, qp, vecs, normalized) -> np.ndarray:
    if vecs.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    if code != DTYPE_F32 and HAVE_NUMKONG:
        try:
            return _cell_distances_nk(metric, qp, vecs)
        except Exception as e:  # pragma: no cover - defensive fallback
            logger.warning("NumKong distance failed (%s); using NumPy for this scan.", e)
    return _cell_distances_np(metric, code, qp, vecs, normalized)


def _cell_distances_nk(metric, qp, vecs) -> np.ndarray:
    metric = (metric or "angular").lower()
    q2 = np.ascontiguousarray(qp).reshape(1, -1)
    if metric == "euclidean":
        d = _nk.cdist(vecs, q2, metric="euclidean")
        return np.asarray(d, dtype=np.float32).reshape(-1)
    if metric == "dot":
        d = _nk.cdist(vecs, q2, metric="dot")
        return (-np.asarray(d, dtype=np.float32).reshape(-1)).astype(np.float32)
    d = _nk.cdist(vecs, q2, metric="angular")
    return np.asarray(d, dtype=np.float32).reshape(-1)


def _cell_distances_np(metric, code, qp, vecs, normalized) -> np.ndarray:
    metric = (metric or "angular").lower()
    q = decode_row(qp, code)
    v = (
        np.asarray(vecs, dtype=np.float32)
        if code != DTYPE_I8
        else (np.asarray(vecs, dtype=np.float32) / I8_SCALE)
    )
    if metric == "euclidean":
        diffs = v - q[None, :]
        return np.sqrt(np.einsum("ij,ij->i", diffs, diffs)).astype(np.float32)
    if metric == "dot":
        return (-(v @ q)).astype(np.float32)
    if normalized and code == DTYPE_F32:
        return (1.0 - np.clip(v @ q, -1.0, 1.0)).astype(np.float32)
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True).astype(np.float32) + 1e-12)
    qn = q / (float(np.linalg.norm(q)) + 1e-12)
    return (1.0 - np.clip(vn @ qn, -1.0, 1.0)).astype(np.float32)
