from __future__ import annotations

import logging
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LYRICS_WHISPER_MIN_FREE_RAM_GB = float(os.environ.get("LYRICS_WHISPER_MIN_FREE_RAM_GB", "2.5"))

LYRICS_WHISPER_LANG_CONFIDENCE = float(os.environ.get("LYRICS_WHISPER_LANG_CONFIDENCE", "0.7"))

WHISPER_MAX_NEW_TOKENS = int(os.environ.get("LYRICS_WHISPER_MAX_NEW_TOKENS", "180"))

WHISPER_REPETITION_PENALTY = float(os.environ.get("LYRICS_WHISPER_REPETITION_PENALTY", "1.15"))

WHISPER_NO_REPEAT_NGRAM = int(os.environ.get("LYRICS_WHISPER_NO_REPEAT_NGRAM", "3"))

WHISPER_COMPRESSION_RATIO_THRESHOLD = float(
    os.environ.get("LYRICS_WHISPER_COMPRESSION_RATIO", "2.4")
)


def _resolve_whisper_threads() -> int:
    raw = os.environ.get('LYRICS_WHISPER_INTRA_OP_THREADS', '').strip()
    if raw == '':
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count // 3)
    try:
        return max(0, int(raw))
    except ValueError:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count // 3)


WHISPER_INTRA_OP_THREADS = _resolve_whisper_threads()

from config import LYRICS_ASR_BEAM_SIZE as WHISPER_BEAM_SIZE

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
WHISPER_CHUNK_SAMPLES = SAMPLE_RATE * 30
WHISPER_FRAMES_PER_CHUNK = WHISPER_CHUNK_SAMPLES // HOP_LENGTH

SOT_TOKEN_ID = 50258
EOT_TOKEN_ID = 50257
TRANSLATE_TOKEN_ID = 50358
TRANSCRIBE_TOKEN_ID = 50359
NO_TIMESTAMPS_TOKEN_ID = 50363
NO_SPEECH_TOKEN_ID = 50362
LANGUAGE_TOKEN_START = 50259
LANGUAGE_TOKEN_END = 50358

_SUPPRESS_TOKEN_IDS: Tuple[int, ...] = tuple(range(LANGUAGE_TOKEN_START, LANGUAGE_TOKEN_END + 1))

_PAST_KEY_VALUES_PREFIX = 'past_key_values.'


def _log_softmax_row(logits_row: np.ndarray) -> np.ndarray:
    x = logits_row.astype(np.float64, copy=False)
    x = x - x.max()
    log_norm = np.log(np.exp(x).sum())
    return (x - log_norm).astype(np.float32, copy=False)


def _softmax_row(logits_row: np.ndarray) -> np.ndarray:
    x = logits_row.astype(np.float64, copy=False)
    x = x - x.max()
    exp = np.exp(x)
    return (exp / exp.sum()).astype(np.float32, copy=False)


def _apply_repetition_penalty(
    logits_row: np.ndarray, tokens: List[int], penalty: float
) -> np.ndarray:
    if penalty == 1.0 or not tokens:
        return logits_row
    out = logits_row.copy()
    seen: Set[int] = set(tokens)
    for tok in seen:
        v = out[tok]
        out[tok] = (v / penalty) if v > 0 else (v * penalty)
    return out


def _no_repeat_banned_tokens(tokens: List[int], n: int) -> Set[int]:
    if n <= 0 or len(tokens) < n - 1:
        return set()
    if n == 1:
        return set(tokens)
    prefix = tuple(tokens[-(n - 1) :])
    banned: Set[int] = set()
    end = len(tokens) - n + 1
    for i in range(end):
        if tuple(tokens[i : i + n - 1]) == prefix:
            banned.add(tokens[i + n - 1])
    return banned


def _compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    encoded = text.encode('utf-8')
    if not encoded:
        return 0.0
    return len(encoded) / max(1, len(zlib.compress(encoded)))


def _check_free_ram_or_raise() -> None:
    try:
        import psutil
    except Exception:
        return
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < LYRICS_WHISPER_MIN_FREE_RAM_GB:
        raise RuntimeError(
            f"Refusing to load Whisper-small: only {available_gb:.1f} GB free, "
            f"need at least {LYRICS_WHISPER_MIN_FREE_RAM_GB:.1f} GB. "
            "Tune via LYRICS_WHISPER_MIN_FREE_RAM_GB."
        )


def _get_mel_filters() -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE / 2,
        norm='slaney',
        htk=False,
    ).astype(np.float32)


def _log_mel_spectrogram(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    import librosa

    if len(wav) < WHISPER_CHUNK_SAMPLES:
        padded = np.zeros(WHISPER_CHUNK_SAMPLES, dtype=np.float32)
        padded[: len(wav)] = wav.astype(np.float32, copy=False)
        wav = padded
    elif len(wav) > WHISPER_CHUNK_SAMPLES:
        wav = wav[:WHISPER_CHUNK_SAMPLES]

    stft = librosa.stft(
        wav.astype(np.float32, copy=False),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window='hann',
        center=True,
        pad_mode='reflect',
    )
    magnitudes = np.abs(stft[:, :WHISPER_FRAMES_PER_CHUNK]) ** 2
    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if log_spec.shape[1] < WHISPER_FRAMES_PER_CHUNK:
        pad_cols = WHISPER_FRAMES_PER_CHUNK - log_spec.shape[1]
        log_spec = np.pad(
            log_spec, ((0, 0), (0, pad_cols)), mode='constant', constant_values=log_spec.min()
        )
    return log_spec.astype(np.float32, copy=False)


def _build_language_token_map(tokenizer) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for tok_id in range(LANGUAGE_TOKEN_START, LANGUAGE_TOKEN_END):
        try:
            piece = tokenizer.id_to_token(tok_id)
        except Exception:
            piece = None
        if not piece:
            continue
        if piece.startswith('<|') and piece.endswith('|>'):
            code = piece[2:-2].strip().lower()
            if code:
                out[tok_id] = code
    return out


class _OnnxWhisperPipeline:
    def __init__(self, model_dir: str, intra_op_threads: int):
        import onnxruntime as ort

        path = Path(model_dir)
        if not path.is_dir():
            raise RuntimeError(
                f"Whisper-small ONNX model dir not found: {model_dir}. "
                "Extract the project release tarball "
                "lyrics_model_whisper.tar.gz into /app/model so that "
                "whisper-small-onnx/ sits at /app/model/whisper-small-onnx."
            )

        encoder_path = path / "encoder_model.onnx"
        decoder_path = path / "decoder_model_merged.onnx"
        for required in (encoder_path, decoder_path):
            if not required.is_file():
                raise RuntimeError(f"Whisper file missing: {required}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if int(intra_op_threads) > 0:
            sess_opts.intra_op_num_threads = int(intra_op_threads)
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False

        if int(intra_op_threads) > 0:
            for env_key in (
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS',
            ):
                os.environ.setdefault(env_key, str(int(intra_op_threads)))

        intra_op_label = (
            'auto' if int(intra_op_threads) <= 0 else str(sess_opts.intra_op_num_threads)
        )
        logger.info(
            "Whisper-small: loading ONNX sessions from %s "
            "(intra_op_threads=%s, inter_op_threads=1, mode=SEQUENTIAL, "
            "arena=disabled, mem_pattern=disabled)",
            path,
            intra_op_label,
        )

        try:
            from tasks.analysis_helper import create_onnx_session

            self.encoder_session = create_onnx_session(
                str(encoder_path), sess_options=sess_opts, label='whisper_encoder'
            )
            self.decoder_session = create_onnx_session(
                str(decoder_path), sess_options=sess_opts, label='whisper_decoder'
            )
        except Exception as exc:
            logger.warning('Whisper: provider helper unavailable (%s) - CPU only', exc)
            self.encoder_session = ort.InferenceSession(
                str(encoder_path), sess_opts, providers=['CPUExecutionProvider']
            )
            self.decoder_session = ort.InferenceSession(
                str(decoder_path), sess_opts, providers=['CPUExecutionProvider']
            )
        logger.info(
            'Whisper-small active providers: encoder=%s decoder=%s',
            self.encoder_session.get_providers()[0],
            self.decoder_session.get_providers()[0],
        )

        self.decoder_input_names: Set[str] = {inp.name for inp in self.decoder_session.get_inputs()}
        self.decoder_output_names: List[str] = [o.name for o in self.decoder_session.get_outputs()]

        sample_pkv = next(
            inp
            for inp in self.decoder_session.get_inputs()
            if inp.name.startswith(_PAST_KEY_VALUES_PREFIX)
            and '.decoder.' in inp.name
            and inp.name.endswith('.key')
        )
        shape = list(sample_pkv.shape)
        self.num_heads = int(shape[1]) if isinstance(shape[1], int) and shape[1] > 0 else 12
        self.head_dim = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 64

        self.present_to_past: Dict[str, str] = {}
        for name in self.decoder_output_names:
            if name.startswith('present.'):
                past_name = _PAST_KEY_VALUES_PREFIX + name[len('present.') :]
                if past_name in self.decoder_input_names:
                    self.present_to_past[name] = past_name

        from tokenizers import Tokenizer

        tok_path = path / "tokenizer.json"
        if not tok_path.is_file():
            raise RuntimeError(f"Whisper tokenizer.json missing: {tok_path}")
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.lang_token_to_code: Dict[int, str] = _build_language_token_map(self.tokenizer)
        if not self.lang_token_to_code:
            logger.warning(
                "Whisper-small: could not extract language token map from "
                "tokenizer; language detection will fall back to 'en'."
            )

        self.mel_filters = _get_mel_filters()
        logger.info(
            "Whisper-small: pipeline ready (layers via past_kv discovery, "
            "num_heads=%s, head_dim=%s, %d language tokens mapped)",
            self.num_heads,
            self.head_dim,
            len(self.lang_token_to_code),
        )

    def _encode(self, log_mel: np.ndarray) -> np.ndarray:
        input_features = log_mel[np.newaxis, :, :].astype(np.float32, copy=False)
        outputs = self.encoder_session.run(None, {'input_features': input_features})
        return outputs[0]

    def _empty_past_kv(self) -> Dict[str, np.ndarray]:
        past_kv: Dict[str, np.ndarray] = {}
        for name in self.decoder_input_names:
            if not name.startswith(_PAST_KEY_VALUES_PREFIX):
                continue
            past_kv[name] = np.zeros((1, self.num_heads, 1, self.head_dim), dtype=np.float32)
        return past_kv

    def _decode_step(
        self,
        input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        past_kv: Dict[str, np.ndarray],
        use_cache: bool,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feed: Dict[str, np.ndarray] = {'input_ids': input_ids}
        if 'encoder_hidden_states' in self.decoder_input_names:
            feed['encoder_hidden_states'] = encoder_hidden_states
        if 'use_cache_branch' in self.decoder_input_names:
            feed['use_cache_branch'] = np.array([use_cache], dtype=bool)
        for name, arr in past_kv.items():
            if name in self.decoder_input_names:
                feed[name] = arr
        outputs = self.decoder_session.run(None, feed)
        named = dict(zip(self.decoder_output_names, outputs))
        return named['logits'], named

    def _detect_language(self, encoder_hidden_states: np.ndarray) -> Tuple[str, float]:
        input_ids = np.array([[SOT_TOKEN_ID]], dtype=np.int64)
        past_kv = self._empty_past_kv()
        logits, _ = self._decode_step(input_ids, encoder_hidden_states, past_kv, use_cache=False)
        last = logits[0, -1, :].copy()
        masked = np.full_like(last, -1e30, dtype=np.float32)
        for tok_id in self.lang_token_to_code:
            if 0 <= tok_id < last.shape[0]:
                masked[tok_id] = last[tok_id]
        probs = _softmax_row(masked)
        top_id = int(np.argmax(probs))
        top_prob = float(probs[top_id])
        code = self.lang_token_to_code.get(top_id, 'en')
        return code, top_prob

    def _greedy_decode(
        self,
        logits: np.ndarray,
        past_kv: Dict[str, np.ndarray],
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int,
    ) -> Tuple[str, float]:
        generated: List[int] = []
        token_logprobs: List[float] = []
        next_token, lp = self._sample_next(logits, generated)
        if next_token == EOT_TOKEN_ID:
            return '', float('-inf')
        for _ in range(max_new_tokens - 1):
            generated.append(next_token)
            token_logprobs.append(lp)
            input_ids = np.array([[next_token]], dtype=np.int64)
            logits, named = self._decode_step(
                input_ids, encoder_hidden_states, past_kv, use_cache=True
            )
            past_kv = self._absorb_present(past_kv, named, on_first_step=False)
            next_token, lp = self._sample_next(logits, generated)
            if next_token == EOT_TOKEN_ID:
                break
        avg_logprob = float(np.mean(token_logprobs)) if token_logprobs else float('-inf')
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip(), avg_logprob

    def _init_beams(
        self, logits: np.ndarray, past_kv: Dict[str, np.ndarray], k: int
    ) -> List[Dict[str, Any]]:
        first_log_probs = self._compute_log_probs(logits, [])
        top_init = np.argpartition(first_log_probs, -k)[-k:]
        top_init = top_init[np.argsort(-first_log_probs[top_init])]
        beams: List[Dict[str, Any]] = []
        for tok_id in top_init:
            tok_id = int(tok_id)
            beams.append(
                {
                    'tokens': [tok_id],
                    'log_prob_sum': float(first_log_probs[tok_id]),
                    'past_kv': dict(past_kv),
                    'finished': tok_id == EOT_TOKEN_ID,
                }
            )
        return beams

    def _expand_beam(
        self,
        beam: Dict[str, Any],
        parent_idx: int,
        encoder_hidden_states: np.ndarray,
        k: int,
    ) -> List[Tuple[float, int, Optional[int], Optional[Dict[str, np.ndarray]]]]:
        if beam['finished']:
            return [(beam['log_prob_sum'], parent_idx, None, None)]
        last_tok = beam['tokens'][-1]
        input_ids = np.array([[last_tok]], dtype=np.int64)
        logits, named = self._decode_step(
            input_ids, encoder_hidden_states, beam['past_kv'], use_cache=True
        )
        new_past_kv = self._absorb_present(dict(beam['past_kv']), named, on_first_step=False)
        log_probs = self._compute_log_probs(logits, beam['tokens'])
        top_idx = np.argpartition(log_probs, -k)[-k:]
        out: List[Tuple[float, int, Optional[int], Optional[Dict[str, np.ndarray]]]] = []
        for tok_id in top_idx:
            tok_id = int(tok_id)
            new_log_prob = beam['log_prob_sum'] + float(log_probs[tok_id])
            out.append((new_log_prob, parent_idx, tok_id, new_past_kv))
        return out

    def _select_beams(
        self,
        beams: List[Dict[str, Any]],
        candidates: List[Tuple[float, int, Optional[int], Optional[Dict[str, np.ndarray]]]],
        k: int,
    ) -> List[Dict[str, Any]]:
        candidates.sort(key=lambda c: c[0], reverse=True)
        new_beams: List[Dict[str, Any]] = []
        for log_prob_sum, parent_idx, tok_id, new_past_kv in candidates[:k]:
            parent = beams[parent_idx]
            if tok_id is None:
                new_beams.append(parent)
                continue
            new_beams.append(
                {
                    'tokens': parent['tokens'] + [tok_id],
                    'log_prob_sum': log_prob_sum,
                    'past_kv': new_past_kv,
                    'finished': tok_id == EOT_TOKEN_ID,
                }
            )
        return new_beams

    def _finalize_beams(self, beams: List[Dict[str, Any]]) -> Tuple[str, float]:
        def _score(b: Dict[str, Any]) -> float:
            n = max(1, len(b['tokens']))
            return float(b['log_prob_sum']) / n

        best = max(beams, key=_score)
        tokens: List[int] = list(best['tokens'])
        if tokens and tokens[-1] == EOT_TOKEN_ID:
            tokens = tokens[:-1]
        avg_logprob = float(best['log_prob_sum']) / max(1, len(tokens) or 1)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text.strip(), avg_logprob

    def _beam_decode(
        self,
        logits: np.ndarray,
        past_kv: Dict[str, np.ndarray],
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int,
        k: int,
    ) -> Tuple[str, float]:
        beams = self._init_beams(logits, past_kv, k)
        for _ in range(max_new_tokens - 1):
            if all(b['finished'] for b in beams):
                break
            candidates: List[Tuple[float, int, Optional[int], Optional[Dict[str, np.ndarray]]]] = []
            for parent_idx, beam in enumerate(beams):
                candidates.extend(
                    self._expand_beam(beam, parent_idx, encoder_hidden_states, k)
                )
            beams = self._select_beams(beams, candidates, k)
        return self._finalize_beams(beams)

    def _decode_chunk(
        self,
        encoder_hidden_states: np.ndarray,
        lang_token_id: int,
        max_new_tokens: int,
        beam_size: int,
    ) -> Tuple[str, float]:
        prompt_ids = [SOT_TOKEN_ID, lang_token_id, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID]
        past_kv = self._empty_past_kv()

        input_ids = np.array([prompt_ids], dtype=np.int64)
        logits, named = self._decode_step(
            input_ids, encoder_hidden_states, past_kv, use_cache=False
        )
        past_kv = self._absorb_present(past_kv, named, on_first_step=True)

        k = max(1, int(beam_size))

        if k == 1:
            return self._greedy_decode(logits, past_kv, encoder_hidden_states, max_new_tokens)

        return self._beam_decode(logits, past_kv, encoder_hidden_states, max_new_tokens, k)

    def _absorb_present(
        self,
        past_kv: Dict[str, np.ndarray],
        named_outputs: Dict[str, np.ndarray],
        on_first_step: bool,
    ) -> Dict[str, np.ndarray]:
        for present_name, past_name in self.present_to_past.items():
            if present_name not in named_outputs:
                continue
            is_encoder = '.encoder.' in present_name
            if is_encoder and not on_first_step:
                continue
            past_kv[past_name] = named_outputs[present_name]
        return past_kv

    def _compute_log_probs(self, logits: np.ndarray, generated: List[int]) -> np.ndarray:
        raw = logits[0, -1, :].copy()
        for tok in _SUPPRESS_TOKEN_IDS:
            if 0 <= tok < raw.shape[0]:
                raw[tok] = -1e30
        raw = _apply_repetition_penalty(raw, generated, WHISPER_REPETITION_PENALTY)
        for tok in _no_repeat_banned_tokens(generated, WHISPER_NO_REPEAT_NGRAM):
            if 0 <= tok < raw.shape[0]:
                raw[tok] = -1e30
        return _log_softmax_row(raw)

    def _sample_next(self, logits: np.ndarray, generated: List[int]) -> Tuple[int, float]:
        log_probs = self._compute_log_probs(logits, generated)
        next_token = int(np.argmax(log_probs))
        return next_token, float(log_probs[next_token])

    def _resolve_language(
        self, first_enc: np.ndarray, language: Optional[str]
    ) -> Tuple[str, float]:
        if language:
            return language.strip().lower(), 1.0
        detected_code, detected_prob = self._detect_language(first_enc)
        logger.info(
            "Whisper-small: detected language=%r confidence=%.3f (threshold=%.2f)",
            detected_code,
            detected_prob,
            LYRICS_WHISPER_LANG_CONFIDENCE,
        )
        return detected_code, detected_prob

    def _resolve_lang_token_id(self, detected_code: str) -> int:
        code_to_token = {v: k for k, v in self.lang_token_to_code.items()}
        lang_token_id = code_to_token.get(detected_code)
        if lang_token_id is None:
            logger.warning(
                "Whisper-small: language code %r has no matching <|xx|> "
                "token; falling back to 'en' for decoding.",
                detected_code,
            )
            lang_token_id = code_to_token.get('en', LANGUAGE_TOKEN_START)
        return lang_token_id

    def _process_chunk(
        self,
        enc: np.ndarray,
        chunk_idx: int,
        n_chunks: int,
        chunk_samples: int,
        lang_token_id: int,
        max_new_tokens: int,
    ) -> Tuple[str, float]:
        chunk_seconds = chunk_samples / SAMPLE_RATE
        text, avg_lp = self._decode_chunk(
            enc,
            lang_token_id,
            max_new_tokens,
            beam_size=WHISPER_BEAM_SIZE,
        )
        cleaned = text.strip()
        dropped_by_compression = False
        if (
            cleaned
            and WHISPER_COMPRESSION_RATIO_THRESHOLD > 0
            and _compression_ratio(cleaned) > WHISPER_COMPRESSION_RATIO_THRESHOLD
        ):
            logger.warning(
                "Whisper-small: compression ratio %.2f > %.2f - "
                "dropping repetition-collapsed chunk (%d chars)",
                _compression_ratio(cleaned),
                WHISPER_COMPRESSION_RATIO_THRESHOLD,
                len(cleaned),
            )
            cleaned = ''
            avg_lp = float('-inf')
            dropped_by_compression = True
        preview = (cleaned[:80] + '…') if len(cleaned) > 80 else cleaned
        logger.info(
            "Whisper-small: chunk %d/%d (%.2fs, %d samples) -> %d chars, avg_logprob=%s%s | %r",
            chunk_idx + 1,
            n_chunks,
            chunk_seconds,
            chunk_samples,
            len(cleaned),
            f"{avg_lp:.3f}" if avg_lp != float('-inf') else '-inf',
            ' (DROPPED by compression)' if dropped_by_compression else '',
            preview,
        )
        return cleaned, avg_lp

    def transcribe(
        self, wav: np.ndarray, language: Optional[str] = None, max_new_tokens: Optional[int] = None
    ) -> Dict[str, object]:
        if wav is None or wav.size == 0:
            return {
                "text": "",
                "language": language or "",
                "duration": 0.0,
                "avg_logprob": float('-inf'),
            }
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        if max_new_tokens is None:
            max_new_tokens = WHISPER_MAX_NEW_TOKENS

        audio_duration = len(wav) / SAMPLE_RATE
        intra_op_label = 'auto' if WHISPER_INTRA_OP_THREADS <= 0 else str(WHISPER_INTRA_OP_THREADS)
        logger.info(
            "Whisper-small: starting transcription "
            "(beam_size=%d, max_new_tokens=%d, repetition_penalty=%.2f, "
            "no_repeat_ngram=%d, compression_ratio_threshold=%.2f, "
            "lang_confidence_threshold=%.2f, intra_op_threads=%s, "
            "language_hint=%r, audio_duration=%.2fs)",
            WHISPER_BEAM_SIZE,
            max_new_tokens,
            WHISPER_REPETITION_PENALTY,
            WHISPER_NO_REPEAT_NGRAM,
            WHISPER_COMPRESSION_RATIO_THRESHOLD,
            LYRICS_WHISPER_LANG_CONFIDENCE,
            intra_op_label,
            language,
            audio_duration,
        )
        t0 = time.time()

        n_chunks = max(1, int(np.ceil(len(wav) / WHISPER_CHUNK_SAMPLES)))
        windows = [
            wav[i * WHISPER_CHUNK_SAMPLES : (i + 1) * WHISPER_CHUNK_SAMPLES]
            for i in range(n_chunks)
        ]

        first_mel = _log_mel_spectrogram(windows[0], self.mel_filters)
        first_enc = self._encode(first_mel)
        detected_code, detected_prob = self._resolve_language(first_enc, language)
        if not language and detected_prob < LYRICS_WHISPER_LANG_CONFIDENCE:
            logger.info(
                "Whisper-small: language confidence %.3f < %.2f - "
                "dropping transcript, treating as instrumental",
                detected_prob,
                LYRICS_WHISPER_LANG_CONFIDENCE,
            )
            return {
                "text": "",
                "language": "",
                "duration": audio_duration,
                "avg_logprob": float('-inf'),
            }

        lang_token_id = self._resolve_lang_token_id(detected_code)

        texts: List[str] = []
        chunk_logprobs: List[float] = []
        encoder_outputs: List[np.ndarray] = [first_enc]
        for w in windows[1:]:
            mel = _log_mel_spectrogram(w, self.mel_filters)
            encoder_outputs.append(self._encode(mel))

        for chunk_idx, enc in enumerate(encoder_outputs):
            cleaned, avg_lp = self._process_chunk(
                enc,
                chunk_idx,
                len(encoder_outputs),
                len(windows[chunk_idx]),
                lang_token_id,
                max_new_tokens,
            )
            if cleaned:
                texts.append(cleaned)
            if avg_lp != float('-inf'):
                chunk_logprobs.append(avg_lp)

        full_text = " ".join(texts).strip()
        elapsed = time.time() - t0
        avg_logprob = float(np.mean(chunk_logprobs)) if chunk_logprobs else float('-inf')
        logger.info(
            "Whisper-small: %.1fs audio in %.1fs (RTF=%.2f, %d chunks, "
            "beam=%d, lang=%r, avg_logprob=%.2f)",
            audio_duration,
            elapsed,
            elapsed / max(audio_duration, 0.001),
            len(encoder_outputs),
            WHISPER_BEAM_SIZE,
            detected_code,
            avg_logprob,
        )
        return {
            "text": full_text,
            "language": detected_code,
            "duration": audio_duration,
            "avg_logprob": avg_logprob,
        }


_pipeline: Optional[_OnnxWhisperPipeline] = None
_pipeline_dir: Optional[str] = None
_load_lock = threading.Lock()


class WhisperLoadRefused(RuntimeError):
    pass


def load_whisper_model() -> _OnnxWhisperPipeline:
    global _pipeline, _pipeline_dir
    try:
        from config import LYRICS_WHISPER_MODEL_DIR
    except Exception:
        LYRICS_WHISPER_MODEL_DIR = '/app/model/whisper-small-onnx'

    if _pipeline is not None and _pipeline_dir == LYRICS_WHISPER_MODEL_DIR:
        return _pipeline
    with _load_lock:
        if _pipeline is not None and _pipeline_dir == LYRICS_WHISPER_MODEL_DIR:
            return _pipeline
        try:
            _check_free_ram_or_raise()
        except RuntimeError as exc:
            raise WhisperLoadRefused(str(exc)) from exc
        _pipeline = _OnnxWhisperPipeline(
            LYRICS_WHISPER_MODEL_DIR, intra_op_threads=WHISPER_INTRA_OP_THREADS
        )
        _pipeline_dir = LYRICS_WHISPER_MODEL_DIR
        return _pipeline


def transcribe(
    wav: np.ndarray, sr: int, language: Optional[str] = None
) -> Dict[str, object]:
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    try:
        pipeline = load_whisper_model()
    except WhisperLoadRefused as exc:
        logger.warning("Whisper-small load refused: %s", exc)
        return {
            "text": "",
            "language": "",
            "duration": len(wav) / SAMPLE_RATE,
            "avg_logprob": float('-inf'),
        }
    return pipeline.transcribe(wav, language=language)


def is_loaded() -> bool:
    return _pipeline is not None


def unload() -> bool:
    global _pipeline, _pipeline_dir
    pipeline = _pipeline
    if pipeline is None and _pipeline_dir is None:
        return False
    _pipeline = None
    _pipeline_dir = None
    try:
        for attr in (
            'encoder_session',
            'decoder_session',
            'tokenizer',
            'mel_filters',
            'lang_token_to_code',
            'decoder_input_names',
            'decoder_output_names',
            'present_to_past',
        ):
            try:
                setattr(pipeline, attr, None)
            except Exception:
                logger.exception("Error dropping Whisper pipeline.%s", attr)
    finally:
        try:
            import gc

            del pipeline
            gc.collect()
        except Exception:
            logger.exception("Error during Whisper pipeline GC")
        try:
            from tasks.memory_utils import comprehensive_memory_cleanup

            comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=True)
        except Exception:
            logger.exception("Error during ONNX memory pool reset on Whisper unload")
    logger.info("Whisper-small: pipeline unloaded (~1.5 GB freed)")
    return True


def reset_session() -> None:
    unload()
