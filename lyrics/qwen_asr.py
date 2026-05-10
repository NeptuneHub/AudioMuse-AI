"""Qwen3-ASR-0.6B (ONNX CPU build) inference for the lyrics pipeline.

Replaces the previous openai-whisper integration. Uses ONNX Runtime only —
no PyTorch dependency for ASR. The ONNX model files are expected in the
directory pointed to by ``config.LYRICS_QWEN_ASR_MODEL_DIR`` (default
``/app/model/Qwen3-ASR-0.6B-ONNX-CPU``).

Per-worker isolation, not cluster-wide locking
----------------------------------------------
Each RQ worker process loads its own copy of the model and keeps it in
memory across jobs (singleton). Concurrent workers all load independently —
that's intentional, since each worker has its own k8s memory budget. To
keep memory pressure manageable when many workers come online at once:

1. **Tight thread caps** — ``intra_op_num_threads = cpu_count // 6``,
   ``inter_op_num_threads = 1``, ``ExecutionMode.ORT_SEQUENTIAL``, and
   pinned BLAS env vars. With four sessions this stays at or below the
   physical CPU count.
2. **Reduced ONNX memory** — ``enable_cpu_mem_arena = False`` and
   ``enable_mem_pattern = False`` so the runtime doesn't cache large
   transient buffers between calls.
3. **Memory-mapped embedding matrix** — ``np.memmap`` instead of
   ``np.fromfile`` saves ~600 MB of RSS per process; the kernel pages it
   in on demand.
4. **Per-worker free-RAM check** — refuses to load when ``psutil`` reports
   less than ``LYRICS_ASR_MIN_FREE_RAM_GB`` (default 6 GB) available.
   The job logs a warning and returns an empty transcription so the
   downstream pipeline applies the instrumental sentinel. With multiple
   workers spinning up simultaneously, the late ones see less free RAM
   and naturally throttle — no explicit cluster lock required.
5. **Per-call gc.collect()** — frees per-call ONNX scratch buffers
   between transcriptions to keep idle RSS low.

Adapted from the community ONNX inference script:
    https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Bulletproofing knobs (env-tunable) ──────────────────────────────────

# Refuse to load when free RAM is below this threshold. Default 6 GB
# leaves headroom for the 5.7 GB load + activations.
LYRICS_ASR_MIN_FREE_RAM_GB = float(os.environ.get("LYRICS_ASR_MIN_FREE_RAM_GB", "6"))


# ── Constants (Qwen3-ASR-0.6B internals) ────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100  # encoder-conv chunk window (frames)

# Special token IDs (from the Qwen3-ASR tokenizer)
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645      # EOS
ENDOFTEXT_ID = 151643   # EOS alt
NEWLINE_ID = 198        # '\n'

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024


def _resolve_thread_cap(num_threads: Optional[int]) -> int:
    """Pick a per-session ONNX thread cap.

    Four ONNX sessions are held concurrently. To stay safely below the CPU
    count we divide by 6 (4 sessions × 1.5x margin). Floor of 1.
    """
    if num_threads and num_threads > 0:
        return max(1, num_threads // 6)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // 6)


# ── Pre-load defenses ───────────────────────────────────────────────────

def _check_free_ram_or_raise() -> None:
    """Refuse to load when free RAM is below the configured threshold.

    Raises ``RuntimeError`` so the caller can convert the failure into an
    empty transcription (which the lyrics pipeline treats as instrumental).
    """
    try:
        import psutil
    except Exception:
        # psutil unavailable — skip the check rather than spuriously refusing.
        return
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < LYRICS_ASR_MIN_FREE_RAM_GB:
        raise RuntimeError(
            f"Refusing to load Qwen3-ASR: only {available_gb:.1f} GB free, "
            f"need at least {LYRICS_ASR_MIN_FREE_RAM_GB:.1f} GB. "
            "Tune via LYRICS_ASR_MIN_FREE_RAM_GB."
        )




# ── Mel spectrogram (Whisper-compatible, librosa-only) ──────────────────

def _compute_mel_spectrogram(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    import librosa
    stft = librosa.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window="hann", center=True, pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2
    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def _get_mel_filters() -> np.ndarray:
    import librosa
    return librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        fmin=0, fmax=SAMPLE_RATE // 2, norm="slaney", htk=False,
    ).astype(np.float32)


def _conv_output_lengths(input_lengths: np.ndarray) -> np.ndarray:
    """Output lengths after the encoder's three stride-2 convolutions."""
    lengths = input_lengths
    for _ in range(3):
        lengths = (lengths - 1) // 2 + 1
    return lengths


# ── Silence-based chunking for long audio ───────────────────────────────

SILENCE_THRESHOLD_DB = -40
SILENCE_HOP_SEC = 0.1


def _find_silence_split_points(wav: np.ndarray, target_sec: int = 30) -> List[int]:
    import librosa
    min_sec = target_sec // 2
    max_sec = int(target_sec * 1.5)
    total_samples = len(wav)
    if total_samples <= max_sec * SAMPLE_RATE:
        return []

    hop_samples = int(SILENCE_HOP_SEC * SAMPLE_RATE)
    rms = librosa.feature.rms(
        y=wav, frame_length=hop_samples * 2, hop_length=hop_samples,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_silent = rms_db < SILENCE_THRESHOLD_DB

    split_points: List[int] = []
    cursor = 0
    while cursor + max_sec * SAMPLE_RATE < total_samples:
        search_start_sec = max(0, cursor / SAMPLE_RATE + min_sec)
        search_end_sec = cursor / SAMPLE_RATE + max_sec
        target_abs_sec = cursor / SAMPLE_RATE + target_sec

        frame_start = int(search_start_sec / SILENCE_HOP_SEC)
        frame_end = min(int(search_end_sec / SILENCE_HOP_SEC), len(is_silent))
        frame_target = int(target_abs_sec / SILENCE_HOP_SEC)

        silent_frames = np.where(is_silent[frame_start:frame_end])[0] + frame_start
        if len(silent_frames) > 0:
            best_idx = int(np.argmin(np.abs(silent_frames - frame_target)))
            split_sample = int(silent_frames[best_idx] * hop_samples)
        else:
            split_sample = int(target_abs_sec * SAMPLE_RATE)

        split_sample = min(split_sample, total_samples)
        split_points.append(split_sample)
        cursor = split_sample
    return split_points


# ── Tokenizer wrapper ───────────────────────────────────────────────────

class _Tokenizer:
    def __init__(self, tokenizer_path: Optional[str]):
        if not tokenizer_path or not Path(tokenizer_path).exists():
            raise RuntimeError(
                "Qwen3-ASR tokenizer.json not found. The Daumee ONNX repo "
                "ships it at the repo root — verify the model directory."
            )
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


# ── ONNX inference pipeline ─────────────────────────────────────────────

class _OnnxAsrPipeline:
    """Long-lived singleton holding the ONNX sessions + embedding matrix."""

    def __init__(self, onnx_dir: str, num_threads: int):
        import onnxruntime as ort
        onnx_path = Path(onnx_dir)
        if not onnx_path.is_dir():
            raise RuntimeError(
                f"Qwen3-ASR ONNX model dir not found: {onnx_dir}. "
                "Download it with: huggingface-cli download "
                "Daumee/Qwen3-ASR-0.6B-ONNX-CPU --local-dir <path>"
            )

        # Repo layout: ONNX files in onnx_models/, tokenizer.json at root.
        # Older flat layout: everything at the root. Support both.
        nested = onnx_path / "onnx_models"
        models_dir = nested if nested.is_dir() else onnx_path

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Sequential execution mode + tight thread caps so 4 sessions don't
        # collectively oversubscribe the CPU and crash the worker.
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = max(1, int(num_threads))
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3
        # Memory-pressure relief: stop ONNX from keeping large transient
        # arenas around between calls. Slightly slower per call, much lower
        # idle RSS — the trade-off we want on a constrained k3s node.
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False

        # Belt-and-suspenders: also pin BLAS / OpenMP thread pools that some
        # numpy/librosa code paths use during mel/STFT preprocessing.
        for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(env_key, str(max(1, int(num_threads))))

        # Prefer INT8 quantized decoders when available — same accuracy, much
        # less RAM during decode.
        if (models_dir / "decoder_init.int8.onnx").exists():
            decoder_init = "decoder_init.int8.onnx"
            decoder_step = "decoder_step.int8.onnx"
            decoder_kind = "INT8"
        else:
            decoder_init = "decoder_init.onnx"
            decoder_step = "decoder_step.onnx"
            decoder_kind = "FP32"

        logger.info(
            "Qwen3-ASR: loading ONNX sessions from %s "
            "(decoder=%s, intra_op_threads=%s, inter_op_threads=1, mode=SEQUENTIAL)",
            models_dir, decoder_kind, sess_opts.intra_op_num_threads,
        )

        # Load all four sessions with the same SessionOptions.
        self.encoder_conv = ort.InferenceSession(
            str(models_dir / "encoder_conv.onnx"), sess_opts,
            providers=["CPUExecutionProvider"])
        self.encoder_transformer = ort.InferenceSession(
            str(models_dir / "encoder_transformer.onnx"), sess_opts,
            providers=["CPUExecutionProvider"])
        self.decoder_init = ort.InferenceSession(
            str(models_dir / decoder_init), sess_opts,
            providers=["CPUExecutionProvider"])
        self.decoder_step = ort.InferenceSession(
            str(models_dir / decoder_step), sess_opts,
            providers=["CPUExecutionProvider"])

        embed_path = models_dir / "embed_tokens.bin"
        logger.info("Qwen3-ASR: memory-mapping embedding matrix (%.0f MB)",
                    embed_path.stat().st_size / 1e6)
        # np.memmap instead of np.fromfile — saves ~600 MB of RSS per process;
        # the kernel pages rows in on demand. Lookups (self.embed_tokens[ids])
        # still produce a regular ndarray view.
        self.embed_tokens = np.memmap(
            str(embed_path), dtype=np.float32, mode='r',
            shape=(VOCAB_SIZE, HIDDEN_SIZE),
        )

        self.mel_filters = _get_mel_filters()

        # Tokenizer can live next to the ONNX files (flat layout) or at the
        # repo root (nested layout). Check both.
        tokenizer_candidates = [models_dir / "tokenizer.json", onnx_path / "tokenizer.json"]
        tokenizer_path = next((p for p in tokenizer_candidates if p.exists()), None)
        self.tokenizer = _Tokenizer(str(tokenizer_path) if tokenizer_path else None)
        logger.info("Qwen3-ASR: pipeline ready")

    def _encode_audio(self, mel: np.ndarray) -> np.ndarray:
        mel_len = mel.shape[1]
        chunk_num = int(np.ceil(mel_len / CHUNK_SIZE))
        chunk_lengths = []
        for i in range(chunk_num):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, mel_len)
            chunk_lengths.append(end - start)

        max_chunk_len = max(chunk_lengths)
        padded = np.zeros((chunk_num, 1, N_MELS, max_chunk_len), dtype=np.float32)
        start = 0
        for i, cl in enumerate(chunk_lengths):
            padded[i, 0, :, :cl] = mel[:, start:start + cl]
            start += cl

        lens_after_cnn = _conv_output_lengths(np.array(chunk_lengths))
        conv_out = self.encoder_conv.run(None, {"padded_mel_chunks": padded})[0]

        features = [conv_out[i, :l, :] for i, l in enumerate(lens_after_cnn)]
        hidden_states = np.concatenate(features, axis=0)

        total_tokens = hidden_states.shape[0]
        attn_mask = np.zeros((1, 1, total_tokens, total_tokens), dtype=np.float32)
        encoder_output = self.encoder_transformer.run(None, {
            "hidden_states": hidden_states,
            "attention_mask": attn_mask,
        })[0]
        return encoder_output  # [N, HIDDEN_SIZE]

    def _build_prompt_ids(self, num_audio_tokens: int, language: Optional[str]) -> List[int]:
        ids: List[int] = []
        ids += [IM_START_ID] + self.tokenizer.encode("system") + [NEWLINE_ID, IM_END_ID, NEWLINE_ID]
        ids += [IM_START_ID] + self.tokenizer.encode("user") + [NEWLINE_ID]
        ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * num_audio_tokens + [AUDIO_END_ID]
        ids += [IM_END_ID, NEWLINE_ID]
        ids += [IM_START_ID] + self.tokenizer.encode("assistant") + [NEWLINE_ID]
        if language:
            ids += self.tokenizer.encode(f"language {language}<asr_text>")
        return ids

    def _embed_and_fuse(self, token_ids: List[int], audio_features: np.ndarray) -> np.ndarray:
        ids_array = np.array(token_ids)
        embeds = self.embed_tokens[ids_array]
        audio_positions = np.where(ids_array == AUDIO_PAD_ID)[0]
        if len(audio_positions) != audio_features.shape[0]:
            raise RuntimeError(
                f"Qwen3-ASR audio token count mismatch: "
                f"{len(audio_positions)} placeholders vs {audio_features.shape[0]} features"
            )
        embeds[audio_positions] = audio_features
        return embeds[np.newaxis, :, :]

    def _transcribe_chunk(self, wav: np.ndarray, language: Optional[str],
                          max_new_tokens: int) -> Dict[str, object]:
        mel = _compute_mel_spectrogram(wav, self.mel_filters)
        audio_features = self._encode_audio(mel)
        num_audio_tokens = audio_features.shape[0]

        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        input_embeds = self._embed_and_fuse(token_ids, audio_features)
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, present_keys, present_values = self.decoder_init.run(None, {
            "input_embeds": input_embeds,
            "position_ids": position_ids,
        })

        # Track per-token log-probabilities so we can return an avg_logprob
        # confidence score (same signal Whisper uses internally to detect
        # hallucinations).
        def _log_softmax_pick(step_logits: np.ndarray, picked: int) -> float:
            row = step_logits[0, -1, :].astype(np.float64)
            row -= row.max()
            log_norm = np.log(np.exp(row).sum())
            return float(row[picked] - log_norm)

        next_token = int(np.argmax(logits[0, -1, :]))
        token_logprobs: List[float] = [_log_softmax_pick(logits, next_token)]
        generated = [next_token]
        cur_pos = seq_len

        for _ in range(max_new_tokens - 1):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break
            token_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            pos = np.array([[cur_pos]], dtype=np.int64)
            logits, present_keys, present_values = self.decoder_step.run(None, {
                "input_embeds": token_embed,
                "position_ids": pos,
                "past_keys": present_keys,
                "past_values": present_values,
            })
            next_token = int(np.argmax(logits[0, -1, :]))
            token_logprobs.append(_log_softmax_pick(logits, next_token))
            generated.append(next_token)
            cur_pos += 1

        if generated and generated[-1] in (IM_END_ID, ENDOFTEXT_ID):
            generated = generated[:-1]
            token_logprobs = token_logprobs[:-1]

        avg_logprob = float(np.mean(token_logprobs)) if token_logprobs else float('-inf')

        raw_text = self.tokenizer.decode(generated)
        # Debug log so we can see exactly what Qwen emitted (including the
        # language prefix when present). Truncated to keep logs readable.
        logger.debug("Qwen3-ASR raw decode (avg_logprob=%.2f): %r",
                     avg_logprob, raw_text[:200])

        # Parse "language <lang><asr_text>...." prefix that the model emits.
        parsed_lang = ""
        parsed_text = raw_text
        if "language " in raw_text and "<asr_text>" in raw_text:
            head, _, tail = raw_text.partition("<asr_text>")
            if head.startswith("language "):
                parsed_lang = head[len("language "):].strip()
            parsed_text = tail
        elif language:
            parsed_lang = language

        return {
            "text": parsed_text.strip(),
            "language": parsed_lang,
            "avg_logprob": avg_logprob,
        }

    def transcribe(self, wav: np.ndarray, language: Optional[str] = None,
                   max_new_tokens: int = 512, chunk_sec: int = 30) -> Dict[str, object]:
        """Transcribe a 16kHz mono float32 numpy array.

        Returns dict with ``text``, ``language``, ``duration``, and
        ``avg_logprob`` (mean log-probability of generated tokens; close to
        0 = confident, very negative = uncertain / hallucinated).
        """
        if wav is None or wav.size == 0:
            return {"text": "", "language": language or "", "duration": 0.0,
                    "avg_logprob": float('-inf')}
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        audio_duration = len(wav) / SAMPLE_RATE
        split_points = _find_silence_split_points(wav, target_sec=chunk_sec)

        if not split_points:
            t0 = time.time()
            result = self._transcribe_chunk(wav, language, max_new_tokens)
            result["duration"] = audio_duration
            elapsed = time.time() - t0
            logger.info("Qwen3-ASR: %.1fs audio in %.1fs (RTF=%.2f, avg_logprob=%.2f)",
                        audio_duration, elapsed,
                        elapsed / max(audio_duration, 0.001),
                        result.get("avg_logprob", float('-inf')))
            return result

        boundaries = [0] + split_points + [len(wav)]
        num_chunks = len(boundaries) - 1
        logger.info("Qwen3-ASR: %.1fs audio split into %d sub-chunks",
                    audio_duration, num_chunks)

        texts: List[str] = []
        detected_lang = language or ""
        chunk_logprobs: List[float] = []
        t0 = time.time()
        for i in range(num_chunks):
            chunk_wav = wav[boundaries[i]:boundaries[i + 1]]
            chunk_result = self._transcribe_chunk(chunk_wav, language, max_new_tokens)
            if chunk_result["text"]:
                texts.append(chunk_result["text"].strip())
            cur_lp = chunk_result.get("avg_logprob")
            if cur_lp is not None and cur_lp != float('-inf'):
                chunk_logprobs.append(cur_lp)
            if not detected_lang and chunk_result["language"]:
                detected_lang = chunk_result["language"]

        full_text = " ".join(t for t in texts if t)
        elapsed = time.time() - t0
        avg_logprob = float(np.mean(chunk_logprobs)) if chunk_logprobs else float('-inf')
        logger.info("Qwen3-ASR: %.1fs audio in %.1fs (RTF=%.2f, %d chunks, avg_logprob=%.2f)",
                    audio_duration, elapsed,
                    elapsed / max(audio_duration, 0.001), num_chunks, avg_logprob)
        return {
            "text": full_text,
            "language": detected_lang,
            "duration": audio_duration,
            "avg_logprob": avg_logprob,
        }


# ── Module-level singleton accessor ─────────────────────────────────────

_pipeline: Optional[_OnnxAsrPipeline] = None
_pipeline_dir: Optional[str] = None


class AsrLoadRefused(RuntimeError):
    """Raised when load is refused due to memory pressure.

    The lyrics pipeline catches this and treats the song as instrumental
    instead of crashing the worker (or, worse, the node).
    """


def load_asr_model(num_threads: Optional[int] = None) -> _OnnxAsrPipeline:
    """Return the cached pipeline, building it on first call.

    Load sequence:

    1. Reuse the cached pipeline if this worker process already loaded it
       (singleton — once loaded, every subsequent job in this worker reuses
       the same in-memory model).
    2. Refuse if free RAM is below ``LYRICS_ASR_MIN_FREE_RAM_GB``. Each
       worker checks independently; concurrent workers that arrive when
       memory is already tight back off and treat their songs as
       instrumental rather than pushing the node into OOM.
    3. Only then construct the heavy pipeline.

    ``num_threads`` is the worker-level thread budget; this loader divides
    it by 6 internally to size each ONNX session, since four sessions run
    on the same CPU pool.

    Raises ``AsrLoadRefused`` when the RAM gate trips. Lower-level errors
    during actual model construction (missing files, etc.) propagate as
    ``RuntimeError``.
    """
    global _pipeline, _pipeline_dir
    try:
        from config import LYRICS_QWEN_ASR_MODEL_DIR
    except Exception:
        LYRICS_QWEN_ASR_MODEL_DIR = '/app/model/Qwen3-ASR-0.6B-ONNX-CPU'

    if _pipeline is not None and _pipeline_dir == LYRICS_QWEN_ASR_MODEL_DIR:
        return _pipeline

    # Defense layer: pre-load free-RAM check.
    try:
        _check_free_ram_or_raise()
    except RuntimeError as exc:
        raise AsrLoadRefused(str(exc)) from exc

    per_session_threads = _resolve_thread_cap(num_threads)
    _pipeline = _OnnxAsrPipeline(LYRICS_QWEN_ASR_MODEL_DIR,
                                 num_threads=per_session_threads)
    _pipeline_dir = LYRICS_QWEN_ASR_MODEL_DIR
    return _pipeline


def transcribe(wav: np.ndarray, sr: int, language: Optional[str] = None,
               num_threads: Optional[int] = None) -> Dict[str, object]:
    """Transcribe a numpy audio buffer with the cached pipeline.

    Returns a dict shaped like the previous Whisper integration:
    ``{'text': str, 'language': str, 'duration': float}``. On
    ``AsrLoadRefused`` (RAM pressure), returns an empty transcription so
    the caller treats the song as instrumental.
    """
    if sr != SAMPLE_RATE:
        # The pipeline is fixed at 16kHz. Resample if a different sr arrives.
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    try:
        pipeline = load_asr_model(num_threads=num_threads)
    except AsrLoadRefused as exc:
        logger.warning("Qwen3-ASR load refused: %s", exc)
        return {"text": "", "language": "", "duration": len(wav) / SAMPLE_RATE}

    try:
        result = pipeline.transcribe(wav, language=language)
    finally:
        # Free per-call ONNX scratch buffers and any large intermediate
        # numpy arrays before the next job arrives. Cheap, ~ms.
        gc.collect()
    return result
