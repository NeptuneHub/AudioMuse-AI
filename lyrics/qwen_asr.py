"""Qwen3-ASR-0.6B (ONNX CPU build) inference for the lyrics pipeline.

Replaces the previous openai-whisper integration. Uses ONNX Runtime only —
no PyTorch dependency for ASR. The ONNX model files are expected in the
directory pointed to by ``config.LYRICS_QWEN_ASR_MODEL_DIR`` (default
``/app/model/Qwen3-ASR-0.6B-ONNX-CPU``).

Decoder quality
---------------
The community ONNX builds we consume don't ship with a tuned generation
config, and a pure-greedy decoder hallucinates badly on music (locking
into "rock 'n' roll, fun." style attractors). To match the defenses
Whisper uses by default we layer in:

* **Beam search** (``LYRICS_ASR_BEAM_SIZE``, default 2) — k parallel
  hypotheses each with their own KV cache; length-normalized log-prob
  picks the winner. Default 2 catches most music-ASR hallucinations
  caused by locally-greedy-but-globally-wrong token picks at ~2×
  decoder cost. Set to 1 for pure greedy (fastest), 4 for Whisper-
  style maximum quality (~4× slower). Beams run sequentially — the
  community ONNX export bakes batch=1 into internal ScatterElements
  ops and rejects any batched run, so we don't try.
* **Repetition penalty** (``LYRICS_ASR_REPETITION_PENALTY``, default
  1.15) — HF-style penalty on raw logits for any token already
  generated.
* **No-repeat n-gram** (``LYRICS_ASR_NO_REPEAT_NGRAM``, default 3) —
  hard ban on tokens that would re-create a 3-gram already present.
  Single most effective fix for stuck loops.
* **Compression-ratio gate** (``LYRICS_ASR_COMPRESSION_RATIO``, default
  2.4) — post-decode zlib check: if the chunk's text compresses below
  this ratio it's a runaway loop, so we drop it and let the lyrics
  pipeline treat the song as instrumental (Whisper retries with higher
  temperature; we don't have temperature sampling, so we drop instead).

Per-worker isolation, not cluster-wide locking
----------------------------------------------
Each RQ worker process loads its own copy of the model and keeps it in
memory across jobs (singleton). Concurrent workers all load independently —
that's intentional, since each worker has its own k8s memory budget. To
keep memory pressure manageable when many workers come online at once:

1. **Thread cap from env** — ``intra_op_num_threads`` controlled by
   ``LYRICS_ASR_INTRA_OP_THREADS`` (default ``0`` = ORT auto-detect,
   matching CLAP's behavior — uses all cores when one Qwen worker is
   active). Set to a positive int to pin (e.g. ``cpu_count // 3`` if
   you run 3 concurrent Qwen workers and need to leave headroom).
   ``inter_op_num_threads = 1`` and ``ExecutionMode.ORT_SEQUENTIAL``
   are always set; the four sessions still run sequentially.
2. **ONNX arena + memory-pattern disabled** — ``enable_cpu_mem_arena
   = False`` and ``enable_mem_pattern = False`` so the runtime doesn't
   accumulate transient buffers between calls. With multiple concurrent
   workers the arena was creeping idle RSS upward and compounding with
   the lower RAM gate, eventually pushing the node into swap. Predictable
   memory ceiling > marginal per-call speedup.
3. **Memory-mapped embedding matrix** — ``np.memmap`` instead of
   ``np.fromfile`` saves ~600 MB of RSS per process; the kernel pages it
   in on demand.
4. **Per-worker free-RAM check** — refuses to load when ``psutil`` reports
   less than ``LYRICS_ASR_MIN_FREE_RAM_GB`` (default 5 GB) available.
   Sized above the model's measured ~3 GB peak RSS with real headroom
   for activation memory and the transient doubling that happens while
   a fresh load maps in. The job logs a warning and returns an empty
   transcription so the downstream pipeline applies the instrumental
   sentinel. With multiple workers spinning up simultaneously, the late
   ones see less free RAM and naturally throttle — no explicit cluster
   lock required.

Adapted from the community ONNX inference script:
    https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU
"""

from __future__ import annotations

import logging
import os
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Bulletproofing knobs (env-tunable) ──────────────────────────────────

# Refuse to load when free RAM is below this threshold. Default 5 GB:
# the INT8 decoder build has ~3 GB peak RSS (encoder_transformer ~700 MB,
# two decoder INT8 sessions ~600 MB each, mmap'd embed_tokens ~600 MB
# paged on demand) plus ~500 MB-1 GB of activation / Python / numpy
# overhead during decode. 5 GB leaves real headroom — important when
# 2-3 workers race the gate concurrently, because each fresh load
# transiently doubles RSS while the model is being mapped in.
# Lowering toward 3.5-4 risks the original ``crash the entire k3s
# node'' compound-pressure scenario.
LYRICS_ASR_MIN_FREE_RAM_GB = float(os.environ.get("LYRICS_ASR_MIN_FREE_RAM_GB", "5"))

# ── Decoder quality knobs (env-tunable) ─────────────────────────────────
# These mirror the standard HuggingFace `generate()` knobs (and Whisper's
# defaults) that the upstream community ONNX builds don't ship with a
# tuned config for. They turn the previous pure-greedy decoder into a
# beam search with repetition controls and a compression-ratio sanity
# gate — the same set of defenses Whisper uses against music
# hallucinations.

# Beam search width. 1 = pure greedy (fastest but most error-prone),
# 4 = Whisper's default (highest quality, ~4x slower). Default 2 is the
# practical sweet spot: catches most of the locally-optimal-but-globally-
# wrong choices that produce music-ASR hallucinations, at ~2x decoder
# cost vs greedy. Each extra beam costs one extra decoder_step.run per
# generated token plus ~30-60 MB of KV cache at a full 30s chunk.
BEAM_SIZE = int(os.environ.get("LYRICS_ASR_BEAM_SIZE", "2"))

# HF-style repetition penalty applied to raw logits before softmax.
# Values >1 push down recently-seen tokens. 1.15 is the value Qwen-LLM
# repos commonly recommend for ASR-like transcription tasks (1.0 = off).
REPETITION_PENALTY = float(os.environ.get("LYRICS_ASR_REPETITION_PENALTY", "1.15"))

# Hard ban on any token that would re-create an n-gram already present
# in the decoded sequence. The single most effective fix for stuck
# loops like "rock 'n' roll, fun. rock 'n' roll, fun." (0 = off).
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("LYRICS_ASR_NO_REPEAT_NGRAM", "3"))

# zlib-compression-ratio sanity check. If the decoded text compresses
# below this ratio it's suspiciously repetitive — the failure mode where
# the model collapses into "the the the the". Whisper uses 2.4 as the
# threshold; we treat the chunk as instrumental and skip it instead of
# persisting a hallucinated transcript. (Set <= 0 to disable.)
COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("LYRICS_ASR_COMPRESSION_RATIO", "2.4"))

# Hard cap on tokens generated per 30s chunk. 128 covers ~25-50 lyric
# lines per chunk (most chunks emit < 80 tokens before EOS). Lower
# value bounds the worst case when the decoder fails to find an EOS
# (which happens more with the n-gram ban active, since common stop-
# token patterns can get banned). Raising past ~200 burns time without
# improving quality — real lyrics rarely exceed it.
MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("LYRICS_ASR_MAX_NEW_TOKENS", "128"))


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
    """Resolve the intra-op thread cap (``0`` = let ORT auto-detect).

    Default behavior matches ``tasks/clap_analyzer.py`` (also CPU-only
    ONNX): we don't set ``intra_op_num_threads`` at all and let ONNX
    Runtime auto-detect, which lights up every available core. CLAP has
    been running this way without crashing — Qwen behaves the same when
    only one Qwen worker is active at a time.

    Override via ``LYRICS_ASR_INTRA_OP_THREADS`` env var:

      * ``0`` (default) — ORT auto-detect, uses all cores
      * any positive int — explicit cap (e.g. ``cpu_count // 3`` to
        leave room for 3 concurrent workers without oversubscribing,
        or a fixed ``8`` to bound RAM-per-worker on huge boxes)

    The ``num_threads`` argument is accepted for backward compatibility
    but ignored — config comes from env.
    """
    raw = os.environ.get('LYRICS_ASR_INTRA_OP_THREADS', '0').strip()
    try:
        value = int(raw)
    except ValueError:
        value = 0
    return max(0, value)


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


# ── Decoder helpers (logits processors + sanity checks) ─────────────────

def _log_softmax_row(logits_row: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax on a 1-D logit row → float32."""
    x = logits_row.astype(np.float64, copy=False)
    x = x - x.max()
    log_norm = np.log(np.exp(x).sum())
    return (x - log_norm).astype(np.float32, copy=False)


def _apply_repetition_penalty(logits_row: np.ndarray,
                              tokens: List[int],
                              penalty: float) -> np.ndarray:
    """HF-style repetition penalty applied to RAW logits (pre-softmax).

    For each token already generated:
      * positive logit → divide by penalty (push down)
      * negative logit → multiply by penalty (push further down)

    Returns a fresh array; caller's logits are untouched.
    """
    if penalty == 1.0 or not tokens:
        return logits_row
    out = logits_row.copy()
    seen: Set[int] = set(tokens)
    for tok in seen:
        v = out[tok]
        if v > 0:
            out[tok] = v / penalty
        else:
            out[tok] = v * penalty
    return out


def _no_repeat_banned_tokens(tokens: List[int], n: int) -> Set[int]:
    """Return token IDs that would create a duplicate n-gram if appended.

    Equivalent to HF's NoRepeatNGramLogitsProcessor. Cheap O(len(tokens))
    scan — fine for sequences up to a few thousand tokens.
    """
    if n <= 0 or len(tokens) < n - 1:
        return set()
    if n == 1:
        # Degenerate: ban every token already seen.
        return set(tokens)
    prefix = tuple(tokens[-(n - 1):])
    banned: Set[int] = set()
    end = len(tokens) - n + 1
    for i in range(end):
        if tuple(tokens[i:i + n - 1]) == prefix:
            banned.add(tokens[i + n - 1])
    return banned


def _compression_ratio(text: str) -> float:
    """zlib compression ratio of ``text`` (utf-8). Whisper uses this as a
    repetition sanity check: well-formed prose lands around 1.5–2.0,
    runaway loops collapse the ratio upward (3.0+).

    Returns 0.0 for empty input so the caller's threshold check never
    fires on legitimately empty transcripts.
    """
    if not text:
        return 0.0
    encoded = text.encode('utf-8')
    if not encoded:
        return 0.0
    compressed = zlib.compress(encoded)
    return len(encoded) / max(1, len(compressed))


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
                "Extract the project release tarball "
                "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/lyrics_model_qwen3_asr.tar.gz "
                "into /app/model (the bundle ships Qwen3-ASR-0.6B-ONNX-CPU/ at the archive root)."
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
        # When num_threads == 0, leave ORT to auto-detect (uses all
        # cores — same default as CLAP's CLAP_PYTHON_MULTITHREADS=False
        # path). When > 0, pin to that exact value.
        if int(num_threads) > 0:
            sess_opts.intra_op_num_threads = int(num_threads)
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3
        # Disable the ONNX runtime's CPU arena + shape-derived memory
        # pattern. With multiple concurrent workers the arena was
        # creeping idle RSS upward over time and compounding with the
        # lower per-worker RAM gate — eventually pushing the node into
        # swap. Per-call allocation is slightly slower but the memory
        # ceiling is predictable, which matters more than peak speed
        # when the alternative is the whole server going unresponsive.
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False

        # Belt-and-suspenders: also pin BLAS / OpenMP thread pools that some
        # numpy/librosa code paths use during mel/STFT preprocessing — but
        # only when num_threads > 0 (explicit cap). When 0 (ORT auto), let
        # BLAS auto-detect too so it matches the unbounded CLAP-style
        # behavior we want.
        if int(num_threads) > 0:
            for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                os.environ.setdefault(env_key, str(int(num_threads)))

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

        intra_op_label = (
            'auto' if int(num_threads) <= 0
            else str(sess_opts.intra_op_num_threads)
        )
        logger.info(
            "Qwen3-ASR: loading ONNX sessions from %s "
            "(decoder=%s, intra_op_threads=%s, inter_op_threads=1, mode=SEQUENTIAL)",
            models_dir, decoder_kind, intra_op_label,
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

    def _decode_beam(self,
                     init_logits: np.ndarray,
                     init_keys: np.ndarray,
                     init_values: np.ndarray,
                     prefill_len: int,
                     max_new_tokens: int,
                     beam_size: int,
                     repetition_penalty: float,
                     no_repeat_ngram_size: int) -> Tuple[List[int], float]:
        """Beam-search decode starting from decoder_init outputs.

        Maintains ``beam_size`` candidate sequences in parallel, each with
        its own KV cache. Each step makes one ``decoder_step.run`` call
        per active beam (sequential — the community ONNX export of
        Qwen3-ASR-0.6B-CPU bakes batch=1 into internal ScatterElements
        ops and rejects any batch>1 input, so we don't try to batch).

        Per-beam decode at k=2 is RTF ~0.1 on the test corpus, which
        is plenty fast. Length-normalized log-prob picks the winner at
        the end (mirrors HF's default beam scorer).
        """
        # Apply logits processors to the prefill row before picking the
        # initial expansions. (Repetition penalty / no-repeat-ngram are
        # no-ops at step 0 because no tokens have been generated yet.)
        first_log_probs = _log_softmax_row(init_logits[0, -1, :])

        # Top-`beam_size` initial candidates.
        k = max(1, beam_size)
        if k == 1:
            top_init = np.array([int(np.argmax(first_log_probs))])
        else:
            top_init = np.argpartition(first_log_probs, -k)[-k:]
            top_init = top_init[np.argsort(-first_log_probs[top_init])]

        beams: List[Dict[str, object]] = []
        for tok_id in top_init:
            tok_id = int(tok_id)
            beams.append({
                'tokens': [tok_id],
                'log_prob_sum': float(first_log_probs[tok_id]),
                # init_keys/init_values are (1, ...) arrays — every beam
                # starts pointing at the same prefill cache and will
                # diverge after the first decoder_step.
                'past_keys': init_keys,
                'past_values': init_values,
                'finished': tok_id in (IM_END_ID, ENDOFTEXT_ID),
            })

        cur_pos = prefill_len  # next position id to feed

        for _ in range(max_new_tokens - 1):
            if all(b['finished'] for b in beams):
                break

            candidates: List[Tuple[float, int, Optional[int],
                                   Optional[np.ndarray], Optional[np.ndarray]]] = []

            # Carry forward finished beams as candidates with no extension.
            for parent_idx, beam in enumerate(beams):
                if beam['finished']:
                    candidates.append((beam['log_prob_sum'], parent_idx,
                                       None, None, None))
                    continue

                # One decoder_step.run per active beam. Sequential —
                # see method docstring for why we don't batch.
                token_embed = self.embed_tokens[beam['tokens'][-1]][
                    np.newaxis, np.newaxis, :].astype(np.float32, copy=False)
                pos_one = np.array([[cur_pos]], dtype=np.int64)
                step_logits, new_keys, new_values = self.decoder_step.run(None, {
                    "input_embeds": token_embed,
                    "position_ids": pos_one,
                    "past_keys": beam['past_keys'],
                    "past_values": beam['past_values'],
                })

                # Own a private copy before mutating — repetition penalty
                # only copies when active, and the no-repeat-ngram ban
                # writes in place.
                raw = step_logits[0, -1, :].copy()
                raw = _apply_repetition_penalty(raw, beam['tokens'], repetition_penalty)
                banned = _no_repeat_banned_tokens(beam['tokens'], no_repeat_ngram_size)
                for tok in banned:
                    raw[tok] = -1e30
                log_probs = _log_softmax_row(raw)

                # Top-`k` continuations from this beam.
                if k == 1:
                    top_idx = np.array([int(np.argmax(log_probs))])
                else:
                    top_idx = np.argpartition(log_probs, -k)[-k:]
                for tok_id in top_idx:
                    tok_id = int(tok_id)
                    new_log_prob = beam['log_prob_sum'] + float(log_probs[tok_id])
                    candidates.append((new_log_prob, parent_idx, tok_id,
                                       new_keys, new_values))

            # Keep the global top ``k`` candidates.
            candidates.sort(key=lambda c: c[0], reverse=True)
            new_beams: List[Dict[str, object]] = []
            for log_prob_sum, parent_idx, tok_id, new_keys, new_values in candidates[:k]:
                parent = beams[parent_idx]
                if tok_id is None:
                    new_beams.append(parent)
                    continue
                new_beams.append({
                    'tokens': parent['tokens'] + [tok_id],
                    'log_prob_sum': log_prob_sum,
                    'past_keys': new_keys,
                    'past_values': new_values,
                    'finished': tok_id in (IM_END_ID, ENDOFTEXT_ID),
                })
            beams = new_beams
            cur_pos += 1

        # Length-normalized score: prefers longer beams when their per-token
        # confidence is comparable, mirroring HF's default beam scorer.
        def _score(b: Dict[str, object]) -> float:
            n = max(1, len(b['tokens']))
            return float(b['log_prob_sum']) / n

        best = max(beams, key=_score)
        tokens: List[int] = list(best['tokens'])
        if tokens and tokens[-1] in (IM_END_ID, ENDOFTEXT_ID):
            tokens = tokens[:-1]

        avg_logprob = float(best['log_prob_sum']) / max(1, len(tokens) or 1)
        return tokens, avg_logprob

    def _transcribe_chunk(self, wav: np.ndarray, language: Optional[str],
                          max_new_tokens: int) -> Dict[str, object]:
        mel = _compute_mel_spectrogram(wav, self.mel_filters)
        audio_features = self._encode_audio(mel)
        num_audio_tokens = audio_features.shape[0]

        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        input_embeds = self._embed_and_fuse(token_ids, audio_features)
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        init_logits, init_keys, init_values = self.decoder_init.run(None, {
            "input_embeds": input_embeds,
            "position_ids": position_ids,
        })

        generated, avg_logprob = self._decode_beam(
            init_logits=init_logits,
            init_keys=init_keys,
            init_values=init_values,
            prefill_len=seq_len,
            max_new_tokens=max_new_tokens,
            beam_size=BEAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )

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

        # Compression-ratio sanity check: catches the failure mode where
        # the decoder collapses into a tight repetition loop ("rock 'n'
        # roll, fun. rock 'n' roll, fun.") that beam search + ngram bans
        # didn't fully prevent. Mirrors Whisper's own defense — but
        # instead of retrying with higher temperature (we don't have
        # temperature sampling), we drop the chunk and let the lyrics
        # pipeline treat the song as instrumental.
        cleaned_text = parsed_text.strip()
        if (cleaned_text and COMPRESSION_RATIO_THRESHOLD > 0
                and _compression_ratio(cleaned_text) > COMPRESSION_RATIO_THRESHOLD):
            ratio = _compression_ratio(cleaned_text)
            logger.warning(
                "Qwen3-ASR: compression ratio %.2f > %.2f — likely a "
                "repetition collapse, dropping chunk (%d chars)",
                ratio, COMPRESSION_RATIO_THRESHOLD, len(cleaned_text),
            )
            cleaned_text = ""
            avg_logprob = float('-inf')

        return {
            "text": cleaned_text,
            "language": parsed_lang,
            "avg_logprob": avg_logprob,
        }

    def transcribe(self, wav: np.ndarray, language: Optional[str] = None,
                   max_new_tokens: Optional[int] = None,
                   chunk_sec: int = 30) -> Dict[str, object]:
        """Transcribe a 16kHz mono float32 numpy array.

        ``max_new_tokens`` caps the generated length per chunk. Default
        comes from ``LYRICS_ASR_MAX_NEW_TOKENS`` env var (128). Tighter
        cap bounds the worst case when the decoder fails to find an EOS
        (which happens more with the n-gram ban active, since common
        stop-token patterns can themselves get banned).

        Returns dict with ``text``, ``language``, ``duration``, and
        ``avg_logprob`` (mean log-probability of generated tokens; close to
        0 = confident, very negative = uncertain / hallucinated).
        """
        if wav is None or wav.size == 0:
            return {"text": "", "language": language or "", "duration": 0.0,
                    "avg_logprob": float('-inf')}
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS_DEFAULT

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


def is_loaded() -> bool:
    """True when the Qwen3-ASR pipeline is currently held in memory."""
    return _pipeline is not None


def unload() -> bool:
    """Drop the cached pipeline and free the four ONNX sessions + memmap.

    Mirrors the per-album release pattern used by CLAP — see
    ``tasks.clap_analyzer.unload_clap_model``. Safe to call when nothing is
    loaded. Each release step runs under its own try/except so a partial
    failure cannot leak the rest. Returns True if anything was actually
    released.
    """
    global _pipeline, _pipeline_dir
    pipeline = _pipeline
    if pipeline is None and _pipeline_dir is None:
        return False
    # Detach from module-level singletons first so concurrent callers can't
    # reuse a half-torn-down pipeline if cleanup raises mid-way.
    _pipeline = None
    _pipeline_dir = None
    try:
        for attr in ('encoder_conv', 'encoder_transformer',
                     'decoder_init', 'decoder_step',
                     'embed_tokens', 'mel_filters', 'tokenizer'):
            try:
                setattr(pipeline, attr, None)
            except Exception:
                logger.exception("Error dropping Qwen3-ASR pipeline.%s", attr)
    finally:
        # Always run GC + memory-pool reset even if some attribute drop above
        # raised — the partially-dropped pipeline still needs to be reclaimed.
        try:
            import gc
            del pipeline
            gc.collect()
        except Exception:
            logger.exception("Error during Qwen3-ASR pipeline GC")
        try:
            from tasks.memory_utils import comprehensive_memory_cleanup
            comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=True)
        except Exception:
            logger.exception("Error during ONNX memory pool reset on Qwen3-ASR unload")
    logger.info("Qwen3-ASR: pipeline unloaded (~3 GB freed)")
    return True


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

    ``num_threads`` is accepted for backward compatibility but ignored —
    the per-split intra-op cap is always ``os.cpu_count() // 3``. See
    :func:`_resolve_thread_cap` for the rationale (sessions and splits
    are all sequential, so we only need to leave headroom for *other*
    workers on the box).

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

    return pipeline.transcribe(wav, language=language)
