from __future__ import annotations

import logging
import os
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LYRICS_ASR_MIN_FREE_RAM_GB = float(os.environ.get("LYRICS_ASR_MIN_FREE_RAM_GB", "5"))

from config import LYRICS_ASR_BEAM_SIZE as BEAM_SIZE

REPETITION_PENALTY = float(os.environ.get("LYRICS_ASR_REPETITION_PENALTY", "1.15"))

NO_REPEAT_NGRAM_SIZE = int(os.environ.get("LYRICS_ASR_NO_REPEAT_NGRAM", "3"))

COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("LYRICS_ASR_COMPRESSION_RATIO", "2.4"))

MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("LYRICS_ASR_MAX_NEW_TOKENS", "128"))

CHUNK_SEC_DEFAULT = int(os.environ.get("LYRICS_ASR_CHUNK_SEC", "30"))

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100

AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
NEWLINE_ID = 198

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024

def _resolve_thread_cap(num_threads: Optional[int]) -> int:
    raw = os.environ.get('LYRICS_ASR_INTRA_OP_THREADS', '').strip()
    if raw == '':
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count // 3)
    try:
        value = int(raw)
    except ValueError:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count // 3)
    return max(0, value)

def _check_free_ram_or_raise() -> None:
    try:
        import psutil
    except Exception:
        return
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < LYRICS_ASR_MIN_FREE_RAM_GB:
        raise RuntimeError(
            f"Refusing to load Qwen3-ASR: only {available_gb:.1f} GB free, "
            f"need at least {LYRICS_ASR_MIN_FREE_RAM_GB:.1f} GB. "
            "Tune via LYRICS_ASR_MIN_FREE_RAM_GB."
        )

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
    lengths = input_lengths
    for _ in range(3):
        lengths = (lengths - 1) // 2 + 1
    return lengths

def _log_softmax_row(logits_row: np.ndarray) -> np.ndarray:
    x = logits_row.astype(np.float64, copy=False)
    x = x - x.max()
    log_norm = np.log(np.exp(x).sum())
    return (x - log_norm).astype(np.float32, copy=False)

def _apply_repetition_penalty(logits_row: np.ndarray,
                              tokens: List[int],
                              penalty: float) -> np.ndarray:
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
    if n <= 0 or len(tokens) < n - 1:
        return set()
    if n == 1:
        return set(tokens)
    prefix = tuple(tokens[-(n - 1):])
    banned: Set[int] = set()
    end = len(tokens) - n + 1
    for i in range(end):
        if tuple(tokens[i:i + n - 1]) == prefix:
            banned.add(tokens[i + n - 1])
    return banned

def _compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    encoded = text.encode('utf-8')
    if not encoded:
        return 0.0
    compressed = zlib.compress(encoded)
    return len(encoded) / max(1, len(compressed))

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

class _OnnxAsrPipeline:
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

        self.intra_op_threads = int(num_threads)
        nested = onnx_path / "onnx_models"
        models_dir = nested if nested.is_dir() else onnx_path

        try:
            from tasks.analysis_helper import get_provider_options
            provider_opts = get_provider_options()
        except Exception as exc:
            logger.warning("Qwen3-ASR: provider helper unavailable (%s) — CPU only", exc)
            provider_opts = [('CPUExecutionProvider', {})]
        use_cuda = provider_opts[0][0] == 'CUDAExecutionProvider'

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if int(num_threads) > 0:
            sess_opts.intra_op_num_threads = int(num_threads)
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3
        if not use_cuda:
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.enable_mem_pattern = False

        if int(num_threads) > 0:
            for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                os.environ.setdefault(env_key, str(int(num_threads)))

        fp32_available = (models_dir / "decoder_init.onnx").exists()
        int8_available = (models_dir / "decoder_init.int8.onnx").exists()
        if use_cuda and fp32_available:
            decoder_init = "decoder_init.onnx"
            decoder_step = "decoder_step.onnx"
            decoder_kind = "FP32"
        elif int8_available:
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
            "(decoder=%s, ep=%s, intra_op_threads=%s, inter_op_threads=1, mode=SEQUENTIAL)",
            models_dir, decoder_kind, provider_opts[0][0], intra_op_label,
        )

        def _make_session(rel_path: str, providers, prov_opts):
            return ort.InferenceSession(
                str(models_dir / rel_path), sess_opts,
                providers=providers, provider_options=prov_opts,
            )

        providers = [p[0] for p in provider_opts]
        provider_options = [p[1] for p in provider_opts]
        try:
            self.encoder_conv = _make_session("encoder_conv.onnx", providers, provider_options)
            self.encoder_transformer = _make_session("encoder_transformer.onnx", providers, provider_options)
            self.decoder_init = _make_session(decoder_init, providers, provider_options)
            self.decoder_step = _make_session(decoder_step, providers, provider_options)
        except Exception as exc:
            if not use_cuda:
                raise
            logger.warning(
                "Qwen3-ASR: CUDA session load failed (%s) — retrying on CPU", exc,
            )
            use_cuda = False
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.enable_mem_pattern = False
            if int8_available:
                decoder_init = "decoder_init.int8.onnx"
                decoder_step = "decoder_step.int8.onnx"
                decoder_kind = "INT8"
            providers = ['CPUExecutionProvider']
            provider_options = [{}]
            logger.info("Qwen3-ASR: CPU fallback (decoder=%s)", decoder_kind)
            self.encoder_conv = _make_session("encoder_conv.onnx", providers, provider_options)
            self.encoder_transformer = _make_session("encoder_transformer.onnx", providers, provider_options)
            self.decoder_init = _make_session(decoder_init, providers, provider_options)
            self.decoder_step = _make_session(decoder_step, providers, provider_options)

        self._use_cuda = use_cuda

        embed_path = models_dir / "embed_tokens.bin"
        logger.info("Qwen3-ASR: memory-mapping embedding matrix (%.0f MB)",
                    embed_path.stat().st_size / 1e6)
        self.embed_tokens = np.memmap(
            str(embed_path), dtype=np.float32, mode='r',
            shape=(VOCAB_SIZE, HIDDEN_SIZE),
        )

        self.mel_filters = _get_mel_filters()

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
        return encoder_output

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
        first_log_probs = _log_softmax_row(init_logits[0, -1, :])

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
                'past_keys': init_keys,
                'past_values': init_values,
                'finished': tok_id in (IM_END_ID, ENDOFTEXT_ID),
            })

        cur_pos = prefill_len

        for _ in range(max_new_tokens - 1):
            if all(b['finished'] for b in beams):
                break

            candidates: List[Tuple[float, int, Optional[int],
                                   Optional[np.ndarray], Optional[np.ndarray]]] = []

            for parent_idx, beam in enumerate(beams):
                if beam['finished']:
                    candidates.append((beam['log_prob_sum'], parent_idx,
                                       None, None, None))
                    continue

                token_embed = self.embed_tokens[beam['tokens'][-1]][
                    np.newaxis, np.newaxis, :].astype(np.float32, copy=False)
                pos_one = np.array([[cur_pos]], dtype=np.int64)
                step_logits, new_keys, new_values = self.decoder_step.run(None, {
                    "input_embeds": token_embed,
                    "position_ids": pos_one,
                    "past_keys": beam['past_keys'],
                    "past_values": beam['past_values'],
                })

                raw = step_logits[0, -1, :].copy()
                raw = _apply_repetition_penalty(raw, beam['tokens'], repetition_penalty)
                banned = _no_repeat_banned_tokens(beam['tokens'], no_repeat_ngram_size)
                for tok in banned:
                    raw[tok] = -1e30
                log_probs = _log_softmax_row(raw)

                if k == 1:
                    top_idx = np.array([int(np.argmax(log_probs))])
                else:
                    top_idx = np.argpartition(log_probs, -k)[-k:]
                for tok_id in top_idx:
                    tok_id = int(tok_id)
                    new_log_prob = beam['log_prob_sum'] + float(log_probs[tok_id])
                    candidates.append((new_log_prob, parent_idx, tok_id,
                                       new_keys, new_values))

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
        logger.debug("Qwen3-ASR raw decode (avg_logprob=%.2f): %r",
                     avg_logprob, raw_text[:200])

        parsed_lang = ""
        parsed_text = raw_text
        if "language " in raw_text and "<asr_text>" in raw_text:
            head, _, tail = raw_text.partition("<asr_text>")
            if head.startswith("language "):
                parsed_lang = head[len("language "):].strip()
            parsed_text = tail
        elif language:
            parsed_lang = language

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
                   chunk_sec: Optional[int] = None) -> Dict[str, object]:
        if wav is None or wav.size == 0:
            return {"text": "", "language": language or "", "duration": 0.0,
                    "avg_logprob": float('-inf')}
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS_DEFAULT
        if chunk_sec is None:
            chunk_sec = CHUNK_SEC_DEFAULT

        audio_duration = len(wav) / SAMPLE_RATE
        intra_op_label = ('auto' if self.intra_op_threads <= 0
                          else str(self.intra_op_threads))
        logger.info(
            "Qwen3-ASR: starting transcription "
            "(beam_size=%d, max_new_tokens=%d, repetition_penalty=%.2f, "
            "no_repeat_ngram=%d, compression_ratio_threshold=%.2f, "
            "intra_op_threads=%s, chunk_sec=%d, language_hint=%r, "
            "audio_duration=%.2fs)",
            BEAM_SIZE, max_new_tokens, REPETITION_PENALTY,
            NO_REPEAT_NGRAM_SIZE, COMPRESSION_RATIO_THRESHOLD,
            intra_op_label, chunk_sec, language, audio_duration,
        )
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
        logger.info("Qwen3-ASR: %.1fs audio split into %d sub-chunks (chunk_sec=%d)",
                    audio_duration, num_chunks, chunk_sec)

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

_pipeline: Optional[_OnnxAsrPipeline] = None
_pipeline_dir: Optional[str] = None

class AsrLoadRefused(RuntimeError):
    pass

def is_loaded() -> bool:
    return _pipeline is not None

def unload() -> bool:
    global _pipeline, _pipeline_dir
    pipeline = _pipeline
    if pipeline is None and _pipeline_dir is None:
        return False
    used_cuda = bool(getattr(pipeline, '_use_cuda', False))
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
        try:
            import gc
            del pipeline
            gc.collect()
        except Exception:
            logger.exception("Error during Qwen3-ASR pipeline GC")
        try:
            from tasks.memory_utils import comprehensive_memory_cleanup
            comprehensive_memory_cleanup(force_cuda=used_cuda, reset_onnx_pool=True)
        except Exception:
            logger.exception("Error during ONNX memory pool reset on Qwen3-ASR unload")
    logger.info("Qwen3-ASR: pipeline unloaded (~3 GB freed)")
    return True

def load_asr_model(num_threads: Optional[int] = None) -> _OnnxAsrPipeline:
    global _pipeline, _pipeline_dir
    try:
        from config import LYRICS_QWEN_ASR_MODEL_DIR
    except Exception:
        LYRICS_QWEN_ASR_MODEL_DIR = '/app/model/Qwen3-ASR-0.6B-ONNX-CPU'

    if _pipeline is not None and _pipeline_dir == LYRICS_QWEN_ASR_MODEL_DIR:
        return _pipeline

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
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    try:
        pipeline = load_asr_model(num_threads=num_threads)
    except AsrLoadRefused as exc:
        logger.warning("Qwen3-ASR load refused: %s", exc)
        return {"text": "", "language": "", "duration": len(wav) / SAMPLE_RATE}

    return pipeline.transcribe(wav, language=language)
