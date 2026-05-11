"""Hand-rolled multilingual-to-English translator using raw ONNX Runtime.

Loads the ``Helsinki-NLP/opus-mt-mul-en`` ONNX bundle (encoder +
``decoder_model_merged.onnx``) and the matching SentencePiece tokenizer
via ``transformers.MarianTokenizer`` (which only requires the
``sentencepiece`` package — no torch).

Why transformers and not bare ``tokenizers``: the marian bundle ships
SentencePiece pieces (``source.spm`` / ``target.spm``) instead of a
fast ``tokenizer.json`` file, and ``MarianTokenizer`` handles all the
marian-specific encode/decode quirks (target-language tokens, special
tokens, etc.) for us. The transformers PyPI package itself does NOT
require torch.

The merged decoder file handles both the cold prefill step and the
cached step via a ``use_cache_branch`` boolean input.

Bundle layout under ``${LYRICS_TRANSLATOR_ONNX_DIR}`` (default
``/app/model/opus-mt-mul-en-onnx``):

    encoder_model.onnx
    decoder_model_merged.onnx
    source.spm / target.spm    (SentencePiece tokenizer pieces)
    tokenizer_config.json
    vocab.json
    config.json
    generation_config.json     (optional — for special token ids)
    special_tokens_map.json    (optional)

Bundle is downloaded at Docker build time from the project's GitHub
release; no official upstream ONNX export exists for opus-mt-mul-en.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = '/app/model/opus-mt-mul-en-onnx'
_DEFAULT_MAX_NEW_TOKENS = 512

_lock = threading.Lock()
_state: Dict[str, Any] = {
    'model_dir': None,            # type: Optional[str]
    'tokenizer': None,
    'encoder_session': None,
    'decoder_session': None,
    'decoder_input_names': (),    # tuple[str, ...]
    'decoder_output_names': (),   # tuple[str, ...]
    'past_key_template': {},      # name -> (shape, dtype) for empty past_kv
    'present_to_past': {},        # mapping from present.* output name -> past.* input name
    'pad_token_id': 0,
    'eos_token_id': 0,
    'decoder_start_token_id': 0,
    'num_layers': 0,
    'num_heads': 0,
    'head_dim': 0,
}


# ── Helpers ─────────────────────────────────────────────────────────────

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logger.warning('Could not parse %s: %s', path, exc)
        return None


def _build_empty_past_template(decoder_session, num_heads: int, head_dim: int
                               ) -> Tuple[Dict[str, Tuple[Tuple[int, ...], Any]],
                                          Dict[str, str], int]:
    """Inspect decoder inputs to build an empty past_key_values template.

    Returns ``(template, present_to_past_map, num_layers)``.

    The template maps each ``past_key_values.X.{decoder|encoder}.{key|value}``
    input name to a ``(shape, dtype)`` pair. Symbolic dims are resolved as:

    * ``batch_size``                        → 1
    * ``num_heads`` / ``encoder_attention_heads`` / etc. → ``num_heads`` arg
    * ``head_dim`` / ``head_size``          → ``head_dim`` arg
    * any sequence-length dim              → 0 (empty cache on first step)

    Using real ``num_heads`` / ``head_dim`` is critical: feeding shape
    ``(1, 0, 0, 0)`` makes the encoder cross-attention matmul fail with
    "right operand cannot broadcast on dim 0".
    """
    template: Dict[str, Tuple[Tuple[int, ...], Any]] = {}
    present_to_past: Dict[str, str] = {}
    num_layers = 0

    for inp in decoder_session.get_inputs():
        name = inp.name
        m = re.match(r'past_key_values\.(\d+)\.(decoder|encoder)\.(key|value)', name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        num_layers = max(num_layers, layer_idx + 1)
        concrete: List[int] = []
        for d in inp.shape:
            if isinstance(d, int):
                concrete.append(d)
                continue
            d_str = str(d).lower()
            if 'batch' in d_str:
                concrete.append(1)
            elif 'head_dim' in d_str or 'head_size' in d_str:
                concrete.append(head_dim)
            elif 'num_heads' in d_str or 'attention_heads' in d_str or 'n_head' in d_str:
                concrete.append(num_heads)
            else:
                # Any sequence-length-like dim (past_sequence_length,
                # encoder_sequence_length, etc.) → 0 = empty cache.
                concrete.append(0)
        template[name] = (tuple(concrete), np.float32)

    for out in decoder_session.get_outputs():
        name = out.name
        m = re.match(r'present\.(\d+)\.(decoder|encoder)\.(key|value)', name)
        if not m:
            continue
        past_name = name.replace('present.', 'past_key_values.', 1)
        present_to_past[name] = past_name

    return template, present_to_past, num_layers


def _load_translator(model_dir: Optional[str] = None):
    """Cache and return all the artifacts needed for translation."""
    target_dir = model_dir or os.environ.get(
        'LYRICS_TRANSLATOR_ONNX_DIR', _DEFAULT_MODEL_DIR)

    if (_state['model_dir'] == target_dir
            and _state['decoder_session'] is not None):
        return _state

    with _lock:
        if (_state['model_dir'] == target_dir
                and _state['decoder_session'] is not None):
            return _state

        if not os.path.isdir(target_dir):
            raise RuntimeError(
                f'Translator ONNX dir not found at {target_dir}. '
                'Download lyrics_model_marian.tar.gz from the project '
                'GitHub release and extract here.')

        import onnxruntime as ort
        from transformers import MarianTokenizer

        # MarianTokenizer reads source.spm / target.spm + vocab.json + the
        # tokenizer_config.json shipped in the bundle. No torch involved.
        for required in ('source.spm', 'target.spm', 'tokenizer_config.json',
                         'vocab.json'):
            p = os.path.join(target_dir, required)
            if not os.path.isfile(p):
                raise RuntimeError(
                    f'Marian tokenizer file missing: {p}. The bundle should '
                    'contain SentencePiece pieces + vocab.json.')
        tokenizer = MarianTokenizer.from_pretrained(
            target_dir, local_files_only=True)

        sess_opts = ort.SessionOptions()
        # Use ORT's default graph optimization level (matches what optimum
        # uses internally). The original ORT_ENABLE_ALL setting tripped the
        # matmul broadcast bug when we were also incorrectly overwriting the
        # encoder past_kv from dummy 0-batch outputs — that's now fixed.
        sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 6)
        sess_opts.inter_op_num_threads = 1

        encoder_path = os.path.join(target_dir, 'encoder_model.onnx')
        decoder_path = os.path.join(target_dir, 'decoder_model_merged.onnx')
        for p in (encoder_path, decoder_path):
            if not os.path.isfile(p):
                raise RuntimeError(f'Translator ONNX file missing: {p}')

        logger.info('Loading translator encoder: %s', encoder_path)
        encoder_session = ort.InferenceSession(
            encoder_path, sess_options=sess_opts,
            providers=['CPUExecutionProvider'])
        logger.info('Loading translator merged decoder: %s', decoder_path)
        decoder_session = ort.InferenceSession(
            decoder_path, sess_options=sess_opts,
            providers=['CPUExecutionProvider'])

        decoder_input_names = tuple(inp.name for inp in decoder_session.get_inputs())
        decoder_output_names = tuple(out.name for out in decoder_session.get_outputs())

        # Read attention-head shape from config.json. Required to size the
        # empty past_key_values dummy tensors fed on the first decode step;
        # without these the encoder cross-attention matmul crashes with
        # "right operand cannot broadcast on dim 0".
        cfg = _read_json(os.path.join(target_dir, 'config.json')) or {}
        d_model = int(cfg.get('d_model') or cfg.get('hidden_size') or 512)
        num_heads = int(cfg.get('decoder_attention_heads')
                        or cfg.get('num_attention_heads')
                        or cfg.get('encoder_attention_heads') or 8)
        head_dim = int(cfg.get('d_kv') or (d_model // max(num_heads, 1)))

        past_template, present_to_past, num_layers = (
            _build_empty_past_template(decoder_session, num_heads, head_dim))

        # Special-token ids: prefer generation_config.json, fall back to config.json,
        # then to safe defaults (0 = pad in marian).
        gen = _read_json(os.path.join(target_dir, 'generation_config.json')) or {}
        pad_token_id = int(gen.get('pad_token_id', cfg.get('pad_token_id', 0)) or 0)
        eos_token_id = int(gen.get('eos_token_id', cfg.get('eos_token_id', 0)) or 0)
        decoder_start_token_id = int(
            gen.get('decoder_start_token_id',
                    cfg.get('decoder_start_token_id', pad_token_id)) or pad_token_id)

        _state.update({
            'model_dir': target_dir,
            'tokenizer': tokenizer,
            'encoder_session': encoder_session,
            'decoder_session': decoder_session,
            'decoder_input_names': decoder_input_names,
            'decoder_output_names': decoder_output_names,
            'past_key_template': past_template,
            'present_to_past': present_to_past,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'decoder_start_token_id': decoder_start_token_id,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'head_dim': head_dim,
        })
        logger.info(
            'Translator ONNX ready (dir=%s, layers=%s, num_heads=%s, head_dim=%s, '
            'decoder_start=%s, eos=%s, pad=%s)',
            target_dir, num_layers, num_heads, head_dim,
            decoder_start_token_id, eos_token_id, pad_token_id)
        return _state


# Sentence-break punctuation we split on at layer 2. Covers Latin
# (. ! ? , ; :) and CJK (。！？．，、；：) — Thai/Lao/etc. don't have
# sentence-terminal punctuation but their lyrics still come line-broken,
# which layer 1 handles. Splitting AFTER each match keeps the punctuation
# with the preceding fragment (preserving translation context).
_SENTENCE_BREAK_RE = re.compile(r'(?<=[。！？．，、；：\.\!\?,;:])\s*')


def _translator_chunk_chars_default() -> int:
    """Resolve the layer-3 hard cap, falling back through config → env → 200."""
    try:
        from config import LYRICS_TRANSLATOR_CHUNK_CHARS as _cfg_max
        return int(_cfg_max)
    except Exception:
        try:
            return int(os.environ.get('LYRICS_TRANSLATOR_CHUNK_CHARS', '200'))
        except Exception:
            return 200


def _split_for_translator(text: str, max_chars: Optional[int] = None) -> List[str]:
    """Split ``text`` into translator-safe chunks of <= ``max_chars`` characters.

    Three layers, applied in order:

    * Layer 1 — split on newlines. Lyrics from ASR / LRC files / APIs are
      almost always line-broken (one line per sung phrase), which alone
      keeps each fragment small.
    * Layer 2 — if a single line is still too long, split on Latin + CJK
      sentence-internal punctuation (. ! ? , ; : 。！？．，、；：). Catches
      the long-CJK-line case where ASR didn't insert line breaks.
    * Layer 3 — hard char-window cap. Reached only when a fragment has
      neither line breaks nor punctuation (very rare for real lyrics).
      Translation quality at the seam degrades, but it stops the model
      from silently truncating past its 512-token context window.

    Returns an empty list for empty input.
    """
    if not text or not text.strip():
        return []
    if max_chars is None:
        max_chars = _translator_chunk_chars_default()
    max_chars = max(1, int(max_chars))

    chunks: List[str] = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            chunks.append(line)
            continue
        # Layer 2: split on sentence-break punctuation.
        for piece in _SENTENCE_BREAK_RE.split(line):
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= max_chars:
                chunks.append(piece)
                continue
            # Layer 3: hard char window.
            for i in range(0, len(piece), max_chars):
                window = piece[i:i + max_chars].strip()
                if window:
                    chunks.append(window)
    return chunks


def _generate(state: Dict[str, Any], input_ids: np.ndarray,
              attention_mask: np.ndarray, max_new_tokens: int) -> List[int]:
    """Run encoder + autoregressive merged decoder; return generated token ids."""
    encoder_session = state['encoder_session']
    decoder_session = state['decoder_session']
    decoder_input_names = state['decoder_input_names']
    present_to_past = state['present_to_past']
    decoder_start = state['decoder_start_token_id']
    eos = state['eos_token_id']
    num_heads = state['num_heads']
    head_dim = state['head_dim']

    # 1) Encode
    encoder_outputs = encoder_session.run(
        ['last_hidden_state'],
        {'input_ids': input_ids, 'attention_mask': attention_mask},
    )
    encoder_hidden_states = encoder_outputs[0]  # (1, src_seq, hidden)

    # 2) Build initial dummy past_key_values arrays.
    #    Optimum's merged decoder convention: on the very first call (where
    #    use_cache_branch=False), every past_kv tensor must have seq dim == 1
    #    — not 0 and not src_seq. The values themselves are ignored because
    #    the model recomputes from encoder_hidden_states / input_ids, but the
    #    shape participates in static graph constraints inside the If
    #    subgraph. Seq=0 or seq=src_seq trips
    #    "right operand cannot broadcast on dim 0" inside encoder_attn.
    #
    #    Verified by capturing optimum.onnxruntime.ORTModelForSeq2SeqLM's
    #    feed dict on the same merged decoder file.
    past_kv: Dict[str, np.ndarray] = {}
    for name in decoder_input_names:
        if not name.startswith('past_key_values.'):
            continue
        past_kv[name] = np.zeros((1, num_heads, 1, head_dim), dtype=np.float32)

    # 3) Greedy decode loop
    decoder_input_ids = np.array([[decoder_start]], dtype=np.int64)
    generated: List[int] = []
    use_cache_branch_input = 'use_cache_branch' in decoder_input_names

    for step in range(max_new_tokens):
        feed: Dict[str, np.ndarray] = {
            'input_ids': decoder_input_ids,
            'encoder_attention_mask': attention_mask,
        }
        if 'encoder_hidden_states' in decoder_input_names:
            feed['encoder_hidden_states'] = encoder_hidden_states
        for name, arr in past_kv.items():
            if name in decoder_input_names:
                feed[name] = arr
        if use_cache_branch_input:
            # False on first step (build cache), True afterwards (use cache)
            feed['use_cache_branch'] = np.array([step > 0], dtype=bool)

        outputs = decoder_session.run(None, feed)
        named_outputs = dict(zip(state['decoder_output_names'], outputs))
        logits = named_outputs['logits']
        # logits shape: (1, seq, vocab) — only the last position matters.
        # Apply the same "forbidden tokens" filter that transformers'
        # `generate()` applies via its LogitsProcessor pipeline: the model
        # frequently puts the highest probability on decoder_start_token_id
        # (== pad_token_id for marian) because that's a structural artifact
        # of the export, NOT a real translation. Masking those before
        # argmax recovers the actual translation token (e.g. 12899 = "Hello").
        last_logits = logits[0, -1, :].copy()
        if 0 <= decoder_start < last_logits.shape[0]:
            last_logits[decoder_start] = -1e30
        pad_id = state['pad_token_id']
        if 0 <= pad_id < last_logits.shape[0] and pad_id != decoder_start:
            last_logits[pad_id] = -1e30
        next_token = int(np.argmax(last_logits))
        if next_token == eos:
            break
        generated.append(next_token)

        # Promote present.* outputs to past_key_values.* inputs for the next
        # step. CRITICAL: when use_cache_branch=True (every step after the
        # prefill), the merged decoder returns DUMMY 0-batch tensors for the
        # encoder.* outputs because the cross-attention cache doesn't change
        # — the model expects the caller to KEEP the encoder past_kv from
        # the prefill step and only refresh the decoder side. Overwriting
        # past_kv.encoder.* with the dummy output would crash the next
        # encoder cross-attention matmul ("right operand cannot broadcast
        # on dim 0").
        on_first_step = (step == 0)
        for present_name, past_name in present_to_past.items():
            if present_name not in named_outputs:
                continue
            is_encoder = '.encoder.' in present_name
            if is_encoder and not on_first_step:
                continue  # Keep the prefill's encoder cache intact.
            past_kv[past_name] = named_outputs[present_name]

        decoder_input_ids = np.array([[next_token]], dtype=np.int64)

    return generated


def translate_to_english(text: str, source_lang: Optional[str] = None,
                         max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS) -> str:
    """Translate ``text`` to English. Returns ``''`` on any failure.

    ``source_lang`` is accepted for API compatibility with the previous
    per-language translator but is unused — the multilingual model
    auto-detects from the input.
    """
    if not text or not text.strip():
        return ''

    try:
        state = _load_translator()
    except Exception as exc:
        logger.warning('Translator not ready (%s); dropping lyrics', exc)
        return ''

    tokenizer = state['tokenizer']
    pieces: List[str] = []
    for chunk in _split_for_translator(text):
        try:
            # MarianTokenizer returns numpy arrays directly when asked.
            encoded = tokenizer(chunk, return_tensors='np', truncation=True,
                                max_length=max_new_tokens)
            input_ids = encoded['input_ids'].astype(np.int64, copy=False)
            attention_mask = encoded['attention_mask'].astype(np.int64, copy=False)
            if input_ids.size == 0:
                continue

            generated = _generate(state, input_ids, attention_mask, max_new_tokens)
            translated = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not translated:
                logger.warning('Translator returned empty chunk; dropping lyrics')
                return ''
            pieces.append(translated)
        except Exception as exc:
            logger.warning('Translator inference failed (%s); dropping lyrics', exc)
            return ''

    return ' '.join(pieces)


def is_loaded() -> bool:
    """True when the Marian translator encoder/decoder sessions are cached."""
    return (_state.get('encoder_session') is not None
            or _state.get('decoder_session') is not None
            or _state.get('tokenizer') is not None)


def reset_session() -> None:
    """Drop the cached sessions + tokenizer (memory cleanup)."""
    with _lock:
        for k in ('encoder_session', 'decoder_session', 'tokenizer'):
            _state[k] = None
        _state['model_dir'] = None
        _state['decoder_input_names'] = ()
        _state['decoder_output_names'] = ()
        _state['past_key_template'] = {}
        _state['present_to_past'] = {}
