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
    'model_dir': None,
    'tokenizer': None,
    'encoder_session': None,
    'decoder_session': None,
    'decoder_input_names': (),
    'decoder_output_names': (),
    'past_key_template': {},
    'present_to_past': {},
    'pad_token_id': 0,
    'eos_token_id': 0,
    'decoder_start_token_id': 0,
    'num_layers': 0,
    'num_heads': 0,
    'head_dim': 0,
}

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
        sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 6)
        sess_opts.inter_op_num_threads = 1

        encoder_path = os.path.join(target_dir, 'encoder_model.onnx')
        decoder_path = os.path.join(target_dir, 'decoder_model_merged.onnx')
        for p in (encoder_path, decoder_path):
            if not os.path.isfile(p):
                raise RuntimeError(f'Translator ONNX file missing: {p}')

        logger.info('Loading translator encoder: %s', encoder_path)
        try:
            from tasks.analysis_helper import create_onnx_session
            encoder_session = create_onnx_session(
                encoder_path, sess_options=sess_opts, label='marian_encoder')
            logger.info('Loading translator merged decoder: %s', decoder_path)
            decoder_session = create_onnx_session(
                decoder_path, sess_options=sess_opts, label='marian_decoder')
        except Exception as exc:
            logger.warning('Marian: provider helper unavailable (%s) — CPU only', exc)
            encoder_session = ort.InferenceSession(
                encoder_path, sess_options=sess_opts,
                providers=['CPUExecutionProvider'])
            logger.info('Loading translator merged decoder: %s', decoder_path)
            decoder_session = ort.InferenceSession(
                decoder_path, sess_options=sess_opts,
                providers=['CPUExecutionProvider'])
        logger.info('Marian active providers: encoder=%s decoder=%s',
                    encoder_session.get_providers()[0],
                    decoder_session.get_providers()[0])

        decoder_input_names = tuple(inp.name for inp in decoder_session.get_inputs())
        decoder_output_names = tuple(out.name for out in decoder_session.get_outputs())

        cfg = _read_json(os.path.join(target_dir, 'config.json')) or {}
        d_model = int(cfg.get('d_model') or cfg.get('hidden_size') or 512)
        num_heads = int(cfg.get('decoder_attention_heads')
                        or cfg.get('num_attention_heads')
                        or cfg.get('encoder_attention_heads') or 8)
        head_dim = int(cfg.get('d_kv') or (d_model // max(num_heads, 1)))

        past_template, present_to_past, num_layers = (
            _build_empty_past_template(decoder_session, num_heads, head_dim))

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

_SENTENCE_BREAK_RE = re.compile(r'(?<=[。！？．，、；：\.\!\?,;:])\s*')

def _translator_chunk_chars_default() -> int:
    try:
        from config import LYRICS_TRANSLATOR_CHUNK_CHARS as _cfg_max
        return int(_cfg_max)
    except Exception:
        try:
            return int(os.environ.get('LYRICS_TRANSLATOR_CHUNK_CHARS', '200'))
        except Exception:
            return 200

def _split_for_translator(text: str, max_chars: Optional[int] = None) -> List[str]:
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
        for piece in _SENTENCE_BREAK_RE.split(line):
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= max_chars:
                chunks.append(piece)
                continue
            for i in range(0, len(piece), max_chars):
                window = piece[i:i + max_chars].strip()
                if window:
                    chunks.append(window)
    return chunks

def _generate(state: Dict[str, Any], input_ids: np.ndarray,
              attention_mask: np.ndarray, max_new_tokens: int) -> List[int]:
    encoder_session = state['encoder_session']
    decoder_session = state['decoder_session']
    decoder_input_names = state['decoder_input_names']
    present_to_past = state['present_to_past']
    decoder_start = state['decoder_start_token_id']
    eos = state['eos_token_id']
    num_heads = state['num_heads']
    head_dim = state['head_dim']

    encoder_outputs = encoder_session.run(
        ['last_hidden_state'],
        {'input_ids': input_ids, 'attention_mask': attention_mask},
    )
    encoder_hidden_states = encoder_outputs[0]

    past_kv: Dict[str, np.ndarray] = {}
    for name in decoder_input_names:
        if not name.startswith('past_key_values.'):
            continue
        past_kv[name] = np.zeros((1, num_heads, 1, head_dim), dtype=np.float32)

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
            feed['use_cache_branch'] = np.array([step > 0], dtype=bool)

        outputs = decoder_session.run(None, feed)
        named_outputs = dict(zip(state['decoder_output_names'], outputs))
        logits = named_outputs['logits']
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

        on_first_step = (step == 0)
        for present_name, past_name in present_to_past.items():
            if present_name not in named_outputs:
                continue
            is_encoder = '.encoder.' in present_name
            if is_encoder and not on_first_step:
                continue
            past_kv[past_name] = named_outputs[present_name]

        decoder_input_ids = np.array([[next_token]], dtype=np.int64)

    return generated

def translate_to_english(text: str, source_lang: Optional[str] = None,
                         max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS) -> str:
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
    return (_state.get('encoder_session') is not None
            or _state.get('decoder_session') is not None
            or _state.get('tokenizer') is not None)

def reset_session() -> None:
    with _lock:
        for k in ('encoder_session', 'decoder_session', 'tokenizer'):
            _state[k] = None
        _state['model_dir'] = None
        _state['decoder_input_names'] = ()
        _state['decoder_output_names'] = ()
        _state['past_key_template'] = {}
        _state['present_to_past'] = {}
