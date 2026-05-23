"""Throwaway sanity check for the gte-multilingual lyrics embedding.

Invents 5 short songs in 5 languages (3 about love, 2 about war), embeds them
with lyrics/gte_onnx.py, then ranks them by cosine similarity against the query
word "love". Multilingual love songs should float to the top regardless of
language; war songs should sink.

Run from the repo root with the venv active:
    source .venv/bin/activate
    python throwaway_lyrics_search.py
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault('LYRICS_GTE_ONNX_PATH', 'model/gte-multilingual-base-int8.onnx')
os.environ.setdefault('LYRICS_GTE_TOKENIZER_DIR', 'model/gte-multilingual-base')

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
log = logging.getLogger('lyrics-search-demo')

import numpy as np

from lyrics.gte_onnx import embed_text

SONGS = {
    'Italian - Love': (
        "Ti amo come il sole ama il mattino, "
        "il tuo nome è una carezza sul mio cuore. "
        "Resta con me per sempre, amore mio, "
        "le tue mani sono la mia casa."
    ),
    'English - Love': (
        "I love you more than words can ever say, "
        "your gentle heart is the home where I belong. "
        "Hold me close and never let me go, "
        "our love is a tender, endless song."
    ),
    'Spanish - Love': (
        "Te amo con toda mi alma y mi corazón, "
        "tu mirada es la luz de mi mañana. "
        "Eres mi amor, mi vida y mi canción, "
        "quédate conmigo cada semana."
    ),
    'France - War': (
        "Les canons grondent sur le champ de bataille, "
        "les soldats tombent sous la fumée et le feu. "
        "La guerre dévore les villes en ruine, "
        "le sang coule sur la terre de Dieu."
    ),
    'Chinese - War': (
        "战争的硝烟笼罩着大地，"
        "士兵在炮火中倒下流血。"
        "城市化为废墟，钢铁与火焰， "
        "战鼓敲响，敌人在前线厮杀。"
    ),
}

QUERIES = ['love', 'war', 'dog', 'amore', 'amor', '爱', 'guerra', '战争']


def _search(query: str, song_vecs: dict) -> None:
    qvec = embed_text(query)
    ranked = sorted(
        ((name, float(np.dot(qvec, vec))) for name, vec in song_vecs.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    log.info('=' * 56)
    log.info('SEARCH RESULTS for query %r (cosine, higher = closer):', query)
    log.info('%-6s %-18s %-8s %s', 'rank', 'song', 'score', 'theme')
    log.info('-' * 56)
    for i, (name, score) in enumerate(ranked, start=1):
        theme = 'LOVE' if name.endswith('Love') else 'WAR'
        log.info('%-6d %-18s %.4f   %s', i, name, score, theme)
    log.info('=' * 56)


def main() -> None:
    log.info('Embedding %d songs...', len(SONGS))
    song_vecs = {}
    for name, lyric in SONGS.items():
        vec = embed_text(lyric)
        song_vecs[name] = vec
        log.info('embedded %-16s dim=%s norm=%.4f', name, vec.shape[0],
                 float(np.linalg.norm(vec)))

    for query in QUERIES:
        log.info('Embedding query word: %r', query)
        _search(query, song_vecs)


if __name__ == '__main__':
    main()
