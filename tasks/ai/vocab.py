# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Fuzzy vocabulary normalizer for AI-emitted filter labels.

Maps free-form genre/mood/voice/scale/tempo/energy terms from the LLM onto
the fixed catalog vocab in config, so ``planner`` can build a filter the
database understands. Combines curated alias tables, rapidfuzz matching, and
an optional WordNet synonym fallback.

Main Features:
* Routes each mood label into mood_vector vs other_features and splits out vocal-type tags into a separate voices list; tempo/energy phrases resolve to numeric BPM/energy ranges. Exact canonical mood labels (config.OTHER_FEATURE_LABELS) win over energy/tempo alias phrases so 'relaxed' stays a mood.
* Fuzzy remap (rapidfuzz WRatio, cutoff 75, min length 4) plus gender-aware WordNet expansion for vocalist synonyms; unrecognized labels are dropped with a note rather than passed through.
* female_voice_exclusions turns a male-only voices list into the female vocal tags to exclude (score below TAG_EXCLUDE_SCORE), shared by the SQL filter and the re-rank.
* parse_tag_score_pairs is the shared parser for the DB's comma-separated 'tag:score' strings (mood_vector/other_features): entries without ':' or with a non-numeric score are silently skipped.
* OUT_OF_VOCAB_GENRES and SOUND_DESCRIPTORS list genre and atmosphere words that no metadata field holds, so the planner can match them by sound; ACTIVITY_TEMPO gives workout-style words a tempo floor; INSTRUMENT_WORDS maps instrument words and synonyms (sax, fiddle, tabla) to DCLAP SAE concept terms; None marks timbres the SAE confuses (rhodes/electric piano fires on acoustic piano, synth on ambient pads, choir, organ, drums), which are matched by sound only.
"""

import functools
import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

import config


def parse_tag_score_pairs(text: Optional[str]) -> Dict[str, float]:
    parsed = {}
    for part in (text or '').split(','):
        key, separator, raw_value = part.partition(':')
        if not separator:
            continue
        try:
            parsed[key.strip()] = float(raw_value)
        except ValueError:
            continue
    return parsed


_MOOD_VOCAB_FROM_MOODVECTOR_RAW = [
    'female vocalists',
    'female vocalist',
    'male vocalists',
    '60s',
    '70s',
    '80s',
    '90s',
    '00s',
    'beautiful',
    'chillout',
    'chill',
    'Mellow',
    'sexy',
    'catchy',
    'oldies',
    'easy listening',
    'instrumental',
    'guitar',
    'sad',
    'happy',
    'party',
]
MOOD_VOCAB_FROM_MOODVECTOR = [m for m in _MOOD_VOCAB_FROM_MOODVECTOR_RAW if m in config.MOOD_LABELS]

OTHER_FEATURE_VOCAB = list(config.OTHER_FEATURE_LABELS)

_GENRE_FROM_MOODS_RAW = [
    'rock',
    'pop',
    'alternative',
    'indie',
    'electronic',
    'dance',
    'alternative rock',
    'jazz',
    'metal',
    'classic rock',
    'soul',
    'indie rock',
    'electronica',
    'folk',
    'punk',
    'blues',
    'hard rock',
    'ambient',
    'acoustic',
    'experimental',
    'Hip-Hop',
    'country',
    'funk',
    'electro',
    'heavy metal',
    'Progressive rock',
    'rnb',
    'indie pop',
    'House',
]
_GENRE_VOCAB_SET = set(config.STRATIFIED_GENRES) | {
    g for g in _GENRE_FROM_MOODS_RAW if g in config.MOOD_LABELS
}
GENRE_VOCAB = sorted(_GENRE_VOCAB_SET, key=lambda s: s.lower())


ALIAS_MOOD = {
    'female': ['female vocalists', 'female vocalist'],
    'females': ['female vocalists', 'female vocalist'],
    'female voice': ['female vocalists', 'female vocalist'],
    'female vox': ['female vocalists', 'female vocalist'],
    'female vocal': ['female vocalists', 'female vocalist'],
    'female vocals': ['female vocalists', 'female vocalist'],
    'female singer': ['female vocalists', 'female vocalist'],
    'female singers': ['female vocalists', 'female vocalist'],
    'female voices': ['female vocalists', 'female vocalist'],
    'female lead': ['female vocalists', 'female vocalist'],
    'female vocalists': ['female vocalists', 'female vocalist'],
    'female vocalist': ['female vocalists', 'female vocalist'],
    'girl singer': ['female vocalists', 'female vocalist'],
    'girl singers': ['female vocalists', 'female vocalist'],
    'woman': ['female vocalists', 'female vocalist'],
    'women': ['female vocalists', 'female vocalist'],
    'woman singer': ['female vocalists', 'female vocalist'],
    'women singers': ['female vocalists', 'female vocalist'],
    'male': ['male vocalists'],
    'males': ['male vocalists'],
    'male voice': ['male vocalists'],
    'male vox': ['male vocalists'],
    'male vocal': ['male vocalists'],
    'male vocals': ['male vocalists'],
    'male singer': ['male vocalists'],
    'male singers': ['male vocalists'],
    'male voices': ['male vocalists'],
    'male lead': ['male vocalists'],
    'male vocalists': ['male vocalists'],
    'male vocalist': ['male vocalists'],
    'man': ['male vocalists'],
    'men': ['male vocalists'],
    'man singer': ['male vocalists'],
    'men singers': ['male vocalists'],
    'boy singer': ['male vocalists'],
    'sad song': ['sad'],
    'sad songs': ['sad'],
    'happy song': ['happy'],
    'happy songs': ['happy'],
    'party song': ['party'],
    'party songs': ['party'],
    'romantic': ['beautiful', 'sexy'],
    'sexy song': ['sexy'],
    'chill song': ['chill', 'chillout', 'Mellow'],
    'chillout song': ['chillout'],
    'mellow song': ['Mellow'],
    'beautiful song': ['beautiful'],
    'catchy song': ['catchy'],
    '60s music': ['60s'],
    '70s music': ['70s'],
    '80s music': ['80s'],
    '90s music': ['90s'],
    '00s music': ['00s'],
    'sixties': ['60s'],
    'seventies': ['70s'],
    'eighties': ['80s'],
    'nineties': ['90s'],
    'oldies music': ['oldies'],
    'easy listening music': ['easy listening'],
    'instrumental song': ['instrumental'],
    'instrumental songs': ['instrumental'],
}

ALIAS_GENRE = {
    'hiphop': 'Hip-Hop',
    'hip hop': 'Hip-Hop',
    'rap': 'Hip-Hop',
    'r&b': 'rnb',
    'r and b': 'rnb',
    'edm': 'electronic',
    'electronica music': 'electronica',
    'electronic music': 'electronic',
    'rock music': 'rock',
    'pop music': 'pop',
    'jazz music': 'jazz',
    'metal music': 'metal',
    'indie music': 'indie',
    'punk music': 'punk',
    'blues music': 'blues',
    'country music': 'country',
    'house music': 'House',
    'funk music': 'funk',
    'soul music': 'soul',
    'folk music': 'folk',
    'ambient music': 'ambient',
    'classic rock music': 'classic rock',
    'hard rock music': 'hard rock',
    'heavy metal music': 'heavy metal',
    'progressive rock music': 'Progressive rock',
    'alternative rock music': 'alternative rock',
    'indie rock music': 'indie rock',
    'indie pop music': 'indie pop',
    'dance music': 'dance',
    'dance songs': 'dance',
    'dance tracks': 'dance',
    'dance hits': 'dance',
    'dance party': 'dance',
    'dance floor': 'dance',
    'dancefloor': 'dance',
    'dance pop': 'dance',
    'dance anthems': 'dance',
    'soulful': 'soul',
    'funky': 'funk',
    'bluesy': 'blues',
    'jazzy': 'jazz',
    'folky': 'folk',
    'punky': 'punk',
}

OUT_OF_VOCAB_GENRES = (
    'techno', 'trance', 'drum and bass', 'drum n bass', 'dnb', 'dubstep', 'jungle', 'breakbeat',
    'hardstyle', 'idm', 'glitch', 'deep house', 'tech house', 'trip hop', 'trip-hop', 'downtempo',
    'synthwave', 'synth-pop', 'synthpop', 'synth pop', 'darkwave', 'new wave', 'vaporwave',
    'chillwave', 'lo-fi', 'lofi', 'lo fi', 'shoegaze', 'dream pop', 'post-rock', 'post rock',
    'post-punk', 'post punk', 'math rock', 'grunge', 'emo', 'screamo', 'hardcore', 'metalcore',
    'nu metal', 'death metal', 'black metal', 'doom metal', 'thrash', 'power metal',
    'symphonic metal', 'stoner rock', 'psychedelic', 'krautrock', 'surf rock', 'garage rock',
    'rockabilly', 'britpop', 'gothic', 'industrial', 'ebm', 'reggae', 'reggaeton', 'dancehall',
    'dub', 'ska', 'latin', 'salsa', 'bachata', 'cumbia', 'merengue', 'tango', 'bossa nova', 'samba',
    'flamenco', 'fado', 'afrobeat', 'afrobeats', 'amapiano', 'highlife', 'k-pop', 'j-pop',
    'city pop', 'disco', 'italo disco', 'eurodance', 'hyperpop', 'phonk', 'trap', 'drill',
    'grime', 'boom bap', 'gospel', 'classical', 'baroque', 'orchestral', 'opera', 'choral',
    'soundtrack', 'film score', 'chanson', 'swing', 'big band', 'bebop', 'smooth jazz',
    'acid jazz', 'neo soul', 'doo-wop', 'bluegrass', 'americana', 'celtic',
    'klezmer', 'polka', 'world music', 'new age', 'chiptune', 'shanty', 'lullaby',
)

SOUND_DESCRIPTORS = (
    'dark', 'moody', 'brooding', 'haunting', 'eerie', 'creepy', 'spooky', 'sinister', 'ominous',
    'dreamy', 'ethereal', 'atmospheric', 'cinematic', 'epic', 'majestic', 'triumphant',
    'anthemic', 'melancholic', 'melancholy', 'bittersweet', 'nostalgic', 'wistful', 'somber',
    'gloomy', 'hopeful', 'euphoric', 'blissful', 'sunny', 'bright', 'warm', 'lush', 'gritty',
    'raw', 'heavy', 'hypnotic', 'trippy', 'spacey', 'cosmic', 'meditative', 'meditation',
    'spiritual', 'mystical', 'tribal', 'groovy', 'sensual', 'sultry', 'seductive', 'romantic',
    'tender', 'intimate', 'minimal', 'minimalist', 'cozy', 'rainy',
    'stormy', 'angry', 'furious', 'rebellious', 'playful', 'quirky', 'whimsical', 'mysterious',
    'noir', 'retro', 'vintage', 'futuristic', 'chaotic', 'soothing',
    'peaceful', 'serene', 'tranquil', 'lonely', 'heartbroken', 'uplifting', 'inspirational',
)

INSTRUMENT_WORDS = {
    'viola': 'viola', 'violas': 'viola', 'violin': 'violin', 'violins': 'violin', 'fiddle': 'violin',
    'fiddles': 'violin', 'cello': 'cello', 'cellos': 'cello', 'violoncello': 'cello',
    'double bass': 'double bass', 'upright bass': 'double bass', 'contrabass': 'double bass',
    'strings': 'strings section', 'string section': 'strings section', 'strings section': 'strings section',
    'string quartet': 'strings section', 'harp': 'harp', 'harps': 'harp', 'mandolin': 'mandolin',
    'ukulele': 'ukulele', 'uke': 'ukulele', 'banjo': 'banjo', 'sitar': 'sitar',
    'acoustic guitar': 'acoustic guitar', 'acoustic guitars': 'acoustic guitar',
    'piano': 'piano', 'pianos': 'piano', 'grand piano': 'piano', 'accordion': 'accordion',
    'harmonica': 'harmonica', 'trumpet': 'trumpet', 'trumpets': 'trumpet', 'trombone': 'trombone',
    'trombones': 'trombone', 'brass': 'brass section', 'horns': 'brass section',
    'horn section': 'brass section', 'brass section': 'brass section', 'saxophone': 'saxophone',
    'saxophones': 'saxophone', 'sax': 'saxophone', 'clarinet': 'clarinet', 'flute': 'flute',
    'flutes': 'flute', 'woodwind': 'woodwind', 'woodwinds': 'woodwind', 'orchestra': 'orchestra',
    'orchestral': 'orchestra', 'tabla': 'percussion', 'congas': 'percussion', 'bongos': 'percussion',
    'percussion': 'percussion',
    'electric piano': None, 'rhodes': None, 'wurlitzer': None, 'keyboard': None, 'keyboards': None,
    'organ': None, 'hammond': None, 'synthesizer': None, 'synthesizers': None, 'synth': None,
    'synths': None, 'synth bass': None, 'electric guitar': None, 'electric guitars': None,
    'steel guitar': None, 'pedal steel': None, 'slide guitar': None, 'bass guitar': None,
    'guitar': None, 'guitars': None, 'drums': None, 'choir': None, 'choirs': None, 'choral': None,
    'steelpan': None, 'steel drums': None,
}
MOOD_WORDS = {
    'relaxed': (
        'chill', 'chilled', 'chillout', 'relaxing', 'relaxed', 'calm', 'mellow', 'laid back',
        'laid-back', 'soothing', 'peaceful', 'rilassante', 'rilassanti', 'tranquilla', 'relajante',
        'tranquila', 'entspannt', 'entspannte', 'entspannende', 'calme', 'relaxante',
    ),
    'sad': (
        'sad', 'melancholic', 'melancholy', 'heartbroken', 'depressing', 'triste', 'tristi',
        'tristes', 'traurig', 'traurige',
    ),
    'happy': (
        'happy', 'cheerful', 'joyful', 'feel-good', 'feel good', 'uplifting', 'felice', 'felici',
        'allegra', 'allegre', 'feliz', 'alegre', 'alegres', 'joyeux', 'joyeuse', 'frohliche',
        'froehliche', 'gut gelaunt',
    ),
    'party': ('party', 'partying', 'festa', 'fiesta', 'fete'),
    'danceable': (
        'danceable', 'groovy', 'to dance to', 'bailar', 'ballare', 'tanzen', 'danser', 'dansante',
    ),
    'aggressive': ('aggressive', 'angry', 'furious', 'rage', 'aggressiva', 'aggressivo', 'wutend'),
}

ACTIVITY_TEMPO = {
    'workout': (120.0, 200.0),
    'running': (120.0, 200.0),
    'jogging': (120.0, 200.0),
    'cardio': (120.0, 200.0),
    'gym': (115.0, 200.0),
}

ALIAS_SCALE = {
    'major key': 'major',
    'minor key': 'minor',
    'in major': 'major',
    'in minor': 'minor',
    'major scale': 'major',
    'minor scale': 'minor',
}

ALIAS_TEMPO = {
    'slow': (40.0, 90.0),
    'slow song': (40.0, 90.0),
    'slow songs': (40.0, 90.0),
    'medium tempo': (90.0, 130.0),
    'mid tempo': (90.0, 130.0),
    'midtempo': (90.0, 130.0),
    'fast': (130.0, 200.0),
    'fast song': (130.0, 200.0),
    'fast songs': (130.0, 200.0),
    'upbeat': (110.0, 200.0),
    'uptempo': (110.0, 200.0),
}

ALIAS_ENERGY = {
    'chill': (0.0, 0.35),
    'chilled': (0.0, 0.35),
    'mellow': (0.0, 0.4),
    'calm': (0.0, 0.35),
    'quiet': (0.0, 0.3),
    'relaxed': (0.0, 0.4),
    'soft': (0.0, 0.35),
    'low energy': (0.0, 0.35),
    'high energy': (0.65, 1.0),
    'energetic': (0.65, 1.0),
    'intense': (0.7, 1.0),
    'powerful': (0.7, 1.0),
    'upbeat': (0.55, 1.0),
    'workout': (0.65, 1.0),
    'running': (0.65, 1.0),
    'jogging': (0.6, 1.0),
    'cardio': (0.65, 1.0),
    'gym': (0.65, 1.0),
    'cheer me up': (0.55, 1.0),
    'cheer up': (0.55, 1.0),
    'cheerful': (0.55, 1.0),
    'uplifting': (0.55, 1.0),
    'feel-good': (0.55, 1.0),
    'feel good': (0.55, 1.0),
}


_MOOD_VOCAB_LOWER = {m.lower(): m for m in MOOD_VOCAB_FROM_MOODVECTOR}
_OTHER_FEATURE_VOCAB_LOWER = {m.lower(): m for m in OTHER_FEATURE_VOCAB}
_GENRE_VOCAB_LOWER = {g.lower(): g for g in GENRE_VOCAB}


def _classify_label(label: str) -> Tuple[List[str], List[str]]:
    low = label.strip().lower()
    mv = [_MOOD_VOCAB_LOWER[low]] if low in _MOOD_VOCAB_LOWER else []
    of = [_OTHER_FEATURE_VOCAB_LOWER[low]] if low in _OTHER_FEATURE_VOCAB_LOWER else []
    return mv, of


_FUZZY_REMAP_CUTOFF = 75
_FUZZY_REMAP_MIN_LEN = 4


_GENDER_FEMALE_RE = re.compile(r'\b(female|woman|women|girl|girls|lady|ladies)\b')
_GENDER_MALE_RE = re.compile(r'\b(male|man|men|boy|boys|gentleman|gentlemen)\b')
_VOCAL_HINT_RE = re.compile(
    r'\b(singer|singers|vocalist|vocalists|vocaliser|vocalizer|voice|voices|vocal|vocals)\b'
)


def _wn_lemmas(synset) -> List[str]:
    try:
        lems = synset.lemmas() or []
    except Exception:
        return []
    return [lem for lem in lems if isinstance(lem, str)]


def _wn_related(synset) -> List:
    related: List = []
    for method_name in ('hypernyms', 'similar', 'also'):
        try:
            fn = getattr(synset, method_name, None)
            if fn is None or not callable(fn):
                continue
            related.extend(list(fn() or []))
        except Exception:
            continue
    return related


def _wn_definition(synset) -> str:
    try:
        d = synset.definition()
    except Exception:
        return ''
    if isinstance(d, str):
        return d.lower()
    return ''


@functools.lru_cache(maxsize=512)
def _wordnet_synonyms(value: str) -> Tuple[str, ...]:
    if not value or not isinstance(value, str):
        return ()
    try:
        import wn

        synsets = wn.synsets(value.strip(), lang='en')
    except Exception:
        return ()

    out: List[str] = []
    seen = {value.strip().lower()}

    def _add(name: str) -> None:
        n = (name or '').replace('_', ' ').strip().lower()
        if n and n not in seen:
            out.append(n)
            seen.add(n)

    for syn in synsets:
        for lemma in _wn_lemmas(syn):
            _add(lemma)

        related = _wn_related(syn)
        for rel in related:
            for lemma in _wn_lemmas(rel):
                _add(lemma)

        gloss = _wn_definition(syn)
        if not gloss:
            continue
        if _GENDER_FEMALE_RE.search(gloss):
            gender_prefix = 'female'
        elif _GENDER_MALE_RE.search(gloss):
            gender_prefix = 'male'
        else:
            continue

        vocal_bases = set()
        for source in [syn] + related:
            for lem in _wn_lemmas(source):
                base = (lem or '').replace('_', ' ').strip().lower()
                if base and _VOCAL_HINT_RE.search(base):
                    vocal_bases.add(base)
        if _VOCAL_HINT_RE.search(gloss):
            vocal_bases.update({'singer', 'vocalist', 'voice'})

        for base in vocal_bases:
            _add(f'{gender_prefix} {base}')
            if not base.endswith('s'):
                _add(f'{gender_prefix} {base}s')

    return tuple(out)


def normalize_mood(value: str, notes: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    if not value or not isinstance(value, str):
        return [], []
    key = value.strip().lower()
    if not key:
        return [], []

    alias_hits = ALIAS_MOOD.get(key)
    if alias_hits:
        mv: List[str] = []
        of: List[str] = []
        for label in alias_hits:
            cmv, cof = _classify_label(label)
            mv.extend(x for x in cmv if x not in mv)
            of.extend(x for x in cof if x not in of)
        return mv, of

    cmv, cof = _classify_label(key)
    if cmv or cof:
        return cmv, cof

    candidates = list(_MOOD_VOCAB_LOWER.keys()) + list(_OTHER_FEATURE_VOCAB_LOWER.keys())
    if len(key) >= _FUZZY_REMAP_MIN_LEN:
        hit = process.extractOne(
            key, candidates, scorer=fuzz.WRatio, score_cutoff=_FUZZY_REMAP_CUTOFF
        )
        if hit:
            mv2, of2 = _classify_label(hit[0])
            if notes is not None and (mv2 or of2):
                target = mv2 or of2
                notes.append(
                    f"vocab_normalizer remapped mood '{value}' -> {target} (fuzzy {int(hit[1])})"
                )
            return mv2, of2

    for syn in _wordnet_synonyms(key):
        alias_hits = ALIAS_MOOD.get(syn)
        if alias_hits:
            mv3: List[str] = []
            of3: List[str] = []
            for label in alias_hits:
                cmv3, cof3 = _classify_label(label)
                mv3.extend(x for x in cmv3 if x not in mv3)
                of3.extend(x for x in cof3 if x not in of3)
            if mv3 or of3:
                if notes is not None:
                    notes.append(
                        f"vocab_normalizer expanded mood '{value}' via wordnet synonym '{syn}' -> alias -> {(mv3 or of3)}"
                    )
                return mv3, of3
        cmv3, cof3 = _classify_label(syn)
        if cmv3 or cof3:
            if notes is not None:
                notes.append(
                    f"vocab_normalizer expanded mood '{value}' via wordnet synonym '{syn}' -> {(cmv3 or cof3)}"
                )
            return cmv3, cof3
    return [], []


def normalize_genre(value: str, notes: Optional[List[str]] = None) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    key = value.strip().lower()
    if not key:
        return None

    alias = ALIAS_GENRE.get(key)
    if alias:
        return alias

    if key in _GENRE_VOCAB_LOWER:
        return _GENRE_VOCAB_LOWER[key]

    if len(key) >= _FUZZY_REMAP_MIN_LEN:
        hit = process.extractOne(
            key,
            list(_GENRE_VOCAB_LOWER.keys()),
            scorer=fuzz.WRatio,
            score_cutoff=_FUZZY_REMAP_CUTOFF,
        )
        if hit:
            mapped = _GENRE_VOCAB_LOWER[hit[0]]
            if notes is not None:
                notes.append(
                    f"vocab_normalizer remapped genre '{value}' -> '{mapped}' (fuzzy {int(hit[1])})"
                )
            return mapped
    return None


def normalize_scale(value: str) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in ('major', 'minor'):
        return key
    return ALIAS_SCALE.get(key)


def normalize_tempo_phrase(value: str) -> Optional[Tuple[float, float]]:
    if not value or not isinstance(value, str):
        return None
    return ALIAS_TEMPO.get(value.strip().lower())


def normalize_energy_phrase(value: str) -> Optional[Tuple[float, float]]:
    if not value or not isinstance(value, str):
        return None
    return ALIAS_ENERGY.get(value.strip().lower())


_VOICE_LABELS_SET = {'female vocalists', 'female vocalist', 'male vocalists'}
TAG_EXCLUDE_SCORE = 0.3


def female_voice_exclusions(voices) -> List[str]:
    lows = {v.strip().lower() for v in voices or [] if isinstance(v, str) and v.strip()}
    if not lows or not all(v.startswith('male') for v in lows):
        return []
    return [v for v in config.VOICE_VOCAB if v.lower().startswith('female')]


def normalize_mood_list(values) -> dict:
    if not values:
        return {
            'mood_vector': [],
            'voices': [],
            'other_features': [],
            'energy_min': None,
            'energy_max': None,
            'tempo_min': None,
            'tempo_max': None,
            'dropped': [],
            'notes': [],
        }
    mv_all: List[str] = []
    voices_all: List[str] = []
    of_all: List[str] = []
    e_min = e_max = t_min = t_max = None
    dropped: List[str] = []
    notes: List[str] = []

    for raw in values:
        if not isinstance(raw, str):
            continue
        low = raw.strip().lower()
        if low in _OTHER_FEATURE_VOCAB_LOWER:
            canonical = _OTHER_FEATURE_VOCAB_LOWER[low]
            if canonical not in of_all:
                of_all.append(canonical)
            continue
        e = normalize_energy_phrase(raw)
        if e is not None:
            e_min = e[0] if e_min is None else min(e_min, e[0])
            e_max = e[1] if e_max is None else max(e_max, e[1])
            continue
        t = normalize_tempo_phrase(raw)
        if t is not None:
            t_min = t[0] if t_min is None else min(t_min, t[0])
            t_max = t[1] if t_max is None else max(t_max, t[1])
            continue
        mv, of = normalize_mood(raw, notes=notes)
        if not mv and not of:
            dropped.append(raw)
            notes.append(f"vocab_normalizer dropped unrecognized mood '{raw}'")
            continue
        for m in mv:
            if m in _VOICE_LABELS_SET:
                if m not in voices_all:
                    voices_all.append(m)
            elif m not in mv_all:
                mv_all.append(m)
        for o in of:
            if o not in of_all:
                of_all.append(o)
    return {
        'mood_vector': mv_all,
        'voices': voices_all,
        'other_features': of_all,
        'energy_min': e_min,
        'energy_max': e_max,
        'tempo_min': t_min,
        'tempo_max': t_max,
        'dropped': dropped,
        'notes': notes,
    }


def normalize_genre_list(values) -> dict:
    if not values:
        return {'genres': [], 'dropped': [], 'notes': []}
    out: List[str] = []
    dropped: List[str] = []
    notes: List[str] = []
    for raw in values:
        if not isinstance(raw, str):
            continue
        g = normalize_genre(raw, notes=notes)
        if g is None:
            dropped.append(raw)
            notes.append(f"vocab_normalizer dropped unrecognized genre '{raw}'")
            continue
        if g not in out:
            out.append(g)
    return {'genres': out, 'dropped': dropped, 'notes': notes}


def normalize_voices_list(values) -> dict:
    if not values:
        return {'voices': [], 'dropped': [], 'notes': []}
    voices_out: List[str] = []
    dropped: List[str] = []
    notes: List[str] = []
    for raw in values:
        if not isinstance(raw, str):
            continue
        mv, _ = normalize_mood(raw, notes=notes)
        matched = [m for m in mv if m in _VOICE_LABELS_SET]
        if not matched:
            dropped.append(raw)
            notes.append(f"vocab_normalizer dropped unrecognized voice '{raw}'")
            continue
        for m in matched:
            if m not in voices_out:
                voices_out.append(m)
    return {'voices': voices_out, 'dropped': dropped, 'notes': notes}
