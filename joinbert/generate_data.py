"""
Template-driven generator for JointBERT training data.

Why this exists
---------------
Hand-writing 5000 well-labeled NLU examples is slow and produces repetitive
data. This script combines:
    * a vocabulary of artists, songs, genres, moods, decades, lyric topics, ...
    * a template bank per intent (with inline Rasa-style `[value](type)` slots)
    * combinatorial multi-intent compositions

...to produce a balanced, diverse `training_data.json` for `train.py`.

Run:
    python generate_data.py             # default: 5000 examples
    python generate_data.py --n 8000    # override

Determinism: seeded with SEED below so the same vocab + templates produce the
same file. Change SEED if you want a fresh sample.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "training_data.json"
SEED = 1337

# Target distribution: more ai_brainstorm for improved brainstorming coverage
# ai_brainstorm increased to 15% (more variety: events + artist-hits + radio keywords)
SHARE = {
    "song_similarity":  0.125,
    "artist_similarity": 0.125,
    "text_search":       0.120,
    "song_alchemy":      0.120,
    "ai_brainstorm":     0.150,  # INCREASED: now 15% for better brainstorm coverage
    "search_database":   0.135,
    "lyrics_search":     0.125,
    "multi":             0.115,
}

# ======================================================================
# VOCABULARY
# ======================================================================
ARTISTS: list[str] = [
    "The Beatles", "Queen", "Pink Floyd", "Led Zeppelin", "The Rolling Stones",
    "David Bowie", "Bob Dylan", "Bruce Springsteen", "Elton John", "Michael Jackson",
    "Prince", "Madonna", "Stevie Wonder", "Frank Sinatra", "Tom Waits",
    "Neil Young", "Joni Mitchell", "Leonard Cohen", "Lou Reed", "Iggy Pop",
    "The Doors", "Jimi Hendrix", "Janis Joplin", "Eric Clapton", "Carlos Santana",
    "Eagles", "Fleetwood Mac", "The Who", "The Kinks", "The Velvet Underground",
    "Nirvana", "Pearl Jam", "Soundgarden", "Alice in Chains", "Foo Fighters",
    "Radiohead", "Muse", "Coldplay", "U2", "Oasis",
    "Blur", "The Smiths", "Joy Division", "New Order", "Depeche Mode",
    "The Cure", "Pixies", "Sonic Youth", "Arcade Fire", "Arctic Monkeys",
    "The Strokes", "Interpol", "The Killers", "Vampire Weekend", "Tame Impala",
    "MGMT", "LCD Soundsystem", "Daft Punk", "The Chemical Brothers", "The Prodigy",
    "Aphex Twin", "Boards of Canada", "Massive Attack", "Portishead", "Björk",
    "Beck", "Beastie Boys", "Run DMC", "Public Enemy", "A Tribe Called Quest",
    "Wu-Tang Clan", "Nas", "JAY-Z", "Eminem", "Kanye West",
    "Drake", "Kendrick Lamar", "J Cole", "Travis Scott", "Tyler the Creator",
    "Frank Ocean", "The Weeknd", "Childish Gambino", "Anderson Paak", "Mac Miller",
    "Beyoncé", "Rihanna", "Adele", "Taylor Swift", "Lady Gaga",
    "Billie Eilish", "Lorde", "Lana Del Rey", "Florence and the Machine", "Sia",
    "Dua Lipa", "Olivia Rodrigo", "Doja Cat", "Ariana Grande", "Bruno Mars",
    "Ed Sheeran", "Sam Smith", "Hozier", "John Mayer", "Mumford and Sons",
    "Imagine Dragons", "Maroon 5", "OneRepublic", "Twenty One Pilots", "Bon Iver",
    "Sufjan Stevens", "Fleet Foxes", "The National", "Phoebe Bridgers", "Mitski",
    "Mac DeMarco", "Father John Misty", "Big Thief", "Arctic Lake", "Slowdive",
    "Metallica", "Iron Maiden", "Black Sabbath", "Deep Purple", "AC DC",
    "Guns N Roses", "Megadeth", "Slayer", "Pantera", "Tool",
    "A Perfect Circle", "System of a Down", "Korn", "Slipknot", "Rammstein",
    "Mastodon", "Opeth", "Gojira", "Lamb of God", "Avenged Sevenfold",
    "Miles Davis", "John Coltrane", "Charlie Parker", "Duke Ellington", "Louis Armstrong",
    "Ella Fitzgerald", "Billie Holiday", "Nina Simone", "Thelonious Monk", "Dave Brubeck",
    "Herbie Hancock", "Chet Baker", "Bill Evans", "Kamasi Washington", "Snarky Puppy",
    "Bach", "Mozart", "Beethoven", "Chopin", "Debussy",
    "Stravinsky", "Mahler", "Wagner", "Vivaldi", "Tchaikovsky",
    "Glenn Gould", "Ludovico Einaudi", "Hans Zimmer", "Ennio Morricone", "John Williams",
    "Bob Marley", "Peter Tosh", "Burning Spear", "Toots and the Maytals", "Lee Scratch Perry",
    "ABBA", "Bee Gees", "Donna Summer", "Earth Wind and Fire", "Kool and the Gang",
    "Toto", "Journey", "Boston", "Foreigner", "Styx",
]

# Famous (song, artist) pairs used for song_similarity and a few alchemy items.
SONGS: list[tuple[str, str]] = [
    ("Bohemian Rhapsody", "Queen"),
    ("Hotel California", "Eagles"),
    ("Stairway to Heaven", "Led Zeppelin"),
    ("Imagine", "John Lennon"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Wonderwall", "Oasis"),
    ("Smooth Criminal", "Michael Jackson"),
    ("Billie Jean", "Michael Jackson"),
    ("Beat It", "Michael Jackson"),
    ("Thriller", "Michael Jackson"),
    ("Creep", "Radiohead"),
    ("Karma Police", "Radiohead"),
    ("No Surprises", "Radiohead"),
    ("Paranoid Android", "Radiohead"),
    ("Black", "Pearl Jam"),
    ("Alive", "Pearl Jam"),
    ("Even Flow", "Pearl Jam"),
    ("Take Five", "Dave Brubeck"),
    ("So What", "Miles Davis"),
    ("Sweet Child o Mine", "Guns N Roses"),
    ("November Rain", "Guns N Roses"),
    ("Hey Jude", "The Beatles"),
    ("Let It Be", "The Beatles"),
    ("Yesterday", "The Beatles"),
    ("Come Together", "The Beatles"),
    ("Africa", "Toto"),
    ("Mr Brightside", "The Killers"),
    ("Clocks", "Coldplay"),
    ("Yellow", "Coldplay"),
    ("Fix You", "Coldplay"),
    ("One", "U2"),
    ("With or Without You", "U2"),
    ("Comfortably Numb", "Pink Floyd"),
    ("Wish You Were Here", "Pink Floyd"),
    ("Money", "Pink Floyd"),
    ("Time", "Pink Floyd"),
    ("Riders on the Storm", "The Doors"),
    ("Light My Fire", "The Doors"),
    ("Don't Stop Believin'", "Journey"),
    ("Thunderstruck", "AC DC"),
    ("Back in Black", "AC DC"),
    ("Highway to Hell", "AC DC"),
    ("Enter Sandman", "Metallica"),
    ("Nothing Else Matters", "Metallica"),
    ("Master of Puppets", "Metallica"),
    ("Sad But True", "Metallica"),
    ("The Trooper", "Iron Maiden"),
    ("Run to the Hills", "Iron Maiden"),
    ("Fear of the Dark", "Iron Maiden"),
    ("Paranoid", "Black Sabbath"),
    ("Iron Man", "Black Sabbath"),
    ("Wonderful Tonight", "Eric Clapton"),
    ("Layla", "Eric Clapton"),
    ("Like a Rolling Stone", "Bob Dylan"),
    ("Blowin' in the Wind", "Bob Dylan"),
    ("Born to Run", "Bruce Springsteen"),
    ("Dancing in the Dark", "Bruce Springsteen"),
    ("Purple Rain", "Prince"),
    ("Kiss", "Prince"),
    ("Vogue", "Madonna"),
    ("Like a Prayer", "Madonna"),
    ("Material Girl", "Madonna"),
    ("Heroes", "David Bowie"),
    ("Space Oddity", "David Bowie"),
    ("Life on Mars", "David Bowie"),
    ("Hallelujah", "Leonard Cohen"),
    ("Suzanne", "Leonard Cohen"),
    ("Hurt", "Johnny Cash"),
    ("Folsom Prison Blues", "Johnny Cash"),
    ("Crazy", "Patsy Cline"),
    ("My Way", "Frank Sinatra"),
    ("New York New York", "Frank Sinatra"),
    ("Rolling in the Deep", "Adele"),
    ("Someone Like You", "Adele"),
    ("Hello", "Adele"),
    ("Halo", "Beyoncé"),
    ("Single Ladies", "Beyoncé"),
    ("Crazy in Love", "Beyoncé"),
    ("Shape of You", "Ed Sheeran"),
    ("Perfect", "Ed Sheeran"),
    ("Bad Guy", "Billie Eilish"),
    ("Ocean Eyes", "Billie Eilish"),
    ("Royals", "Lorde"),
    ("Born to Die", "Lana Del Rey"),
    ("Summertime Sadness", "Lana Del Rey"),
    ("Video Games", "Lana Del Rey"),
    ("Get Lucky", "Daft Punk"),
    ("One More Time", "Daft Punk"),
    ("Around the World", "Daft Punk"),
    ("Heart of Glass", "Blondie"),
    ("Sweet Dreams", "Eurythmics"),
    ("Take On Me", "a-ha"),
    ("Never Gonna Give You Up", "Rick Astley"),
    ("Africa", "Toto"),
    ("Tiny Dancer", "Elton John"),
    ("Rocket Man", "Elton John"),
    ("Your Song", "Elton John"),
    ("Bennie and the Jets", "Elton John"),
]

GENRES: list[str] = [
    "rock", "pop", "metal", "jazz", "blues", "classical", "country", "folk",
    "electronic", "techno", "house", "trance", "ambient", "hip hop", "rap",
    "R&B", "soul", "funk", "disco", "reggae", "ska", "punk", "indie",
    "indie rock", "alternative", "shoegaze", "grunge", "hardcore", "death metal",
    "black metal", "thrash metal", "progressive rock", "psychedelic", "synthwave",
    "lo-fi", "garage", "new wave", "post-punk", "post-rock", "dream pop",
    "K-pop", "J-pop", "latin", "bossa nova", "salsa", "tango", "flamenco",
    "gospel", "bluegrass", "drum and bass", "dubstep", "trap", "afrobeat",
]

MOODS: list[str] = [
    "happy", "sad", "danceable", "aggressive", "party", "relaxed",
    "melancholic", "uplifting", "dark", "dreamy", "romantic", "nostalgic",
    "epic", "playful", "intense", "soothing", "energetic", "moody",
    # Vocal styles
    "female vocalist", "male vocalist",
    "female singer", "male singer",
    "female vocal", "male vocal",
    "female voice", "male voice",
]

TEMPOS = ["slow", "fast", "medium"]
ENERGIES = ["low", "high", "medium", "calm", "intense"]
KEYS = ["C", "D", "E", "F", "G", "A", "B", "C#", "D#", "F#", "G#", "A#"]
SCALES = ["major", "minor"]

DECADES = ["50s", "60s", "70s", "80s", "90s", "2000s", "2010s", "2020s"]
RECENT_PHRASES = [
    "last year", "this year", "recent", "last 2 years", "last 3 years",
    "last 5 years", "last 10 years", "the past decade",
]
YEARS_SINGLE = [str(y) for y in range(1965, 2025, 2)]
YEARS_FROM = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]
YEARS_TO   = ["1999", "2004", "2009", "2014", "2019", "2024"]

RATINGS = ["5 stars", "4 stars", "favorites", "favourites", "top rated", "best rated"]

ALBUMS = [
    "Dark Side of the Moon", "The Wall", "Wish You Were Here", "Abbey Road",
    "Sgt. Pepper's Lonely Hearts Club Band", "Revolver", "Pet Sounds",
    "Thriller", "Back in Black", "Rumours", "Nevermind", "OK Computer",
    "Kid A", "In Rainbows", "The Joshua Tree", "Born to Run",
    "Songs in the Key of Life", "Blue", "Kind of Blue", "A Love Supreme",
    "London Calling", "Purple Rain", "Highway 61 Revisited", "Blonde on Blonde",
    "Hotel California", "Master of Puppets", "Appetite for Destruction",
    "The Velvet Underground & Nico", "Exile on Main St", "Born in the U.S.A.",
]

DESCRIPTIONS = [
    "piano music with rain sounds", "romantic dreamy guitar tracks",
    "energetic workout music with heavy bass", "chill lofi beats for studying",
    "upbeat acoustic ukulele songs", "heavy distorted guitars",
    "ambient electronic with synthesizers", "melancholic violin pieces",
    "funky bass lines and groovy beats", "sad piano ballads",
    "happy whistling tunes", "soft jazz with saxophone",
    "aggressive metal with double bass drums", "smooth R&B with sensual vocals",
    "acoustic folk with harmonica", "cinematic orchestral epic soundtrack",
    "reggae vibes with offbeat guitar", "psychedelic rock with phaser effects",
    "tropical beach house", "atmospheric vocal-driven indie",
    "deep dark techno", "smooth bossa nova guitar",
    "soulful gospel choir vocals", "whispery female vocals with reverb",
    "fuzzy garage rock", "shimmering dream pop", "driving krautrock",
    "minimalist piano with cello", "chiptune retro 8-bit", "lush string arrangements",
    "punchy brass section", "hypnotic kosmische synths", "wistful folk guitar",
    "slick funk grooves", "raw blues harmonica", "atmospheric post-rock crescendos",
    "thumping kick drums and synth stabs", "glitchy IDM textures",
    "spacey synth pads", "vibrato-heavy crooner vocals",
]

DESC_VIBES = [
    "chill", "upbeat", "moody", "intense", "dreamy", "haunting", "uplifting",
    "smooth", "raw", "warm", "icy", "lush", "minimal", "epic",
]

LYRIC_TOPICS = [
    "heartbreak", "love", "loneliness", "rebellion", "summer love",
    "growing up", "freedom", "loss", "redemption", "joy", "anger",
    "regret", "hope", "death", "addiction", "war", "friendship", "betrayal",
    "leaving home", "coming home", "small town life", "the road",
    "broken dreams", "young love", "first love", "missing someone",
    "missing home", "feeling lonely", "feeling lost", "fighting back",
    "letting go", "moving on", "starting over", "city life", "the rain",
    "the ocean", "the night", "midnight drives", "drinking alone",
    "Sunday mornings", "summer nights", "winter blues", "Christmas",
    "memories of childhood", "your first crush", "growing old",
    "political protest", "social justice", "feminism", "race", "class struggle",
    "spirituality", "doubt", "faith", "addiction recovery", "mental health",
]

EVENTS = [
    # Chart & Radio keywords
    "Grammy winners", "Billboard top 10", "Billboard hot 100",
    "radio hits", "top radio songs", "radio classics", "radio play",
    "chart-toppers", "chart hits", "trending pop songs", "number one hits",
    # Viral & Trending
    "viral TikTok hits", "viral songs", "viral hits", "trending songs",
    "songs trending on Spotify", "TikTok viral", "internet famous songs",
    # Awards & Recognition
    "Oscar winning songs", "Grammy award winners", "MTV Music Awards",
    "Billboard Music Awards", "Brit Awards winners", "Grammys all-time greats",
    # Awards/Prestige keywords
    "award-winning songs", "platinum records", "gold records", "award-winning hits",
    "bestselling songs of all time", "bestselling albums", "blockbuster hits",
    # Curated Lists & Classics
    "Christmas classics", "summer anthems of 2024", "Halloween playlist",
    "best wedding songs", "karaoke classics", "iconic movie soundtracks",
    "funeral songs", "songs you must hear before you die",
    # Festival & Industry
    "Coachella 2024", "Eurovision winners", "Super Bowl halftime",
    "the Woodstock festival", "Glastonbury headliner classics",
    "Lollapalooza highlights", "MTV Unplugged classics",
    # Magazine/Publication Top Lists
    "Rolling Stone top 500", "Pitchfork best new music", "Billboard year-end charts",
    "best bossa nova ever recorded", "essential punk songs",
    # Time-based top lists
    "anthems of the 2000s", "greatest songs of the 80s", "best of the 90s",
    "hits of the 2020s", "top songs of the decade", "timeless classics",
    # Famous/Legendary
    "legendary artists", "greatest musicians", "iconic bands",
    "legendary performances", "all-time classics", "timeless hits",
    # Radio-specific variants
    "top 40 hits", "adult contemporary hits", "rock radio classics",
    "country radio hits", "hip hop radio hits", "pop radio hits",
    "urban radio hits", "oldies but goodies", "classic hits",
]

# ======================================================================
# TEMPLATES (each contains inline `[value](type)` slots)
# ======================================================================
# NEW: multi-song song_similarity (2 or 3 song+artist pairs).
# Dispatcher pairs songs with artists by order: song[0]+artist[0], song[1]+artist[1], ...
SONG_SIM_MULTI_2_TEMPLATES = [
    "songs like [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "tracks similar to [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "find more like [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "give me [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "I want songs like [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "play stuff like [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "anything similar to [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
    "vibes like [{s1}](song) by [{a1}](artist) and [{s2}](song) by [{a2}](artist)",
]
SONG_SIM_MULTI_3_TEMPLATES = [
    "songs like [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist) and [{s3}](song) by [{a3}](artist)",
    "tracks similar to [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist) and [{s3}](song) by [{a3}](artist)",
    "find more like [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist), and [{s3}](song) by [{a3}](artist)",
    "play stuff like [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist) and [{s3}](song) by [{a3}](artist)",
    "I want vibes of [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist) and [{s3}](song) by [{a3}](artist)",
    "similar to [{s1}](song) by [{a1}](artist) and also [{s2}](song) by [{a2}](artist) and [{s3}](song) by [{a3}](artist)",
    "music like [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist), or [{s3}](song) by [{a3}](artist)",
    "give me more tracks like [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist), and [{s3}](song) by [{a3}](artist)",
    "anything that sounds like [{s1}](song) by [{a1}](artist), [{s2}](song) by [{a2}](artist), or [{s3}](song) by [{a3}](artist)",
]

# NEW: Artist-dash format (artist before song, separated by dash)
SONG_SIM_ARTIST_DASH_TEMPLATES = [
    "similar songs to [{artist}](artist) - [{song}](song)",
    "songs like [{artist}](artist) - [{song}](song)",
    "more like [{artist}](artist) - [{song}](song)",
    "find music similar to [{artist}](artist) - [{song}](song)",
    "play stuff like [{artist}](artist) - [{song}](song)",
    "tracks in the vibe of [{artist}](artist) - [{song}](song)",
    "I love [{artist}](artist) - [{song}](song), more like that",
    "give me [{artist}](artist) - [{song}](song) style music",
]

SONG_SIM_ARTIST_DASH_MULTI_2 = [
    "songs like [{a1}](artist) - [{s1}](song) and [{a2}](artist) - [{s2}](song)",
    "more like [{a1}](artist) - [{s1}](song) and [{a2}](artist) - [{s2}](song)",
    "similar to [{a1}](artist) - [{s1}](song) and also [{a2}](artist) - [{s2}](song)",
    "find me [{a1}](artist) - [{s1}](song) and [{a2}](artist) [{s2}](song) style",
]

# No-separator format (artist and song concatenated like "ed sheeran 2step")
SONG_SIM_NO_SEP_TEMPLATES = [
    "songs like [{artist}](artist) [{song}](song)",
    "more [{artist}](artist) [{song}](song)",
    "similar to [{artist}](artist) [{song}](song)",
    "vibes of [{artist}](artist) [{song}](song)",
]

SONG_SIM_TEMPLATES = [
    "songs like [{song}](song) by [{artist}](artist)",
    "find tracks similar to [{song}](song) by [{artist}](artist)",
    "more like [{song}](song) by [{artist}](artist)",
    "what sounds like [{song}](song) by [{artist}](artist)",
    "give me songs in the style of [{song}](song) by [{artist}](artist)",
    "similar to [{song}](song) by [{artist}](artist)",
    "play things like [{song}](song) by [{artist}](artist)",
    "find me similar tracks to [{song}](song) by [{artist}](artist)",
    "more songs in the vein of [{song}](song) by [{artist}](artist)",
    "vibes like [{song}](song) by [{artist}](artist)",
    "play stuff like [{song}](song) by [{artist}](artist)",
    "tracks similar to [{song}](song) by [{artist}](artist)",
    "songs reminiscent of [{song}](song) by [{artist}](artist)",
    "find more like [{song}](song) by [{artist}](artist)",
    "play similar to [{song}](song) by [{artist}](artist)",
    "music like [{song}](song) by [{artist}](artist)",
    "stuff like [{song}](song) by [{artist}](artist)",
    "more like [{song}](song) by [{artist}](artist) please",
    "I want songs like [{song}](song) by [{artist}](artist)",
    "tracks in the style of [{song}](song) by [{artist}](artist)",
    "give me vibes of [{song}](song) by [{artist}](artist)",
    "more in the mood of [{song}](song) by [{artist}](artist)",
    "anything that sounds like [{song}](song) by [{artist}](artist)",
    "tracks reminiscent of [{song}](song) by [{artist}](artist)",
    "I'm looking for songs like [{song}](song) by [{artist}](artist)",
    "songs that feel like [{song}](song) by [{artist}](artist)",
    "more songs like [{song}](song) by [{artist}](artist)",
    "anything similar to [{song}](song) by [{artist}](artist)",
    "show me tracks like [{song}](song) by [{artist}](artist)",
    "make a playlist around [{song}](song) by [{artist}](artist)",
]

ARTIST_SIM_TEMPLATES = [
    "songs by [{artist}](artist)",
    "tracks from [{artist}](artist)",
    "more music from [{artist}](artist)",
    "play me [{artist}](artist)",
    "everything by [{artist}](artist)",
    "artists similar to [{artist}](artist)",
    "stuff like [{artist}](artist)",
    "music from [{artist}](artist)",
    "play [{artist}](artist) and similar",
    "songs in the style of [{artist}](artist)",
    "I want [{artist}](artist)",
    "tracks similar to [{artist}](artist)",
    "artists like [{artist}](artist)",
    "play more [{artist}](artist)",
    "give me [{artist}](artist)",
    "music by [{artist}](artist)",
    "anything from [{artist}](artist)",
    "more [{artist}](artist) please",
    "songs from [{artist}](artist)",
    "give me [{artist}](artist) tracks",
    "I want music by [{artist}](artist)",
    "more songs from [{artist}](artist)",
    "play me some [{artist}](artist)",
    "what's in your library by [{artist}](artist)",
    "more like [{artist}](artist)",
    "fans of [{artist}](artist) would love",
    "anyone who likes [{artist}](artist)",
    "build me a [{artist}](artist) playlist",
    "I'm in the mood for [{artist}](artist)",
    "today I feel like [{artist}](artist)",
]

# NEW: multi-artist artist_similarity. Use neutral conjunctions ("and",
# comma-lists) — NEVER alchemy keywords ("meets"/"blend"/"combine"/"mix"/"+").
# The dispatcher fires one artist_similarity call per extracted artist.
ARTIST_SIM_MULTI_2_TEMPLATES = [
    "songs similar to [{a1}](artist) and [{a2}](artist)",
    "play me [{a1}](artist) and [{a2}](artist)",
    "I want [{a1}](artist) and [{a2}](artist)",
    "music from [{a1}](artist) and [{a2}](artist)",
    "give me [{a1}](artist) and [{a2}](artist)",
    "tracks by [{a1}](artist) and [{a2}](artist)",
    "fans of [{a1}](artist) and [{a2}](artist) would love",
    "play [{a1}](artist) or [{a2}](artist)",
    "anything by [{a1}](artist) and [{a2}](artist)",
    "songs from [{a1}](artist), [{a2}](artist)",
    "stuff like [{a1}](artist) and [{a2}](artist)",
    "more music from [{a1}](artist) and [{a2}](artist)",
    "artists similar to [{a1}](artist) and [{a2}](artist)",
]
ARTIST_SIM_MULTI_3_TEMPLATES = [
    "songs similar to [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "play me [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "I want [{a1}](artist), [{a2}](artist), and [{a3}](artist)",
    "music from [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "give me [{a1}](artist), [{a2}](artist), [{a3}](artist)",
    "tracks by [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "fans of [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "songs from [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "stuff like [{a1}](artist), [{a2}](artist), [{a3}](artist)",
    "artists similar to [{a1}](artist), [{a2}](artist) and [{a3}](artist)",
    "play me [{a1}](artist), [{a2}](artist), or [{a3}](artist)",
    "I'm in the mood for [{a1}](artist), [{a2}](artist), and [{a3}](artist)",
    "more music from [{a1}](artist), [{a2}](artist), and [{a3}](artist)",
    "songs in the style of [{a1}](artist), [{a2}](artist), or [{a3}](artist)",
    "anything from [{a1}](artist), [{a2}](artist), or [{a3}](artist)",
]

TEXT_SEARCH_TEMPLATES = [
    "[{desc}](description)",
    "songs with [{desc}](description)",
    "I want [{desc}](description)",
    "play [{desc}](description)",
    "music for [{desc}](description)",
    "give me [{desc}](description)",
    "find me [{desc}](description)",
    "anything with [{desc}](description)",
    "tracks featuring [{desc}](description)",
    "play me some [{desc}](description)",
    "I'm looking for [{desc}](description)",
    "I need [{desc}](description)",
    "show me [{desc}](description)",
    "search for [{desc}](description)",
    "let me hear [{desc}](description)",
    "in the mood for [{desc}](description)",
    "something like [{desc}](description)",
]

# Alchemy templates take 2-3 add_artists and (optionally) 1 subtract_artist.
ALCHEMY_ADD_2_TEMPLATES = [
    "mix of [{a1}](add_artist) and [{a2}](add_artist)",
    "blend [{a1}](add_artist) and [{a2}](add_artist)",
    "combine [{a1}](add_artist) with [{a2}](add_artist)",
    "[{a1}](add_artist) meets [{a2}](add_artist)",
    "[{a1}](add_artist) crossed with [{a2}](add_artist)",
    "[{a1}](add_artist) plus [{a2}](add_artist)",
    "[{a1}](add_artist) and [{a2}](add_artist) combined",
    "mash [{a1}](add_artist) with [{a2}](add_artist)",
    "play like [{a1}](add_artist) + [{a2}](add_artist)",
    "[{a1}](add_artist) merged with [{a2}](add_artist)",
    "blend of [{a1}](add_artist) and [{a2}](add_artist)",
    "give me a mix of [{a1}](add_artist) and [{a2}](add_artist)",
]
ALCHEMY_ADD_3_TEMPLATES = [
    "[{a1}](add_artist) meets [{a2}](add_artist) meets [{a3}](add_artist)",
    "[{a1}](add_artist) plus [{a2}](add_artist) plus [{a3}](add_artist)",
    "blend of [{a1}](add_artist) [{a2}](add_artist) and [{a3}](add_artist)",
    "mix [{a1}](add_artist) [{a2}](add_artist) and [{a3}](add_artist)",
    "[{a1}](add_artist) crossed with [{a2}](add_artist) and [{a3}](add_artist)",
    "play like [{a1}](add_artist) + [{a2}](add_artist) + [{a3}](add_artist)",
]
ALCHEMY_SUBTRACT_TEMPLATES = [
    "[{a1}](add_artist) plus [{a2}](add_artist) minus [{sub}](subtract_artist)",
    "[{a1}](add_artist) and [{a2}](add_artist) but not [{sub}](subtract_artist)",
    "mix of [{a1}](add_artist) and [{a2}](add_artist) without [{sub}](subtract_artist)",
    "blend [{a1}](add_artist) with [{a2}](add_artist) excluding [{sub}](subtract_artist)",
]
ALCHEMY_SUBTRACT_GENRE_TEMPLATES = [
    "[{a1}](add_artist) but not [{g}](subtract_genre)",
    "[{a1}](add_artist) without [{g}](subtract_genre)",
    "[{a1}](add_artist) excluding [{g}](subtract_genre)",
    "[{a1}](add_artist) minus the [{g}](subtract_genre)",
    "[{a1}](add_artist) plus [{a2}](add_artist) but not [{g}](subtract_genre)",
]

BRAINSTORM_TEMPLATES = [
    "[{event}](event)",
    "best of [{event}](event)",
    "songs from [{event}](event)",
    "play me [{event}](event)",
    "give me [{event}](event)",
    "[{event}](event) playlist",
    "the best [{event}](event)",
    "iconic [{event}](event)",
    "classics of [{event}](event)",
    "highlights from [{event}](event)",
    "famous [{event}](event)",
    "essential [{event}](event)",
    "I want [{event}](event)",
    "play [{event}](event) for me",
    "show me [{event}](event)",
    "I'm in the mood for [{event}](event)",
    "make me a [{event}](event) playlist",
]

# NEW: Artist-specific hits/top songs -> ai_brainstorm (NOT artist_similarity)
# Distinguish from artist_similarity with keywords: top/best/greatest/iconic/essential/classic/famous/most-popular/radio/chart/blockbuster/platinum
BRAINSTORM_ARTIST_HITS_TEMPLATES = [
    # Top/Best variants
    "top songs of [{artist}](artist)",
    "best hits by [{artist}](artist)",
    "greatest hits of [{artist}](artist)",
    "most famous songs of [{artist}](artist)",
    "iconic songs by [{artist}](artist)",
    "essential [{artist}](artist) tracks",
    "classic songs of [{artist}](artist)",
    "most popular [{artist}](artist) songs",
    # Radio-specific
    "top radio songs of [{artist}](artist)",
    "radio hits by [{artist}](artist)",
    "radio classics of [{artist}](artist)",
    "radio play favorites from [{artist}](artist)",
    # Chart/Commercial keywords
    "chart-topping songs of [{artist}](artist)",
    "blockbuster hits by [{artist}](artist)",
    "platinum records of [{artist}](artist)",
    "bestselling songs of [{artist}](artist)",
    # Award/Recognition
    "award-winning songs of [{artist}](artist)",
    "Grammy awarded tracks by [{artist}](artist)",
    # Famous/Legendary
    "legendary [{artist}](artist) songs",
    "famous [{artist}](artist) tracks",
    "timeless classics by [{artist}](artist)",
    "all-time greatest [{artist}](artist) songs",
    # Direct hit requests
    "give me the hits of [{artist}](artist)",
    "what are the best songs by [{artist}](artist)",
    "[{artist}](artist) greatest tracks ever",
    "best of [{artist}](artist) playlist",
    "make me a [{artist}](artist) hits playlist",
    "[{artist}](artist) top charting songs",
    "most iconic [{artist}](artist) songs",
]

LYRICS_TEMPLATES = [
    "songs about [{topic}](lyrics_query)",
    "lyrics about [{topic}](lyrics_query)",
    "tracks with lyrics about [{topic}](lyrics_query)",
    "find songs about [{topic}](lyrics_query)",
    "music with [{topic}](lyrics_query) lyrics",
    "songs whose lyrics talk about [{topic}](lyrics_query)",
    "give me songs about [{topic}](lyrics_query)",
    "I want songs about [{topic}](lyrics_query)",
    "tracks that mention [{topic}](lyrics_query)",
    "lyrics that mention [{topic}](lyrics_query)",
    "anything about [{topic}](lyrics_query) lyrically",
    "songs with words about [{topic}](lyrics_query)",
    "music about [{topic}](lyrics_query)",
    "play songs about [{topic}](lyrics_query)",
    "search lyrics for [{topic}](lyrics_query)",
    "find me lyrics about [{topic}](lyrics_query)",
    "what songs are about [{topic}](lyrics_query)",
    "songs with [{topic}](lyrics_query) in the lyrics",
    "I'm looking for songs about [{topic}](lyrics_query)",
    "lyrically focused on [{topic}](lyrics_query)",
]
LYRICS_TWO_TOPIC_TEMPLATES = [
    "songs about [{t1}](lyrics_query) and [{t2}](lyrics_query)",
    "lyrics about [{t1}](lyrics_query) and [{t2}](lyrics_query)",
    "tracks about [{t1}](lyrics_query) or [{t2}](lyrics_query)",
    "music about both [{t1}](lyrics_query) and [{t2}](lyrics_query)",
]

# search_database templates – sample a subset of slots to keep variety high.
DB_SINGLE_SLOT_TEMPLATES = [
    "[{genre}](genre) songs",
    "[{genre}](genre) music",
    "play [{genre}](genre)",
    "give me [{genre}](genre)",
    "I want [{genre}](genre)",
    "show me [{genre}](genre) tracks",
    "[{mood}](mood) songs",
    "[{mood}](mood) tracks",
    "play something [{mood}](mood)",
    "I'm feeling [{mood}](mood)",
    "[{tempo}](tempo) songs",
    "[{tempo}](tempo) tracks",
    "[{energy}](energy) energy music",
    "music with [{energy}](energy) energy",
    "music from the [{decade}](time_range)",
    "songs from the [{decade}](time_range)",
    "songs from [{year}](year)",
    "music from [{year}](year)",
    "music from [{recent}](time_range)",
    "songs from [{recent}](time_range)",
    "[{rating}](rating)",
    "my [{rating}](rating)",
    "tracks from album [{album}](album)",
    "songs from [{album}](album) album",
    "music in [{key}](key) [{scale}](scale)",
]
DB_TWO_SLOT_TEMPLATES = [
    "[{genre}](genre) songs from the [{decade}](time_range)",
    "[{genre}](genre) music from [{recent}](time_range)",
    "[{genre}](genre) tracks rated [{rating}](rating)",
    "[{mood}](mood) [{genre}](genre) songs",
    "[{tempo}](tempo) [{genre}](genre) tracks",
    "[{tempo}](tempo) [{mood}](mood) songs",
    "[{genre}](genre) songs from [{year}](year)",
    "[{genre}](genre) songs from [{yfrom}](year) to [{yto}](year)",
    "[{genre}](genre) with [{energy}](energy) energy",
    "[{mood}](mood) [{tempo}](tempo) songs",
    "[{rating}](rating) [{genre}](genre) tracks",
    "[{genre}](genre) songs in [{key}](key) [{scale}](scale)",
    "[{mood}](mood) [{genre}](genre) from the [{decade}](time_range)",
    "[{energy}](energy) energy [{genre}](genre) songs",
    "music from the [{decade}](time_range) rated [{rating}](rating)",
    "[{genre}](genre) [{mood}](mood) [{tempo}](tempo)",
    "[{genre}](genre) with [{mood}](mood) from [{year}](year)",
    "[{mood}](mood) [{genre}](genre) music",
    "songs with [{mood}](mood) from the [{decade}](time_range)",
]
DB_THREE_SLOT_TEMPLATES = [
    "[{tempo}](tempo) [{genre}](genre) from the [{decade}](time_range)",
    "[{mood}](mood) [{genre}](genre) with [{energy}](energy) energy",
    "[{genre}](genre) songs from [{decade}](time_range) rated [{rating}](rating)",
    "[{tempo}](tempo) [{mood}](mood) [{genre}](genre) tracks",
    "[{rating}](rating) [{tempo}](tempo) [{genre}](genre)",
]

# Multi-tool composition templates
MULTI_ARTIST_DESC = [
    ("songs like [{artist}](artist) but more [{vibe}](description)",        ["artist_similarity", "text_search"]),
    ("[{artist}](artist) but [{vibe}](description)",                        ["artist_similarity", "text_search"]),
    ("more [{artist}](artist) but [{vibe}](description) sounding",          ["artist_similarity", "text_search"]),
    ("[{vibe}](description) tracks similar to [{artist}](artist)",          ["artist_similarity", "text_search"]),
    ("give me [{artist}](artist) but a bit more [{vibe}](description)",     ["artist_similarity", "text_search"]),
]
MULTI_ARTIST_GENRE = [
    ("[{genre}](genre) songs like [{artist}](artist)",                      ["artist_similarity", "search_database"]),
    ("[{genre}](genre) tracks similar to [{artist}](artist)",               ["artist_similarity", "search_database"]),
    ("[{artist}](artist) and other [{genre}](genre)",                       ["artist_similarity", "search_database"]),
    ("similar to [{artist}](artist) and also [{genre}](genre)",             ["artist_similarity", "search_database"]),
]
MULTI_ARTIST_TIME = [
    ("[{artist}](artist) from the [{decade}](time_range)",                  ["artist_similarity", "search_database"]),
    ("[{artist}](artist) tracks from [{recent}](time_range)",               ["artist_similarity", "search_database"]),
    ("early [{artist}](artist) from [{decade}](time_range)",                ["artist_similarity", "search_database"]),
]
MULTI_ARTIST_RATING = [
    ("[{artist}](artist) tracks rated [{rating}](rating)",                  ["artist_similarity", "search_database"]),
    ("my [{rating}](rating) by [{artist}](artist)",                         ["artist_similarity", "search_database"]),
    ("[{artist}](artist) — only the [{rating}](rating)",                    ["artist_similarity", "search_database"]),
]
MULTI_EVENT_ARTIST = [
    ("best [{event}](event) similar to [{artist}](artist)",                 ["ai_brainstorm", "artist_similarity"]),
    ("[{event}](event) from [{artist}](artist)",                            ["ai_brainstorm", "artist_similarity"]),
    ("[{event}](event) like [{artist}](artist)",                            ["ai_brainstorm", "artist_similarity"]),
]
MULTI_SONG_DESC = [
    ("songs like [{song}](song) by [{artist}](artist) but [{vibe}](description)",
        ["song_similarity", "text_search"]),
    ("tracks similar to [{song}](song) by [{artist}](artist) — more [{vibe}](description)",
        ["song_similarity", "text_search"]),
]
MULTI_LYRICS_GENRE = [
    ("[{genre}](genre) songs about [{topic}](lyrics_query)",                ["lyrics_search", "search_database"]),
    ("[{genre}](genre) with lyrics about [{topic}](lyrics_query)",          ["lyrics_search", "search_database"]),
    ("[{genre}](genre) tracks whose lyrics mention [{topic}](lyrics_query)", ["lyrics_search", "search_database"]),
]
MULTI_LYRICS_ARTIST = [
    ("[{artist}](artist) songs about [{topic}](lyrics_query)",              ["lyrics_search", "artist_similarity"]),
    ("[{artist}](artist) with lyrics about [{topic}](lyrics_query)",        ["lyrics_search", "artist_similarity"]),
    ("[{artist}](artist) tracks whose lyrics are about [{topic}](lyrics_query)",
        ["lyrics_search", "artist_similarity"]),
]
MULTI_LYRICS_TIME = [
    ("songs about [{topic}](lyrics_query) from the [{decade}](time_range)",
        ["lyrics_search", "search_database"]),
    ("[{decade}](time_range) songs about [{topic}](lyrics_query)",
        ["lyrics_search", "search_database"]),
]
# NEW: song_similarity + artist_similarity combo. Dispatcher pairs the FIRST
# song with the FIRST artist (song_sim), then routes any remaining artists to
# artist_sim.
MULTI_SONG_ARTIST = [
    ("songs like [{song}](song) by [{a1}](artist) and more by [{a2}](artist)",
        ["song_similarity", "artist_similarity"]),
    ("tracks similar to [{song}](song) by [{a1}](artist) and also [{a2}](artist)",
        ["song_similarity", "artist_similarity"]),
    ("find more like [{song}](song) by [{a1}](artist) and stuff from [{a2}](artist)",
        ["song_similarity", "artist_similarity"]),
    ("songs like [{song}](song) by [{a1}](artist) plus more [{a2}](artist)",
        ["song_similarity", "artist_similarity"]),
    ("give me [{song}](song) by [{a1}](artist) and anything by [{a2}](artist)",
        ["song_similarity", "artist_similarity"]),
    ("vibes of [{song}](song) by [{a1}](artist) plus [{a2}](artist) tracks",
        ["song_similarity", "artist_similarity"]),
]

# ======================================================================
# GENERATION HELPERS
# ======================================================================
INLINE_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def is_valid(annotated: str) -> bool:
    """Reject anything where bracket annotations didn't expand cleanly."""
    # Must contain at least one [..](..) annotation
    if not INLINE_RE.search(annotated):
        return False
    # Strip annotations and verify no stray brackets remain
    stripped = INLINE_RE.sub(lambda m: m.group(1), annotated)
    if "[" in stripped or "]" in stripped or "(" in stripped or ")" in stripped:
        return False
    return True


def emit(text: str, intents: list[str], pool: set[str], out: list[dict]) -> None:
    if not is_valid(text):
        return
    if text in pool:
        return
    pool.add(text)
    out.append({"text": text, "intents": intents})


def gen_song_similarity(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    """Mix of single-pair (50%), two-pair (30%), three-pair (20%) with varied formats."""
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 5:
        tries += 1
        # Three-way choice: pair size, then format variant
        # INCREASED multi-pair weight: single 50% → two 30% → three 20%
        pair_mode = rng.choices(["single", "two", "three"], weights=[0.50, 0.30, 0.20])[0]
        song, artist = None, None

        if pair_mode == "single":
            song, artist = rng.choice(SONGS)
            # Choose format: standard (70%), dash (20%), no-sep (10%)
            fmt = rng.choices(
                ["standard", "dash", "no_sep"],
                weights=[0.70, 0.20, 0.10]
            )[0]
            if fmt == "standard":
                tpl = rng.choice(SONG_SIM_TEMPLATES)
                text = tpl.format(song=song, artist=artist)
            elif fmt == "dash":
                tpl = rng.choice(SONG_SIM_ARTIST_DASH_TEMPLATES)
                text = tpl.format(artist=artist, song=song)
            else:  # no_sep
                tpl = rng.choice(SONG_SIM_NO_SEP_TEMPLATES)
                text = tpl.format(artist=artist, song=song)
        elif pair_mode == "two":
            (s1, a1), (s2, a2) = rng.sample(SONGS, 2)
            # Choose format: standard multi-2 (80%) or dash multi-2 (20%)
            fmt = rng.choices(["standard", "dash"], weights=[0.80, 0.20])[0]
            if fmt == "standard":
                tpl = rng.choice(SONG_SIM_MULTI_2_TEMPLATES)
                text = tpl.format(s1=s1, a1=a1, s2=s2, a2=a2)
            else:  # dash
                tpl = rng.choice(SONG_SIM_ARTIST_DASH_MULTI_2)
                text = tpl.format(a1=a1, s1=s1, a2=a2, s2=s2)
        else:  # three
            (s1, a1), (s2, a2), (s3, a3) = rng.sample(SONGS, 3)
            tpl = rng.choice(SONG_SIM_MULTI_3_TEMPLATES)
            text = tpl.format(s1=s1, a1=a1, s2=s2, a2=a2, s3=s3, a3=a3)
        emit(text, ["song_similarity"], pool, out)


def gen_artist_similarity(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    """Mix of single (55%), two-artist (30%), three-artist (15%) templates."""
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 5:
        tries += 1
        # INCREASED multi-artist weight: single 55% → two 30% → three 15%
        mode = rng.choices(["single", "two", "three"], weights=[0.55, 0.30, 0.15])[0]
        if mode == "single":
            artist = rng.choice(ARTISTS)
            tpl = rng.choice(ARTIST_SIM_TEMPLATES)
            text = tpl.format(artist=artist)
        elif mode == "two":
            a1, a2 = rng.sample(ARTISTS, 2)
            tpl = rng.choice(ARTIST_SIM_MULTI_2_TEMPLATES)
            text = tpl.format(a1=a1, a2=a2)
        else:
            a1, a2, a3 = rng.sample(ARTISTS, 3)
            tpl = rng.choice(ARTIST_SIM_MULTI_3_TEMPLATES)
            text = tpl.format(a1=a1, a2=a2, a3=a3)
        emit(text, ["artist_similarity"], pool, out)


def gen_text_search(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 5:
        tries += 1
        desc = rng.choice(DESCRIPTIONS)
        tpl = rng.choice(TEXT_SEARCH_TEMPLATES)
        text = tpl.format(desc=desc)
        emit(text, ["text_search"], pool, out)


def gen_song_alchemy(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 5:
        tries += 1
        mode = rng.choices(
            ["add2", "add3", "sub_artist", "sub_genre"],
            weights=[0.45, 0.30, 0.10, 0.15],
        )[0]
        if mode == "add2":
            a1, a2 = rng.sample(ARTISTS, 2)
            tpl = rng.choice(ALCHEMY_ADD_2_TEMPLATES)
            text = tpl.format(a1=a1, a2=a2)
        elif mode == "add3":
            a1, a2, a3 = rng.sample(ARTISTS, 3)
            tpl = rng.choice(ALCHEMY_ADD_3_TEMPLATES)
            text = tpl.format(a1=a1, a2=a2, a3=a3)
        elif mode == "sub_artist":
            a1, a2, sub = rng.sample(ARTISTS, 3)
            tpl = rng.choice(ALCHEMY_SUBTRACT_TEMPLATES)
            text = tpl.format(a1=a1, a2=a2, sub=sub)
        else:  # sub_genre
            tpl = rng.choice(ALCHEMY_SUBTRACT_GENRE_TEMPLATES)
            if "{a2}" in tpl:
                a1, a2 = rng.sample(ARTISTS, 2)
                text = tpl.format(a1=a1, a2=a2, g=rng.choice(GENRES))
            else:
                text = tpl.format(a1=rng.choice(ARTISTS), g=rng.choice(GENRES))
        emit(text, ["song_alchemy"], pool, out)


def gen_brainstorm(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 5:
        tries += 1
        # Choose brainstorm type: event-based (60%) or artist-hits (40%)
        # INCREASED artist-hits: now more examples for "top songs", "radio hits", "chart-toppers", etc.
        btype = rng.choices(["event", "artist_hits"], weights=[0.60, 0.40])[0]
        if btype == "event":
            event = rng.choice(EVENTS)
            tpl = rng.choice(BRAINSTORM_TEMPLATES)
            text = tpl.format(event=event)
        else:  # artist_hits
            artist = rng.choice(ARTISTS)
            tpl = rng.choice(BRAINSTORM_ARTIST_HITS_TEMPLATES)
            text = tpl.format(artist=artist)
        emit(text, ["ai_brainstorm"], pool, out)


def gen_lyrics_search(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 5:
        tries += 1
        if rng.random() < 0.15:
            t1, t2 = rng.sample(LYRIC_TOPICS, 2)
            tpl = rng.choice(LYRICS_TWO_TOPIC_TEMPLATES)
            text = tpl.format(t1=t1, t2=t2)
        else:
            topic = rng.choice(LYRIC_TOPICS)
            tpl = rng.choice(LYRICS_TEMPLATES)
            text = tpl.format(topic=topic)
        emit(text, ["lyrics_search"], pool, out)


def gen_search_database(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    target = len(out) + n
    tries = 0
    while len(out) < target and tries < n * 6:
        tries += 1
        bucket = rng.choices(["one", "two", "three"], weights=[0.35, 0.45, 0.20])[0]
        ctx = {
            "genre": rng.choice(GENRES), "mood": rng.choice(MOODS),
            "tempo": rng.choice(TEMPOS), "energy": rng.choice(ENERGIES),
            "key": rng.choice(KEYS), "scale": rng.choice(SCALES),
            "decade": rng.choice(DECADES), "year": rng.choice(YEARS_SINGLE),
            "yfrom": rng.choice(YEARS_FROM), "yto": rng.choice(YEARS_TO),
            "recent": rng.choice(RECENT_PHRASES), "rating": rng.choice(RATINGS),
            "album": rng.choice(ALBUMS),
        }
        if bucket == "one":
            tpl = rng.choice(DB_SINGLE_SLOT_TEMPLATES)
        elif bucket == "two":
            tpl = rng.choice(DB_TWO_SLOT_TEMPLATES)
        else:
            tpl = rng.choice(DB_THREE_SLOT_TEMPLATES)
        try:
            text = tpl.format(**ctx)
        except KeyError:
            continue
        emit(text, ["search_database"], pool, out)


def gen_multi(n: int, rng: random.Random, pool: set[str], out: list[dict]) -> None:
    target = len(out) + n
    tries = 0
    groups: list[tuple[list[tuple[str, list[str]]], float]] = [
        (MULTI_ARTIST_DESC,   0.14),
        (MULTI_ARTIST_GENRE,  0.12),
        (MULTI_ARTIST_TIME,   0.12),
        (MULTI_ARTIST_RATING, 0.09),
        (MULTI_EVENT_ARTIST,  0.09),
        (MULTI_SONG_DESC,     0.07),
        (MULTI_LYRICS_GENRE,  0.10),
        (MULTI_LYRICS_ARTIST, 0.09),
        (MULTI_LYRICS_TIME,   0.06),
        (MULTI_SONG_ARTIST,   0.12),
    ]
    templates = [g for g, _ in groups]
    weights = [w for _, w in groups]
    while len(out) < target and tries < n * 6:
        tries += 1
        group = rng.choices(templates, weights=weights)[0]
        tpl, intents = rng.choice(group)
        song, artist_for_song = rng.choice(SONGS)
        ctx = {
            "artist": rng.choice(ARTISTS), "vibe": rng.choice(DESC_VIBES),
            "genre": rng.choice(GENRES), "decade": rng.choice(DECADES),
            "recent": rng.choice(RECENT_PHRASES), "rating": rng.choice(RATINGS),
            "event": rng.choice(EVENTS), "topic": rng.choice(LYRIC_TOPICS),
            "song": song,
        }
        # MULTI_SONG_DESC uses both song and artist (the artist is the song's owner)
        if "{song}" in tpl:
            ctx["artist"] = artist_for_song
        # MULTI_SONG_ARTIST uses {a1} (song's owner) + {a2} (different artist)
        if "{a1}" in tpl or "{a2}" in tpl:
            ctx["a1"] = artist_for_song
            a2_pool = [a for a in ARTISTS if a != artist_for_song]
            ctx["a2"] = rng.choice(a2_pool)
        try:
            text = tpl.format(**ctx)
        except KeyError:
            continue
        emit(text, list(intents), pool, out)


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Target number of examples")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool: set[str] = set()
    out: list[dict] = []

    plan = {k: int(round(args.n * v)) for k, v in SHARE.items()}
    print(f"[generate] target per intent: {plan}")

    gen_song_similarity(plan["song_similarity"], rng, pool, out)
    gen_artist_similarity(plan["artist_similarity"], rng, pool, out)
    gen_text_search(plan["text_search"], rng, pool, out)
    gen_song_alchemy(plan["song_alchemy"], rng, pool, out)
    gen_brainstorm(plan["ai_brainstorm"], rng, pool, out)
    gen_search_database(plan["search_database"], rng, pool, out)
    gen_lyrics_search(plan["lyrics_search"], rng, pool, out)
    gen_multi(plan["multi"], rng, pool, out)

    # Shuffle so multi-intent and single-intent examples interleave during training
    rng.shuffle(out)

    counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    multi_count = 0
    for ex in out:
        for i in ex["intents"]:
            counts[i] = counts.get(i, 0) + 1
        if len(ex["intents"]) > 1:
            multi_count += 1
        for m in INLINE_RE.finditer(ex["text"]):
            t = m.group(2)
            entity_counts[t] = entity_counts.get(t, 0) + 1

    payload = {
        "_format_doc": "Each example uses INLINE Rasa-style entity annotations: '[value](type)'. Generated by generate_data.py.",
        "_tool_list": [
            "song_similarity", "text_search", "artist_similarity", "song_alchemy",
            "ai_brainstorm", "search_database", "lyrics_search",
        ],
        "_entity_types": sorted(entity_counts.keys()),
        "_stats": {
            "total_examples": len(out),
            "multi_intent_examples": multi_count,
            "examples_per_intent": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "entity_type_counts": dict(sorted(entity_counts.items(), key=lambda kv: -kv[1])),
            "seed": args.seed,
        },
        "examples": out,
    }

    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[generate] wrote {len(out)} examples to {args.out}")
    print(f"[generate] multi-intent: {multi_count}")
    print(f"[generate] intents:  {payload['_stats']['examples_per_intent']}")
    print(f"[generate] entities: {payload['_stats']['entity_type_counts']}")


if __name__ == "__main__":
    main()
