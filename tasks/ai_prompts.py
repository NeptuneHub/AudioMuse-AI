"""Centralized AI prompt templates and prompt builders.

All business-level prompts used by the AudioMuse-AI features live here, so
that:

* `tasks/ai_api*.py` stay generic transports (no embedded prompts).
* Adding or tweaking a prompt does not require touching transport code.
* Migrating a feature to a new provider only requires plugging the existing
  prompt into the new transport.

Public entry points:
    creative_prompt_template          -- clustering / playlist naming prompt template
    build_ai_brainstorm_prompt(...)   -- ai_brainstorm free-text prompt
    build_artist_hits_prompt(...)     -- artist-hits free-text prompt
    build_vibe_match_prompt(...)      -- vibe_match free-text prompt
"""
from typing import Dict, List, Optional


# --- Clustering / playlist naming ---------------------------------------------

creative_prompt_template = (
    "You are an expert music collector and MUST give a title to this playlist.\n"
    "The title MUST represent the mood and the activity of when you are listening to the playlist.\n"
    "The title MUST use ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "The title MUST be within the range of 5 to 40 characters long.\n"
    "No special fonts or emojis.\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '\U0001D5DD\U0001D5C2\U0001D5C8 \U0001D5C2\U0001D5CB\U0001D5C8\U0001D5C7\U0001D5C2 \U0001D5C9\U0001D5CB\U0001D5C8\U0001D5C7\U0001D5C2' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n\n"
    "This is the playlist:\n{song_list_sample}\n\n"
)


# --- Free-text MCP prompts (artist hits, brainstorm, vibe match) --------------

def build_artist_hits_prompt(artist: str) -> str:
    return f"""You are a music expert. List the most famous and popular songs by the artist "{artist}".

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of song titles
2. Include 15-25 of their most famous songs
3. Use exact song titles as they appear on albums
4. Format: ["Song Title 1", "Song Title 2", ...]
5. NO explanations, NO numbering, ONLY the JSON array

Example format:
["Song A", "Song B", "Song C"]

List the famous songs by {artist} now:"""


def build_ai_brainstorm_prompt(user_request: str) -> str:
    return f"""You are a music expert with extensive knowledge of songs, artists, and music history. 

User request: "{user_request}"

TASK: Use your knowledge to suggest 25-35 specific songs (with exact artist names) that match this request.

Think about:
- If they want songs similar to an artist \u2192 suggest songs by that artist AND similar artists
- If they want a genre/mood \u2192 suggest famous songs in that genre/mood
- If they want popular/radio hits \u2192 suggest well-known mainstream songs
- If they want a time period \u2192 suggest songs from that era
- If they want a vibe \u2192 suggest songs that match that feeling

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of objects
2. Each object MUST have "title" and "artist" fields
3. Be specific with exact song titles and artist names (as they appear in databases)
4. Include variety - different artists when possible
5. Format: [{{"title": "Song Name", "artist": "Artist Name"}}, ...]
6. NO explanations, NO numbering, ONLY the JSON array

Example format:
[
  {{"title": "All the Small Things", "artist": "blink-182"}},
  {{"title": "Basket Case", "artist": "Green Day"}},
  {{"title": "American Idiot", "artist": "Green Day"}}
]

Suggest songs for "{user_request}" now:"""


def build_vibe_match_prompt(vibe_description: str) -> str:
    return f"""You are a music database expert. The user wants songs matching this vibe: "{vibe_description}"

Analyze this vibe and return a JSON object with search criteria for a music database.

Database schema:
- mood_vector: Contains genres like 'rock', 'pop', 'jazz', etc.
- other_features: Contains moods like 'danceable', 'party', 'relaxed', etc.
- energy: 0.01-0.15 (higher = more energetic)
- tempo: 40-200 BPM

Return ONLY this JSON structure:
{{
    "genres": ["genre1", "genre2"],
    "moods": ["mood1", "mood2"],
    "energy_min": 0.05,
    "energy_max": 0.12,
    "tempo_min": 100,
    "tempo_max": 140
}}

Return the JSON now:"""
