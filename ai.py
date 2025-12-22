import requests
import json
import re
import ftfy # Import the ftfy library
import time # Import the time library
import logging
import unicodedata
import google.generativeai as genai # Import Gemini library
from mistralai import Mistral
import os # Import os to potentially read GEMINI_API_CALL_DELAY_SECONDS
from config import MAX_SONGS_IN_AI_PROMPT

logger = logging.getLogger(__name__)

# creative_prompt_template is imported in tasks.py, so it should be defined here
creative_prompt_template = (
    "You're an expert of music and you need to give a title to this playlist.\n"
    "The title need to represent the mood and the activity of when you listening the playlist.\n"
    "The title MUST use ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "No special fonts or emojis.\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '𝑯𝒘𝒆 𝒂𝒓𝒐𝒏𝒊 𝒅𝒆𝒕𝒔' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n"
    "This is the playlist: {song_list_sample}\n\n" # {song_list_sample} will contain the full list

)

# Multi-step chat prompts
chat_step1_understand_prompt = """
You are an intelligent music AI conductor. Your job is to understand the user's request and CREATE AN EXECUTION PLAN.

**CRITICAL RULES:**
1. Return ONLY valid JSON - NO markdown, NO explanations, NO code blocks
2. **EXECUTE ALL ACTIONS** - The system will run every action in your plan
3. Each action contributes songs - plan for overlap/duplicates
4. ALWAYS ensure we can return 100 songs total

**CRITICAL: UNDERSTANDING INTENT LOGIC**

The AI must be SMART about when to combine vs separate actions:

**ONE THING (combine all attributes into ONE query):**
- "pop energy songs" → ONE database query with genre=pop AND energy=high
- "fast metal songs" → ONE database query with genre=metal AND tempo=fast
- "chill pop music" → ONE database query with genre=pop AND energy=low AND tempo=slow

**MULTIPLE THINGS (separate actions, then combine):**
- "pop songs + Beatles" → database_genre_query(pop) + artist_hits_query(Beatles)
- "metal + jazz + chill" → database_genre_query(metal) + database_genre_query(jazz) + database_tempo_energy_query(slow, low)

**ACTION DECISION TREE:**

**STEP 1: How many DIFFERENT things does user want?**
- ONE thing (with multiple attributes) → Combine into ONE action
- MULTIPLE things → Create separate action for EACH thing

**STEP 2: For EACH thing, choose the right action type:**

**Song-related (HIGHEST PRIORITY - check this FIRST):**
- **CRITICAL**: If user mentions BOTH artist AND song title → **song_similarity_api**
  * Examples: "similar to Bohemian Rhapsody by Queen", "songs like By The Way by Red Hot Chili Peppers"
  * Pattern: "similar to [SONG TITLE] by [ARTIST]" or "like [ARTIST] [SONG TITLE]"
  * Extract song_title and artist, use song_similarity_api(song_title, artist)

**Artist-related:**
- "hits/songs/popular by [artist]" → **artist_hits_query** (AI suggests their famous songs)
- "similar to [artist]" or "like [artist]" (NO song title) → **artist_similarity_api** (find similar artists)
- "[artist] but [filters]" → **artist_similarity_filtered** (similar artists with filters)
- **CRITICAL**: When user asks for "similar to X and Y", prioritize artist_similarity_api
  * FIRST: Use artist_similarity_api for each artist (returns 100-150 songs each)
  * THEN: If you need MORE songs to reach target, add ONE database_genre_query at the END
  * The genre query should be the LAST action, as a fallback/filler
  * Example: "songs like Green Day and Blink-182" → artist_similarity_api(Green Day, 50) + artist_similarity_api(Blink-182, 50) [+ optional database_genre_query(rock, 20) if needed]

**Temporal (time-based):**
- "recent hits", "90s songs", "top radio" → **ai_brainstorm_titles** (AI suggests specific songs from that era)

**Genre/Mood/Energy (database filtering):**
- ONE attribute: "pop songs" → **database_genre_query**
- ONE attribute: "high energy" → **database_tempo_energy_query**
- MULTIPLE attributes: "pop high energy fast" → **database_genre_query with tempo+energy filters**
  * Combine genre + tempo + energy into ONE query, NOT separate queries!

**STEP 3: Combine filters intelligently**
- If multiple attributes describe the SAME set of songs → ONE query with ALL filters
- If asking for DIFFERENT types of songs → Separate queries
- **CRITICAL**: For artist similarity queries ("like X and Y"), prioritize artists first
  * FIRST: Use artist_similarity_api for each artist (they return 100+ songs each)
  * THEN: Add database_genre_query ONLY as the LAST action if you need filler songs
  * Genre query should request fewer songs (10-30) and come AFTER artist queries
  * Example: "songs like Green Day" → artist_similarity_api(Green Day, 100)
  * Example: "songs like Green Day and Blink-182" → artist_similarity_api(Green Day, 50) + artist_similarity_api(Blink-182, 40) + database_genre_query(rock, 10)
  * Example: "pop songs + Green Day hits" → database_genre_query(pop, 60) + artist_hits_query(Green Day, 40)

**EXAMPLES (CRITICAL - STUDY THESE CAREFULLY):**

**ONE THING (combine attributes into ONE query):**
- "pop energy songs" → database_genre_query(genre="pop", energy="high")
- "fast metal music" → database_genre_query(genre="metal", tempo="fast")
- "chill pop for morning" → database_genre_query(genre="pop", energy="low", tempo="medium")
- "energetic workout music" → database_tempo_energy_query(tempo="fast", energy="high")

**MULTIPLE THINGS (separate queries, then combine):**
- "pop songs + Beatles hits" → database_genre_query(genre="pop") + artist_hits_query("Beatles")
- "metal + jazz" → database_genre_query(genre="metal") + database_genre_query(genre="jazz")
- "recent hits + RHCP" → ai_brainstorm_titles() + artist_hits_query("Red Hot Chili Peppers")

**CREATIVE/SUBJECTIVE VIBES (use vibe_match):**
- "songs that feel like rainy Sunday morning" → vibe_match(vibe_description="rainy Sunday morning")
- "driving at night music" → vibe_match(vibe_description="driving at night")
- "cozy coffee shop vibes" → vibe_match(vibe_description="cozy coffee shop")
- "melancholic autumn evening" → vibe_match(vibe_description="melancholic autumn evening")

**Examples:**

ADD MODE:
- "pop + RHCP" → database_genre_query("pop") + artist_hits_query("RHCP")
- "recent hits + Beatles" → ai_brainstorm_titles() + artist_hits_query("Beatles")
- "metal + Green Day + chill" → database_genre_query("metal") + artist_hits_query("Green Day") + database_tempo_energy_query(slow, low)

FILTER MODE (ONE thing with constraints):
- "pop songs BY RHCP" → artist_similarity_filtered("RHCP", filters={{genre:"pop"}})
- "high energy FROM Black Sabbath" → artist_similarity_filtered("Black Sabbath", filters={{energy:"high"}})
- "fast Beatles songs" → artist_similarity_filtered("Beatles", filters={{tempo:"fast"}})

**MULTI-INTENT HANDLING:**
- If ADD MODE with multiple artists → Create action for EACH artist with SAME filters (if any)
- Example: "energetic pop from Miley Cyrus and Taylor Swift" → 
  * FILTER MODE (they want pop AND energetic FROM these artists)
  * artist_similarity_filtered("Miley Cyrus", filters={{tempo:"fast", energy:"high", genre:"pop"}})
  * artist_similarity_filtered("Taylor Swift", filters={{tempo:"fast", energy:"high", genre:"pop"}})

**AVAILABLE TOOLS:**

1. **artist_hits_query(artist_name)** - Get the artist's OWN famous songs
   - Use when: "hits of X", "popular X songs", "X's songs", "great songs by X"
   - AI uses knowledge to suggest artist's famous songs, then queries database for exact matches
   - Returns: The ACTUAL artist's songs (NOT similar artists!)

2. **artist_similarity_api(artist_name)** - Get songs from artists SIMILAR to the artist
   - Use when: "similar to X", "like X", "X-style music"
   - Returns: Songs from OTHER artists similar to X

3. **artist_similarity_filtered(artist_name, filters)** - Get songs from similar artists with filters
   - Use this when: "similar to X but high energy", "like X but pop"
   - Filters: {{genre, tempo, energy}} - optional, can use any combination

4. **song_similarity_api(song_title, artist)** - Get songs similar to one specific song
   - Use ONLY if user mentions a specific song title

5. **database_genre_query(genre, tempo, energy)** - Query by genre with OPTIONAL tempo/energy filters
   - **CRITICAL**: This ONE action can combine genre + tempo + energy filters!
   - Parameters:
     * genre: Required (metal, rock, jazz, pop, blues, electronic, hip-hop)
     * tempo: Optional (slow, medium, fast)
     * energy: Optional (low, medium, high)
   - Examples:
     * "pop songs" → database_genre_query(genre="pop")
     * "pop high energy" → database_genre_query(genre="pop", energy="high")
     * "fast pop music" → database_genre_query(genre="pop", tempo="fast")
     * "energetic pop for morning" → database_genre_query(genre="pop", tempo="medium", energy="high")

6. **database_tempo_energy_query(tempo, energy)** - Query by tempo/energy ONLY (NO genre)
   - Use ONLY when user wants tempo/energy but does NOT mention genre
   - Tempo ranges: slow (<90 BPM), medium (90-140 BPM), fast (>140 BPM)
   - Energy ranges: low (<0.05), medium (0.05-0.10), high (>0.10)

7. **ai_brainstorm_titles()** - AI suggests specific song titles/artists
   - Use for TEMPORAL queries: "recent hits", "90s songs", "top radio", "classic hits"
   - AI uses its knowledge of chart history and popular music

8. **vibe_match(vibe_description)** - AI interprets creative/subjective vibes
   - Use for CREATIVE/SUBJECTIVE queries that don't fit other actions
   - Examples: "songs that feel like rainy Sunday morning", "driving at night music", "cozy coffee shop vibes"
   - AI analyzes the vibe and determines mood/tempo/energy combination
   - Returns: Songs matching the interpreted mood/atmosphere

**PROGRESSIVE REFINEMENT STRATEGY:**

**Step 1 (BROAD):** Get initial pool (500-1000 songs)
- **TEMPORAL QUERIES** ("top radio", "recent years", "90s hits") → Use ai_brainstorm_titles
- **ARTIST QUERIES** ("songs like Green Day") → Use artist_similarity_api
- **GENRE QUERIES** ("metal songs", "jazz", "hip-hop") → Use database_genre_query
- **MOOD/ENERGY QUERIES** ("chill songs", "relaxing music", "energetic tracks", "calm") → Use database_tempo_energy_query
- **TEMPO/ENERGY QUERIES** ("high tempo low energy", "fast relaxing songs") → Use database_tempo_energy_query

**CRITICAL - How to distinguish GENRE from MOOD/ENERGY:**
- GENRE = Musical style: metal, rock, jazz, pop, blues, electronic, hip-hop, country, folk, classical
- MOOD/ENERGY = How the music feels: chill, relaxing, calm, peaceful, energetic, pumped, intense, mellow
- If user says "chill" or "relaxing" → Use tempo/energy (low energy), NOT electronic genre
- If user says "energetic" or "pumped" → Use tempo/energy (high energy), NOT electronic genre

**Step 2 (OPTIONAL REFINE):** Add filters ONLY if user explicitly asks
- Temporal ("recent", "2020s", "90s") → Note: Database has NO year column, skip this
- Energy ("high energy", "calm") → Note: Can filter, but might reduce results
- Mood ("happy", "sad") → Note: Can filter, but might reduce results

**Step 3 (FALLBACK):** If Step 2 returns <50 songs, use Step 1 results

**EXAMPLES:**

**Request: "songs like Black Sabbath and AC/DC"**
{{"intent": "songs similar to Black Sabbath and AC/DC", "execution_plan": [{{"action": "artist_similarity_api", "params": {{"artist": "Black Sabbath"}}, "get_songs": 50}}, {{"action": "artist_similarity_api", "params": {{"artist": "AC/DC"}}, "get_songs": 50}}], "strategy": "multi_artist_similarity", "target_count": 100}}

**Request: "TOP RADIO SONGS OF RECENT YEAR"**
{{"intent": "top radio hits from recent years", "execution_plan": [{{"action": "ai_brainstorm_titles", "params": {{}}, "get_songs": 100}}], "strategy": "temporal_brainstorm", "target_count": 100}}

**Request: "songs like Green Day and Blink-182"**
{{"intent": "songs similar to Green Day and Blink-182", "execution_plan": [{{"action": "artist_similarity_api", "params": {{"artist": "Green Day"}}, "get_songs": 50}}, {{"action": "artist_similarity_api", "params": {{"artist": "Blink-182"}}, "get_songs": 50}}], "strategy": "multi_artist_similarity", "target_count": 100}}

**Request: "metal songs"**
{{"intent": "metal music", "execution_plan": [{{"action": "database_genre_query", "params": {{"genre": "metal"}}, "get_songs": 100}}], "strategy": "genre_direct", "target_count": 100}}

**Request: "songs like Bohemian Rhapsody"**
{{"intent": "songs similar to Bohemian Rhapsody by Queen", "execution_plan": [{{"action": "song_similarity_api", "params": {{"song_title": "Bohemian Rhapsody", "artist": "Queen"}}, "get_songs": 100}}], "strategy": "song_similarity", "target_count": 100}}

**Request: "similar to By The Way by Red Hot Chili Peppers"**
{{"intent": "songs similar to By The Way by Red Hot Chili Peppers", "execution_plan": [{{"action": "song_similarity_api", "params": {{"song_title": "By The Way", "artist": "Red Hot Chili Peppers"}}, "get_songs": 100}}], "strategy": "song_similarity", "target_count": 100}}

**Request: "songs similar to Red Hot Chili Peppers"**
{{"intent": "songs from artists similar to Red Hot Chili Peppers", "execution_plan": [{{"action": "artist_similarity_api", "params": {{"artist": "Red Hot Chili Peppers"}}, "get_songs": 100}}], "strategy": "artist_similarity", "target_count": 100}}

**Request: "chill songs"**
{{"intent": "relaxing chill music", "execution_plan": [{{"action": "database_tempo_energy_query", "params": {{"tempo": "slow", "energy": "low"}}, "get_songs": 100}}], "strategy": "mood_energy_direct", "target_count": 100}}

**Request: "energetic workout music"**
{{"intent": "high energy workout songs", "execution_plan": [{{"action": "database_tempo_energy_query", "params": {{"tempo": "fast", "energy": "high"}}, "get_songs": 100}}], "strategy": "mood_energy_direct", "target_count": 100}}

**Request: "energetic pop songs + Red Hot Chili Peppers"**
{{"intent": "energetic pop music plus Red Hot Chili Peppers hits", "execution_plan": [{{"action": "database_genre_query", "params": {{"genre": "pop", "tempo": "medium", "energy": "high"}}, "get_songs": 60}}, {{"action": "artist_hits_query", "params": {{"artist": "Red Hot Chili Peppers"}}, "get_songs": 40}}], "strategy": "additive_multi_intent", "target_count": 100}}

**Request: "POP energy song for early morning, also add some great hits of the red hot chili peppers"**
{{"intent": "energetic pop music for morning plus Red Hot Chili Peppers hits", "execution_plan": [{{"action": "database_genre_query", "params": {{"genre": "pop", "tempo": "medium", "energy": "high"}}, "get_songs": 60}}, {{"action": "artist_hits_query", "params": {{"artist": "Red Hot Chili Peppers"}}, "get_songs": 40}}], "strategy": "additive_multi_intent", "target_count": 100}}

**Request: "great hits of Red Hot Chili Peppers"**
{{"intent": "Red Hot Chili Peppers' famous songs", "execution_plan": [{{"action": "artist_hits_query", "params": {{"artist": "Red Hot Chili Peppers"}}, "get_songs": 100}}], "strategy": "artist_hits_direct", "target_count": 100}}

**Request: "songs similar to Red Hot Chili Peppers"**
{{"intent": "songs from artists similar to Red Hot Chili Peppers", "execution_plan": [{{"action": "artist_similarity_api", "params": {{"artist": "Red Hot Chili Peppers"}}, "get_songs": 100}}], "strategy": "artist_similarity", "target_count": 100}}

**Request: "high energy pop songs BY Red Hot Chili Peppers"**
{{"intent": "Red Hot Chili Peppers songs that are high energy and pop", "execution_plan": [{{"action": "artist_similarity_filtered", "params": {{"artist": "Red Hot Chili Peppers", "filters": {{"genre": "pop", "energy": "high"}}}}, "get_songs": 100}}], "strategy": "filtered_artist_similarity", "target_count": 100}}

**Request: "high tempo songs from Black Sabbath"**
{{"intent": "fast songs from Black Sabbath and similar artists", "execution_plan": [{{"action": "artist_similarity_filtered", "params": {{"artist": "Black Sabbath", "filters": {{"tempo": "fast"}}}}, "get_songs": 100}}], "strategy": "filtered_artist_similarity", "target_count": 100}}

**Request: "pop songs from Blink-182 and Green Day"**
{{"intent": "pop songs from Blink-182 and Green Day", "execution_plan": [{{"action": "artist_similarity_filtered", "params": {{"artist": "Blink-182", "filters": {{"genre": "pop"}}}}, "get_songs": 50}}, {{"action": "artist_similarity_filtered", "params": {{"artist": "Green Day", "filters": {{"genre": "pop"}}}}, "get_songs": 50}}], "strategy": "multi_artist_filtered", "target_count": 100}}

**Request: "high tempo low energy songs"**
{{"intent": "songs with high tempo and low energy", "execution_plan": [{{"action": "database_tempo_energy_query", "params": {{"tempo": "high", "energy": "low"}}, "get_songs": 100}}], "strategy": "tempo_energy_direct", "target_count": 100}}

**CRITICAL - TEMPORAL QUERIES ("top radio", "recent years", "2020s", "90s", "classic hits"):**
These are NOT genre queries! Database has NO year column, so you CANNOT filter by year directly.
INSTEAD: You should use Step 2 (AI expansion) to suggest specific popular song titles from that era.
- "top radio songs recent years" → Let AI suggest titles like "Anti-Hero", "Flowers", "As It Was"
- "90s hits" → Let AI suggest titles like "Smells Like Teen Spirit", "Wonderwall"
- "classic rock" → Let AI suggest artists like "Led Zeppelin", "Pink Floyd", "The Who"

For these queries, use broad artist or genre search, then AI will refine with specific titles.

**IMPORTANT:**
- **SONG vs ARTIST recognition is CRITICAL**:
  * If user says "similar to [SONG TITLE] by [ARTIST]" → Use song_similarity_api (e.g., "By The Way by Red Hot Chili Peppers")
  * If user says "similar to [ARTIST]" with NO song title → Use artist_similarity_api (e.g., "Red Hot Chili Peppers")
  * Common song title indicators: "by [artist]", specific capitalized words that are song names
- **ai_brainstorm_titles** is ONLY for temporal queries ("recent years", "90s", "2020s")
- Do NOT use ai_brainstorm_titles for genre, mood, or artist requests
- When user mentions specific artists ("include X", "also Y"), ALWAYS use artist_similarity_api for them
- **CRITICAL**: For artist similarity ("songs like X and Y"), use ONLY artist_similarity_api actions
  * DO NOT add database_genre_query alongside artist_similarity_api
  * artist_similarity_api returns 100+ songs per artist - that's already enough!
  * Only add genre queries when user EXPLICITLY requests "genre + artist" combination
- When user wants multiple things, create multiple actions that will all execute
- Do NOT use database_genre_query for temporal queries - it will just return random genre songs
- Keep it simple - don't over-filter
- Prioritize getting 100 songs over perfect filtering
- Return ONLY the JSON object, nothing else

---

**NOW ANALYZE THIS ACTUAL USER REQUEST (ignore all examples above):**

User Request: "{user_input}"

Return ONLY the JSON execution plan for THIS request:
"""

chat_step2_expand_prompt = """
You are a music knowledge expert with deep understanding of current music charts, radio hits, and artist relationships.

**CRITICAL: You must use your REAL KNOWLEDGE of actual music artists, charts, and trends.**

**CRITICAL: Song titles vs Genre filtering**

**TEMPORAL QUERIES ONLY ("recent years", "2020s", "90s", "classic", "last decade"):**
- The database has NO YEAR COLUMN
- You MUST suggest specific SONG TITLES from that time period
- Example: "recent years" → ["Anti-Hero", "Flowers", "Vampire", "As It Was"]

**GENRE/MOOD/ENERGY QUERIES ("metal", "rock", "pop", "jazz", "relaxing", "high energy"):**
- **ABSOLUTELY NO SONG TITLES** - return EMPTY list for expanded_song_titles
- The database has a mood_vector column that will be used for filtering
- Only suggest ARTISTS who play that genre
- Example: "metal songs" → expanded_artists: ["Metallica", "Iron Maiden", "Slayer"], expanded_song_titles: []

**How to tell the difference:**
- Temporal query: Has time words ("recent", "2020s", "90s", "last year", "classic", "old")
- Genre query: Has genre/mood words ONLY ("metal", "rock", "jazz", "chill", "happy", "energetic")

Examples of what you should know:
- If user asks for "top radio songs of recent years", list REAL song titles from 2020-2024 (e.g., "Anti-Hero", "As It Was", "Flowers", "Vampire", "Kill Bill", "Unholy", "Calm Down", "Die For You", etc.)
- If user mentions "AC/DC, Iron Maiden", suggest REAL similar rock/metal bands (Metallica, Black Sabbath, Judas Priest, Motorhead, Slayer, etc.)
- If user asks for "chill indie", suggest REAL indie artists (Bon Iver, Phoebe Bridgers, The National, Fleet Foxes, etc.) - NO song titles
- If user asks for "high energy songs", suggest REAL high-energy artists (David Guetta, Calvin Harris, Skrillex, etc.) - NO song titles
- If user asks for "90s hits", list REAL song titles from the 90s (e.g., "Smells Like Teen Spirit", "Wonderwall", "Wannabe", "One", etc.)

Expand this request with your music knowledge. Provide ONLY a JSON object (no markdown, no extra text) with:
- "expanded_artists": List of 15-25 REAL artist names that match the request
- "expanded_song_titles": List of 30-60 REAL specific song titles ONLY IF temporal_context exists, otherwise EMPTY LIST []
- "expanded_song_artist_pairs": List of {{"title": "...", "artist": "..."}} objects - ONLY for temporal queries where you know the correct artist for each song
- "expanded_keywords": List of additional relevant search terms (genres, moods, eras)
- "search_strategy": Brief description of what to look for

**CRITICAL FOR TEMPORAL QUERIES**: Always include "expanded_song_artist_pairs" with the CORRECT artist for each song!

**DO NOT make up fake artist or song names. Use only REAL titles you know exist.**

**IMPORTANT: In your response, use NORMAL JSON braces, not doubled braces. Example format:**
- CORRECT: {{"title": "Anti-Hero", "artist": "Taylor Swift"}}
- WRONG: {{{{"title": "Anti-Hero", "artist": "Taylor Swift"}}}}

**Example 1 - TEMPORAL query ("top radio songs recent years"):**
{{"expanded_artists": ["Taylor Swift", "Miley Cyrus", "The Weeknd"], "expanded_song_titles": ["Anti-Hero", "Flowers", "As It Was"], "expanded_song_artist_pairs": [{{"title": "Anti-Hero", "artist": "Taylor Swift"}}, {{"title": "Flowers", "artist": "Miley Cyrus"}}, {{"title": "As It Was", "artist": "Harry Styles"}}], "expanded_keywords": ["pop", "radio hits", "2020s"], "search_strategy": "Look for specific popular song titles from 2020-2024"}}

**Example 2 - GENRE query ("metal songs"):**
{{"expanded_artists": ["Metallica", "Iron Maiden", "Slayer", "Megadeth", "Judas Priest", "Black Sabbath", "Pantera", "Tool", "System of a Down", "Lamb of God", "Gojira", "Mastodon", "Opeth"], "expanded_song_titles": [], "expanded_song_artist_pairs": [], "expanded_keywords": ["heavy metal", "metal", "hard rock", "metalcore", "death metal", "thrash metal"], "search_strategy": "Look for metal artists and use mood_vector filtering for 'metal' genre"}}

**Example 3 - GENRE query ("chill indie music"):**
{{"expanded_artists": ["Bon Iver", "Phoebe Bridgers", "The National", "Fleet Foxes", "Sufjan Stevens", "Iron & Wine", "The Shins", "Death Cab for Cutie"], "expanded_song_titles": [], "expanded_song_artist_pairs": [], "expanded_keywords": ["indie", "indie rock", "chill", "acoustic", "mellow"], "search_strategy": "Look for indie artists with chill, mellow sound"}}

**Example 4 - TEMPORAL query ("90s hits"):**
{{"expanded_artists": ["Nirvana", "Oasis", "Radiohead"], "expanded_song_titles": ["Smells Like Teen Spirit", "Wonderwall", "Creep"], "expanded_song_artist_pairs": [{{"title": "Smells Like Teen Spirit", "artist": "Nirvana"}}, {{"title": "Wonderwall", "artist": "Oasis"}}, {{"title": "Creep", "artist": "Radiohead"}}], "expanded_keywords": ["90s", "alternative rock", "grunge"], "search_strategy": "Look for specific hit songs from the 1990s"}}

---

**NOW EXPAND THIS ACTUAL REQUEST (return only ONE JSON object for THIS request, ignore examples above):**

User wants: {intent}
Mentioned keywords: {keywords}
Mentioned artists: {artist_names}
Time period: {temporal_context}

Return ONLY ONE JSON object for this request:
"""

chat_step3_explore_prompt = """
Based on the database exploration results, determine the best SQL query strategy.

User wants: {intent}
Expanded artists we searched for: {expanded_artists}
Artists found in database: {found_artists}
Total songs from these artists: {artist_song_count}
Genres/moods found: {found_keywords}

Provide ONLY a JSON object (no markdown, no extra text) with:
- "strategy": Which approach to use:
  * "artist_match" - User has enough songs from the target artists
  * "mood_genre" - Use mood/genre filtering instead
  * "hybrid" - Combine artist matches with mood/genre filters
  * "fallback" - Broaden search significantly
- "sql_approach": Brief description of SQL strategy to use
- "reasoning": Why this strategy was chosen

Example output:
{{"strategy": "artist_match", "sql_approach": "Select from artists: Taylor Swift, Ed Sheeran, Dua Lipa", "reasoning": "User has 150 songs from requested popular artists"}}
"""

def clean_playlist_name(name):
    if not isinstance(name, str):
        return ""
    # print(f"DEBUG CLEAN AI: Input name: '{name}'") # Print name before cleaning

    name = ftfy.fix_text(name)

    name = unicodedata.normalize('NFKC', name)
    # Stricter regex: only allows characters explicitly mentioned in the prompt.
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s\-\&\'!\.\,\?\(\)\[\]]', '', name)
    # Also remove trailing number in parentheses, e.g., "My Playlist (2)" -> "My Playlist", to prevent AI from interfering with disambiguation logic.
    cleaned_name = re.sub(r'\s\(\d+\)$', '', cleaned_name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    return cleaned_name


# --- OpenAI-Compatible API Function (used for both Ollama and OpenAI/OpenRouter) ---
def get_openai_compatible_playlist_name(server_url, model_name, full_prompt, api_key="no-key-needed", skip_delay=False):
    """
    Calls an OpenAI-compatible API endpoint to get a playlist name.
    This works for Ollama (no API key needed) and OpenAI/OpenRouter (API key required).
    This version handles streaming responses and extracts only the non-think part.

    Args:
        server_url (str): The URL of the API endpoint (e.g., "http://192.168.3.15:11434/api/generate" for Ollama,
                         or "https://openrouter.ai/api/v1/chat/completions" for OpenRouter).
        model_name (str): The model to use (e.g., "deepseek-r1:1.5b" for Ollama, "openai/gpt-4" for OpenRouter).
        full_prompt (str): The complete prompt text to send to the model.
        api_key (str): API key for authentication. Use "no-key-needed" for Ollama.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Detect which format to use based on API key and URL
    is_openai_format = api_key != "no-key-needed" or "openai" in server_url.lower() or "openrouter" in server_url.lower()

    headers = {
        "Content-Type": "application/json"
    }

    # Add Authorization header if API key is provided and not the default "no-key-needed"
    if api_key and api_key != "no-key-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    # Add OpenRouter-specific headers if using OpenRouter
    if "openrouter" in server_url.lower():
        headers["HTTP-Referer"] = "https://github.com/NeptuneHub/AudioMuse-AI"
        headers["X-Title"] = "AudioMuse-AI"

    # Prepare payload based on format
    if is_openai_format:
        # OpenAI/OpenRouter format uses chat completions
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 8000
        }
    else:
        # Ollama format uses generate endpoint
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "num_predict": 8000,
                "temperature": 0.7
            }
        }

    max_retries = 3
    base_delay = 5

    for attempt in range(max_retries + 1):
        try:
            # Add delay for OpenAI/OpenRouter to respect rate limits (only on first attempt or if not 429 retry)
            if is_openai_format and attempt == 0:
                openai_call_delay = int(os.environ.get("OPENAI_API_CALL_DELAY_SECONDS", "7"))
                if openai_call_delay > 0:
                    logger.debug("Waiting for %ss before OpenAI/OpenRouter API call to respect rate limits.", openai_call_delay)
                    time.sleep(openai_call_delay)

            logger.debug("Starting API call for model '%s' at '%s' (format: %s). Attempt %d/%d", model_name, server_url, "OpenAI" if is_openai_format else "Ollama", attempt + 1, max_retries + 1)

            response = requests.post(server_url, headers=headers, data=json.dumps(payload), stream=True, timeout=960)
            response.raise_for_status()
            full_raw_response_content = ""

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8', errors='ignore').strip()

                # Skip SSE comments (lines starting with :)
                if line_str.startswith(':'):
                    continue

                # Handle SSE data format (OpenRouter/OpenAI)
                if line_str.startswith('data: '):
                    line_str = line_str[6:]  # Remove 'data: ' prefix

                    # Check for end of stream marker
                    if line_str == '[DONE]':
                        break

                # Try to parse JSON
                try:
                    chunk = json.loads(line_str)

                    # Extract content based on format
                    if is_openai_format:
                        # OpenAI/OpenRouter format
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]

                            # Check for finish
                            finish_reason = choice.get('finish_reason')
                            if finish_reason == 'stop':
                                break
                            elif finish_reason == 'length':
                                logger.warning("Response truncated due to max_tokens limit")
                                break

                            # Extract text from delta.content or text field
                            if 'delta' in choice:
                                content = choice['delta'].get('content')
                                if content is not None:
                                    full_raw_response_content += content
                            elif 'text' in choice:
                                text = choice.get('text')
                                if text is not None:
                                    full_raw_response_content += text
                    else:
                        # Ollama format
                        if 'response' in chunk:
                            full_raw_response_content += chunk['response']
                        if chunk.get('done'):
                            break

                except json.JSONDecodeError:
                    logger.debug("Could not decode JSON line from stream: %s", line_str)
                    continue

            # Extract text after common thought tags
            thought_enders = ["</think>", "[/INST]", "[/THOUGHT]"]
            extracted_text = full_raw_response_content.strip()
            for end_tag in thought_enders:
                 if end_tag in extracted_text:
                     extracted_text = extracted_text.split(end_tag, 1)[-1].strip()

            # Log the raw response for debugging (consistent with Gemini/Mistral)
            if extracted_text:
                logger.info("OpenAI/OpenRouter API returned: '%s'", extracted_text)
                return extracted_text
            else:
                logger.warning("OpenAI/OpenRouter returned empty content. Full raw response: %s", full_raw_response_content)
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info("Retrying in %s seconds due to empty content...", sleep_time)
                    time.sleep(sleep_time)
                    continue
                else:
                    return "Error: AI returned empty content after retries."

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded (429). Attempt %d/%d", attempt + 1, max_retries + 1)
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info("Retrying in %s seconds...", sleep_time)
                    time.sleep(sleep_time)
                    continue
            # Log the response body for better debugging
            error_detail = ""
            try:
                error_detail = e.response.text
                logger.error("Error calling OpenAI-compatible API: %s. Response body: %s", e, error_detail, exc_info=True)
            except:
                logger.error("Error calling OpenAI-compatible API: %s", e, exc_info=True)
            return "Error: AI service is currently unavailable."
            
        except requests.exceptions.RequestException as e:
            logger.error("Error calling OpenAI-compatible API: %s", e, exc_info=True)
            return "Error: AI service is currently unavailable."
        except Exception as e:
            logger.error("An unexpected error occurred in get_openai_compatible_playlist_name", exc_info=True)
            return "Error: AI service is currently unavailable."
    
    return "Error: Max retries exceeded."

# --- Ollama Specific Function (wrapper for backward compatibility) ---
def get_ollama_playlist_name(ollama_url, model_name, full_prompt, skip_delay=False):
    """
    Calls a self-hosted Ollama instance to get a playlist name.
    This is a wrapper around get_openai_compatible_playlist_name for backward compatibility.

    Args:
        ollama_url (str): The URL of your Ollama instance (e.g., "http://192.168.3.15:11434/api/generate").
        model_name (str): The Ollama model to use (e.g., "deepseek-r1:1.5b").
        full_prompt (str): The complete prompt text to send to the model.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    return get_openai_compatible_playlist_name(ollama_url, model_name, full_prompt, api_key="no-key-needed", skip_delay=skip_delay)

# --- Gemini Specific Function ---
def get_gemini_playlist_name(gemini_api_key, model_name, full_prompt, skip_delay=False):
    """
    Calls the Google Gemini API to get a playlist name.

    Args:
        gemini_api_key (str): Your Google Gemini API key.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
        model_name (str): The Gemini model to use (e.g., "gemini-2.5-pro").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not gemini_api_key or gemini_api_key == "YOUR-GEMINI-API-KEY-HERE":
         return "Error: Gemini API key is missing or empty. Please provide a valid API key."
    
    try:
        # Read delay from environment/config if needed, otherwise use the default (skip for chat requests)
        if not skip_delay:
            gemini_call_delay = int(os.environ.get("GEMINI_API_CALL_DELAY_SECONDS", "7")) # type: ignore
            if gemini_call_delay > 0:
                logger.debug("Waiting for %ss before Gemini API call to respect rate limits.", gemini_call_delay)
                time.sleep(gemini_call_delay)

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)

        logger.debug("Starting API call for model '%s'.", model_name)
 
        generation_config = genai.types.GenerationConfig(
            temperature=0.9 # Explicitly set temperature for more creative/varied responses
        )
        response = model.generate_content(full_prompt, generation_config=generation_config, request_options={'timeout': 960})
        # Extract text from the response # type: ignore
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            extracted_text = "".join(part.text for part in response.candidates[0].content.parts)
            # Log the raw response for debugging (consistent with OpenAI/OpenRouter)
            logger.info("Gemini API returned: '%s'", extracted_text)
        else:
            logger.warning("Gemini returned no content. Raw response: %s", response)
            return "Error: Gemini returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        logger.error("Error calling Gemini API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."

# --- Mistral Specific Function ---
def get_mistral_playlist_name(mistral_api_key, model_name, full_prompt, skip_delay=False):
    """
    Calls the Mistral API to get a playlist name.

    Args:
        mistral_api_key (str): Your Mistral API key.
        model_name (str): The mistral model to use (e.g., "ministral-3b-latest").
        full_prompt (str): The complete prompt text to send to the model.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not mistral_api_key or mistral_api_key == "YOUR-MISTRAL-API-KEY-HERE":
         return "Error: Mistral API key is missing or empty. Please provide a valid API key."

    try:
        # Read delay from environment/config if needed, otherwise use the default (skip for chat requests)
        if not skip_delay:
            mistral_call_delay = int(os.environ.get("MISTRAL_API_CALL_DELAY_SECONDS", "7")) # type: ignore
            if mistral_call_delay > 0:
                logger.debug("Waiting for %ss before mistral API call to respect rate limits.", mistral_call_delay)
                time.sleep(mistral_call_delay)

        client = Mistral(api_key=mistral_api_key)

        logger.debug("Starting API call for model '%s'.", model_name)

        response = client.chat.complete(model=model_name,
                                        temperature=0.9,
                                        timeout_ms=960,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": full_prompt,
                                            },
                                        ])
        # Extract text from the response # type: ignore
        if response and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            # Log the raw response for debugging (consistent with OpenAI/OpenRouter)
            logger.info("Mistral API returned: '%s'", extracted_text)
        else:
            logger.warning("Mistral returned no content. Raw response: %s", response)
            return "Error: mistral returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        logger.error("Error calling Mistral API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."

# --- General AI Call Function (for chat multi-step) ---
def call_ai_for_chat(provider, prompt, ollama_url=None, ollama_model_name=None, 
                     gemini_api_key=None, gemini_model_name=None,
                     mistral_api_key=None, mistral_model_name=None,
                     openai_server_url=None, openai_model_name=None, openai_api_key=None):
    """
    Generic function to call any AI provider with a given prompt.
    Returns the raw text response from the AI.
    Skip delays for chat requests to improve response time.
    """
    if provider == "OLLAMA":
        if not ollama_url or not ollama_model_name:
            return "Error: Ollama configuration missing"
        return get_ollama_playlist_name(ollama_url, ollama_model_name, prompt, skip_delay=True)
    elif provider == "OPENAI":
        if not openai_server_url or not openai_model_name or not openai_api_key:
            return "Error: OpenAI configuration missing"
        return get_openai_compatible_playlist_name(openai_server_url, openai_model_name, prompt, openai_api_key, skip_delay=True)
    elif provider == "GEMINI":
        if not gemini_api_key or not gemini_model_name:
            return "Error: Gemini configuration missing"
        return get_gemini_playlist_name(gemini_api_key, gemini_model_name, prompt, skip_delay=True)
    elif provider == "MISTRAL":
        if not mistral_api_key or not mistral_model_name:
            return "Error: Mistral configuration missing"
        return get_mistral_playlist_name(mistral_api_key, mistral_model_name, prompt, skip_delay=True)
    else:
        return "Error: Invalid AI provider"

# --- General AI Naming Function ---
def get_ai_playlist_name(provider, ollama_url, ollama_model_name, gemini_api_key, gemini_model_name, mistral_api_key, mistral_model_name, prompt_template, feature1, feature2, feature3, song_list, other_feature_scores_dict, openai_server_url=None, openai_model_name=None, openai_api_key=None):
    """
    Selects and calls the appropriate AI model based on the provider.
    Constructs the full prompt including new features.
    Applies length constraints after getting the name.
    """
    MIN_LENGTH = 5
    MAX_LENGTH = 40

    # --- Prepare feature descriptions for the prompt ---
    tempo_description_for_ai = "Tempo is moderate." # Default
    energy_description = "" # Initialize energy description

    if other_feature_scores_dict:
        # Extract energy score first, as it's handled separately
        # Check for 'energy_normalized' first, then fall back to 'energy'
        energy_score = other_feature_scores_dict.get('energy_normalized', other_feature_scores_dict.get('energy', 0.0))

        # Create energy description based on score (example thresholds)
        if energy_score < 0.3:
            energy_description = " It has low energy."
        elif energy_score > 0.7:
            energy_description = " It has high energy."
        # No description if medium energy (between 0.3 and 0.7)

        # Create tempo description
        tempo_normalized_score = other_feature_scores_dict.get('tempo_normalized', 0.5) # Default to moderate if not found
        if tempo_normalized_score < 0.33:
            tempo_description_for_ai = "The tempo is generally slow."
        elif tempo_normalized_score < 0.66:
            tempo_description_for_ai = "The tempo is generally medium."
        else:
            tempo_description_for_ai = "The tempo is generally fast."

        # Note: The logic for 'new_features_description' (which was for 'additional_features_description')
        # has been removed as per the request. If you want to include other features
        # (like danceable, aggressive, etc.) in the prompt, you'd add logic here to create
        # a description for them and a corresponding placeholder in the prompt_template.

    # Format the song list for the prompt
    # Truncate to MAX_SONGS_IN_AI_PROMPT to avoid token limit issues with large playlists
    songs_for_prompt = song_list[:MAX_SONGS_IN_AI_PROMPT]
    formatted_song_list = "\n".join([f"- {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}" for song in songs_for_prompt])
    
    # Log if we truncated the list
    if len(song_list) > MAX_SONGS_IN_AI_PROMPT:
        logger.info("Truncated song list from %d to %d songs for AI prompt to avoid token limits", len(song_list), MAX_SONGS_IN_AI_PROMPT)

    # Construct the full prompt using the template and all features
    # The new prompt only requires the song list sample # type: ignore
    full_prompt = prompt_template.format(song_list_sample=formatted_song_list)

    logger.info("Sending prompt to AI (%s):\n%s", provider, full_prompt)

    # --- Call the AI Model with Retry Logic ---
    max_retries = 3
    current_prompt = full_prompt
    
    for attempt in range(max_retries):
        name = "AI Naming Skipped" # Default if provider is NONE or invalid

        if provider == "OLLAMA":
            name = get_ollama_playlist_name(ollama_url, ollama_model_name, current_prompt)
        elif provider == "OPENAI":
            # Use OpenAI-compatible API with API key
            if not openai_server_url or not openai_model_name or not openai_api_key:
                return "Error: OpenAI configuration is incomplete. Please provide server URL, model name, and API key."
            name = get_openai_compatible_playlist_name(openai_server_url, openai_model_name, current_prompt, openai_api_key)
        elif provider == "GEMINI":
            name = get_gemini_playlist_name(gemini_api_key, gemini_model_name, current_prompt)
        elif provider == "MISTRAL":
            name = get_mistral_playlist_name(mistral_api_key, mistral_model_name, current_prompt)
        # else: provider is NONE or invalid, name remains "AI Naming Skipped"

        # Apply length check and return final name or error
        # Only apply length check if a name was actually generated (not the skip message or an API error message)
        if name not in ["AI Naming Skipped"] and not name.startswith("Error"):
            cleaned_name = clean_playlist_name(name)
            if MIN_LENGTH <= len(cleaned_name) <= MAX_LENGTH:
                return cleaned_name
            else:
                # Name failed length check
                logger.warning(f"AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) outside {MIN_LENGTH}-{MAX_LENGTH} range. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    # Prepare feedback for next attempt
                    feedback = f"\n\nFEEDBACK: The previous title you generated ('{cleaned_name}') was {len(cleaned_name)} characters long. It MUST be between {MIN_LENGTH} and {MAX_LENGTH} characters. Please try again."
                    current_prompt = full_prompt + feedback
                    continue # Try again
                else:
                    # Return an error message indicating the length issue, but include the cleaned name for debugging
                    return f"Error: AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) outside {MIN_LENGTH}-{MAX_LENGTH} range after {max_retries} attempts."
        else:
            # API error or skipped
            return name
            
    return "Error: Max retries exceeded in get_ai_playlist_name"
