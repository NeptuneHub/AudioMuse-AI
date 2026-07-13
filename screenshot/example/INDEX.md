# AudioMuse-AI - UI Example Screenshots (fake demo data)

All data shown is **synthetic/fake** - none of the real ~180k-song library is exposed, so these can be shared publicly.

- Library stats are faked (e.g. 2,437 songs / 612 artists / 348 albums).
- Artists / titles / albums are invented (e.g. *Velvet Gravity - The Velvet Owls - Slow Motion Skies*).
- The Music Map is reduced to a 280-point sample.
- Usernames, worker hosts, media-server URLs/tokens are faked or redacted.

**LLM provider** (shown on both AI pages): OpenAI / OpenRouter · `https://api.atlascloud.ai/v1/chat/completions` · `deepseek-ai/DeepSeek-V3-0324`

## Highlights
- `02_index_2_clustering_playlists.png` - Clustering results with **AI-generated playlist names** (Midnight Velvet Grooves, Sunlit Acoustic Mornings, …).
- `03_chat_2_similar_song_result.png` - Instant Playlist: *"Similar to Midnight Harbor by The Velvet Echo"* → full AI pipeline + a 14-song generated playlist.

## Populated search results (all fake data)
Every search page also has a screenshot showing a real, populated result:
- `04_similarity_4_result.png` - Playlist from Similar Song → similar-tracks table
- `05_artist_similarity_2_result.png` - Artist Similarity → 12 similar artists with scores
- `06_path_3_result.png` - Song Path → path table + transition graph
- `07_alchemy_4_result.png` - Song Alchemy → results
- `08_clap_search_2_results.png` - Text Search (DCLAP) → results list
- `09_lyrics_search_4_axis_result.png` / `_5_text_result.png` / `_6_song_result.png` - Lyrics Search results for all three tabs (By Axis / By Text / By Song)

## All files
| File | Page / state |
|------|--------------|
| 00_login.png | Login |
| 01_dashboard.png | Dashboard (faked stats + charts) |
| 02_index_1_config_openai.png | Analysis & Clustering - config (LLM = OpenAI/atlascloud/DeepSeek) |
| 02_index_2_clustering_playlists.png | Analysis & Clustering - **AI-named playlists** |
| 02_index_3_advanced.png | Analysis & Clustering - Advanced view |
| 03_chat_1_openai_config.png | Instant Playlist - LLM config |
| 03_chat_2_similar_song_result.png | Instant Playlist - **similar-song result** |
| 04_similarity_1_song.png / _2_mood / _3_anchor | Playlist from Similar Song - 3 modes |
| 05_artist_similarity_1_autocomplete.png | Artist Similarity - autocomplete |
| 06_path_1_form.png / _2_autocomplete | Song Path |
| 07_alchemy_1_alchemy.png / _2_anchors / _3_radio | Song Alchemy - 3 tabs |
| 08_clap_search_1_form.png / _2_results | Text Search (DCLAP) |
| 09_lyrics_search_1_axis.png / _2_text / _3_song | Lyrics Search - 3 tabs |
| 10_map.png | Music Map (280-point fake sample) |
| 11_sonic_fingerprint.png | Sonic Fingerprint (creds masked) |
| 13_cleaning.png | Cleaning |
| 14_cron.png | Scheduled Tasks |
| 15_backup.png | Backup & Restore |
| 16_provider_migration.png | Provider Migration |
| 17_setup.png | Setup Wizard (creds masked) |
| 18_users.png | Users (usernames faked) |
