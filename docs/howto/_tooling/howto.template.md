# AudioMuse-AI — How-To Guide

*Where Music Takes Shape.* A walkthrough of every page and feature, with screenshots.

**Application version {{VERSION}}**

> **About the screenshots.** Every track title, artist and album name shown in this guide has been replaced with neutral placeholders (`Song Title 1`, `Artist Name 1`, `Album Name 1`) to avoid reproducing copyrighted metadata. Server URLs, user IDs, tokens and passwords are redacted. A few "result" screenshots use representative placeholder data so the success state can be shown without running heavy jobs.

> **The golden rule:** AudioMuse-AI can only recommend tracks it has *analysed*. Run [Analysis and Clustering](#analysis-and-clustering) first (or schedule it under [Scheduled Tasks](#scheduled-tasks)); every other feature draws on that data.

## Contents

**Getting started**
- [Logging in](#logging-in)
- [Dashboard](#dashboard)

**Build your library data**
- [Analysis and Clustering](#analysis-and-clustering)

**Create playlists**
- [Instant Playlist](#instant-playlist)
- [Playlist from Similar Song](#playlist-from-similar-song)
- [Artist Similarity](#artist-similarity)
- [Song Path](#song-path)
- [Song Alchemy](#song-alchemy)
- [Text Search (DCLAP)](#text-search-dclap)
- [Lyrics Search](#lyrics-search)
- [Sonic Fingerprint](#sonic-fingerprint)

**Explore**
- [Music Map](#music-map)
- [Waveform](#waveform)

**Administration (admin only)**
- [Scheduled Tasks](#scheduled-tasks)
- [Database Cleaning](#database-cleaning)
- [Backup and Restore](#backup-and-restore)
- [Provider Migration](#provider-migration)
- [Setup Wizard](#setup-wizard)
- [Users](#users)

---

## Logging in

**Route:** `/login`

When authentication is enabled, AudioMuse-AI presents a sign-in screen. Enter the username and password of an account created in the [Setup Wizard](#setup-wizard) or on the [Users](#users) page. A session cookie keeps you signed in for 8 hours.

**How to use it**
1. Open the AudioMuse-AI URL in your browser.
2. Type your **username** and **password**.
3. Click **Sign In** — you land on the Dashboard.

![Login screen](screenshots/00-login.png)
*The sign-in screen. Admins see every page; normal users see everything except the Administration menu.*

> **Roles.** **Admins** have full access and can manage other users. **Normal users** can use all the playlist / exploration tools but cannot see the Administration menu (Analysis, Cleaning, Scheduled Tasks, Backup, Provider Migration, Setup, Users).

---

## Dashboard

**Route:** `/`

The landing page summarises your library and the system at a glance: key counts (songs, artists, albums and how much of the library is indexed for each model), the live queue workers, the last completed batch tasks, content charts (genres, mood coverage, tempo) and the configured scheduled tasks. It refreshes automatically every 30 seconds.

**What to look for**
1. **Total Songs / Artists / Albums** — with the percentage indexed by each model (Musicnn, CLAP, GMM).
2. **Queue Workers** — each worker container runs a high-priority and a default worker, so one node usually shows two rows.
3. **Last 10 completed batch tasks** — type, status, duration and a short note per run.
4. **Content / Library charts** — genre distribution, mood coverage and tempo profile of the analysed library.

![Dashboard](screenshots/01-dashboard.png)
*Key numbers, queue workers, recent tasks, genre/mood/tempo charts and scheduled tasks.*

> **Tip:** Low index percentages mean you still have library to analyse — head to [Analysis and Clustering](#analysis-and-clustering).

---

## Analysis and Clustering

**Route:** `/analysis` · **admin only**

This is the engine room. **Analysis** scans recently-added albums on your media server and computes the acoustic features, embeddings and mood vectors that power every other feature. **Clustering** then groups the analysed library into automatically-named playlists (optionally using an AI provider to name them). Both run as background jobs whose progress you can follow live.

**How to use it**
1. Set **Number of Recent Albums** to analyse (small for a first run, or large to catch up).
2. Click **Start Analysis** and watch *Task Status* — type, running time, progress bar and a live log.
3. Set the clustering parameters (algorithm, number of playlists, clustering runs) and, optionally, an **AI provider** for playlist names.
4. Click **Start Clustering**; when it finishes, the generated playlists appear under *Generated Playlists*.
5. Use **Fetch Playlists** at any time to list the playlists already generated.

![Analysis run result](screenshots/02a-analysis-result.png)
*Analysis parameters and the task status after an analysis run completes.*

![Clustering run result](screenshots/02b-clustering-result.png)
*Clustering parameters, AI naming, and the resulting auto-generated playlists.*

> **Basic vs Advanced.** Toggle the **Advanced** view to expose the full set of tunables; **Basic** keeps just the essentials. A run can be stopped with **Cancel**.

> **Tip:** Prefer set-and-forget? Configure recurring runs on the [Scheduled Tasks](#scheduled-tasks) page instead of starting them by hand.

---

## Instant Playlist

**Route:** `/chat/`

Describe the vibe you want in plain language and let an AI assistant build a playlist for you. Behind the scenes it plans a chain of internal tools (similarity, search, brainstorming, lyrics) and returns up to 100 matching songs from *your* library, which you can then push to your media server as a real playlist.

**How to use it**
1. Choose an **AI Provider** (Ollama, OpenAI-compatible, Gemini, Mistral…). The model and server fields adapt to your choice.
2. Type what you want — e.g. *"Calm rainy-day piano songs for focus"* — or click a suggestion chip (Similar Song, Multi-Song Mix, From Artist, Sound/Vibe, Lyrics Theme…).
3. Click **Get Playlist Idea** and watch the live pipeline steps.
4. Review the generated list, give it a name, and click **Create Playlist** to save it on your media server.

![Instant Playlist form](screenshots/03-instant-playlist.png)
*Pick an AI provider, type a request (or tap a suggestion chip), then ask for a playlist.*

![Instant Playlist AI result](screenshots/03b-instant-playlist-result.png)
*The AI returns a tool chain, a collapsible raw log, and the generated playlist.*

> **Tip:** Keep prompts short and specific — naming 1–3 seed songs or artists plus a couple of filters (genre, year, mood) works better than a long paragraph.

---

## Playlist from Similar Song

**Route:** `/similarity`

Pick a seed — a specific **song**, a **mood** centroid, or a saved **anchor** — and AudioMuse-AI finds the tracks that sound closest to it, ranked by acoustic distance. Turn the results into a playlist in one click.

**How to use it**
1. Use the mode toggle to pick **Song**, **Mood** or **Anchor**.
2. In Song mode, start typing (3+ characters) and pick a track from the autocomplete.
3. Set **Number of results**; optionally enable **Radius Similarity**.
4. Click **Find Similar Tracks** to see the ranked list.
5. Give the playlist a name and click **Create Playlist on Media Server**.

![Song autocomplete](screenshots/04b-similarity-autocomplete.png)
*Start typing to find a seed song; suggestions appear with title, artist and album.*

![Similar tracks results](screenshots/04-similarity.png)
*Ranked similar tracks with mood/genre tags and a distance badge (smaller = more similar).*

---

## Artist Similarity

**Route:** `/artist_similarity`

Find artists whose overall sound profile resembles a chosen artist, using Gaussian Mixture Models. Optionally see *why* they match — the shared "components" of their sound, with representative example songs from each — and build a playlist from an artist's tracks or the matching songs.

**How to use it**
1. Type an artist name (2+ characters) and select it from the list.
2. Set how many similar artists to return; keep **Show component matches** ticked to see the reasoning.
3. Click **Find Similar Artists**.
4. Use **Show All Songs** or **Show Matches** on any row to expand its tracks.
5. Name the playlist and click **Create Playlist on Media Server**.

![Artist search](screenshots/05b-artist-autocomplete.png)
*Type to find an artist; the suggestion shows each artist's track count.*

![Similar artists results](screenshots/05-artist-similarity.png)
*Ranked similar artists with a similarity score and expandable song / component matches.*

---

## Song Path

**Route:** `/path`

Build a smooth, gradually-shifting sequence of songs that travels from a **start** point to an **end** point. Each endpoint can be a song, a lyrics-seed song, a mood or an anchor. The result is charted as a feature progression and a 2-D route, and can be saved as a playlist.

**How to use it**
1. Choose the **Start** endpoint (e.g. Song mode) and select a track from the autocomplete.
2. Choose the **End** endpoint the same way.
3. Set **Number of steps** (and optionally fix the playlist size).
4. Click **Find Path** to see the ordered journey and its charts.
5. Name it and click **Create Playlist on Media Server**.

![Song Path results](screenshots/06-song-path.png)
*A computed path between two songs, with progression charts and the ordered track list.*

> **Tip:** Great for parties and long drives — start mellow and end energetic (or vice-versa) for a set that evolves naturally.

---

## Song Alchemy

**Route:** `/alchemy`

"Vector maths" for music. **Include** songs/artists/anchors/moods to add their character and **Exclude** others to subtract it; AudioMuse-AI computes the resulting centroid and returns the nearest songs, plotted on a 2-D projection. You can save the result as a playlist or store the centroid as a reusable **anchor**.

**How to use it**
1. In each item card pick a type (song, artist, anchor, mood), search for it, and set it to **Include** or **Exclude**.
2. Add more rows as needed and tune **Number of results**, **Temperature** and **Subtract distance**.
3. Click **Run Alchemy** to compute the recommendations and the projection plot.
4. Save the results as a playlist, or click **Save Anchor** to reuse this centroid elsewhere.

![Song Alchemy results](screenshots/07-song-alchemy.png)
*Include/Exclude item cards, the 2-D projection plot and the resulting recommended songs.*

> **Note:** Anchors created here appear as a seed option across Similarity, Song Path and Alchemy.

---

## Text Search (DCLAP)

**Route:** `/clap_search`

Search your library by *describing the sound* in free text. Powered by CLAP text-to-audio embeddings, it matches your description (mood, genre, instrument, energy) directly against the audio of your tracks — no tags required.

**How to use it**
1. Type a description — e.g. *"energetic upbeat saxophone jazz"* — or tap a suggested example tag.
2. Set the result **limit** and click **Search**.
3. Review the matches (each with a similarity score), then name and create a playlist.

![DCLAP text search results](screenshots/08-dclap-search.png)
*A natural-language audio search and its ranked, similarity-scored results.*

> **Note:** Requires CLAP indexing to be enabled and built for your library.

---

## Lyrics Search

**Route:** `/lyrics_search`

Find songs by what they're *about*, focusing on the lyrics rather than the groove. Three complementary modes are available as tabs.

**By Axis — compose facets.** Pick a value for one or more lyrical "axes" (setting, social dynamic, emotional valence, narrative temporality, thematic weight) to compose a precise search.

![Lyrics search by axis](screenshots/09a-lyrics-axis.png)
*Select one value per axis; leave an axis on "None" to ignore it.*

**By Text — semantic free-text.** Type a free-form description of the lyrics; matching is by *meaning*, not exact words.

![Lyrics search by text](screenshots/09b-lyrics-text.png)
*"Love and heartbreak in the city at night" surfaces songs even if those exact words never appear.*

**By Song — lyrically and acoustically similar.** Pick a seed song and find tracks that share both its lyrical meaning (75%) and its sound (25%).

![Lyrics search by song](screenshots/09c-lyrics-song.png)
*The SemGrove index blends lyrics and audio similarity around a seed song.*

> **Note:** Lyrics features require lyrics analysis and the relevant indexes to be built. Any tab's results can be saved as a media-server playlist.

---

## Sonic Fingerprint

**Route:** `/sonic_fingerprint`

Generate a personalised playlist from *your own* listening history on the media server, plus a radar "fingerprint" of your mood traits and a top-genres breakdown.

**How to use it**
1. Enter your media-server **username/user-ID** (the default user is pre-filled). A token/password is only needed for other users.
2. Set the **Number of results**.
3. Click **Generate My Sonic Fingerprint** — the radar and recommendations appear.
4. Name and create a playlist from the results.

![Sonic Fingerprint form](screenshots/11-sonic-fingerprint.png)
*Enter your credentials and how many songs to include.*

![Sonic Fingerprint result](screenshots/11b-sonic-fingerprint-result.png)
*The mood radar, your top genres, and the recommended tracks.*

> **Note:** The radar shows the share of your songs leaning toward each trait (danceable, aggressive, happy, party, relaxed, sad).

---

## Music Map

**Route:** `/map`

An interactive 2-D scatter plot of your whole library, projected from the song embeddings and coloured by dominant genre/mood. Pan, zoom, lasso-select clusters, search for a track to highlight it, and build playlists or draw listening paths directly on the map.

**How to use it**
1. Choose a map size (25% loads fastest; 100% shows everything).
2. Hover a point to see its track; **lasso** or click points to add them to the selection.
3. Use the search box to find a song and highlight it on the map.
4. With 2–10 songs selected, click **Song Path** to trace a route between them, or **Create playlist** from the whole selection.

![Music Map](screenshots/10-music-map.png)
*The library as a sound-space; each dot is a track, coloured by genre.*

![Music Map with a path drawn](screenshots/10b-music-map-path.png)
*A path drawn across the map connecting a sequence of selected songs.*

> **Tip:** Use the legend below the map to hide/show individual genres and declutter the view.

---

## Waveform

**Route:** `/waveform`

Visualise the amplitude waveform of any single track in your library. AudioMuse-AI downloads and analyses the file, then renders its loudness shape over time.

**How to use it**
1. Start typing in the search box and pick a track.
2. Click **Generate Waveform** (this can take a few seconds while the file is fetched and analysed).
3. The waveform appears, captioned with the number of sample points.

![Waveform visualization](screenshots/12-waveform.png)
*The rendered waveform for a selected track, with its title and artist above.*

---

## Scheduled Tasks

**Route:** `/cron` · **admin only**

Automate Analysis, Clustering and Sonic Fingerprint runs with cron expressions, so your library stays up to date without manual effort.

**How to use it**
1. Enter a cron expression for each task (e.g. `0 2 * * 0-5` = 02:00 on weekdays).
2. Tick **Enable** for the schedules you want active.
3. Click **Save Schedules**.

![Scheduled Tasks](screenshots/14-scheduled-tasks.png)
*One cron expression and an enable toggle per task type.*

> **Note:** Sensible defaults — Analysis nightly at 02:00 (except Saturday), Clustering Saturday at 02:00, Sonic Fingerprint Saturday at 01:00. They start disabled.

---

## Database Cleaning

**Route:** `/cleaning` · **admin only**

Scan the media server and remove database rows for albums/tracks that no longer exist there — keeping AudioMuse-AI's data in sync with your actual library.

**How to use it**
1. Click **Start Database Cleaning**.
2. Watch the status; a summary reports how many orphaned albums/tracks were found and deleted.

![Database Cleaning](screenshots/13-cleaning.png)
*A single action that scans for and removes orphaned entries.*

> **Heads up:** this permanently deletes the analysis data for tracks that are gone from your server. Run it after you've removed or moved music, not on a healthy library.

---

## Backup and Restore

**Route:** `/backup` · **admin only**

Download a full database dump, or restore from a previous one. Analysis can be expensive to recompute, so a backup is cheap insurance.

**How to use it**
1. **Create Backup** — runs a full dump and downloads the `.sql` file to your browser.
2. **Restore** — choose a backup file, type the confirmation phrase, and click **Restore**.

![Backup and Restore](screenshots/15-backup-restore.png)
*Create a backup (downloads a .sql dump) or upload one to restore.*

> **Restore replaces everything.** It wipes and recreates the database, then restarts the app and workers. Only restore a dump you trust, and ideally take a fresh backup first.

---

## Provider Migration

**Route:** `/provider-migration` · **admin only**

Switched the media server in front of the *same* music library (e.g. replaced Navidrome with Emby)? This 6-step wizard rewrites every track's internal ID to the matching ID on the new provider, so your analysis, embeddings and local playlists keep pointing at the right songs.

**The steps**
1. **Back up** your database and confirm you've stored it.
2. **Choose the new provider**, enter its credentials, and **Test Connection**.
3. **Automatic matching** — match every track against the new provider (re-runnable).
4. **Manual album matching** (optional) — fix any albums that didn't match automatically.
5. **Finalize** the dry run and review the counts.
6. **Execute** — apply the migration.

![Provider Migration wizard](screenshots/16-provider-migration.png)
*The guided wizard: back up → choose provider → match → review → finalize → execute.*

> **Destructive final step.** Executing rewrites IDs and *deletes* tracks that don't exist on the new provider as orphans. Best results come when both providers expose the library from the same file paths. Always back up first.

---

## Setup Wizard

**Route:** `/setup` · **admin only**

The first-run and re-configuration screen. Connect your media server, configure authentication, tune advanced options, and optionally wire up lyrics-provider APIs.

**How to use it**
1. Pick your **media server type** and fill in its URL and credentials, then **Test connection**.
2. Choose which libraries to scan (or scan all).
3. Configure **Authentication** (enable/disable, admin account, JWT secret, API token).
4. Expand **Advanced configuration** and **Lyrics API** if needed.
5. Click **Save configuration** — the app applies the settings and restarts.

![Setup Wizard](screenshots/17-setup-wizard.png)
*Media-server connection and authentication settings (secrets redacted in this screenshot).*

> **Note:** Once an admin account exists, manage additional accounts on the [Users](#users) page rather than here.

---

## Users

**Route:** `/users` · **admin only**

Manage the accounts that can sign in. Admins can add normal users or other admins, change any password, and delete accounts. Normal users only see — and can only change the password of — their own account.

**How to use it**
1. Review the user table (username, role, created date).
2. Click **Add user**, fill in the username, password and role, then **Save user**.
3. Use **Change password** or **Delete** on any row to manage existing accounts.

![Users configuration](screenshots/18-users.png)
*The user table and the "Add user" panel (usernames anonymised in this screenshot).*

> **Note:** Passwords are stored hashed (argon2) and never shown again. You can't delete your own account, and at least one admin must always remain.

---

*AudioMuse-AI — How-To Guide · application version {{VERSION}} · screenshots use placeholder metadata to avoid copyright.*
