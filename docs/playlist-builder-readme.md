# Playlist Builder

The Playlist Builder is a powerful feature in AudioMuse-AI that lets you create intelligent playlists using smart filters, extend existing playlists with similar songs, or combine both approaches for maximum flexibility.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
  - [Smart Filters](#smart-filters)
  - [Existing Playlist Mode](#existing-playlist-mode)
  - [Extend Playlist](#extend-playlist)
  - [Track Weights](#track-weights)
  - [Source Playlist Drawer](#source-playlist-drawer)
  - [Include/Exclude Songs](#includeexclude-songs)
  - [Statistics Display](#statistics-display)
  - [Web Player](#web-player)
  - [Saving Playlists](#saving-playlists)
- [Workflows](#workflows)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Access the Playlist Builder from the main navigation menu or visit `/playlist_builder`.

### Three Ways to Build Playlists

1. **Smart Filters Only**: Search your library using filters (artist, mood, BPM, etc.) and save matching songs
2. **Extend Existing Playlist**: Select a playlist and find similar songs to add
3. **Smart Filter → Extend**: Use filter results as a starting point, then find similar songs

---

## Features

### Smart Filters

Build dynamic playlists by filtering your music library based on various attributes.

#### Available Filter Types

| Field | Type | Description |
|-------|------|-------------|
| **Album Title** | Text | Filter by album name |
| **Artist** | Text | Filter by artist name |
| **Track Title** | Text | Filter by song title |
| **BPM** | Range | Tempo ranges: 0-80, 80-100, 100-120, 120-140, 140-160, 160+ |
| **Energy** | Range | Low, Medium, or High energy levels |
| **Key** | Dropdown | Musical key: C, C#, D, D#, E, F, F#, G, G#, A, A#, B |
| **Scale** | Dropdown | Major or Minor |
| **Mood** | Dropdown | Dynamically loaded from analyzed tracks |

#### Filter Operators

**Text fields** (Album, Artist, Title):
- `contains` - Partial match (e.g., "rock" matches "Classic Rock")
- `does not contain` - Exclude partial matches
- `is` - Exact match
- `is not` - Exclude exact matches

**Range/Dropdown fields** (BPM, Energy, Key, Scale, Mood):
- `is` - Exact match only

#### Match Modes

- **Match ALL rules** (AND logic): Songs must match every filter
- **Match ANY rule** (OR logic): Songs matching at least one filter are included

#### Using Smart Filters

1. Click **+ Add Filter** to add a filter row
2. Select a field, operator, and value
3. Add more filters as needed
4. Choose your match mode (ALL or ANY)
5. Click **Search** to find matching songs
6. Exclude unwanted songs by clicking **Exclude**
7. Save your playlist or send results to Extend mode

---

### Existing Playlist Mode

Extend playlists you've already created on your media server.

#### Fetching Playlists

1. Switch to the **Existing Playlist** tab
2. Click **Fetch Playlists** to sync with your media server
3. Wait for the sync to complete (status shows "Done!")
4. Select a playlist from the dropdown

The dropdown shows each playlist with its song count (e.g., "Chill Vibes (25 songs)").

#### Smart Filter Results Option

After running a Smart Filter search, your results appear in the playlist dropdown as **"Smart Filter Results (N songs)"**. This lets you use filter results as a source for finding similar songs.

---

### Extend Playlist

Find songs similar to your source playlist using AI-powered similarity search.

#### How It Works

The system calculates a "centroid" (average musical fingerprint) from your source songs and finds tracks with similar characteristics. This uses the 200-dimensional embeddings generated during audio analysis.

#### Controls

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **Maximum Songs** | 1-200 | 50 | Maximum recommendations to return |
| **Similarity Threshold** | 0.0-1.0 | 0.50 | Distance threshold (lower = more similar) |

**Similarity Threshold Explained:**
- **0.0-0.3**: Very similar songs (close musical fingerprints)
- **0.3-0.5**: Moderately similar (balanced recommendations)
- **0.5-1.0**: More diverse results (wider musical range)

#### Finding Similar Songs

1. Select a source (existing playlist or Smart Filter results)
2. Adjust Maximum Songs and Similarity Threshold
3. Click **Find Similar Songs**
4. Review recommendations in the results table
5. Click **Include** to add songs to your extended playlist
6. Click **Exclude** to remove songs from recommendations

---

### Track Weights

Control how much influence each track has on recommendations by assigning weights.

#### How Weights Work

Weights determine how heavily a track influences the centroid (average musical fingerprint) used for finding similar songs. Higher weights mean that track's characteristics dominate the recommendations.

| Weight | Influence | Use Case |
|--------|-----------|----------|
| **x1** | Normal (default) | Standard influence |
| **x2** | Double | Slightly favor this track |
| **x4** | Quadruple | Moderately favor this track |
| **x8** | 8x | Strongly favor this track |
| **x16** | 16x | Very strongly favor this track |
| **x32** | 32x | Heavily favor this track |
| **x64** | 64x | Very heavily favor this track |
| **x128** | 128x | Dominant influence |
| **x256** | 256x | Extreme influence |
| **x512** | 512x | Near-total influence |
| **x1024** | 1024x | Maximum influence |

#### Weight Impact Example

With 10 tracks where 9 have weight x1 and 1 has weight x1024:
- The x1024 track contributes **99.1%** of the centroid
- The other 9 tracks combined contribute only **0.9%**

This means recommendations will be heavily biased toward songs similar to the high-weight track.

#### Using Weights

1. Click the **Weight** button on any track (shows "x1" by default)
2. Each click cycles through: x1 → x2 → x4 → x8 → x16 → x32 → x64 → x128 → x256 → x512 → x1024 → x1
3. Weights > x1 are highlighted in blue
4. Recommendations automatically recalculate after 500ms

#### Weight Button Locations

| Location | Affects |
|----------|---------|
| **Source Playlist Drawer** | How much each source track influences recommendations |
| **Recommended Songs Table** | Pre-set weight for when you include the track |

---

### Source Playlist Drawer

A collapsible drawer shows your source playlist tracks, allowing you to adjust weights and remove tracks before finding recommendations.

#### Drawer Features

- **Collapsed by default** - Click header to expand/collapse
- **Track count badge** - Shows number of source tracks
- **Play button** - Preview tracks before modifying
- **Weight button** - Adjust track influence (see [Track Weights](#track-weights))
- **Remove button** - Remove track from source set

#### Remove vs Exclude

| Action | Effect | Use When |
|--------|--------|----------|
| **Remove** | Drops track from source set | Track was incorrectly included by filters |
| **Exclude** | Pushes recommendations away from similar tracks | You want to avoid songs like this one |

**Key Difference**: Remove simply ignores the track. Exclude actively steers recommendations away from similar songs.

#### Accessing the Drawer

The drawer appears automatically when you:
1. Use **Smart Filter → Extend**: Run a Smart Filter search, then click **Send to Extend Playlist**
2. Use **Existing Playlist**: Select a playlist from the dropdown and click **Find Similar Songs**

Both workflows give you full access to track weights and the Remove button.

Click the drawer header to expand and view/modify source tracks.

---

### Include/Exclude Songs

Fine-tune your playlist by including or excluding individual songs.

#### In Smart Filter Mode

- All matching songs are included by default
- Click **Exclude** to remove a song from the final playlist
- Excluded songs show a red "EXCLUDED" badge
- **Total** count updates to reflect exclusions

#### In Extend Mode

- Recommended songs are NOT included by default
- Click **Include** to add a song to your extended playlist (green highlight)
- Click **Exclude** to remove a song from recommendations
- Included songs appear at the top of the results
- **Total** count shows source playlist + included songs

#### Visual Indicators

| State | Background | Badge |
|-------|------------|-------|
| Included | Green | "INCLUDED" |
| Excluded | Red (60% opacity) | "EXCLUDED" |
| Neutral | Default | Action buttons |

---

### Statistics Display

The stats bar shows real-time counts as you build your playlist.

#### Smart Filter Mode

| Stat | Meaning |
|------|---------|
| **Selected Songs** | Number of songs matching your filters |
| **Total Songs** | Selected minus excluded (what will be saved) |

#### Extend Mode

| Stat | Meaning |
|------|---------|
| **Selected Songs** | Number of songs in source playlist |
| **Total Songs** | Source playlist + newly included songs |

---

### Web Player

Preview songs directly in the browser before including them.

#### Controls

| Button | Action |
|--------|--------|
| **Play/Pause** | Start or pause playback |
| **Seek -10s** | Jump back 10 seconds |
| **Seek +10s** | Jump forward 10 seconds |
| **Stop** | Stop playback and hide player |

#### Features

- Progress bar with click-to-seek
- Current time and duration display
- Track title and artist shown
- Streams directly from your media server
- Dark theme for visibility

---

### Saving Playlists

Save your curated playlist to your media server.

#### Steps

1. Enter a name for your playlist
2. Click **Save to Media Server**
3. Wait for confirmation message

#### What Gets Saved

| Mode | Songs Saved |
|------|-------------|
| **Smart Filter** | All matching songs except excluded |
| **Extend** | Source playlist + explicitly included songs |

#### Supported Media Servers

- Jellyfin
- Navidrome
- Emby
- Lyrion (formerly Logitech Media Server)

---

## Workflows

### Workflow 1: Smart Filter Only

Best for: Creating playlists based on specific criteria (e.g., "all jazz songs in C major")

```
1. Add filters (Artist contains "Miles Davis", Key is "C")
2. Set match mode to ALL
3. Click Search
4. Exclude any unwanted songs
5. Enter playlist name
6. Click Save to Media Server
```

### Workflow 2: Extend Existing Playlist

Best for: Discovering new songs similar to a playlist you already love

```
1. Switch to Existing Playlist tab
2. Click Fetch Playlists
3. Select your playlist
4. Adjust similarity threshold (lower = more similar)
5. Click Find Similar Songs
6. Include songs you like
7. Enter new playlist name
8. Click Save to Media Server
```

### Workflow 3: Smart Filter → Extend

Best for: Maximum control—filter first, then expand with similar songs

```
1. Create Smart Filters (e.g., Mood is "happy", Energy is "High")
2. Click Search
3. Review matches and exclude unwanted songs
4. Click Send to Extend Playlist
5. System finds similar songs automatically
6. Include additional songs from recommendations
7. Enter playlist name
8. Click Save to Media Server
```

### Workflow 4: Weighted Recommendations

Best for: Finding songs similar to a specific "anchor" track within a broader set

```
1. Create Smart Filters to get a set of songs
2. Click Send to Extend Playlist
3. Expand the Source Playlist drawer
4. Find your anchor track and set its weight to x64 or higher (up to x1024)
5. Optionally Remove tracks that don't fit your vision
6. Wait for recommendations to recalculate
7. Include songs from recommendations
8. Enter playlist name
9. Click Save to Media Server
```

**Example**: You have 20 rock songs but want recommendations heavily influenced by one particular blues-rock track. Set that track to x512 or x1024—recommendations will strongly favor similar blues-influenced songs.

---

## Technical Details

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/filter_options` | GET | Get available filter values (moods, keys, etc.) |
| `/api/extend_playlist` | POST | Search or find similar songs |
| `/api/save_extended_playlist` | POST | Save playlist to media server |
| `/api/playlists` | GET | Get all playlists from database |
| `/api/analysis/fetch_playlists` | POST | Trigger media server sync |

### Centroid-Based Similarity

The extend feature uses centroid-based nearest neighbor search:

1. **Positive Centroid**: Weighted average embedding of source + included songs
2. **Negative Centroid**: Average embedding of excluded songs (weighted 0.5x)
3. **Query Vector**: Positive centroid minus weighted negative centroid
4. **Search**: Voyager HNSW index finds nearest neighbors to query vector

This means excluding songs actively steers recommendations away from similar tracks.

### Weighted Centroid Calculation

When track weights are applied, the centroid uses a weighted mean formula:

```
weighted_centroid = Σ(weight_i × vector_i) / Σ(weight_i)
```

Where:
- `weight_i` is the track's weight (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, or 1024)
- `vector_i` is the track's 200-dimensional embedding

**Key behaviors:**
- Default weight is 1 for all tracks (equivalent to unweighted mean)
- Excluded songs always use unweighted mean (exclusion strength is separate)
- Weights are validated server-side to prevent invalid values
- Auto-recalculation triggers 500ms after weight changes (debounced)

### Dark Mode

The Playlist Builder fully supports dark mode. All components use CSS variables for theming:

- Automatic adaptation to system/app theme preference
- High contrast for accessibility
- Consistent styling across all UI elements

---

## Troubleshooting

### Common Issues

#### "No songs found matching the filters"

- Check that your filters aren't too restrictive
- Try using "any" match mode instead of "all"
- Verify songs have been analyzed with the required metadata

#### "Please add at least one filter"

- Add at least one filter row before clicking Search
- Click the **+ Add Filter** button to add a filter

#### "Please select a playlist"

- Select a playlist from the dropdown before clicking Find Similar Songs
- Click Fetch Playlists if the dropdown is empty

#### Fetch Playlists shows "Failed"

- Check your media server connection settings
- Verify your media server is running and accessible
- Check the browser console for detailed error messages

#### No recommendations returned

- Try increasing the Similarity Threshold (e.g., from 0.5 to 0.7)
- Increase Maximum Songs
- Ensure your library has been analyzed

#### Web Player not working

- Check that stream URLs are configured correctly
- Verify your media server supports streaming
- Check browser console for CORS or authentication errors

### Error Messages

| Message | Cause | Solution |
|---------|-------|----------|
| "Search failed" | API error | Check server logs |
| "Internal error" | Backend exception | Check Flask logs |
| "No songs to save" | All songs excluded or no results | Include at least one song |
| "Please provide a playlist name" | Empty name field | Enter a playlist name |

---

## Version History

- **v0.9.0**: Added Track Weights feature and Source Playlist Drawer for fine-tuning recommendations
- **v0.8.0**: Added Smart Filter functionality, Include/Exclude actions, and improved statistics display
- **v0.7.0**: Initial Playlist Builder with Extend functionality
