# AudioMuse-AI with Navidrome

This guide shows how to deploy AudioMuse-AI next to Navidrome and how to install
the Navidrome plugin so that sonic features show up right inside Navidrome. Once
it works you get Instant Mix (similar songs), Radio (similar artists) and related
artist info in the Navidrome web UI and in compatible clients like Symfonium,
Feishin, Substreamer, Tempo and Sonixd.

Two repositories are involved:

- Core app: https://github.com/NeptuneHub/AudioMuse-AI
- Navidrome plugin: https://github.com/NeptuneHub/AudioMuse-AI-NV-plugin

## What you need

- The **latest** version of all three pieces, kept aligned: the latest Navidrome,
  the latest AudioMuse-AI core, and the latest Navidrome plugin. A version
  mismatch is the most common cause of errors, so always update all three
  together.
- Docker and Docker Compose (or a native build, see the note in Step 1).
- Navidrome and AudioMuse-AI able to reach each other over the network.

## Step 1 - Deploy AudioMuse-AI

AudioMuse-AI runs as a small stack: the Flask app, a worker, PostgreSQL and
Redis. It needs only three things set by environment variable: **PostgreSQL**,
**Redis** and the timezone (**TZ**). In the Docker Compose example, Postgres and
Redis are already included as services and wired for you, so in `.env` you
normally set just the database user, password and TZ. Everything else is
configured later in the Setup Wizard.

Download the Docker Compose file and the env template (the second command saves
it straight to `.env`):

```bash
wget https://raw.githubusercontent.com/NeptuneHub/AudioMuse-AI/refs/heads/main/deployment/docker-compose.yaml
wget https://raw.githubusercontent.com/NeptuneHub/AudioMuse-AI/refs/heads/main/deployment/.env.example -O .env
```

Edit `.env`:

```env
TZ=UTC
POSTGRES_USER=audiomuse
POSTGRES_PASSWORD=changeme
```

Then start the stack:

```bash
docker compose up -d
```

> Prefer not to use Docker? There are native builds for macOS, Windows and Linux
> on the releases page: https://github.com/NeptuneHub/AudioMuse-AI/releases .
> They bundle PostgreSQL and Redis, so you just download and run. Always grab the
> latest release.

Open `http://<your-host>:8000`. On first start the Setup Wizard appears:

1. Pick **Navidrome** as the media server and enter your Navidrome URL, user and
   password.
2. Under the AudioMuse-AI authentication section, set usern, password and **API token**. The API token will be needed for the plugin.
3. Save and finish the wizard.

**Run a first analysis (do this before anything else).** AudioMuse-AI can only
find similar songs after it has analysed them. Right after setup, start a new
analysis from the main page and let it finish. On a large library this takes a
while. When it is done, confirm it works by using **Similar Song** on any track
in the AudioMuse-AI UI.

## Step 2 - Enable plugins in Navidrome

Add these variables to your Navidrome container. The order in `ND_AGENTS`
matters: Navidrome uses the first agent that supports sonic similarity, so keep
`audiomuseai` first.

```yaml
services:
  navidrome:
    image: deluan/navidrome:latest
    ports:
      - "4533:4533"
    environment:
      - ND_PLUGINS_ENABLED=true
      - ND_PLUGINS_AUTORELOAD=true
      - ND_AGENTS=audiomuseai,lastfm,deezer
    volumes:
      - ./data:/data
      - /path/to/music:/music:ro
```

## Step 3 - Install the plugin file

1. Download the latest `audiomuseai.ndp` from the plugin releases:
   https://github.com/NeptuneHub/AudioMuse-AI-NV-plugin/releases
2. Copy it into the Navidrome plugins folder, `/data/plugins` by default.
3. Restart Navidrome (with `ND_PLUGINS_AUTORELOAD=true` a reload is often enough).

```bash
mkdir -p ./data/plugins
cp audiomuseai.ndp ./data/plugins/
docker compose restart navidrome
```

## Step 4 - Configure the plugin

In the Navidrome web UI go to **Settings > Plugins**, open AudioMuse-AI and set:

- **AudioMuse-AI API URL**: the address where Navidrome reaches the core app, for
  example `http://192.168.1.50:8000`. Use a host and port that the Navidrome
  container can actually reach, not `localhost`, unless they share the same
  network namespace.
- **API token**: the same token you set in the Setup Wizard in Step 1.

Save. The plugin is now active.

## Step 5 - Check that it works

Test both sides:

- In AudioMuse-AI, use **Similar Song** on a track. It should return results.
- In Navidrome, use **Instant Mix** on the same track. It should build a queue of
  similar songs.

Then check the logs.

Navidrome log (look for the plugin name and no errors):

```bash
docker compose logs -f navidrome | grep audiomuseai
```

You should see lines with `plugin=audiomuseai` and no error next to them.

AudioMuse-AI Flask log (look for the incoming requests from the plugin):

```bash
docker compose logs -f audiomuse-ai-flask
```

You should see the API calls arriving each time you trigger Instant Mix or Radio
in Navidrome.

## Troubleshooting

- **401 Unauthorized in the Navidrome log**: the API token is missing or wrong.
  Set the same token in the plugin and in the AudioMuse-AI Setup Wizard.
- **No requests in the AudioMuse-AI log**: Navidrome cannot reach the API URL.
  Check the host, port and network path, and confirm the URL from inside the
  Navidrome container.
- **Empty or poor results**: the library is not analysed yet, or not fully. Run
  the analysis in AudioMuse-AI and wait for it to finish.
- **Odd errors in general**: update Navidrome, the plugin and the AudioMuse-AI
  core to their latest versions and keep them aligned. Most issues come from one
  of the three being out of date.
