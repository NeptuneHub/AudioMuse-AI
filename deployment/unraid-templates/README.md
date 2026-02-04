# 🎵 AudioMuse-AI - Unraid Templates

## Overview

These XML templates allow you to easily deploy AudioMuse-AI on Unraid Server.

AudioMuse-AI is an open-source solution for automatic playlist generation for Jellyfin, Navidrome, LMS, Lyrion and Emby. It analyzes your audio files locally using Librosa and ONNX.

## Architecture

```
┌────────────────────┐     ┌────────────────────┐
│   AudioMuse-AI     │────►│      Redis         │
│   (Flask + Worker) │     │   Port: 6380       │
│   Port: 8000       │     └────────────────────┘
│                    │
│                    │     ┌────────────────────┐
│                    │────►│    PostgreSQL      │
└────────────────────┘     │   Port: 5435       │
         │                 └────────────────────┘
         │
         ▼
┌────────────────────┐
│  Media Server      │
│  (Jellyfin/etc)    │
└────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `AudioMuse-AI.xml` | Main container (Flask app + Worker) |
| `AudioMuse-AI-Redis.xml` | Redis container for job queue |
| `AudioMuse-AI-PostgreSQL.xml` | PostgreSQL container for data storage |

## Installation

### 1. Copy templates to Unraid

```bash
# Download the templates
wget -P /boot/config/plugins/dockerMan/templates-user/ \
  https://raw.githubusercontent.com/ptichalouf/AudioMuse-AI/main/deployment/unraid-templates/AudioMuse-AI.xml \
  https://raw.githubusercontent.com/ptichalouf/AudioMuse-AI/main/deployment/unraid-templates/AudioMuse-AI-Redis.xml \
  https://raw.githubusercontent.com/ptichalouf/AudioMuse-AI/main/deployment/unraid-templates/AudioMuse-AI-PostgreSQL.xml
```

Or manually copy the XML files to `/boot/config/plugins/dockerMan/templates-user/`

### 2. Create data directories

```bash
mkdir -p /mnt/user/appdata/audiomuse-ai/{data,temp,redis,postgres}
chown -R nobody:users /mnt/user/appdata/audiomuse-ai
```

### 3. Install containers (ORDER MATTERS!)

1. **First: PostgreSQL**
   - Docker → Add Container → Template: AudioMuse-AI-PostgreSQL
   - Set a strong password
   - Apply

2. **Second: Redis**
   - Docker → Add Container → Template: AudioMuse-AI-Redis
   - Apply

3. **Finally: AudioMuse-AI**
   - Docker → Add Container → Template: AudioMuse-AI
   - Configure your media server settings (see below)
   - Set the same PostgreSQL password
   - Set your Unraid IP for POSTGRES_HOST and REDIS_URL
   - Apply

### 4. Media Server Configuration

#### For Jellyfin

**Find your User ID:**
1. Log into Jellyfin as admin
2. Go to Dashboard → Admin Panel → Users
3. Click on your user
4. The ID is in the URL: `http://.../useredit.html?userId=XXXXXX`

**Create an API Token:**
1. Dashboard → Admin Panel → API Keys
2. Create a new key

**Configure AudioMuse-AI:**
| Variable | Value |
|----------|-------|
| MEDIASERVER_TYPE | `jellyfin` |
| JELLYFIN_URL | `http://YOUR_JELLYFIN_IP:8096` |
| JELLYFIN_USER_ID | Your User ID |
| JELLYFIN_TOKEN | Your API Token |

#### For Navidrome

| Variable | Value |
|----------|-------|
| MEDIASERVER_TYPE | `navidrome` |
| NAVIDROME_URL | `http://YOUR_NAVIDROME_IP:4533` |
| NAVIDROME_USER | Your username |
| NAVIDROME_PASSWORD | Your password |

#### For Lyrion

| Variable | Value |
|----------|-------|
| MEDIASERVER_TYPE | `lyrion` |
| LYRION_URL | `http://YOUR_LYRION_IP:9000` |

### 5. Database Connection

Make sure to configure:

| Variable | Value |
|----------|-------|
| POSTGRES_HOST | Your Unraid IP (e.g., `192.168.1.10`) |
| POSTGRES_PORT | `5435` |
| POSTGRES_USER | `audiomuse` |
| POSTGRES_PASSWORD | Same as PostgreSQL container |
| POSTGRES_DB | `audiomuse` |
| REDIS_URL | `redis://YOUR_UNRAID_IP:6380/0` |

## Usage

1. Access http://YOUR_UNRAID_IP:8000
2. Go to "Analysis and Clustering"
3. Start an analysis of your library
4. Wait for completion
5. Explore features: Clustering, Music Map, Similar Songs...

## Optional: AI-Powered Playlist Naming

You can use Ollama (self-hosted), Gemini, Mistral, or OpenAI to generate creative playlist names.

For Ollama:
| Variable | Value |
|----------|-------|
| AI_MODEL_PROVIDER | `OLLAMA` |
| OLLAMA_SERVER_URL | `http://YOUR_OLLAMA_IP:11434/api/generate` |
| OLLAMA_MODEL_NAME | `mistral:7b` |

## Hardware Requirements

- **CPU**: 4 cores minimum, AVX2 support recommended
- **RAM**: 8 GB minimum
- **Storage**: SSD recommended

## Ports Used

| Service | Port |
|---------|------|
| AudioMuse-AI WebUI | 8000 |
| Redis | 6380 |
| PostgreSQL | 5435 |

## Troubleshooting

### Check logs
```bash
docker logs -f AudioMuse-AI
docker logs -f AudioMuse-AI-Redis
docker logs -f AudioMuse-AI-PostgreSQL
```

### Test PostgreSQL connection
```bash
docker exec -it AudioMuse-AI-PostgreSQL psql -U audiomuse -d audiomuse -c "\dt"
```

### Test Redis connection
```bash
docker exec -it AudioMuse-AI-Redis redis-cli ping
# Should return "PONG"
```

## Resources

- Documentation: https://neptunehub.github.io/AudioMuse-AI/
- GitHub: https://github.com/NeptuneHub/AudioMuse-AI
- Discussions: https://github.com/NeptuneHub/AudioMuse-AI/discussions
