# 🎵 AudioMuse-AI — Self-Hosted AI Music Playlist Generator

> An intelligent, **self-hosted music playlist generator** that uses AI to create personalized playlists based on mood, genre, and listening history — compatible with Jellyfin, Navidrome, Emby & more.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask)
![Redis](https://img.shields.io/badge/Redis-Queue-DC382D?style=flat-square&logo=redis)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-DB-336791?style=flat-square&logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Media Server Support:**
![Jellyfin](https://img.shields.io/badge/Jellyfin-10.11.8-00A4DC?style=flat-square)
![Navidrome](https://img.shields.io/badge/Navidrome-0.61.0-FF9D00?style=flat-square)
![Emby](https://img.shields.io/badge/Emby-4.9.1-52B54B?style=flat-square)

---

## 🎯 What It Does

AudioMuse-AI brings **AI-powered music intelligence** to your self-hosted media server:

- 🎭 **Mood-based playlists** — generate playlists by emotion (happy, focus, chill, workout)
- 🔍 **Semantic music search** — find songs by vibe, not just title/artist
- 🎨 **Artist similarity** — discover similar artists using ML embeddings
- 📊 **Listening analytics** — dashboard with your music taste patterns
- 🔄 **Auto-clustering** — groups your library by sonic characteristics
- 🎼 **Waveform analysis** — visual audio fingerprinting

---

## 🏗️ Architecture

```
Media Server (Jellyfin/Navidrome/Emby)
          │
          ▼
    AudioMuse-AI API (Flask)
          │
    ┌─────┴──────┐
    │            │
  Redis        PostgreSQL
  (Job Queue)  (Music DB)
    │
    ▼
 RQ Workers
    │
  ┌─┴──────────────────┐
  │  AI Analysis        │
  │  ├── CLAP Embeddings│
  │  ├── Mood Detection │
  │  ├── Clustering     │
  │  └── Similarity     │
  └────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask + Python |
| Job Queue | Redis + RQ workers |
| Database | PostgreSQL |
| AI Models | CLAP (audio embeddings), MULAN |
| Audio Analysis | Librosa, SonicFingerprint |
| Containerization | Docker + Docker Compose |
| API Docs | Swagger/Flasgger |

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/RahulRachhoya/AudioMuse-AI
cd AudioMuse-AI

# 2. Configure
cp .env.example .env
# Edit .env with your media server credentials

# 3. Run with Docker
docker compose up -d

# 4. Access UI
open http://localhost:5000
```

---

## 📁 Key Files

```
AudioMuse-AI/
├── app.py                    # Main Flask app
├── app_chat.py               # AI chat interface
├── app_clustering.py         # ML clustering engine
├── app_artist_similarity.py  # Artist similarity model
├── app_clap_search.py        # Semantic audio search
├── app_dashboard.py          # Analytics dashboard
├── config.py                 # Configuration
├── rq_worker.py              # Background job workers
├── Dockerfile                # Container definition
└── deployment/               # K8s / compose configs
```

---

<div align="center">
Built with ❤️ | <a href="https://rahulrachhoya.is-a.dev">rahulrachhoya.is-a.dev</a>
</div>
