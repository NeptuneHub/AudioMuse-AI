![GitHub license](https://img.shields.io/github/license/neptunehub/AudioMuse-AI.svg)
![Latest Tag](https://img.shields.io/github/v/tag/neptunehub/AudioMuse-AI?label=latest-tag)
![Media Server Support: Jellyfin 10.11.3, Navidrome 0.61.0, LMS v3.69.0, Lyrion 9.0.2, Emby 4.9.1.80](https://img.shields.io/badge/Media%20Server-Jellyfin%2010.11.3%2C%20Navidrome%200.61.0%2C%20LMS%20v3.69.0%2C%20Lyrion%209.0.2%2C%20Emby%204.9.1.80-blue?style=flat-square&logo=server&logoColor=white)

<p align="center">
  <a href="https://docs.google.com/forms/d/e/1FAIpQLSfl9Qd5UW-0sI-pcUkFtHWFwwphhErw_4Btao34CPL8TQ93rQ/viewform">
    Please vote for the AudioMuse-AI best functionality
  </a>
</p>

# **AudioMuse-AI - Where Music Takes Shape** 

<p align="center">
  <img src="screenshot/AM-AI-MAP.png?raw=true" alt="AudioMuse-AI Logo" width="480">
</p>


AudioMuse-AI is an open-source, Dockerized environment that brings **automatic playlist generation** to your self-hosted music library. Using tools such as [Librosa](https://github.com/librosa/librosa) and [ONNX](https://onnx.ai/), it performs **sonic analysis** on your audio files locally, allowing you to curate playlists for any mood or occasion without relying on external APIs.

Deploy it easily on your local machine with Docker Compose or Podman, or scale it in a Kubernetes cluster (supports **AMD64** and **ARM64**). It integrates with the main music servers' APIs such as [Jellyfin](https://jellyfin.org), [Navidrome](https://www.navidrome.org/), [LMS](https://github.com/epoupon/lms/tree/master), [Lyrion](https://lyrion.org/), and [Emby](https://emby.media). More integrations may be added in the future.

AudioMuse-AI lets you explore your music library in innovative ways, just **start with an initial analysis**, and you’ll unlock features like:
* **Clustering**: Automatically groups sonically similar songs, creating genre-defying playlists based on the music's actual sound.
* **Instant Playlists**: Simply tell the AI what you want to hear—like "high-tempo, low-energy music" and it will instantly generate a playlist for you.
* **Music Map**: Discover your music collection visually with a vibrant, genre-based 2D map.
* **Playlist from Similar Songs**: Pick a track you love, and AudioMuse-AI will find all the songs in your library that share its sonic signature, creating a new discovery playlist.
* **Song Paths**: Create a seamless listening journey between two songs. AudioMuse-AI finds the perfect tracks to bridge the sonic gap.
* **Sonic Fingerprint**: Generates playlists based on your listening habits, finding tracks similar to what you've been playing most often.
* **Song Alchemy**: Mix your ideal vibe, mark tracks as "ADD" or "SUBTRACT" to get a curated playlist and a 2D preview. Export the final selection directly to your media server.
* **Text Search**: search your song with simple text that can contains mood, instruments and genre like calm piano songs.

More information like [ARCHITECTURE](docs/ARCHITECTURE.md), [ALGORITHM DESCRIPTION](docs/ALGORITHM.md), [DEPLOYMENT STRATEGY](docs/DEPLOYMENT.md), [FAQ](docs/FAQ.md), [GPU DEPLOYMENT](docs/GPU.md), [CONFIGURATION PARAMETERS](docs/PARAMETERS.md) [AUTHENTICATION](docs/AUTH.md) and can be found in the [docs folder](docs).

**The full list or AudioMuse-AI related repository are:** 
  > * [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI): the core application, it run Flask and Worker containers to actually run all the feature;
  > * [AudioMuse-AI Helm Chart](https://github.com/NeptuneHub/AudioMuse-AI-helm): helm chart for easy installation on Kubernetes;
  > * [AudioMuse-AI Plugin for Jellyfin](https://github.com/NeptuneHub/audiomuse-ai-plugin): Jellyfin Plugin;
  > * [AudioMuse-AI Plugin for Navidrome](https://github.com/NeptuneHub/AudioMuse-AI-NV-plugin): Navidrome Plugin;
  > * [AudioMuse-AI MusicServer](https://github.com/NeptuneHub/AudioMuse-AI-MusicServer): Open Subosnic like Music Sever with integrated sonic functionality.

And now just some **NEWS:**
> * **Version 0.9.0** introduce the possibility to create a Song Path with mood. Also Song Alchemy can save is output as an Anchor to be used for future alchemy or as an input of Song Path.
> * **Version 0.9.0** introduce the new DCLAP model, for a faster analysis. **IMPORTANT**: it need to start from a clean db and a new analysis. Learn more on the release note.
> * **Version 0.8.13** introduce the authentication layer. It start initially disabled by default. Learn more on the release note.

## Disclaimer

**Important:** Despite the similar name, this project (**AudioMuse-AI**) is an independent, community-driven effort. It has no official connection to the website audiomuse.ai.

We are **not affiliated with, endorsed by, or sponsored by** the owners of `audiomuse.ai`.

## **Table of Contents**

- [Quick Start Deployment](#quick-start-deployment)
- [Hardware Requirements](#hardware-requirements)
- [Docker Image Tagging Strategy](#docker-image-tagging-strategy)
- [How To Contribute](#how-to-contribute)
- [Star History](#star-history)

## Quick Start Deployment

Get AudioMuse-AI running in minutes with Docker Compose. 

If you need more deployment example take a look at [DEPLOYMENT](docs/DEPLOYMENT.md) page.

For a full list of configuration parameter take a look at [PARAMETERS](docs/PARAMETERS.md) page.

For the architecture design of AudioMuse-AI, take a look to the [ARCHITECTURE](docs/ARCHITECTURE.md) page.

**Prerequisites:**
* Docker and Docker Compose installed
* A running media server (Jellyfin, Navidrome, Lyrion, or Emby)
* See [Hardware Requirements](#hardware-requirements)

**Steps:**

1. **Create your environment file:**
   ```bash
   cp deployment/.env.example deployment/.env
   ```

2. **Edit `.env` with your media server credentials:**
   
   **For Jellyfin:**
   ```env
   MEDIASERVER_TYPE=jellyfin
   JELLYFIN_URL=http://your-jellyfin-server:8096
   JELLYFIN_USER_ID=your-user-id
   JELLYFIN_TOKEN=your-api-token
   ```
   
   **For Navidrome:**
   ```env
   MEDIASERVER_TYPE=navidrome
   NAVIDROME_URL=http://your-navidrome-server:4533
   NAVIDROME_USER=your-username
   NAVIDROME_PASSWORD=your-password
   ```
   
   **For Lyrion:**
   ```env
   MEDIASERVER_TYPE=lyrion
   LYRION_URL=http://your-lyrion-server:9000
   ```

   **For Emby:**
   ```env
   MEDIASERVER_TYPE=emby
   EMBY_URL=http://your-emby-server:8096
   EMBY_USER_ID=your-user-id
   EMBY_TOKEN=your-api-token
   ```

3. **Start the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml up -d
   ```
> Remember to get the correct version for your Music Server.

> docker-compose.yaml is for Jellyfin.
> You also have docker-compose-navidrome.yaml, docker-compose-lyrion.yaml, dokcer-compose-emby.yaml

> Other example are for advanced deployment.
4. **Access the application:**
   Open your browser at `http://localhost:8000`

5. **Run your first analysis:**
   - Navigate to "Analysis and Clustering" page
   - Click "Start Analysis" to scan your library
   - Wait for completion, then explore features like clustering and music map

**Stopping the services:**
```bash
docker compose -f deployment/docker-compose.yaml down
```

> NOTE: by default AudioMuse-AI is deployed WITHOUT authentication layer and its suited only for LOCAL deployment. If you want to configure it have a look to the  [AUTHENTICATION](docs/AUTH.md) docs. If you enable the Authentication Layer, you need to be sure that any plugin used support and use the AudioMuse-AI API TOKEN

## **Hardware Requirements**

AudioMuse-AI has been tested on:
* **Intel**: HP Mini PC with Intel i5-6500, 16 GB RAM and NVMe SSD
* **ARM**: Raspberry Pi 5, 8 GB RAM and NVMe SSD / Mac Mini M4 16GB / Amphere based VM with 4core 8GB ram

**Minimum requirements:**
* CPU: 4-core Intel with AVX2 support (usually produced in 2015 or later) or ARM
* RAM: 8 GB RAM
* DISK: NVME SSD storage

For more information about the GPU deployment requirements have a look to the [GPU](docs/GPU.md) page.

## **Docker Image Tagging Strategy**

Our GitHub Actions workflow automatically builds and publishes Docker images with the following tags:

* **`:latest`**
  Stable build from the **main** branch.
  **Recommended for most users.**

* **`:devel`**
  Development build from the **devel** branch.
  May be unstable — **for testing and development only.**

* **`:X.Y.Z`** (e.g. `:1.0.0`, `:0.1.4-alpha`)
  Immutable images built from **Git release tags**.
  **Ideal for reproducible or pinned deployments.**

* **`-noavx2`** variants
  Experimental images for CPUs **without AVX2 support**, using legacy dependencies.
  **Not recommended** unless required for compatibility.

* **`-nvidia`** variants
  Images that support the use of GPU for both Analysis and Clustering.
  **Not recommended** for old GPU.

## **How To Contribute**

Contributions, issues, and feature requests are welcome\!  

For more details on how to contribute please follow the [Contributing Guidelines](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/CONTRIBUTING.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NeptuneHub/AudioMuse-AI&type=Timeline)](https://www.star-history.com/#NeptuneHub/AudioMuse-AI&Timeline)
