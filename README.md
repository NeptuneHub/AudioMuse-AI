![GitHub license](https://img.shields.io/github/license/neptunehub/AudioMuse-AI.svg)
![Latest Tag](https://img.shields.io/github/v/tag/neptunehub/AudioMuse-AI?label=latest-tag)
![Media Server Support: Navidrome 0.62.0, Jellyfin 12.0, LMS v3.69.0, Lyrion 9.0.2, Emby 4.9.1.80, Plex 1.43.2](https://img.shields.io/badge/Media%20Server-Navidrome%200.62.0%2C%20Jellyfin%2012.0%2C%20LMS%20v3.69.0%2C%20Lyrion%209.0.2%2C%20Emby%204.9.1.80%2C%20Plex%201.43.2-blue?style=flat-square&logo=server&logoColor=white)
<a href="https://www.bestpractices.dev/projects/13329"><img src="https://www.bestpractices.dev/projects/13329/badge"></a>

<p align="center">
<strong>⭐ Leave a star on this project:</strong> One shines alone; together, they make it visible and keep it alive.
</p>

<p align="center">
💛 <a href="https://liberapay.com/NeptuneHub/donate">Donate</a> to shape AudioMuse-AI future by supporting AI licenses, homelab infrastructure, and continuous development.
</p>

# **AudioMuse-AI - Where Music Takes Shape** 

<p align="center">
  <img src="screenshot/AM-AI-MAP.png?raw=true" alt="AudioMuse-AI Logo" width="480">
</p>

AudioMuse-AI is an opensource and self-hosted tool that uses sonic analysis to rediscover forgotten songs in your music library and generate groove-aware playlists that also capture the meaning behind each track, without relying on metadata or external APIs.

You can run it locally with Docker Compose or Podman, deploy it at scale in a Kubernetes cluster (**AMD64** and **ARM64** supported), or use native applications available for **macOS, Windows, and Linux**. It integrates with major self-hosted music servers including [Navidrome](https://www.navidrome.org/), [Jellyfin](https://jellyfin.org), [LMS](https://github.com/epoupon/lms/tree/master), [Lyrion](https://lyrion.org/), [Emby](https://emby.media), and [Plex](https://www.plex.tv/), with more integrations planned.

> **Prefer not to self-host?** [Elestio](https://elest.io/open-source/audiomuse-ai) offers AudioMuse-AI as a managed cloud service, and their [YouTube video](https://www.youtube.com/watch?v=Ow89q6gQ1mM) is a good introduction to the project and its features.

AudioMuse-AI lets you explore your music library in innovative ways, just **start with an initial analysis**, and you’ll unlock features like:
* **Multiple Music Servers** (from `v3.0.0`): connect several media servers - any mix of Navidrome, Jellyfin, LMS, Lyrion, Emby and Plex - to a **single AudioMuse-AI deployment**. Built-in duplicate detection recognizes the same song across servers, so each track is **analyzed only once** and every server shares the result.
* **Clustering**: Automatically groups sonically similar songs, creating genre-defying playlists based on the music's actual sound.
* **Instant Playlists**: Simply tell the AI what you want to hear-like "high-tempo, low-energy music" and it will instantly generate a playlist for you.
* **Music Map**: Discover your music collection visually with a vibrant, genre-based 2D map.
* **Playlist from Similar Songs**: Pick a track you love, and AudioMuse-AI will find all the songs in your library that share its sonic signature, creating a new discovery playlist.
* **Song Paths**: Create a seamless listening journey between two songs. AudioMuse-AI finds the perfect tracks to bridge the sonic gap.
* **Sonic Fingerprint**: Generates playlists based on your listening habits, finding tracks similar to what you've been playing most often.
* **Song Alchemy**: Mix your ideal vibe, mark tracks as "ADD" or "SUBTRACT" to get a curated playlist and a 2D preview. Export the final selection directly to your media server.
* **Text Search**: search your song with simple text that can contains mood, instruments and genre like calm piano songs.
* **Lyrics Search**: search your library by theme, story or meaning, like love songs, not just the sound.

> **Lyrics language support:** the Lyrics Search feature works only with the **72 languages** listed below.
>
> <details>
> <summary>Show the 72 supported languages</summary>
>
> Afrikaans, Albanian, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali, Bulgarian, Burmese, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Lao, Latvian, Lithuanian, Macedonian, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Sinhala, Slovak, Slovenian, Somali, Spanish, Swahili, Swedish, Tagalog, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, Yoruba.
>
> </details>

More information can be found in the [docs folder](docs): [ARCHITECTURE](docs/ARCHITECTURE.md), [ALGORITHM DESCRIPTION](docs/ALGORITHM.md), [MULTIPLE MUSIC SERVERS](docs/MULTI_SERVER.md), [DEPLOYMENT STRATEGY](docs/DEPLOYMENT.md), [NAVIDROME SETUP](docs/NAVIDROME.md), [GPU DEPLOYMENT](docs/GPU.md), [CONFIGURATION PARAMETERS](docs/PARAMETERS.md), [AUTHENTICATION](docs/AUTH.md), [PLUGINS](docs/PLUGIN.md), [ERROR CODES](docs/ERROR_CODES.md) and [FAQ](docs/FAQ.md).

**The full list of AudioMuse-AI related repository are:** 
  > * [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI): the core application, it run Flask and Worker containers to actually run all the feature;
  > * [AudioMuse-AI Helm Chart](https://github.com/NeptuneHub/AudioMuse-AI-helm): helm chart for easy installation on Kubernetes;
  > * [AudioMuse-AI Plugin for Navidrome](https://github.com/NeptuneHub/AudioMuse-AI-NV-plugin): Navidrome Plugin;
  > * [AudioMuse-AI Plugin for Jellyfin](https://github.com/NeptuneHub/audiomuse-ai-plugin): Jellyfin Plugin;
  > * [lyrion-audiomuseai-plugin](https://github.com/JameZUK/lyrion-audiomuseai-plugin): Unofficial Lyrion Plugin by [JameZUK](https://github.com/JameZUK);
  > * [AudioMuse-AI MusicServer](https://github.com/NeptuneHub/AudioMuse-AI-MusicServer): Open Subosnic like Music Sever with integrated sonic functionality.

And now just some **NEWS:**
> * **ghcr.io/neptunehub/audiomuse-ai:devel-nvidia-arm** is the new **experimental** image that support GPU on ARM, created for **DGX Sparl**.
> * **Version 3.2.0** implemented queue on postgresql, this means that Redis is not needed anymore. Just check the new deployment/docker-compose example.
> * **Version 3.0.0** added multiple music server support on a single deployment, with duplicate detection so a song shared by more servers is analyzed only once.
> * **Version 2.6.0** added support for third party plugin. Give a look to the [plugin documentation](docs/PLUGIN.md) to know how to develop one and to the [official 3rd party catalog](https://github.com/NeptuneHub/AudioMuse-AI-plugins). The plugin system requires a persistent volume mounted on both the Flask and worker containers, otherwise installed plugins are lost whenever the containers restart; the deployment example has been updated accordingly.
> * **Version 2.5.0** added Plex Music Server support.

## Disclaimer

> [!IMPORTANT]
> Despite the similar name, this project (**AudioMuse-AI**) is an independent, community-driven effort. It has no official connection to the website audiomuse.ai.

We are **not affiliated with, endorsed by, or sponsored by** the owners of `audiomuse.ai`.

## **Table of Contents**

- [Quick Start Deployment (Containerized)](#quick-start-deployment-containerized)
- [Native Deployment](#native-deployment)
- [Hardware Requirements](#hardware-requirements)
- [Docker Image Tagging Strategy](#docker-image-tagging-strategy)
- [How To Contribute](#how-to-contribute)
- [Code Mirror](#code-mirror)
- [Star History](#star-history)

## Quick Start Deployment (Containerized)

Get AudioMuse-AI running in minutes with Docker Compose. For more deployment examples see the [DEPLOYMENT](docs/DEPLOYMENT.md) page.

From `v1.0.0`, only PostgreSQL and `TZ` are configured via environment variables. Everything else is managed through the browser Setup Wizard and persisted in the database (legacy environment variables are imported automatically on first startup). The Setup Wizard is the landing page of a clean installation and stays available under Administration > Setup Wizard.

**Prerequisites:**
* Docker and Docker Compose installed
* A running media server (Navidrome, Jellyfin, Lyrion, Emby, or Plex)
* See [Hardware Requirements](#hardware-requirements)

**Steps:**

1. **Create your environment file:**
   ```bash
   cp deployment/.env.example deployment/.env
   ```

   You can customize the setup by editing `deployment/.env` before startup. As a minimum, it is suggested to change the default database user and password, but you can also override other PostgreSQL connection parameters if needed:

   ```env
   POSTGRES_PASSWORD=your-secure-password
   ```

2. **Start the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml up -d
   ```

3. **Access the application:**
   - Web UI: `http://localhost:8000`
   - Interactive API documentation (Swagger UI): `http://localhost:8000/apidocs/`
     (when authentication is enabled, log in via the Web UI first - `/apidocs/`
     is gated by the same JWT cookie as the rest of the app.)

4. **Run your first analysis:**
   - Navigate to "Analysis and Clustering" page
   - Click "Start Analysis" to scan your library
   - Wait for completion, then explore features like clustering and music map

5. **Stopping the services:**
```bash
docker compose -f deployment/docker-compose.yaml down
```
> [!IMPORTANT]
> AudioMuse-AI is designed to work with PostgreSQL v15 as in the deployment example. Different versions could cause errors.

## Native Deployment

Prefer not to use Docker? We ship native packages for **macOS, Linux and Windows**, attached to each [release](https://github.com/NeptuneHub/AudioMuse-AI/releases). Each bundles the whole stack (embedded PostgreSQL, web UI and workers), so you don't need Docker or an external database. Once started, open **http://127.0.0.1:8000**.

> The apps are not signed, so your OS may warn you on first launch, see the per-platform notes below for how to allow them.

<details>
<summary><b>macOS</b> - Apple Silicon, <code>AudioMuse-AI-arm64.zip</code> (from <code>v2.1.2</code>)</summary>

- Unzip and move `AudioMuse-AI.app` to `/Applications`.
- Remove the quarantine flag (the app is unsigned), either way:
  - **Terminal:** `xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app`, then double-click - the icon appears in your menu bar.
  - **No Terminal:** double-click and dismiss the warning, then System Settings → Privacy & Security → "Open Anyway", authenticate, and launch again.
- Runs only on Apple Silicon (ARM) on recent macOS (tested on macOS 15.3.1, Mac Mini M4 / 16 GB).

**Files:** data (database, temp audio) in `~/Library/AudioMuse-AI`, log at `~/Library/Logs/AudioMuse-AI/audiomuse.log`
</details>

<details>
<summary><b>Linux</b> - x86_64 / arm64, <code>.deb</code> or <code>.rpm</code> (from <code>v2.1.3</code>)</summary>
  
- **Install as root** (writes to `/opt` and the system app/service dirs):
  - Debian/Ubuntu: `sudo dpkg -i AudioMuse-AI-<arch>-linux.deb` (where `<arch>` is `x86_64` or `aarch64`)
  - Fedora/RHEL: `sudo rpm -i AudioMuse-AI-<arch>-linux.rpm` (where `<arch>` is `x86_64` or `aarch64`)
- **Run as your normal user** (never with `sudo`/root - it stores data in your home and won't start as root):
  - `audiomuse-ai start` (stop with `audiomuse-ai stop`), or auto-start on login with `systemctl --user enable --now audiomuse-ai`.
- Verified on **Debian 12 (bookworm)** (glibc 2.36). The `.rpm` is the same payload, expected to work on recent Fedora / RHEL 9, but too old for RHEL/Rocky/Alma 8 (glibc 2.28). Feedback on RPM-based distros is welcome.

**Files** (under the launching user's home): data (database, temp audio) in `~/.local/share/AudioMuse-AI`, log at `~/.local/state/AudioMuse-AI/logs/audiomuse.log` (newest entries first)
</details>

<details>
<summary><b>Windows</b> - x86_64, <code>AudioMuse-AI-amd64-windows.zip</code> (from <code>v2.1.4</code>)</summary>

- Unzip the portable archive anywhere.
- From a terminal you can start with `AudioMuse-AI.exe start` and stop with `AudioMuse-AI.exe stop`.
- Runs only on x86_64 (Intel/AMD) on Windows 10/11.

**Files:** data (database, temp audio) in `%LOCALAPPDATA%\AudioMuse-AI`, log at `%LOCALAPPDATA%\AudioMuse-AI\logs\audiomuse.log` (newest entries first)
</details>

> [!IMPORTANT]
> Before updating a native version, first stop any running instance.

## **Hardware Requirements**
AudioMuse-AI has been tested on:
* **Intel**: HP Mini PC with Intel i5-6500, 16 GB RAM and NVMe SSD
* **ARM**: Raspberry Pi 5, 8 GB RAM and NVMe SSD / Mac Mini M4 16GB / Amphere based VM with 4core 8GB ram

**Minimum requirements:**
* CPU: 4-core Intel with AVX2 support (usually produced in 2015 or later) or ARM
* RAM: 8 GB RAM
* DISK: NVME SSD storage

For more information about the GPU deployment requirements have a look to the [GPU](docs/GPU.md) page.

> [!IMPORTANT]
> If you use virtualization (e.g. Proxmox), make sure to pass through the host CPU. QEMU's virtual CPU lacks AVX2 support, which will prevent AudioMuse-AI from starting.

## **Docker Image Tagging Strategy**

Our GitHub Actions workflow automatically builds and publishes Docker images with the following tags:

* **`:latest`**
  Last released image.
  **Use it for automatic update.**

* **`:X.Y.Z`** (e.g. `:1.0.0`, `:0.1.4-alpha`)
  Immutable images built from **Git release tags**.
  **Recommended for most users. Pinned deployments: you decide when to update by changing the version manually.**

* **`:devel`**
  Build from main on each commit/pr merged. It's a less stable build.
  **Recommended only for testing and early adopters.**

* **`:pr-<NUMBER>`** (e.g. `:pr-661`)
  Build generated for a specific open **pull request** (non-draft), to preview its changes before they are merged.
  **For reviewing and testing that single PR**

* **`-noavx2`** variants
  Experimental images for CPUs **without AVX2 support**, using legacy dependencies.
  **Not recommended** unless required for compatibility.

* **`-nvidia`** variants
  Images that support the use of GPU for both Analysis and Clustering.
  **Not recommended** for old GPU.
  
* **`-devel-nvidia-arm`** variants
  Images that support the use of GPU on ARM processor for both Analysis and Clustering **EXPERIMENTAL FOR DGX SPARK**.

> Versioning is Major.Minor.Patch release. Eventually (rare) model change that could require a new analysis could happen in Major and Minor release.
> Read the [release note](https://github.com/NeptuneHub/AudioMuse-AI/releases) before any update especially for Major and Minor release.

## **How To Contribute**

Contributions, issues, and feature requests are welcome\!  

For more details on how to contribute please follow the [Contributing Guidelines](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/CONTRIBUTING.md)

## **Code Mirror**

[AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) repository code is mirrored here:
- https://codeberg.org/NeptuneHub/AudioMuse-AI

DO **NOT** USE MIRROR TO RAISE ISSUE, PR OTHER ACTION DIFFERENT FROM GET THE CODE
