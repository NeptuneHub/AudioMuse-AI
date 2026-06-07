![GitHub license](https://img.shields.io/github/license/neptunehub/AudioMuse-AI.svg)
![Latest Tag](https://img.shields.io/github/v/tag/neptunehub/AudioMuse-AI?label=latest-tag)
![Media Server Support: Jellyfin 10.11.8, Navidrome 0.61.0, LMS v3.69.0, Lyrion 9.0.2, Emby 4.9.1.80](https://img.shields.io/badge/Media%20Server-Jellyfin%2010.11.8%2C%20Navidrome%200.61.0%2C%20LMS%20v3.69.0%2C%20Lyrion%209.0.2%2C%20Emby%204.9.1.80-blue?style=flat-square&logo=server&logoColor=white)


# **AudioMuse-AI - Where Music Takes Shape** 

<p align="center">
  <img src="screenshot/AM-AI-MAP.png?raw=true" alt="AudioMuse-AI Logo" width="480">
</p>

AudioMuse-AI is an opensource and self-hosted tool that uses sonic analysis to rediscover forgotten songs in your music library and generate groove-aware playlists that also capture the meaning behind each track, without relying on metadata or external APIs.

You can run it locally with Docker Compose or Podman, or deploy it at scale in a Kubernetes cluster (**AMD64** and **ARM64** supported). It integrates with major self-hosted music servers including [Jellyfin](https://jellyfin.org), [Navidrome](https://www.navidrome.org/), [LMS](https://github.com/epoupon/lms/tree/master), [Lyrion](https://lyrion.org/), and [Emby](https://emby.media), with more integrations planned.

> **Prefer not to self-host?** We're proud that [Elestio](https://elest.io/open-source/audiomuse-ai) picked AudioMuse-AI as a managed cloud service, happy to see the project reach more people.

<p align="center">
  <a href="https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=AudioMuse-AI">
    <img src="screenshot/atlas-cloud.png?raw=true" alt="Atlas Cloud Logo" width="180">
  </a>
</p>

> **Need a hosted LLM provider?** AudioMuse-AI supports OpenAI-compatible APIs through the existing `OPENAI` provider. [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=AudioMuse-AI) is one hosted option you can configure this way; see the [configuration parameters](docs/PARAMETERS.md#openai-compatible-hosted-providers) for details.

AudioMuse-AI lets you explore your music library in innovative ways, just **start with an initial analysis**, and you’ll unlock features like:
* **Clustering**: Automatically groups sonically similar songs, creating genre-defying playlists based on the music's actual sound.
* **Instant Playlists**: Simply tell the AI what you want to hear—like "high-tempo, low-energy music" and it will instantly generate a playlist for you.
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

More information like [ARCHITECTURE](docs/ARCHITECTURE.md), [ALGORITHM DESCRIPTION](docs/ALGORITHM.md), [DEPLOYMENT STRATEGY](docs/DEPLOYMENT.md), [FAQ](docs/FAQ.md), [GPU DEPLOYMENT](docs/GPU.md), [CONFIGURATION PARAMETERS](docs/PARAMETERS.md) [AUTHENTICATION](docs/AUTH.md) and can be found in the [docs folder](docs).

**The full list or AudioMuse-AI related repository are:** 
  > * [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI): the core application, it run Flask and Worker containers to actually run all the feature;
  > * [AudioMuse-AI Helm Chart](https://github.com/NeptuneHub/AudioMuse-AI-helm): helm chart for easy installation on Kubernetes;
  > * [AudioMuse-AI Plugin for Jellyfin](https://github.com/NeptuneHub/audiomuse-ai-plugin): Jellyfin Plugin;
  > * [AudioMuse-AI Plugin for Navidrome](https://github.com/NeptuneHub/AudioMuse-AI-NV-plugin): Navidrome Plugin;
  > * [AudioMuse-AI MusicServer](https://github.com/NeptuneHub/AudioMuse-AI-MusicServer): Open Subosnic like Music Sever with integrated sonic functionality.

And now just some **NEWS:**
> * **Version 2.1.5** introduces the Windows native version. Attached to each release you will find `AudioMuse-AI-amd64-windows.msi` and `AudioMuse-AI-amd64-windows.zip`.
> * **Version 2.1.3** introduces the Linux native version. Attached to each release you will find `.deb` and `.rpm` file.
> * **Version 2.1.2** introduces the MacOS native version. Attached to each release you will find `AudioMuse-AI-arm64.zip`.
> * **Version 2.1.0** re-exports the GTE lyrics model so it produces correct embeddings on **every CPU**. The only affected users are those who analyzed lyrics on an **older CPU without VNNI** (`avx512_vnni`/`avx_vnni`), where the previous model could produce degraded vectors, they should re-analyze the lyrics. To check if your CPU has VNNI, run on the host: `grep -oE 'avx512_vnni|avx_vnni' /proc/cpuinfo | head -1` , if it prints nothing, you have no VNNI and we suggest to re-analyze. Before re-analyzing, drop the old lyrics tables:
> ```bash
> docker compose exec -e PGPASSWORD=audiomusepassword postgres \
>   psql -U audiomuse -d audiomusedb \
>   -c "DROP TABLE IF EXISTS lyrics_embedding; DROP TABLE IF EXISTS lyrics_index_data; DROP TABLE IF EXISTS lyrics_axes_index_data;"
> ```
> * **Version 2.0.0** introduces a new faster and reliable multilangue model for lyrics search. Follow the release note to drop the old lyrics index and re-analyze the lyrics.

## Disclaimer

**Important:** Despite the similar name, this project (**AudioMuse-AI**) is an independent, community-driven effort. It has no official connection to the website audiomuse.ai.

We are **not affiliated with, endorsed by, or sponsored by** the owners of `audiomuse.ai`.

## **Table of Contents**

- [Quick Start Deployment](#quick-start-deployment)
- [Quick Start Deployment MacOS](#quick-start-deployment-macos)
- [Quick Start Deployment Linux](#quick-start-deployment-linux)
- [Quick Start Deployment Windows](#quick-start-deployment-windows)
- [Hardware Requirements](#hardware-requirements)
- [Docker Image Tagging Strategy](#docker-image-tagging-strategy)
- [How To Contribute](#how-to-contribute)
- [Star History](#star-history)

## Quick Start Deployment

Get AudioMuse-AI running in minutes with Docker Compose.

If you need more deployment example take a look at [DEPLOYMENT](docs/DEPLOYMENT.md) page.

For a full list of configuration parameter take a look at [PARAMETERS](docs/PARAMETERS.md) page.

For the architecture design of AudioMuse-AI, take a look to the [ARCHITECTURE](docs/ARCHITECTURE.md) page.

From `v1.0.0`, only PostgreSQL, Redis, and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as lending page and is also available later from the menu under Administration > Setup Wizard.

**Prerequisites:**
* Docker and Docker Compose installed
* A running media server (Jellyfin, Navidrome, Lyrion, or Emby)
* See [Hardware Requirements](#hardware-requirements)

**Steps:**

1. **Create your environment file:**
   ```bash
   cp deployment/.env.example deployment/.env
   ```

   You can customize the setup by editing `deployment/.env` before startup. As a minimum, it is suggested to change the default database user and password, but you can also override other PostgreSQL and Redis connection parameters if needed:

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
     (when authentication is enabled, log in via the Web UI first — `/apidocs/`
     is gated by the same JWT cookie as the rest of the app.)

4. **Run your first analysis:**
   - Navigate to "Analysis and Clustering" page
   - Click "Start Analysis" to scan your library
   - Wait for completion, then explore features like clustering and music map

5. **Stopping the services:**
```bash
docker compose -f deployment/docker-compose.yaml down
```
> **Important:** AudioMuse-AI is designed to work with PostgreSql v15 as in the deployment example. Different version could create error.

## Quick Start Deployment MacOS
Starting from release `v2.1.2` we introduce a MacOS native version. You will find it as `AudioMuse-AI-arm64.zip` attached to the [release](https://github.com/NeptuneHub/AudioMuse-AI/releases).

To run it you have two option:

- **Option A - Terminal:**
  - Unzip and move AudioMuse-AI.app to /Applications.
  - Run in a terminal: `xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app`
  - Double-click the app - the icon appears in your menu bar.

- **Option B - no Terminal:**
  - Move the app to /Applications, double-click, dismiss the warning.
  - System Settings → Privacy & Security → Security → "Open Anyway" next to AudioMuse-AI, authenticate.
  - Launch again.

The core step both share is removing the quarantine flag due to the fact that the app is not signed.

This version run only on Apple Silicon (ARM) processor on recent version of MacOS (tested on MacOS 15.3.1 on MacMini M4 with 16gb ram)

**Files:**
- Data (database, Redis, temp audio): `~/Library/AudioMuse-AI`
- Log: `~/Library/Logs/AudioMuse-AI/audiomuse.log`

## Quick Start Deployment Linux
Starting from release `v2.1.4` we provide a native Linux package (`.deb` and `.rpm`, x86_64 and arm64) attached to the [release](https://github.com/NeptuneHub/AudioMuse-AI/releases).

- **Install as root** (writes to `/opt` and the system app/service dirs):
  - Debian/Ubuntu: `sudo dpkg -i AudioMuse-AI-x86_64.deb`
  - Fedora/RHEL: `sudo rpm -i AudioMuse-AI-x86_64.rpm`
- **Run as your normal user** (never with `sudo`/root — it stores data in your home and will not start as root):
  - `audiomuse-ai start`, then open http://127.0.0.1:8000
  - Or auto-start on login: `systemctl --user enable --now audiomuse-ai`
  - `audiomuse-ai stop` can be used to stop

**Files** (under the launching user's home):
- Data (database, Redis, temp audio): `~/.local/share/AudioMuse-AI`
- Log: `~/.local/state/AudioMuse-AI/logs/audiomuse.log` (newest entries first — read the top)

> **Tested on:** the `.deb` has been verified on **Debian GNU/Linux 12 (bookworm)** (glibc 2.36). The `.rpm` is built from the exact same payload but has not yet been tested on a live RPM-based distribution; it is expected to work on a reasonably recent system (e.g. Fedora / RHEL 9), but older distributions such as RHEL/Rocky/Alma 8 (glibc 2.28) are too old for the bundled binaries. Feedback on RPM-based distros is welcome.

## Quick Start Deployment Windows
Starting from release `v2.1.5` we provide a native Windows package (x86_64) attached to the [release](https://github.com/NeptuneHub/AudioMuse-AI/releases): the MSI installer `AudioMuse-AI-amd64-windows.msi` and the portable archive `AudioMuse-AI-amd64-windows.zip`. Both bundle the whole stack (embedded PostgreSQL, Redis, the web UI and the workers) so you do not need Docker or an external database.

To run it you have two option:

- **Option A - MSI installer:**
  - Double-click `AudioMuse-AI-amd64-windows.msi` and follow the wizard (installs to `C:\Program Files\AudioMuse-AI\` and creates Start Menu shortcuts).
  - Launch **AudioMuse-AI** from the Start Menu, then open http://127.0.0.1:8000

- **Option B - portable zip:**
  - Unzip `AudioMuse-AI-amd64-windows.zip` anywhere.
  - Double-click `AudioMuse-AI.exe` (or run it from a terminal), then open http://127.0.0.1:8000

The app is not signed, so on first run Windows SmartScreen may show a warning, click "More info" then "Run anyway" to continue.

From a terminal you can also control the stack with `AudioMuse-AI.exe start`, `AudioMuse-AI.exe stop`, `AudioMuse-AI.exe status` and `AudioMuse-AI.exe open`.

This version run only on x86_64 (Intel/AMD) processor on Windows 10/11. ARM64 Windows is not supported yet.

**Files:**
- Data (database, Redis, temp audio): `%LOCALAPPDATA%\AudioMuse-AI`
- Log: `%LOCALAPPDATA%\AudioMuse-AI\logs\audiomuse.log` (newest entries first — read the top)

## **Hardware Requirements**
AudioMuse-AI has been tested on:
* **Intel**: HP Mini PC with Intel i5-6500, 16 GB RAM and NVMe SSD
* **ARM**: Raspberry Pi 5, 8 GB RAM and NVMe SSD / Mac Mini M4 16GB / Amphere based VM with 4core 8GB ram

**Minimum requirements:**
* CPU: 4-core Intel with AVX2 support (usually produced in 2015 or later) or ARM
* RAM: 8 GB RAM
* DISK: NVME SSD storage

For more information about the GPU deployment requirements have a look to the [GPU](docs/GPU.md) page.

> **IMPORTANT**: If you use virtualization (e.g. Proxmox), make sure to pass through the host CPU. QEMU's virtual CPU lacks AVX2 support, which will prevent AudioMuse-AI from starting.

## **Docker Image Tagging Strategy**

Our GitHub Actions workflow automatically builds and publishes Docker images with the following tags:

* **`:latest`**
  Last build from the **main** branch.
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

> Versioning is Major.Minor.Patch release. Eventually (rare) model change that could require a new analysis could happen in Major and Minor release.
> Read the [release note](https://github.com/NeptuneHub/AudioMuse-AI/releases) before any update especially for Major and Minor release.

## **How To Contribute**

Contributions, issues, and feature requests are welcome\!  

For more details on how to contribute please follow the [Contributing Guidelines](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/CONTRIBUTING.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NeptuneHub/AudioMuse-AI&type=Timeline)](https://www.star-history.com/#NeptuneHub/AudioMuse-AI&Timeline)
