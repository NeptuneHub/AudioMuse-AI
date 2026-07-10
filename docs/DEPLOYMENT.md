# Deployment strategy

From `v1.0.0`, only PostgreSQL, Redis, and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as lending page and is also available later from the menu under Administration > Setup Wizard.

## Contents

- [Quick Start Deployment on K3S WITH HELM](#quick-start-deployment-on-k3s-with-helm)
- [Quick Start Deployment on K3S](#quick-start-deployment-on-k3s)
- [Local Deployment with Docker Compose](#local-deployment-with-docker-compose)
- [Local Deployment MacOS](#local-deployment-macos)
- [Local Deployment Linux](#local-deployment-linux)
- [Local Deployment Windows](#local-deployment-windows)
- [Local Deployment with Podman Quadlets](#local-deployment-with-podman-quadlets)

## Quick Start Deployment on K3S WITH HELM

The easiest way to install AudioMuse-AI on K3S is with the [AudioMuse-AI Helm Chart repository](https://github.com/NeptuneHub/AudioMuse-AI-helm).

* **Prerequisites:**
  * A running K3S cluster
  * `kubectl` configured for your cluster
  * `helm` installed
  * A media server already installed: Navidrome, Jellyfin, Emby, or Lyrion
  * See the hardware requirements in the documentation

Use the Helm chart for the simplest, most production-ready K3S deploy.

## Quick Start Deployment on K3S

This section covers direct deployment with the `deployment/*.yaml` manifests.

* **Prerequisites:**
  * A running K3S cluster
  * `kubectl` configured for your cluster
  * A media server already installed: Navidrome, Jellyfin, Emby, or Lyrion
  * See the hardware requirements in the documentation

* **Get manifest example:**
  * `deployment/deployment.yaml`

* **Edit the manifest:**
  * Set database secrets in the matching secret object (mandatory; env-only):
    * `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
  * Ensure cluster connection values are correct (mandatory; env-only):
    * `POSTGRES_HOST`, `POSTGRES_PORT`, `REDIS_URL`
  * Optional: set the timezone with `TZ`

* **Deploy:**
  ```bash
  kubectl apply -f deployment/deployment.yaml
  ```

* **Access:**
  * Web UI: `http://<EXTERNAL-IP>:8000`

**Setup Wizard:**
   The first startup a wizard setup will show where you need to configure the Music server authentication, AudioMuse-AI authentication and other optional parameters. They will be saved directly in the database.

## Local Deployment with Docker Compose

AudioMuse-AI provides Docker Compose files example:

- `deployment/docker-compose.yaml` - all music server, cpu only.
- `deployment/docker-compose-nvidia.yaml` - all music server, GPU with fallback to CPU.

**Prerequisites:**
* Docker and Docker Compose installed
* A media server already installed: Navidrome, Jellyfin, Lyrion, or Emby
* See the [hardware requirements](../README.md#hardware-requirements)

**Steps:**
1. **Create your environment file:**
   ```bash
   cp deployment/.env.example deployment/.env
   ```
   You can find the example here: [deployment/.env.example](../deployment/.env.example)

2. **Edit `.env`:**
   * Set your timezone (optional, defaults to UTC):
     ```env
     TZ=UTC
     ```
   * Change credentials for security (mandatory):
     ```env
     POSTGRES_USER=audiomuse
     POSTGRES_PASSWORD=audiomusepassword
     ```
   * Change host ports only if the defaults are already in use on your machine (optional):
     ```env
     POSTGRES_PORT=5432
     REDIS_PORT=6379
     FRONTEND_PORT=8000
     ```
   All other values (database name, internal ports, Redis URL) are hardcoded in the compose file and do not need to be set.

3. **Start the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml up -d
   ```
   Use the matching compose file `docker-compose.yaml`or `docker-compose-nvidia.yaml`.

4. **Access the app:**
   Open `http://localhost:8000` in your browser.

5. **Setup Wizard:**
   The first startup a wizard setup will show where you need to configure the Music server authentication, AudioMuse-AI authentication and other optional parameters. They will be saved directly in the database.

6. **Stop the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml down
   ```

**Note:**
> If you use LMS, create and use the Subsonic API token instead of a password. Other Subsonic-compatible servers may require the same token-based auth.

**Remote worker tip:**
If you deploy a worker on separate hardware, copy your `.env` to that machine and update `WORKER_POSTGRES_HOST` and `WORKER_REDIS_URL` so the worker can reach the main server.

## Local Deployment MacOS

The native MacOS package is shipped as a release asset for Apple Silicon only. It bundles the entire app, embedded PostgreSQL, Redis, and the browser UI so you do not need Docker or an external database for local use.

**Prerequisites:**
* Apple Silicon Mac (M1/M2/M3/M4)
* macOS 15 or later

**Steps:**
1. Download the latest release asset for macOS from the GitHub releases page.
2. Unzip and move `AudioMuse-AI.app` to `/Applications`.
3. Clear the quarantine flag before first launch:
   ```bash
   xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app
   ```
4. Open the app from `/Applications`.

**Alternative no-terminal flow:**
* Double-click the app and dismiss the security warning.
* Go to System Settings → Privacy & Security → Open Anyway for AudioMuse-AI.
* Launch the app again.

> [!IMPORTANT]
> * The app is unsigned, so macOS will require an explicit trust step on first run.
> * The native MacOS build is Apple Silicon only.

**Data and logs:**
* Data: `~/Library/AudioMuse-AI`
* Logs: `~/Library/Logs/AudioMuse-AI/audiomuse.log`

## Local Deployment Linux

The native Linux packages are provided as `.deb` and `.rpm` release assets for x86_64 and aarch64. These packages bundle the full app, embedded PostgreSQL, Redis, and the web UI.

**Prerequisites:**
* A Linux distribution compatible with the packaged binaries
* A matching package manager (`dpkg` for Debian/Ubuntu, `rpm` for Fedora/RHEL)

**Install:**
* Debian/Ubuntu:
  ```bash
  sudo dpkg -i AudioMuse-AI-<arch>-linux.deb
  ```
* Fedora/RHEL:
  ```bash
  sudo rpm -i AudioMuse-AI-<arch>-linux.rpm
  ```
Replace `<arch>` with the release artifact for your CPU (`x86_64` or `aarch64`).

**Run:**
* Start the app as a normal user (do not run as root):
  ```bash
  audiomuse-ai start
  ```
* Open the web UI at `http://127.0.0.1:8000`.
* Enable user session autostart:
  ```bash
  systemctl --user enable --now audiomuse-ai
  ```
* Stop the app:
  ```bash
  audiomuse-ai stop
  ```

**Data and logs:**
* Data: `~/.local/share/AudioMuse-AI`
* Logs: `~/.local/state/AudioMuse-AI/logs/audiomuse.log`

> **Tested on:** Debian GNU/Linux 12 (bookworm) with glibc 2.36. RPMs are expected to work on current Fedora/RHEL systems but may not support older distributions.

> [!NOTE]
> When `systemctl` is used with the `--user` flag, the process is shut down whenever the user logs out. To keep the process alive after logging out, run `loginctl enable-linger yourusername`.

## Local Deployment Windows

The native Windows package is shipped as a release asset for x86_64 only: a portable ZIP archive (`AudioMuse-AI-amd64-windows.zip`). It bundles the full app, embedded PostgreSQL, Redis, and the web UI, so you do not need Docker or an external database for local use.

**Prerequisites:**
* Windows 10 or 11 (x86_64)

**Install (ZIP):**
1. Download the latest `AudioMuse-AI-amd64-windows.zip` from the GitHub releases page.
2. Unzip it anywhere.
3. Double-click `AudioMuse-AI.exe` (or run it from a terminal).
4. Open the web UI at `http://127.0.0.1:8000`.

**Control from a terminal:**
* Start the stack and open the browser:
  ```powershell
  AudioMuse-AI.exe start
  ```
* Print whether it is running or stopped:
  ```powershell
  AudioMuse-AI.exe status
  ```
* Stop the app:
  ```powershell
  AudioMuse-AI.exe stop
  ```

> [!IMPORTANT]
> * The app is unsigned, so Windows SmartScreen may warn on first run - choose "More info" then "Run anyway".
> * The native Windows build is x86_64 only; ARM64 Windows is not supported yet.

**Data and logs:**
* Data: `%LOCALAPPDATA%\AudioMuse-AI`
* Logs: `%LOCALAPPDATA%\AudioMuse-AI\logs\audiomuse.log`

## **Local Deployment with Podman Quadlets**

For an alternative local setup, [Podman Quadlet](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html) files are provided in the `deployment/podman-quadlets` directory.

These files are configured to automatically update AudioMuse-AI using the [latest](../README.md#docker-image-tagging-strategy) stable release and should perform an automatic rollback if the updated image fails to start.

**Prerequisites:**
*   Podman and systemd.
*   Supported music server installed
*   Respect the [hardware requirements](../README.md#hardware-requirements)

**Steps:**
1.  **Navigate to the `deployment/podman-quadlets` directory:**
    ```bash
    cd deployment/podman-quadlets
    ```
2.  **Review and Customize:**

    The `audiomuse-ai-postgres.container` and `audiomuse-redis.container` files are pre-configured with default credentials and settings suitable for local testing. <BR>
    You will need to edit environment variables within `audiomuse-ai-worker.container` and `audiomuse-ai-flask.container` files to reflect your personal credentials and environment.

    Once you've customized the unit files, you will need to copy all of them into a systemd container directory, such as `/etc/containers/systemd/user/`.<BR>

3.  **Start the Services:**
    ```bash
    systemctl --user daemon-reload
    systemctl --user start audiomuse-pod
    ```
    The first command reloads systemd (generating the systemd service files) and the second command starts all AudioMuse services (Flask app, RQ worker, Redis, PostgreSQL).

4.  **Access the Application:**
    Once the containers are up, you can access the web UI at `http://localhost:8000`.

5. **Setup Wizard:**
   The first startup a wizard setup will show where you need to configure the Music server authentication, AudioMuse-AI authentication and other optional parameters. They will be saved directly in the database.
   
6.  **Stopping the Services:**
    ```bash
    systemctl --user stop audiomuse-pod
    ```
      
