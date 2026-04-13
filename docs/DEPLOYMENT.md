# Deployment strategy

From `v1.0.0` the app includes a browser Setup wizard. If the app starts without the required media server or auth values, it will show a simple setup page so you can finish configuration from the UI. Env vars still work as the initial quick-start values, and once setup is complete those settings are saved in the database and can be edited later from the Setup menu.

> **IMPORTANT:** `DATABASE_URL` / `POSTGRES_*` and `REDIS_URL` must remain environment variables.
>
> **IMPORTANT:** After the first startup, setup values are loaded from the database and managed through the Setup wizard. Updating those values in `.env` later will not change the running configuration, except for database and Redis connection settings.


## Quick Start Deployment on K3S WITH HELM

The easiest way to install AudioMuse-AI on K3S is with the [AudioMuse-AI Helm Chart repository](https://github.com/NeptuneHub/AudioMuse-AI-helm).

* **Prerequisites:**
  * A running K3S cluster
  * `kubectl` configured for your cluster
  * `helm` installed
  * A media server already installed: Jellyfin, Emby, Navidrome, or Lyrion
  * See the hardware requirements in the documentation

Use the Helm chart for the simplest, most production-ready K3S deploy.

## Quick Start Deployment on K3S

This section covers direct deployment with the `deployment/*.yaml` manifests.

* **Prerequisites:**
  * A running K3S cluster
  * `kubectl` configured for your cluster
  * A media server already installed: Jellyfin, Emby, Navidrome, or Lyrion
  * See the hardware requirements in the documentation

* **Choose the right manifest:**
  * `deployment/deployment.yaml` — Jellyfin
  * `deployment/deployment-emby.yaml` — Emby
  * `deployment/deployment-navidrome.yaml` — Navidrome
  * `deployment/deployment-lyrion.yaml` — Lyrion

* **Edit the manifest:**
  * Set your media server values (optional; can also be entered later via the UI wizard):
    * Jellyfin: `JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN`
    * Navidrome: `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD`
    * Lyrion: `LYRION_URL`
  * Set database secrets in the matching secret object (mandatory; env-only):
    * `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
  * Ensure cluster connection values are correct (mandatory; env-only):
    * `POSTGRES_HOST`, `POSTGRES_PORT`, `REDIS_URL`
  * Optional AI keys (optional; can also be entered later via the UI wizard):
    * `GEMINI_API_KEY`, `MISTRAL_API_KEY`

* **Deploy:**
  ```bash
  kubectl apply -f deployment/deployment.yaml
  ```

* **Access:**
  * Web UI: `http://<EXTERNAL-IP>:8000`
  * Swagger: `http://<EXTERNAL-IP>:8000/apidocs`

## Local Deployment with Docker Compose

AudioMuse-AI provides Docker Compose files for different media server backends:

- **Jellyfin**: `deployment/docker-compose.yaml`
- **Navidrome**: `deployment/docker-compose-navidrome.yaml`
- **Lyrion**: `deployment/docker-compose-lyrion.yaml`
- **Emby**: `deployment/docker-compose-emby.yaml`

If you want to use the UI wizard instead of editing env vars first, you can skip steps 1–3 and configure the app in the browser after startup.

**Prerequisites:**
* Docker and Docker Compose installed
* A media server already installed: Jellyfin, Navidrome, Lyrion, or Emby
* See the [hardware requirements](../README.md#hardware-requirements)

**Steps:**
1. **Create your environment file:**
   ```bash
   cp deployment/.env.example deployment/.env
   ```
   You can find the example here: [deployment/.env.example](../deployment/.env.example)

2. **Edit `.env` with your environment values (optional; media server/auth can also be set later via the UI wizard):**
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

   **Database and Redis (mandatory; env-only):**
   ```env
   REDIS_URL=redis://localhost:6379/0
   POSTGRES_USER=audiomuse
   POSTGRES_PASSWORD=audiomusepassword
   POSTGRES_HOST=postgres
   POSTGRES_PORT=5432
   POSTGRES_DB=audiomusedb
   ```

   **Optional AI keys (optional; can also be set later via the UI wizard):**
   ```env
   GEMINI_API_KEY=your-gemini-key
   MISTRAL_API_KEY=your-mistral-key
   ```

3. **Add auth values if you want to preconfigure login: (optional, can be done in the UI wizard)**
   ```env
   AUTH_ENABLED=true
   AUDIOMUSE_USER=alice
   AUDIOMUSE_PASSWORD=secret123
   API_TOKEN=api-token
   ```
   We recommend leaving `AUTH_ENABLED=true` enabled for secure local use.

4. **Start the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml up -d
   ```
   Use the matching compose file for your media server: `docker-compose.yaml` for Jellyfin, `docker-compose-navidrome.yaml`, `docker-compose-lyrion.yaml`, or `docker-compose-emby.yaml`.

5. **Access the app:**
   Open `http://localhost:8000` in your browser.

6. **Stop the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml down
   ```

> `DATABASE_URL` / `POSTGRES_*` and `REDIS_URL` are still managed as environment settings and MUST be configured in the compose environment.

**Note:**
> If you use LMS, create and use the Subsonic API token instead of a password. Other Subsonic-compatible servers may require the same token-based auth.

**Remote worker tip:**
If you deploy a worker on separate hardware, copy your `.env` to that machine and update `WORKER_POSTGRES_HOST` and `WORKER_REDIS_URL` so the worker can reach the main server.

## **Local Deployment with Podman Quadlets**

For an alternative local setup, [Podman Quadlet](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html) files are provided in the `deployment/podman-quadlets` directory for interacting with **Navidrome**. The unit files can  be edited for use with **Jellyfin**. 

These files are configured to automatically update AudioMuse-AI using the [latest](../README.md#docker-image-tagging-strategy) stable release and should perform an automatic rollback if the updated image fails to start.

**Prerequisites:**
*   Podman and systemd.
*   `Jellyfin` or `Navidrome` installed.
*   Respect the [hardware requirements](../README.md#hardware-requirements)

**Steps:**
1.  **Navigate to the `deployment/podman-quadlets` directory:**
    ```bash
    cd deployment/podman-quadlets
    ```
2.  **Review and Customize:**

    The `audiomuse-ai-postgres.container` and `audiomuse-redis.container` files are pre-configured with default credentials and settings suitable for local testing. <BR>
    You will need to edit environment variables within `audiomuse-ai-worker.container` and `audiomuse-ai-flask.container` files to reflect your personal credentials and environment.
    * For **Navidrome**, update `NAVIDROME_URL`, `NAVIDROME_USER` and `NAVIDROME_PASSWORD` with your real credentials.  
    * For **Jellyfin** replace these variables with `JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN`; add your real credentials; and change the `MEDIASERVER_TYPE` to `jellyfin`. 

    Once you've customized the unit files, you will need to copy all of them into a systemd container directory, such as `/etc/containers/systemd/user/`.<BR>

3.  **Start the Services:**
    ```bash
    systemctl --user daemon-reload
    systemctl --user start audiomuse-pod
    ```
    The first command reloads systemd (generating the systemd service files) and the second command starts all AudioMuse services (Flask app, RQ worker, Redis, PostgreSQL).
4.  **Access the Application:**
    Once the containers are up, you can access the web UI at `http://localhost:8000`.
5.  **Stopping the Services:**
    ```bash
    systemctl --user stop audiomuse-pod
    ```
      