# Deployment strategy

From `v1.0.0`, only PostgreSQL, Redis, and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as lending page and is also available later from the menu under Administration > Setup Wizard.


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
   The first startup a wizard setup will show where you add to configure the Music server authentication, AudioMsue-AI atuhentication and other optional paramter. They will be saved directly in the database.

## Local Deployment with Docker Compose

AudioMuse-AI provides Docker Compose files example:

- `deployment/docker-compose.yaml` - all music server, cpu only.
- `deployment/docker-compose-nvidia.yaml` - all music server, GPU with fallback to CPU.

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

2. **Edit `.env` with your environment values for **Database and Redis (mandatory; env-only):**
   ```env
   REDIS_URL=redis://localhost:6379/0
   POSTGRES_USER=audiomuse
   POSTGRES_PASSWORD=audiomusepassword
   POSTGRES_HOST=postgres
   POSTGRES_PORT=5432
   POSTGRES_DB=audiomusedb
   TZ=UTC
   ```

3. **Start the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml up -d
   ```
   Use the matching compose file `docker-compose.yaml`or `docker-compose-nvidia.yaml`.

4. **Access the app:**
   Open `http://localhost:8000` in your browser.

5. **Setup Wizard:**
   The first startup a wizard setup will show where you add to configure the Music server authentication, AudioMsue-AI atuhentication and other optional paramter. They will be saved directly in the database.

6. **Stop the services:**
   ```bash
   docker compose -f deployment/docker-compose.yaml down
   ```

**Note:**
> If you use LMS, create and use the Subsonic API token instead of a password. Other Subsonic-compatible servers may require the same token-based auth.

**Remote worker tip:**
If you deploy a worker on separate hardware, copy your `.env` to that machine and update `WORKER_POSTGRES_HOST` and `WORKER_REDIS_URL` so the worker can reach the main server.

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
   The first startup a wizard setup will show where you add to configure the Music server authentication, AudioMsue-AI atuhentication and other optional paramter. They will be saved directly in the database.
   
6.  **Stopping the Services:**
    ```bash
    systemctl --user stop audiomuse-pod
    ```
      