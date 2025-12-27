# Deploymnet strategy
## **Quick Start Deployment on K3S WITH HELM**

The best way to install AudioMuse-AI on K3S (kubernetes) is by [AudioMuse-AI Helm Chart repository](https://github.com/NeptuneHub/AudioMuse-AI-helm)

*  **Prerequisites:**
    *   A running `K3S cluster`.
    *   `kubectl` configured to interact with your cluster.
    *   `helm` installed.
    *   `Jellyfin` or `Navidrome` or `Lyrion` installed.
    *   Respect the HW requirements (look the specific chapter)

You can directly check the Helm Chart repo for more details and deployments examples.

## **Quick Start Deployment on K3S**

This section provides a minimal guide to deploy AudioMuse-AI on a K3S (Kubernetes) cluster by directly using the `deployment` manifests.

* **Prerequisites:**
    *   A running K3S cluster.
    *   `kubectl` configured to interact with your cluster.
    *   `Jellyfin` or `Navidrome` or `Lyrion` installed.
    *   Respect the HW requirements (look the specific chapter)

*  **Jellyfin Configuration:**
    *   Navigate to the `deployment/` directory.
    *   Edit `deployment.yaml` to configure mandatory parameters:
        *   **Secrets:**
            *   `jellyfin-credentials`: Update `api_token` and `user_id`.
            *   `postgres-credentials`: Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
            *   `gemini-api-credentials` (if using Gemini for AI Naming): Update `GEMINI_API_KEY`.
            *   `mistral-api-credentials` (if using Mistral for AI Naming): Update `MISTRAL_API_KEY`.
        *   **ConfigMap (`audiomuse-ai-config`):**
            *   Update `JELLYFIN_URL`.
            *   Ensure `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL` are correct for your setup (defaults are for in-cluster services).

*  **Navidrome/LMS (Open Subsonic API Music Server) Configuration:**
    *   Navigate to the `deployment/` directory.
    *   Edit `deployment-navidrome.yaml` to configure mandatory parameters:
        *   **Secrets:**
            *   `navidrome-credentials`: Update `NAVIDROME_USER` and `NAVIDROME_PASSWORD`.
            *   `postgres-credentials`: Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
            *   `gemini-api-credentials` (if using Gemini for AI Naming): Update `GEMINI_API_KEY`.
            *   `mistral-api-credentials` (if using Mistral for AI Naming): Update `MISTRAL_API_KEY`.
        *   **ConfigMap (`audiomuse-ai-config`):**
            *   Update `NAVIDROME_URL`.
            *   Ensure `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL` are correct for your setup (defaults are for in-cluster services).
        *   > The same instruction used for Navidrome could apply to other Mediaserver that support Subsonic API. LMS for example is supported, only remember to user the Subsonic API token instead of the password.

*  **Lyrion Configuration:**
    *   Navigate to the `deployment/` directory.
    *   Edit `deployment-lyrion.yaml` to configure mandatory parameters:
        *   **Secrets:**
            *   `postgres-credentials`: Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
            *   `gemini-api-credentials` (if using Gemini for AI Naming): Update `GEMINI_API_KEY`.
        *   **ConfigMap (`audiomuse-ai-config`):**
            *   Update `LYRION_URL`.
            *   Ensure `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL` are correct for your setup (defaults are for in-cluster services).
            
*  **Deploy:**
    ```bash
    kubectl apply -f deployment/deployment.yaml
    ```
*  **Access:**
    *   **Main UI:** Access at `http://<EXTERNAL-IP>:8000`
    *   **API Docs (Swagger UI):** Explore the API at `http://<EXTERNAL-IP>:8000/apidocs`
 
## **Local Deployment with Docker Compose**

AudioMuse-AI provides Docker Compose files for different media server backends:

- **Jellyfin**: Use `deployment/docker-compose.yaml`
- **Navidrome**: Use `deployment/docker-compose-navidrome.yaml`
- **Lyrion**: Use `deployment/docker-compose-lyrion.yaml`
- **Emby**: Use `deployment/docker-compose-emby.yaml`

Choose the appropriate file based on your media server setup.

**Prerequisites:**
*   Docker and Docker Compose installed.
*   `Jellyfin` or `Navidrome` or `Lyrion` or `Emby` installed.
*   Respect the [hardware requirements](../README.md#hardware-requirements)
*   Optionally, you can install the `docker-model-plugin` to enable the use of the [Docker Model Runner](https://docs.docker.com/ai/model-runner/get-started/#docker-engine) for running AI models locally. If you choose this setup, use `deployment/docker-compose-dmr.yaml` to configure AudioMuse-AI to communicate with DMR through an OpenAI-compatible API interface.

**Steps:**
1.  **Create your environment file:**
    ```bash
    cp deployment/.env.example deployment/.env
    ```
    you can find the example here: [deployment/.env.example](../deployment/.env.example)
    
2.  **Review and Customize:**
    Edit `.env` and provide the media-server credentials (e.g., `JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN` or `NAVIDROME_*`, `EMBY_*`, `LYRION_URL`) along with any API keys (`GEMINI_API_KEY`, `MISTRAL_API_KEY`). The same values are injected into every compose file, so you only need to edit them here.
3.  **Start the Services:**
    ```bash
    docker compose -f deployment/docker-compose.yaml up -d
    ```
    Swap the compose filename if you're targeting Navidrome (`docker-compose-navidrome.yaml`), Lyrion (`docker-compose-lyrion.yaml`) or Emby (`docker-compose-emby.yaml`). This command starts all services (Flask app, RQ workers, Redis, PostgreSQL) in detached mode (`-d`).

    **IMPORTANT:** both `docker-compose.yaml` and `.env` file need to be in the same directory.
5.  **Access the Application:**
    Once the containers are up, you can access the web UI at `http://localhost:8000`. You can change the value of the used port by changing the FRONTEND_PORT value
6.  **Stopping the Services:**
    ```bash
    docker compose -f deployment/docker-compose.yaml down
    ```
    Swap the compose filename here as well if you started a different variant.
**Note:**
  > If you use LMS instead of the password you need to create and use the Subsonic API token. Additional Subsonic API based Mediaserver could require it in place of the password.

**Remote worker tip:**
If you deploy a worker on different hardware (using `docker-compose-worker.yaml` or `docker-compose-worker-nvidia.yaml`), copy your `.env` to that machine and update `WORKER_POSTGRES_HOST` and `WORKER_REDIS_URL` so the worker can reach the main server.

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
      