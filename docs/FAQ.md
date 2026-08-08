# AudioMuse-AI FAQ

This document provides answers to frequently asked questions (FAQs) about **deploying** and **using** AudioMuse-AI.

## Deployment FAQs

Find answers to common questions about setting up, configuring, and deploying AudioMuse-AI in different environments.

<details>
<summary>Which is the HW requirements?</summary>

> AudioMuse-AI work on both ARM and INTEL architecture. The suggested requirements are 4core and 8gb of ram with SSD. Some very old processor could have issue due to not supported command.  
> If you want to use the -nvidia version we suggest a GPU with 8gb VRAM.

</details>

<details>
<summary>How to deploy AudioMuse-AI?</summary>

> The [readme](../README.md) section has the explanation and multiple examples can be found in the [deployment folder](../deployment/). If you're not able to reach the front-end on **http://YOUR-IP:8000** or the analysis seems to finish without analyzing anything, it usually means that some parameters are missing in your `.env`.
>
> From v1.0.0, only PostgreSQL and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as landing page and is also available later from the menu under Administration > Setup Wizard.

</details>

<details>
<summary>Can AudioMuse-AI support multiple music libraries?</summary>

> Yes, in two different ways.
>
> **Several libraries inside one server.** Each server has a library filter. It is a comma-separated list of libraries or folders to analyze; if it is empty, everything is scanned. For Lyrion use folder paths like "/music/myfolder". For Navidrome, Jellyfin, Emby and Plex use the library or folder names.
>
> **Several servers at the same time.** A single AudioMuse-AI instance can be connected to several media servers at once, including several servers of the same type, for example one Navidrome plus two Jellyfins plus a Plex. Add them under Setup > Music Servers. The same song present on two servers is analyzed only once and mapped to both. See [MULTI_SERVER](MULTI_SERVER.md) for the full model.

</details>

<details>
<summary>The analysis takes too long, can I speed it up?</summary>

> The time needed for the analysis really depends on your HW and how big your music collection is. For big collections (100k+ songs) or old HW, 1 week+ of analysis can be totally normal.
>
> If you want faster analysis, you can disable the text search functionality by setting `CLAP_ENABLED` to false. This will run only the Musicnn model, skipping the CLAP model.
>
> Alternatives include running multiple worker containers in parallel (see the [ARCHITECTURE](ARCHITECTURE.md) page and deployment examples in the `deployment/` folder). GPU analysis is also supported but still experimental (see [GPU DEPLOYMENT](GPU.md)).
>
> Also remember that Automatic Speech Recognition (ASR) of song is the part that take longer, configure Lyrics API on AudioMuse-AI or on your Music server when supported, will speed up the analysis.

</details>

<details>
<summary>Setup Wizard connection test fails when using Jellyfin</summary>

> During the initial setup, the Setup Wizard may fail the connection test when configuring Jellyfin.
>
> This is most commonly caused by incorrect credentials. In Jellyfin, you must use the **User ID (UID)** instead of the username.
>
> You can find instructions on how to retrieve the Jellyfin User ID here: [PARAMETERS](PARAMETERS.md).

</details>

<details>
<summary>How to get the Plex auth token (X-Plex-Token)</summary>

> Plex authenticates with an auth token instead of a username/password.
>
> Sign in to the Plex Web App, open the browser developer tools (F12) and go to the Network tab. Refresh a library, click a request pointing to your server (for example one ending in `/library/sections`), then copy the `X-Plex-Token` value from the request headers or the query string.
>
> Reference: [plexapi.dev authentication](https://plexapi.dev/authentication). See also [PARAMETERS](PARAMETERS.md).

</details>

---

## User Guide FAQs

Learn how to use AudioMuse-AI effectively, from basic features to advanced functionality.

* **NOTE**: Most front-end parameters default value can be configured in the Setup Wizard functionality. See the parameter table in the [PARAMETERS](PARAMETERS.md) page for a complete list.

<details>
<summary>How do I start using AudioMuse-AI?</summary>

> After deployment, the first thing to do is access the AudioMuse-AI frontend, available at **http://YOUR-IP:8000**.
>
> From there, run the **Analysis**, which collects information about your songs and stores it in the local database.
>
> Running the analysis is **mandatory** before you can use any other features.

</details>

<details>
<summary>How long does the analysis take? What if I interrupt it midway?</summary>

> The time required depends on the number of songs and hardware performance. It can take from a few hours to several days.
>
> If interrupted, you can safely restart the process, already analyzed songs are stored in the database, so only missing songs will be processed.

</details>

<details>
<summary>Clustering returns empty playlists, or playlists with only a few songs. How can I fix this?</summary>

> First check that **Automatic Parameter Discovery** is enabled. It is the recommended setting: a few quick probe runs tune the cluster count and the sampling percentile for each of your servers before the real run, which is what usually fixes empty or tiny playlists on its own.
>
> If you prefer to tune by hand, turn it off and adjust these Advanced Parameters:
>
> - **Stratified Sampling Target Percentile**: raises the number of songs included in the clustering sample (set it up to 100 for maximum coverage)
> - **min clusters / max clusters**: fewer clusters means bigger playlists, more clusters means smaller ones
> - **Minimum playlist size**: playlists below this size are dropped at the end, so a high value can leave you with very few playlists

</details>

<details>
<summary>Clustering returns playlists with too many songs. How can I fix this?</summary>

> Raise `min clusters` and `max clusters`, and lower the `Stratified Sampling Target Percentile`, in the advanced parameter view. With Automatic Parameter Discovery on, you can instead lower the maximum playlist size target so the calibration aims for smaller playlists.

</details>

<details>
<summary>Clustering takes a lot of time, how can I run it faster?</summary>

> Reduce the **Clustering Runs** value. The default is 1000 iterations; a few hundred already gives usable results on a small library.
>
> The run also stops enqueuing new batches once several consecutive batches fail to improve the best result, so raising the run count is not always as expensive as it looks.

</details>

<details>
<summary>How to reset the Admin password?</summary>

> From AudioMuse-AI v1.0.0, the Admin password is stored encrypted in the database. The only way to reset it is by accessing the PostgreSQL database and deleting it. See the [AUTHENTICATION](./AUTH.md) docs for more details.

</details>

<details>
<summary>How to backup and restore the database?</summary>

> Backup and restore are available under `Administration > Backup and Restore`.
>
> Important notes:
> * Restore into the same PostgreSQL major version the backup came from. The published Docker Compose examples use `postgres:15-alpine`; the native builds bundle their own PostgreSQL, whose version can differ.
> * For the same reason, a backup is not always interchangeable between a container deployment and a native Linux, Windows or macOS build.
> * If something fails, check the Flask container logs and the files under `/app/backup`.

</details>

<details>
<summary>What happens if my music server IDs change ?</summary>

> AudioMuse-AI depends on stable track IDs provided by the music server. If an action causes IDs to change (e.g. database reset, migration, reinstall, or major update), existing mappings may break and tracks may appear missing, duplicated, or mismatched.
>
> If this happens, the recommended recovery steps are:
> 1. Restore a previous backup of the music server to recover the original track IDs.
> 2. If restore is not possible, try a provider migration to preserve as much identity mapping as possible.
> 3. If the change is partial (e.g. albums moved or deleted), use `Administration > Cleaning` to remove stale entries and just run a new analysis
> 4. If none of the above works, as a last resort, reset the AudioMuse-AI database and run a full new analysis (this will rebuild all mappings from scratch).
>
> **Always create backups of both the music server and AudioMuse-AI database after the first analysis and possible on weekly basis**

</details>
