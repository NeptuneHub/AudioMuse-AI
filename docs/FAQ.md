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
> From v1.0.0, only PostgreSQL, Redis, and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as landing page and is also available later from the menu under Administration > Setup Wizard.

</details>

<details>
<summary>Can AudioMuse-AI support multiple music libraries?</summary>

> Yes, it can support multiple music libraries within a single media server instance (e.g., two separate music folders in one Jellyfin server). However, a single AudioMuse-AI instance cannot connect to multiple different media servers (e.g., one Jellyfin and one Navidrome server) at the same time.
>
> The parameters `MUSIC_LIBRARIES` can be used for matching multiple music libraries on the same music server. It is a comma-separated list of music libraries/folders for analysis. If empty, all libraries/folders are scanned. For Lyrion: Use folder paths like "/music/myfolder". For Jellyfin/Navidrome: Use library/folder names.

</details>

<details>
<summary>The analysis takes too long, can I speed it up?</summary>

> The time needed for the analysis really depends on your HW and how big your music collection is. For big collections (100k+ songs) or old HW, 1 week+ of analysis can be totally normal.
>
> If you want faster analysis, you can disable the text search functionality by setting `CLAP_ENABLED` to false. This will run only the Musicnn model, skipping the CLAP model.
>
> Alternatives include running multiple worker containers in parallel (see the [ARCHITECTURE](ARCHITECTURE.md) page and deployment examples in the `deployment/` folder). GPU analysis is also supported but still experimental (see [GPU DEPLOYMENT](GPU.md)).

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
<summary>Clustering returns empty playlist or with only a few songs. How can I fix this?</summary>

> The default clustering parameters are tuned for collections of around **50,000–100,000 songs**.
>
> If clusters are too small or empty, adjust these Advanced Parameters:
>
> - **Stratified Sampling Target Percentile**: increases number of songs included in clustering (set up to 100 for more coverage)
> - **min clusters / max clusters**: reduce or increase number of clusters and adjust cluster size

</details>

<details>
<summary>Clustering returns clusters with big number of songs. How can I fix this?</summary>

> Increase the `Stratified Sampling Target Percentile`, `min clusters`, and `max clusters` values in the advanced parameter view.

</details>

<details>
<summary>Clustering takes a lot of time, how can I run it faster?</summary>

> By default, clustering runs 5000 iterations. You can reduce the **Clustering Runs** value (e.g., 1000) to speed up execution while keeping acceptable quality.

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
> * Ensure the PostgreSQL version matches the deployment example (e.g., up to v2.1.5 uses `postgres:15-alpine`). Version mismatches may break backup/restore compatibility.
> * Backups may not be interchangeable across different OS/container setups (Linux, Windows, macOS) due to PostgreSQL version differences.
> * If issues occur, check logs in the Flask container under `/app/backup`.

</details>
