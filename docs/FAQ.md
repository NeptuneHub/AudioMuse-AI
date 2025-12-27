# AudioMuse-AI FAQ

This document provides answers to frequently asked questions (FAQs) about **deploying** and **using** AudioMuse-AI.

## Deployment FAQs

Find answers to common questions about setting up, configuring, and deploying AudioMuse-AI in different environments.

### Which is the HW requirements?

AudioMuse-AI work on both ARM and INTEL architecture. The suggested requirements are 4core and 8gb of ram with SSD. Some very old processor could have issue due to not supported command.
If you want to use the -nvidia version we suggest a GPU with 8gb VRAM.

### How to deploy AudioMuse-AI?

The [readme](../README.md) section has the explanation and multiple examples can be found in the [deployment folder](../deployment/). If you're not able to reach the front-end on **[http://YOUR-IP:8000](http://YOUR-IP:8000)** or the analysis seems to finish without analyzing anything, it usually means that some parameters are missing in your `.env`. Docker Compose now pulls shared values from there, so update the media server credentials once and both the Flask and Worker services will receive them automatically.

### Can AudioMuse-AI support multiple music libraries?
Yes, it can support multiple music libraries within a single media server instance (e.g., two separate music folders in one Jellyfin server). However, a single AudioMuse-AI instance cannot connect to multiple different media servers (e.g., one Jellyfin and one Navidrome server) at the same time. 

The ENV variable `MUSIC_LIBRARIES` can be used for match multiple music library on the same music server. Is a Comma-separated list of music libraries/folders for analysis. If empty, all libraries/folders are scanned. For Lyrion: Use folder paths like "/music/myfolder". For Jellyfin/Navidrome: Use library/folder names.	"" (empty - scan all)

### The analysis takes to long, can I speed up it?
The time needed for the analysis really depends from your HW and how big is your music collection. For big collection (100k+ songs) or old HW 1week+ of analysis can be totally normal.
If you want to have a faster analysis you can also disable the text search functionality by putting the env var `CLAP_ENABLED` to false. This will run the analysis only for the Musicnn model skipping the CLAP model.
Alternatives could also be run multiple worker container in parallel, learn more by taking a look to the [ARCHITECTURE](docs/ARCHITECTURE.md) page and to the different example of deployment in `deployment/` folder.
GPU analysis is also supported but still experimental. Take a look to [GPU DEPLOYMENT](docs/GPU.md) page.

## User Guide FAQs

Learn how to use AudioMuse-AI effectively, from basic features to advanced functionality.

* **NOTE**: Most front-end parameters can also be set as environment variables. See the parameter table in the [README.md](../README.md) for a complete list.

### How do I start using AudioMuse-AI?

After deployment, the first thing to do is access the AudioMuse-AI frontend, which is available at **[http://YOUR-IP:8000](http://YOUR-IP:8000)**.
From there, run the **Analysis**. This process collects information about your songs and stores it in your local database.
Running the analysis is **mandatory** before you can use any other features.

### How long does the analysis take? What if I interrupt it midway?

The time required for the analysis depends on several factors, such as the number of songs to analyze and the hardware on which AudioMuse-AI is running.
Depending on these factors, it can take anywhere from a few hours to several days.

The good news is that analyzed songs are stored in the database, so if the process is interrupted, you can restart it and only the missing songs will be analyzed.


### Clustering returns empty playlist or with only a few songs. How can I fix this?

The default clustering parameters are fine-tuned for music collections of around **50,000–100,000 songs**.
If your clusters are too small or is empty, you can adjust the following values in the **Advanced Parameters** view:

* **`Stratified Sampling Target Percentile`**:
  Defines the percentile of songs sampled per genre for clustering.
  The higher this value, the more songs will be clustered. You can set it to **100** to include more songs.

* **`min clusters` and `max clusters`**:
  By default, AudioMuse-AI creates between **40 and 100 clusters (playlists)**.
  Lowering these numbers will result in fewer clusters, each containing more songs.

### Clustering returns clusters with big number of songs. How can I fix this?

In contrast to cluster with few song, you can just raise the `Stratified Sampling Target Percentile`, `min clusters` and `max clusters` values in the advanced parameter view. 

### Clustering takes a lot of time, how can I run it faster?

Clusterign algorithm by default do 5000 run. This means that multiple run are executed and the best is kept. You can lower this number in the front-end in the `Clustering Runs:` to do less run. For example with 1000 run the result should still be good enough and take a reasonable amount of time.