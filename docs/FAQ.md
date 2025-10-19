# AudioMuse-AI FAQ

This document provides answers to frequently asked questions (FAQs) about **deploying** and **using** AudioMuse-AI.

## Deployment FAQs

Find answers to common questions about setting up, configuring, and deploying AudioMuse-AI in different environments.

### Which is the HW requirements?

AudioMuse-AI work on both ARM and INTEL architecture. The suggested requirements are 4core and 8gb of ram with SSD. Some very old processor could have issue due to not supported command.

### How to deploy AudioMuse-AI?

The readme section have multiple example. If you're not able to reach the front-end on **[http://YOUR-IP:8000](http://YOUR-IP:8000)** or the analysis seems finishing without analyze nothing, it means that you miss some important parameter. Remember that in docker compose the same parameter need to be write in two place, one in the Flask section and the other in the Worker section

## User Guide FAQs

Learn how to use AudioMuse-AI effectively — from basic features to advanced functionality.


### How do I start using AudioMuse-AI?

After deployment, the first thing to do is access the AudioMuse-AI frontend, which is available at **[http://YOUR-IP:8000](http://YOUR-IP:8000)**.
From there, run the **Analysis**. This process collects information about your songs and stores it in your local database.
Running the analysis is **mandatory** before you can use any other features.

### How long does the analysis take? What if I interrupt it midway?

The time required for the analysis depends on several factors, such as the number of songs to analyze and the hardware on which AudioMuse-AI is running.
Depending on these factors, it can take anywhere from a few hours to several days.

The good news is that analyzed songs are stored in the database, so if the process is interrupted, you can restart it and only the missing songs will be analyzed.


### Clustering returns clusters with only a few songs. How can I fix this?

The default clustering parameters are fine-tuned for music collections of around **50,000–100,000 songs**.
If your clusters are too small, you can adjust the following values in the **Advanced Parameters** view:

* **`Stratified Sampling Target Percentile`**:
  Defines the percentile of songs sampled per genre for clustering.
  The higher this value, the more songs will be clustered. You can set it to **100** to include all songs.

* **`min clusters` and `max clusters`**:
  By default, AudioMuse-AI creates between **40 and 100 clusters (playlists)**.
  Lowering these numbers will result in fewer clusters, each containing more songs.


### Clustering returns clusters with big number of songs. How can I fix this?

In contrast to cluster with few song, you can just raise the `Stratified Sampling Target Percentile`, `min clusters` and `max clusters` values in the advanced parameter view. 