# Tested HW and Configuration WITH ONNX (from version 0.7.0-beta)
| Issue ID | HW | CONFIGURATION | Supported | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - | **CPU:** Intel: i5 6th gen, i5 8th gen; ARM: VM on cloud (no detail on the specific cpu) | K3S cluster, Docker Image `:0.7.0-beta` | ✅ Yes | - |
| - | **CPU:**  Intel i5-12450H | K3S, Docker Image `:0.7.0-beta` | ✅ Yes | - |
| [#14](https://github.com/NeptuneHub/AudioMuse-AI/issues/104)  | **CPU:**  Intel 12600K | Docker Compose, Jellyfin, Docker Image `:0.7.0-beta`  | ✅ Yes| user need to set at least 4GB of ram to the container |

# Tested HW and Configuration WITH TENSORFLOW (till version 0.6.10-beta)

This table collect all the HW and configuration from the different issue in this repository.

**IMPORTANT:** If a CPU is not present in the table only means is not tested, but is still possible that works. If you have an issue on a particular CPU please raise a ticket.

| Issue ID | HW | CONFIGURATION | Supported | Notes |
| :--- | :--- | :--- | :--- | :--- |
| [#14](https://github.com/NeptuneHub/AudioMuse-AI/issues/14) | **CPU:** Ryzen 5600G | Docker Compose, Jellyfin | ✅ Yes | - |
| [#24](https://github.com/NeptuneHub/AudioMuse-AI/issues/24) | **CPU:** Ryzen 5600G | TrueNAS SCALE, Docker Image v0.6.0-beta / v0.6.2-beta | ✅ Yes | - |
| [#25](https://github.com/NeptuneHub/AudioMuse-AI/issues/25) | **CPU:** i5 6th gen, i5 8th gen, ARM Raspberry PI 5 | K3S cluster, Docker Image `:devel` | ✅ Yes | - |
| [#39](https://github.com/NeptuneHub/AudioMuse-AI/issues/39) | **CPU:** Ryzen 5 3600 | Bazzite (Fedora Atomic), Docker Image v0.6.4-beta | ✅ Yes | - |
| [#55](https://github.com/NeptuneHub/AudioMuse-AI/issues/55) | **CPU:** Intel (32c), **GPU:** Nvidia RTX 3060 | Docker Swarm/single docker, Navidrome | ✅ Yes | Analysis failed for high-res FLACs. Developer added fallbacks, but the issue was ultimately a problem with Navidrome's media library indexing, fixed by the user running a full scan in Navidrome. |
| [#62](https://github.com/NeptuneHub/AudioMuse-AI/issues/62) | **CPU:** Intel Xeon W-2125 | Proxmox-VE LXC (Debian), Docker Image v0.6.5-beta | ✅ Yes | - |
| [#66](https://github.com/NeptuneHub/AudioMuse-AI/issues/66) | **CPU:** E5-2697 | Docker Compose (via Portainer), Navidrome 0.58.0 | ✅ Yes | - |
| [#67](https://github.com/NeptuneHub/AudioMuse-AI/issues/67) | **CPU:** Intel i7-10850H | Arch Linux, Docker Image `latest-nvidia` | ✅ Yes | - |
| [#69](https://github.com/NeptuneHub/AudioMuse-AI/issues/69)| **CPU:** Ryzen 5 PRO 4650G | Ubuntu 24.04.3 LTS, Docker Image v0.6.7-beta | ✅ Yes | - |
| [#73](https://github.com/NeptuneHub/AudioMuse-AI/issues/73) | **CPU:** Intel core i5 1035G1 | Docker Compose, Jellyfin 10.10.7 | ✅ Yes | Database showed zero tracks after analysis. A bug related to float precision on certain CPUs was fixed by casting to `Float32` in the `:devel` branch. |
| [#74](https://github.com/NeptuneHub/AudioMuse-AI/issues/74) | **CPU:** Amd Ryzen 3600 | Docker Compose, Navidrome v0.58.0 | ✅ Yes | - |
| [#65](https://github.com/NeptuneHub/AudioMuse-AI/issues/65) | **CPU:** N100 | Docker Compose, Navidrome | ✅ Yes | - |
| [#93](https://github.com/NeptuneHub/AudioMuse-AI/issues/93) | **CPU:** AMD Ryzen AI 9 HX 370 w/ Radeon 890M 64bit | Podman with docker-compose (v5.6.1), Jellyfin v10.10.7, AudioMuse-AI v0.6.8-beta | 🚧 In Progress | A bug with CPU-specific behavior was fixed in the `:devel` branch by adding ENV TF_ENABLE_ONEDNN_OPTS=0. Probably a new parameter will be added in deployment/yaml file. |
| [#56](https://github.com/NeptuneHub/AudioMuse-AI/issues/56) | **CPU:** Intel Celeron CPU N3160 | Docker Compose, Unraid 7.1.4 | ❌ No | Flask app failed with an `Illegal instruction` error. This is a hardware limitation: TensorFlow requires **AVX CPU support**, which the Celeron N3160 lacks. |

