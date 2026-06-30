# Contributing to AudioMuse-AI

Thank you for considering a contribution to AudioMuse-AI. Open-source projects thrive on the collective effort and expertise of their communities, and every contribution, regardless of size, is highly valued.

The vision of AudioMuse-AI is to bring Sonic Analysis open and free to the higher number of user possible. So each change should aim to bring it more usebul from more and more users.

Remember that contributing not only means develop code, also suggest new feature, highlight a bug or just share your feedback (good or bad is always important) by an [issue](https://github.com/NeptuneHub/AudioMuse-AI/issues) is also contributing.

Multiple information can be found in the [docs](docs/) folder.

> Issues, discussions, PRs, comments in code, and any other interaction MUST be written in English. We reserve the right not to reply to and to close them if they are not.

## High-Level Architecture
To contribute effectively, it is crucial to understand that AudioMuse-AI is not a monolithic program. It is a multi-service, containerized application designed for robustness, scalability, and a clear separation of concerns. This architecture is composed of several core components that work in concert.
* **Flask Web Application (audiomuse-ai-flask):** Here is the front-end of the application, intended as both an HTML page and an API. Here also lives the logic of the services that are synchronous, such as getting similar songs.
* **Redis Queue (RQ) Workers (audiomuse-ai-worker):** This is for tasks that need to be executed asynchronously, such as analyzing songs, performing clustering, or reconstructing the index for similar song searches. With the Redis queue and a Kubernetes architecture, it is possible to spawn more workers to increase scalability and availability.
* **Redis Queue:** where Flask write the job, and Workers check the job to do.
* **PostgreSQL Database (postgres-deployment):** The database. Not only the analysis live here but also the log status o the async task.

## Supported Architecture and Mediaserver
Remember that this software support both Intel and Arm architecture. So avoid code that will not work on both except for very specific case. If you're not able to test on both architecture, add this in the PR description.

Remember that AudioMuse-AI is also shipped as native app for MacOS, Linux and Windows and your change must avoid to brake them-

Remember also that this application support multiple mediaserver. So try to don't introduce change that can distrupt one or the other mediaserver. If you're not able to test on all mediaserver, add this in the PR description.


## **The Codebase Map**

The following table details the most important paths in the repository, their purpose, and the key technologies associated with them.

| Path | Purpose |
| :---- | :---- |
| app.py, app_*.py | The main entry point for the Flask web application. It handles the initialization of the Flask app, database connections, and the registration of API routes and blueprints. |
| tasks/ | **The Core Logic Hub.** This is where the most intensive computations occur. Each API or async task then point to an specific implementation in this directory|
| tasks/mediaserver/ | In this package the generic methods to interact with the mediaservers (`__init__.py`) dispatch to the specific backend (`jellyfin.py`, `navidrome.py`, `emby.py`, `lyrion.py`) |
| tasks/ai/ | All AI / MCP code. `tasks/ai/api.py` is the provider dispatcher (called via `tasks.ai.api.call_with_tools()`), with backends in `tasks/ai/providers/{openai,gemini,mistral,ollama}.py`. Prompts are in `tasks/ai/prompts.py`. MCP tool schemas + dispatcher in `tasks/ai/tools.py`; tool bodies in `tasks/ai/tool_impl.py`. Two-stage planner (intent classifier + plan validation + execution) in `tasks/ai/planner.py`. Vocabulary normalization helpers in `tasks/ai/vocab.py`. |
| config.py | Contains the application's default, non-sensitive configuration parameters. These values serve as fallbacks and can be easily overridden by environment variables, providing a flexible and secure configuration system. |
| Authentication | Configured in `config.py` by `AUTH_ENABLED`, `AUDIOMUSE_USER`, `AUDIOMUSE_PASSWORD`, `API_TOKEN`, and `JWT_SECRET`. Enforcement happens in `app.py` and `app_helepr.py` functionality |
| static/ & templates/ | These directories contain all frontend assets. |
| deployment/ | This contains deployment example but also the supervisord configuration |
| Dockerfile, Dockerfile.nvidia | These files contain the instructions for building the OCI-compatible container images for the application. |
| .github/ | This directory holds GitHub-specific configuration files, such as issue templates, pull request templates, and potentially continuous integration/continuous deployment (CI/CD) workflows.1 |

## **Prerequisites**

The development environment for AudioMuse-AI is fully containerized to ensure consistency and simplify setup. The only required tools are:

* **Git:** For version control and interacting with the GitHub repository.  
* **Docker and Docker Compose:** For building and running the containerized application stack.
* ***Python dependencies:** Need to run unit and integration test locally before push the code.

## **How to compile**

If you have a k3s (kubernetes) cluster at home, I highly recommend to deploy a local registry and then directly deploy the image against it. To do that I suggest to follow my [private-registry how-to](https://github.com/NeptuneHub/k3s-supreme-waffle/tree/main/private-registry)

If you don't have K3s (kubernetes) at home, you can just use docker-compose to compile the `docker-compose.yaml` file on the flight. Remember to point it to your local image by changing this:

```
audiomuse-ai-flask:
    image: ghcr.io/neptunehub/audiomuse-ai:0.6.5-beta
    # ... rest of the service definition
```

to something like this:
```
audiomuse-ai-flask:
    build:
      context: .  # <-- ADD THIS: Tells Docker to look for the Dockerfile in the current directory
      dockerfile: Dockerfile # <-- ADD THIS: Specifies the name of the Dockerfile
    image: audiomuse-ai:dev # <-- CHANGE THIS: Give your local build a new, clear name
    # ... rest of the service definition
```

for both `flask` and `worker` container. Then you can just run&build with this command:

```
docker-compose up --build -d
```

## **PR**
### Before You Start
0. Remember that **AudioMuse-AI is under AGPLv3 license**, your own code, plus any third-party libraries, frameworks, or models you introduce MUST align with AGPLv3
1. **Check existing PRs and issues** to avoid duplicate work
2. **Discuss WHAT you want to implement and HOW first** or in an existing issue (if you want to solve it) or creating a new one if the topic is not already covered (use feature label for feature, bug for bugfix)

### PR Requirements
When submitting a pull request, ensure:

* **Clear description:** Explain what the PR achieves and why the change is needed with **HUMAN generated** text. Also clearly explain how you tested it and how to replicate those tests
* **Link the PR to an existing issue:** in this way you work on something already agreed on and you avoid many rework.
* **Testing:** Verify core features work on at least one architecture (Intel/ARM) and one media server:
  * Analysis and Clustering
  * Instant Playlist
  * Playlist from Similar Song
  * Song Path
  * Sonic Fingerprint
  * *(Basically, test each function in the integrated front-end menu at least once)*
* **Automated tests:** Any PR that adds or changes major functionality MUST include automated tests (unit and/or integration, under `test/`) that cover that functionality. Run `pytest test/unit/` and `pytest test/integration/` locally before pushing, and make sure the CI Unit Tests and Integration Tests workflows pass.
* **AGPLv3 License compliance:** Your code must align with AudioMuse-AI's license
* **CPU Compatibility:** AudioMuse-AI supports both Intel and ARM CPUs, including older Intel processors. PRs that introduce dependencies breaking compatibility with older CPUs will not be merged
* **Documentation:** If needed, update the documentation
* **Static analysis (flake8):** the `Static Analysis` workflow (`.github/workflows/lint-flake8.yml`) runs `flake8 --select=E9,F,W605,E711,E712,E713,E714,E722,E401` on every push/PR to `main` and must pass. It covers syntax errors, all pyflakes checks (undefined/leftover names, unused imports and locals, f-string and `%`-format mistakes), invalid escape sequences, `== None`/`== True` comparisons, bare `except:`, and multiple imports on one line. Run `flake8 .` with that select before pushing. In particular: don't reference undefined names, leftover/renamed variables, or unresolved imports; fix syntax errors; and only keep a `global`/`nonlocal` declaration when the function actually reassigns that name (declaring one for a variable you merely read or mutate in place — `d[k]=v`, `.append()`, `.update()` — will fail the check). Style/formatting rules (line length, whitespace, blank lines) are intentionally not enforced here.
* **logging.exception (ruff TRY400):** the `Ruff` workflow (`.github/workflows/lint-ruff.yml`, config in `ruff.toml`) runs the same rule set plus `TRY400` — inside an `except` block use `logging.exception("msg")`, never `logging.error(...)`. Run `ruff check .` before pushing. `ruff format` (single-quote-preserving) and the un-gated rules (line length, `pep8-naming`) are available to run ad hoc but are not required to pass.
* **Spell check (codespell):** the `Spell Check (codespell)` workflow (`.github/workflows/lint-codespell.yml`) runs `codespell` over the whole repository and must pass. Fix real typos; add genuine false positives (with a short reason) to `.codespellrc`.
* **No emoji/icons in code:** code files must stay plain ASCII — decorative emoji break the Windows console (cp1252) when logged. Use ASCII instead (`OK`, `X`, `->`). The `No Emoji In Code` workflow (`.github/workflows/lint-no-emoji.yml`) enforces this for `.py` using the maintained `emoji` library (run `pip install emoji && python scripts/check_no_emoji.py` locally). Emoji remain fine in HTML/templates and frontend JS.
* **Line endings (LF):** every file must use LF line endings; the `Line Ending Linter` workflow rejects CRLF.

> Important: Prefear opening small PR, focused on specific functionality that directly add value. Avoid to change multiple unrelated functionality to facilitate test.

> Contributions generated with AI are welcome, provided that a qualified human reviewer verifies, tests, and understands the code. AI tools can assist in development, but all pull requests must be submitted by someone capable of ensuring correctness and maintainability. 

> Missing requirements may lead to requests for additional information and, if not provided, the PR may be closed. Regardless of the above, the final decision to merge a pull request is at the maintainer’s discretion.

### How to Open a Draft PR
1. Push your branch to your fork
2. Go to the main repository and click **"New Pull Request"**
3. Select your fork and branch
4. Click the dropdown arrow next to **"Create Pull Request"**
5. Select **"Create Draft Pull Request"**
6. Once ready for review, click **"Ready for review"** in the PR

This workflow helps avoid spending time on PRs that may not align with project goals.

##  **Related Repositories** 
Below the full list of related repository. When you submit a PR avoid to introduce change that could brake one or more of this repository:
  > * [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI): the core application, it run Flask and Worker containers to actually run all the feature;
  > * [AudioMuse-AI Helm Chart](https://github.com/NeptuneHub/AudioMuse-AI-helm): helm chart for easy installation on Kubernetes;
  > * [AudioMuse-AI Plugin for Jellyfin](https://github.com/NeptuneHub/audiomuse-ai-plugin): Jellyfin Plugin;
  > * [AudioMuse-AI Plugin for Navidrome](https://github.com/NeptuneHub/AudioMuse-AI-NV-plugin): Navidrome Plugin;
  > * [AudioMuse-AI MusicServer](https://github.com/NeptuneHub/AudioMuse-AI-MusicServer): Open Subosnic like Music Sever with integrated sonic functionality.


## **Security**

To report a security vulnerability, please do **not** open a public issue or PR. Follow the private process documented in [SECURITY.md](SECURITY.md).

## **Questions**

For any question you can [raise an issue](https://github.com/NeptuneHub/AudioMuse-AI/issues)

