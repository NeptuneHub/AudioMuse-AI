# AudioMuse-AI Plugins

## Introduction

AudioMuse-AI plugins let you add features without touching the core app. A plugin is a small Python package that can add a page, read and write the database, talk to your media server, save settings, and run scheduled jobs. You install and update them from inside AudioMuse-AI.

## Capability

Everything a plugin can do, at a glance. The last column tells you where to find working code: most capabilities are shown live in the SongCounter reference plugin, and the rest have a complete example in this guide.

| # | Capability | How | In the SongCounter example |
|---|---|---|---|
| 1 | Web page + menu entry | `add_blueprint`, `add_menu_item` | Yes: the chart page and its menu link |
| 2 | Settings page (opened from Manage Plugins) | a route called `settings` | Yes: pick which indexes to count |
| 3 | Per-plugin settings storage | `get_setting` / `set_setting` | Yes: saves the chosen indexes |
| 4 | Read the core database | `get_db` | Yes: counts songs and index sizes |
| 5 | Own data tables | `table()` + `on_install` | Yes: `hook_stats` and `index_log` tables |
| 6 | React to each analyzed song | `on_song_analyzed` | Yes: live per-run counter + last song payload |
| 7 | Scheduled (cron) tasks | `add_cron_task` | Yes: the hourly `index_log` snapshot |
| 8 | Extra pip packages, with version pins - **works only on the container image (Docker / Kubernetes), never on the standalone builds** | `requirements` in plugin.json | Yes: matplotlib |
| 9 | Choose the container it runs on | `targets` in plugin.json | Yes: left out, so it runs on both |
| 10 | Publishing, updates and rollback | catalog + `versions` list | Yes: released through the community repo |
| 11 | Background jobs from a page | `enqueue` | No - see "Run a job in the background" |
| 12 | Named worker tasks | `add_task` | No - see "API reference" |
| 13 | Startup hooks | `on_flask_start` / `on_worker_start` | No - see "The plugin lifecycle" |
| 14 | Create media-server playlists | `tasks.mediaserver` | No - see "Create a playlist on your media server" |
| 15 | Extra ONNX execution provider | `register_onnx_provider` | No - see "API reference" (advanced) |

## Installing and managing plugins

You manage plugins from inside AudioMuse-AI. Open the **Plugins** page from the menu. Only an admin can see and use it. The page has three tabs:

* **Installed** shows the plugins you have, their status, and an update button when a newer version exists.
* **Catalog** shows every plugin available from your repositories, with an Install button.
* **Repositories** lets you add or remove plugin catalogs (the default community catalog is always there).

### Install a plugin

Go to the Catalog tab and click **Install**. You must confirm a warning first: plugins run with full application and database permissions, so only install plugins you trust. If the plugin has extra pip dependencies the install can take up to a minute. Do not reload the page while it runs. The worker containers download the plugin and its dependencies at the same time, so the restart afterwards is fast.

### Apply your changes (restart)

Install, update, enable, disable and uninstall all need a restart to take effect. After any of these actions a yellow banner appears: "A restart is required to apply your changes". The banner is remembered on the server, so it stays after a page reload and every admin sees it until the restart happens. You can do several changes in a row and then click **Apply now (restart)** once. AudioMuse-AI restarts and the page reloads by itself after about 20 seconds.

### Update a plugin

AudioMuse-AI checks your repositories for new versions in the background (about once per hour, and every time you open the Catalog tab). When a newer compatible version exists, the Installed tab shows an **Update to vX** button. Click it, then apply the restart. Your plugin settings, its data tables and its scheduled jobs are kept.

### Install an older version (rollback)

When a plugin publishes more than one compatible version, the Catalog tab shows a small version selector next to the Install button. Pick the version you want and click the button: this is how you roll back after a bad update. Rolling back replaces the code only; your settings and data tables stay.

### Enable and disable

Disable turns a plugin off without removing anything. Its pages, menu items and scheduled jobs stop working after the restart, but its code, settings and data stay. Enable it again at any time.

### Uninstall

When you uninstall, AudioMuse-AI first asks you to confirm the uninstall itself. Then it asks one more question: do you also want to delete the plugin's data tables?

* Choose **OK** to drop every table the plugin created. This cannot be undone.
* Choose **Cancel** to keep the tables for later.

The plugin's settings and its entries in Scheduled Tasks are always removed. If you kept the data tables, reinstalling the plugin later finds them again.

## Working example

The best way to learn is to read a real plugin. SongCounter is a small, complete example with a page, a settings page, per-plugin settings, a worker hook that shows a live count of analyzed songs plus the last song's full hook payload, and an hourly cron task that logs the index sizes:

https://github.com/NeptuneHub/AudioMuse-AI-plugins/tree/main/plugins/SongCounter

## Plugin architecture

A plugin needs two files: `plugin.json` says what the plugin is, and `__init__.py` says what it does (it can add more, such as a `tasks.py` or a `templates/` folder). The examples below come from SongCounter, the reference plugin, so the easiest way to start is to copy it.

`plugin.json` is the plugin's whole description. This is a simplified version of SongCounter's (the real one has more releases, and no `targets` because its worker hook needs both containers):

```json
{
  "id": "song_counter",
  "name": "SongCounter",
  "author": "NeptuneHub",
  "description": "Counts analyzed songs and shows them as a bar chart.",
  "requirements": ["matplotlib"],
  "versions": [
    {
      "version": "1.5.0",
      "min_core_version": "2.5.0",
      "changelog": "First public release.",
      "imageUrl": ""
    }
  ]
}
```

The top level is the plugin's identity: `id` (lowercase, matching `^[a-z][a-z0-9_]{1,63}$`, used in its URL and table names), `name`, `author`, `description`, `targets` (which container runs it - optional, see "Choose where the plugin runs"), and `requirements` (extra pip packages). The `versions` list has one entry per release, each with its own `version`, `min_core_version` (the core version that release needs), `changelog`, and `imageUrl`. You add a new entry to the top for each release; the build workflow fills in that release's `sourceUrl` (the code zip) and `checksum` (its md5), so you never write those by hand.

`__init__.py` holds the code. It must define `register(ctx)`, which tells AudioMuse-AI what to add. SongCounter adds one page and a menu item that opens it:

```python
from flask import Blueprint, request, redirect
from plugin.api import get_db, get_setting, set_setting, render_page, manage_plugins_url

bp = Blueprint("song_counter", __name__)

@bp.route("/")
def home():
    label = get_setting("label", "Analyzed songs")
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM score")
    total = cur.fetchone()[0]
    cur.close()
    return render_page(f"<p>{label}: {total}</p>", title="SongCounter")

@bp.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        set_setting("label", request.form.get("label", "Analyzed songs"))
        return redirect(manage_plugins_url())
    label = get_setting("label", "Analyzed songs")
    body = (
        '<form method="post">'
        f'<input name="label" value="{label}">'
        '<button type="submit">Save</button>'
        '</form>'
    )
    return render_page(body, title="SongCounter Settings")

def register(ctx):
    ctx.add_blueprint(bp)
    ctx.add_menu_item("SongCounter", "song_counter.home")
```

The full SongCounter draws its counts as a bar chart with matplotlib, which is why its `plugin.json` lists `matplotlib` under `requirements`.

One naming rule: **the Blueprint name must be your plugin id**, like SongCounter does with `Blueprint("song_counter", __name__)`. The plugin id is unique by definition, so your routes (`song_counter.home`, `song_counter.settings`) can never collide with another plugin. Blueprint names are unique across the whole app: if two plugins use the same name, the second one fails to load with a clear error, and a plugin using any other name gets a warning in the log.

If you want a settings page, add a route called `settings`. AudioMuse-AI opens it from the Settings button on the Manage Plugins page, so it does not add a menu entry for it. If your settings route has a different name, point to it with `ctx.set_settings_page("song_counter.my_settings")`. The settings page is always admin only.

To publish, one more file lists the plugin: the catalog `manifest.json`. It has one small entry per plugin, holding `id`, `name`, `author`, `description`, and a `pluginUrl` that points at that plugin's `plugin.json`. AudioMuse-AI reads the catalog, follows `pluginUrl` to your `plugin.json`, picks the newest version the running core supports, downloads its `sourceUrl` zip (code only, with no `plugin.json` inside), and verifies the `checksum`. You never write the catalog or the `sourceUrl`/`checksum`; the build workflow generates them from your `plugin.json`.

AudioMuse-AI ships with the public community catalog by default. To run your own instead, host a `manifest.json` and add its link under Plugins > Repositories.

## Main capabilities

Import everything you need from `plugin.api`. Below are the most common things a plugin does.

### Read the database

Get a normal database connection and run any query you want.

```python
from plugin.api import get_db

def count_songs():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM score")
    total = cur.fetchone()[0]
    cur.close()
    return total
```

### Read song details

Get title, artist, tempo, key, mood, energy and more for a list of songs.

```python
from plugin.api import get_score_data_by_ids

rows = get_score_data_by_ids(["song-id-1", "song-id-2"])
for row in rows:
    print(row["title"], row["author"], row["tempo"], row["mood_vector"])
```

### Create a playlist on your media server

Pick some songs from the database, then send them to your media server. This works with Jellyfin, Emby, Plex, Navidrome and Lyrion.

```python
from plugin.api import get_db
from tasks.mediaserver import create_or_replace_playlist

def make_fast_playlist():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT item_id FROM score WHERE tempo > 120 LIMIT 50")
    track_ids = [row[0] for row in cur.fetchall()]
    cur.close()
    if track_ids:
        create_or_replace_playlist("Fast Songs", track_ids)
```

The call raises an error when the list is empty, so check that you found songs first.

### Save and read settings

Store small values for your plugin. The admin can also edit them from the Settings button.

```python
from plugin.api import get_setting, set_setting

set_setting("limit", 50)
limit = get_setting("limit", 20)  # 20 is the default if nothing is saved yet
```

### Store your own data in a table

Your plugin can have its own tables. Use `table()` to get a safe, unique name, and create the table once at install time.

```python
from plugin.api import get_db, table

def migrate(db):
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS " + table("runs") + " (ran_at TIMESTAMP DEFAULT now())")
    db.commit()

def register(ctx):
    ctx.on_install(migrate)   # runs at install and again on every update
```

### Run a job on a schedule

Write a function and register it as a cron task. It runs on the worker.

```python
from plugin.api import logger

def daily_job():
    logger.info("my plugin ran")

def register(ctx):
    ctx.add_cron_task("daily", daily_job)
```

Then open Administration > Scheduled Tasks. Every cron task of an enabled plugin is listed there under "Plugin tasks" with its own cron field and an Enable checkbox. The task type is `plugin.<your id>.<task name>` (here `plugin.my_plugin.daily`). Each run gets a row under Active Tasks and is marked success or failure by itself - you do not need to report anything.

### Scheduled tasks and multiple music servers

Music servers hold different catalogues, so a scheduled plugin task runs **once
per music server**, on **every** configured server, one at a time - like every
other batch task in AudioMuse-AI. There is no scope selector to choose from. Your
function does not change: every media-server call inside it - playlist creation,
listening history, downloads - already targets the server of the current run.
Ask who that is, or target another server explicitly, through the API:

```python
from plugin.api import active_server_id, list_servers, use_server

def daily_job():
    logger.info("running against server %s", active_server_id() or "default")
    # ... playlists created here land on the server of this run

    with use_server(some_other_server_id):   # only if you need a specific one
        ...
```

Tracks are stored once in a shared catalogue under a canonical id, so a task
that only reads the database sees the same songs on every run; use
`active_server_id()` when the work is server-specific.

You can also ship a default schedule (disabled, so the admin stays in control) by inserting the cron row in your `on_install` hook. SongCounter does this for its `index_log` task: every hour it stores a timestamped snapshot of the index sizes in its own table, keeps only the last 10 rows, and shows them as a small log on its page.

```python
def migrate(db):
    cur = db.cursor()
    cur.execute(
        "INSERT INTO cron (name, task_type, cron_expr, enabled) VALUES (%s, %s, %s, FALSE) "
        "ON CONFLICT (task_type) DO NOTHING",
        ('plugin.my_plugin.daily', 'plugin.my_plugin.daily', '0 * * * *'),
    )
    db.commit()
```

### Run a job in the background

A page must answer fast. For heavy work, put the job in a function and hand it to the worker with `enqueue`. The route returns right away and the job runs on the worker container.

```python
from flask import Blueprint
from plugin.api import enqueue, render_page, logger

bp = Blueprint("my_plugin", __name__)

def rebuild_report(days):
    logger.info("rebuilding the report for %s days", days)
    # slow work here

@bp.route("/rebuild")
def rebuild():
    enqueue(rebuild_report, 30)
    return render_page("<p>Rebuild started. Check the worker logs.</p>", title="My Plugin")
```

Keyword arguments work too: `enqueue(rebuild_report, days=30)`. Use `queue='high'` if the job should skip the analysis queue. The `logger` output goes to the worker container logs.

One important rule: the job runs on the worker, so the worker must have your plugin's code - do not set `targets` to `["flask"]` (see "Choose where the plugin runs").

### React to a song after analysis

Register a listener with `ctx.on_song_analyzed(func)` and AudioMuse-AI calls it on the worker right after each song finishes analysis (all models run and results saved). This is where you run another model on the audio or store extra information about the song.

Your function receives one dict:

* `item_id` - the media-server id (string). In a multi-server install this is always the **default** server's id (the canonical id used across the database). To reach the same track on another configured server, translate it with `from tasks.mediaserver import registry; registry.translate_ids([item_id], other_server_id)`. See [MULTI_SERVER.md](MULTI_SERVER.md).
* `run_id` - the analysis run's task id. Every song of one "Start Analysis" shares the same `run_id`, so you can count or group per run (reset when it changes).
* `audio_path` - the temporary audio file on disk. It is deleted right after your listener returns, so read it now if you need it.
* `metadata` - `title`, `artist`, `album`, `album_artist`, `year`, `rating`, `file_path`, `album_id`, `album_name`.
* `media_item` - the full raw track object from the media server.
* `analysis` - `{tempo, key, scale, moods, energy}`, or `None` if MusiCNN was skipped for this song.
* `top_moods` - the top moods as `{name: score}`.
* `musicnn_embedding`, `clap_embedding` - numpy arrays, or `None` when that model did not run.

```python
from plugin.api import get_db, table, logger

def on_analyzed(song):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO " + table("seen") + " (item_id, tempo) VALUES (%s, %s) "
        "ON CONFLICT DO NOTHING",
        (song["item_id"], (song["analysis"] or {}).get("tempo")),
    )
    db.commit()
    cur.close()

def migrate(db):
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS " + table("seen") + " (item_id TEXT PRIMARY KEY, tempo REAL)")
    db.commit()

def register(ctx):
    ctx.on_install(migrate)
    ctx.on_song_analyzed(on_analyzed)
```

The listener runs inside the analysis loop, so keep it quick. If the work is heavy (a second model over the whole audio), hand it to `enqueue` instead and copy `audio_path` first, or re-download the audio by `item_id` in the background job, because the temp file is gone once your listener returns. If your listener raises, AudioMuse-AI logs it and moves on - it never breaks the analysis.

SongCounter uses this exact hook to count the songs of the latest analysis run and show the last song's full payload on its page.

### Use an extra pip package

**Extra pip packages work only on the container image (Docker / Kubernetes). The Windows, macOS and Linux standalone builds cannot install them: there a plugin with `requirements` is marked "incompatible".**

If you need a library that is not built in, add it to the top-level `requirements` list in `plugin.json` (this is where SongCounter lists `matplotlib`). AudioMuse-AI installs it for you at install time, then you import it like normal. You can pin an exact version or a range, using the normal pip syntax:

```json
"requirements": ["matplotlib==3.9.2", "requests>=2.31,<3"]
```

When you change a pin in a new release, the update installs the new version. If a dependency fails to install, the install still completes but tells you right away in the response, and the plugin shows a `deps_failed` status until it works.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

This works on Docker and Kubernetes when pip installs are allowed, which is the default (an admin can turn them off with `PLUGIN_ALLOW_PIP=false`). The standalone builds cannot install extra packages - see "What works on each build".

### Choose where the plugin runs (Flask or Worker)

By default a plugin is installed on both the Flask (web) container and the Worker (batch) container. If your plugin only adds pages and menus (Flask) or only adds tasks and cron jobs (Worker), set a top-level `targets` list in `plugin.json` so the other container never downloads the code or installs pip packages it will not use.

Use `["flask"]` for a page-only plugin, `["worker"]` for a task or cron-only plugin, or leave `targets` out to run on both - SongCounter leaves it out, because its page runs on Flask and its analysis hook runs on the worker. This matters most when the worker container has no internet access: a Flask-only plugin then does not try (and fail) to reach GitHub or PyPI from the worker.

## The plugin lifecycle

Understanding when your code runs helps you put it in the right place.

* At every start of AudioMuse-AI, each container the plugin targets imports your `__init__.py` and calls `register(ctx)`. Keep both fast: do not query the database or the network at import time. Register things, nothing more.
* `ctx.on_install(func)` runs at install time and again on every update. Use it for tables (see "Store your own data in a table").
* `ctx.on_flask_start(func)` and `ctx.on_worker_start(func)` run once per start, on the web or worker container, after your plugin has loaded. Use them for warm-up work such as filling a cache or starting a background thread.

If a hook raises an error, it is logged and the start continues.

### When something goes wrong

A plugin can never stop AudioMuse-AI from starting. If your module fails to import, or `register(ctx)` raises, the plugin is marked **error** on the Installed tab with a short message, and the full trace goes to the container log. Every other plugin and the core app keep working. Fix the code, publish or reinstall, and restart.

Two smaller safety nets: a menu item that points at an endpoint that does not exist is hidden (with a warning in the log), and a cron entry for a plugin that was uninstalled or disabled is skipped with a warning instead of failing.

## Test your plugin locally

You do not need the community repository or the build workflow to try your plugin. You can serve a small catalog from your own machine.

1. Zip your plugin's files (code only: `__init__.py` and anything else, but no `plugin.json` inside the zip).
2. Put the zip, your `plugin.json`, and a `manifest.json` in one folder.
3. In `plugin.json`, add `sourceUrl` to the version entry yourself, pointing at the zip. Leave `checksum` out - it is optional, and without it the download is accepted as-is.
4. Serve the folder: `python -m http.server 8000`.
5. In AudioMuse-AI, open Plugins > Repositories and add `http://<your-ip>:8000/manifest.json`. Use an address the container can reach, not `localhost`.
6. Install your plugin from the Catalog tab and apply the restart.

A minimal `manifest.json` looks like this:

```json
{
  "plugins": [
    {
      "id": "my_plugin",
      "name": "MyPlugin",
      "author": "me",
      "description": "Testing.",
      "pluginUrl": "http://<your-ip>:8000/plugin.json"
    }
  ]
}
```

And the version entry in your `plugin.json`:

```json
"versions": [
  {
    "version": "0.1.0",
    "min_core_version": "2.5.0",
    "changelog": "Local test.",
    "sourceUrl": "http://<your-ip>:8000/my_plugin.zip"
  }
]
```

To test a code change, rebuild the zip and click **Reinstall** on the Catalog tab, then apply the restart. You do not need to bump the version while testing. If you edit `plugin.json` itself, click **Refresh catalog** so the change is picked up. Keep `min_core_version` at or below the version you are running, or the plugin will not appear in the catalog.

## How updates work

To release a new version, add a new entry at the top of the `versions` list in your `plugin.json`, with a higher `version`, its `min_core_version` and a `changelog`. The build workflow fills in the zip and checksum, as always. Never change the code of a version that is already published: the workflow refuses to rebuild it, because installed copies re-download it by checksum. Always add a new entry with a bumped version.

Some rules to keep in mind:

* Versions are compared as numbers, part by part: `1.10.0` is newer than `1.9.0`. A leading `v` is ignored.
* Each AudioMuse-AI instance picks the newest release its own core version supports. Users on an older core keep getting your last release that still supports them, so you can raise `min_core_version` without breaking them.
* An update is a fresh install of the new version: the old code is replaced completely.
* Your `on_install` hooks run again on every update (and on a reinstall). Write them so they can run twice without harm - use `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, and so on.

What survives an update: the plugin's settings, its data tables, its Scheduled Tasks entries, and its enabled or disabled state (a disabled plugin stays disabled after an update). Only the code and the manifest are replaced.

## API reference

Everything below comes from `plugin.api`. The one exception is the media-server helper shown earlier, which you import from `tasks.mediaserver`.

| Import | What it does |
|---|---|
| `get_db()` | A normal database connection. Run any query. |
| `get_score_data_by_ids(ids)` | Song details (title, author, tempo, key, mood, energy, ...) for a list of song ids. |
| `get_tracks_by_ids(ids)` | The same details plus each song's analysis embedding. |
| `get_setting(key, default)` / `set_setting(key, value)` | Read and write your plugin's settings. Values must be JSON-friendly. |
| `table(name)` | Your safe table name, `plugin_<your id>__<name>`. |
| `enqueue(func, *args, queue='default')` | Run a function on the worker in the background. |
| `save_task_status(...)` and the `TASK_STATUS_*` constants | Report progress of a long job so it shows under Active Tasks. |
| `render_page(body, title)` | Wrap your HTML in the AudioMuse-AI layout with the app menu. |
| `manage_plugins_url()` | URL of the Manage Plugins page. Good redirect target after saving settings. |
| `logger` | Your log channel. Output goes to the container logs. |
| `config` | Read-only access to the app configuration values. |

And the methods on the `ctx` object in `register(ctx)`:

| Method | What it does |
|---|---|
| `add_blueprint(bp)` | Mount your Flask blueprint at `/plugins/<your id>/`. One blueprint per plugin. |
| `add_menu_item(label, endpoint, admin_only=False)` | Add a link to the app menu. |
| `set_settings_page(endpoint)` | Point the Settings button at your own page (only needed when the route is not called `settings`). |
| `add_cron_task(name, func, queue='default')` | Register a task the admin can schedule as `plugin.<your id>.<name>`. |
| `add_task(name, func, queue='default')` | Register a named worker task. It appears on the Scheduled Tasks page like a cron task, so the admin can schedule it or run it now. |
| `on_install(func)` | Run once at install and on every update. Gets the database connection. |
| `on_flask_start(func)` / `on_worker_start(func)` | Run at every start of that container, after the plugin loads. |
| `register_onnx_provider(name, options, position)` | Advanced: offer an extra ONNX Runtime execution provider (for example a GPU) for analysis. |

## Who can see and manage plugins

AudioMuse-AI keeps a clear line between managing plugins and using them.

* The Manage Plugins page and every install, update, uninstall, enable, disable, settings and apply action are admin only. A normal user cannot reach them.
* An installed plugin's own pages (under `/plugins/<your id>/`) are open to any logged-in user, so a plugin page is a normal feature of the app.
* A plugin's settings page is admin only, even though it lives under the same `/plugins/<your id>/` path. AudioMuse-AI recognises it by its endpoint, not by its URL.

If your plugin has a menu item that only admins should see, pass `admin_only=True` when you add it. Non-admin users never see the link.

```python
def register(ctx):
    ctx.add_blueprint(bp)
    ctx.add_menu_item("Admin Report", "my_plugin.report", admin_only=True)
```

Keep in mind that `admin_only` hides the menu link only. The page URL under `/plugins/<your id>/` stays reachable by any logged-in user. The one page AudioMuse-AI gates to admins for you is the settings page.

## What works on each build

Nearly everything a plugin can do works the same on Docker, Kubernetes and the Windows and macOS standalone builds. The only real difference is extra pip packages.

| Capability | Docker / Kubernetes | Windows / macOS standalone |
|---|---|---|
| Pages, menu items, settings page | Yes | Yes |
| Read and write the database, own tables | Yes | Yes |
| Per-plugin settings | Yes | Yes |
| Create playlists on the media server | Yes | Yes |
| Cron tasks and worker tasks | Yes | Yes |
| Built-in libraries (Flask, numpy, psycopg2, onnxruntime, redis, rq, standard library) | Yes | Yes |
| Extra pip packages (`requirements`) | Yes, when `PLUGIN_ALLOW_PIP` is true (the default) | No, the plugin is marked "incompatible" |

## Configuration for admins

The plugin system works out of the box. These environment variables let an admin change its behavior:

| Variable | Default | What it does |
|---|---|---|
| `PLUGINS_ENABLED` | `true` | Master switch. Set `false` to turn the whole plugin system off, including the Plugins page. |
| `PLUGIN_ALLOW_PIP` | `true` | Allow plugins to install their extra pip packages. Set `false` to only allow plugins that use built-in libraries. |
| `PLUGINS_DIR` | build-specific | Where plugin code and its pip packages live. Mount a persistent volume here. If it is empty at start, plugins are re-downloaded automatically. |
| `PLUGIN_DEFAULT_REPO_URL` | community catalog | The catalog that is always present. Point it at your own `manifest.json` to replace the community one. |
| `PLUGIN_MAX_DOWNLOAD_MB` | `50` | Maximum size of one plugin download. |
| `PLUGIN_CATALOG_REFRESH_INTERVAL` | `3600` | How often (seconds) the catalog is checked for new versions in the background. |
| `PLUGIN_HTTP_FORCE_IPV4` | `true` | Use IPv4 for plugin downloads. Set `false` only on an IPv6-only host. |
| `PLUGIN_HTTP_CONNECT_TIMEOUT` / `PLUGIN_HTTP_READ_TIMEOUT` | `10` / `20` | Seconds to wait when connecting to and reading from a plugin repository. Raise them on a very slow network. |
| `PLUGIN_HTTP_RETRIES` / `PLUGIN_HTTP_BACKOFF` | `4` / `0.5` | How many times a failed download is retried and how fast the wait between tries grows. |

On Kubernetes and Docker the Flask and worker containers each keep their own plugins volume; there is nothing to share between them.

## Troubleshooting

### What the status badge means

Each plugin on the Installed tab has a status badge:

* **ok** - the plugin loaded and is running.
* **error** - the plugin failed to load. A short message shows under the plugin and tells you which container failed (for example "failed on worker: ..."); the full error is in that container's logs. The rest of AudioMuse-AI keeps working.
* **incompatible** - this version needs a newer AudioMuse-AI core, or it needs extra pip packages on a build that cannot install them (a standalone build, or `PLUGIN_ALLOW_PIP=false`).
* **deps_failed** - the plugin code is installed but its pip dependencies could not be satisfied. The message under the plugin shows the reason. This can also happen when two plugins pin conflicting versions of the same package: they share one library folder, so only one pin can win. The plugin still runs, but check that it behaves correctly.
* **pending** - the plugin has not been loaded yet. Apply the restart.

### Where the logs are

Plugin errors never show a full trace in the browser. Look in the container logs instead: the Flask (web) container for pages and installs, the worker container for tasks and cron jobs. On Docker use `docker logs <container>`, on Kubernetes use `kubectl logs <pod>`.

### A plugin is missing from the Catalog

The catalog only shows versions your core can run. If every release of a plugin needs a newer core than yours, the plugin does not appear at all. Update AudioMuse-AI and refresh the catalog.

### Repository or download errors

Downloads retry by themselves with backoff, and GitHub files are also tried from a second CDN (jsDelivr) when GitHub does not answer. The catalog keeps the last good copy, so a short outage does not empty the page. If downloads keep failing, check that the container can reach the internet. By default outbound plugin traffic uses IPv4 only, because many containers have a broken IPv6 path; on an IPv6-only host set `PLUGIN_HTTP_FORCE_IPV4=false`.

### The plugins volume was wiped

Plugin code lives on a volume, but the database remembers every installed plugin and where it came from. If the volume is empty at start, AudioMuse-AI re-downloads each plugin by itself and logs a warning. If a plugin cannot be re-downloaded, reinstall it from the Catalog.

### A plugin page gives 404 right after install

You have not applied the restart yet (see "Apply your changes").
