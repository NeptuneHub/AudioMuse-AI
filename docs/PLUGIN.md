# AudioMuse-AI Plugins

## Introduction

AudioMuse-AI plugins let you add your own features without touching the core app.
A plugin is a small Python package. It can add a new page, read and write the
database, talk to your media server, save settings, and run scheduled jobs. You
install and update plugins from inside AudioMuse-AI, the same way Jellyfin does.

## Working example

The best way to learn is to read a real plugin. SongCounter is a small, complete
example (a page, a settings page, and per-plugin settings):

https://github.com/NeptuneHub/AudioMuse-AI-plugins/tree/main/plugins/SongCounter

## Plugin architecture

A plugin has two files.

`plugin.json` holds the metadata. The `id` must be lowercase.

```json
{ "id": "my_plugin", "name": "My Plugin", "version": "1.0.0", "min_core_version": "2.5.0" }
```

`__init__.py` holds the code. It must have a `register` function that tells
AudioMuse-AI what to add.

```python
from flask import Blueprint
from plugin.api import render_page

bp = Blueprint("my_plugin", __name__)

@bp.route("/")
def home():
    return render_page("<p>Hello world</p>", title="My Plugin")

def register(ctx):
    ctx.add_blueprint(bp)
    ctx.add_menu_item("My Plugin", "my_plugin.home")
```

If you want a settings page, add a route called `settings`. AudioMuse-AI opens it
from the Settings button on the Manage Plugins page, so it does not add a menu
entry for it. If your settings route has a different name, point to it with
`ctx.set_settings_page("my_plugin.my_settings")`. The settings page is always
admin only.

To share your plugin, add it to the public repository, or host your own
`manifest.json` and add its link under Plugins > Repositories. You do not write
the manifest by hand; the build workflow in the plugin repository creates it for
you from your `plugin.json`.

## Main capabilities

Import everything you need from `plugin.api`. Below are the most common things a
plugin does.

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

Pick some songs from the database, then send them to your media server. This
works with Jellyfin, Emby, Plex, Navidrome and Lyrion.

```python
from plugin.api import get_db
from tasks.mediaserver import create_or_replace_playlist

def make_fast_playlist():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT item_id FROM score WHERE tempo > 120 LIMIT 50")
    track_ids = [row[0] for row in cur.fetchall()]
    cur.close()
    create_or_replace_playlist("Fast Songs", track_ids)
```

### Save and read settings

Store small values for your plugin. The admin can also edit them from the
Settings button.

```python
from plugin.api import get_setting, set_setting

set_setting("limit", 50)
limit = get_setting("limit", 20)  # 20 is the default if nothing is saved yet
```

### Store your own data in a table

Your plugin can have its own tables. Use `table()` to get a safe, unique name,
and create the table once at install time.

```python
from plugin.api import get_db, table

def migrate(db):
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS " + table("runs") + " (ran_at TIMESTAMP DEFAULT now())")
    db.commit()

def register(ctx):
    ctx.on_install(migrate)   # runs once when the plugin is installed
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

Then open Administration > Scheduled Tasks and add a schedule with the task type
`plugin.my_plugin.daily` (that is `plugin.<your id>.<task name>`).

### Use an extra pip package

If you need a library that is not built in, list it in `plugin.json`. AudioMuse-AI
installs it for you at startup, then you import it like normal. The SongCounter
example uses `matplotlib` to draw a bar chart of the counts.

```json
{ "id": "my_plugin", "name": "My Plugin", "version": "1.0.0", "min_core_version": "2.5.0", "requirements": ["matplotlib"] }
```

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

This works on Docker and Kubernetes when pip installs are allowed, which is the
default. An admin can turn them off with the `PLUGIN_ALLOW_PIP=false` environment
variable. The Windows and macOS standalone builds cannot install extra packages,
so a plugin that lists `requirements` there is marked as not compatible. Plugins
that only use built-in libraries (Flask, numpy, psycopg2, onnxruntime, redis, rq,
and the standard library) work everywhere.

### Choose where the plugin runs (Flask or Worker)

By default a plugin is installed on both the Flask (web) container and the Worker
(batch) container. If your plugin only adds pages and menus (Flask) or only adds
tasks and cron jobs (Worker), say so with a `targets` list in `plugin.json` so the
other container never downloads the code or installs pip packages it will not use.

```json
{ "id": "my_plugin", "name": "My Plugin", "version": "1.0.0", "min_core_version": "2.5.0",
  "targets": ["flask"], "requirements": ["matplotlib"] }
```

Use `["flask"]` for a page-only plugin (like SongCounter), `["worker"]` for a
task/cron-only plugin, or leave `targets` out to run on both. This matters most
when the worker container has no internet access: a Flask-only plugin then does
not try (and fail) to reach GitHub or PyPI from the worker.

## Who can see and manage plugins

AudioMuse-AI keeps a clear line between managing plugins and using them.

* The Manage Plugins page and every install, update, uninstall, enable, disable,
  settings and apply action are admin only. A normal user cannot reach them.
* An installed plugin's own pages (under `/plugins/<your id>/`) are open to any
  logged-in user, so a plugin page is a normal feature of the app.
* A plugin's settings page is admin only, even though it lives under the same
  `/plugins/<your id>/` path. AudioMuse-AI recognises it by its endpoint, not by
  its URL.

If your plugin has a menu item that only admins should see, pass `admin_only=True`
when you add it. Non-admin users never see the link.

```python
def register(ctx):
    ctx.add_blueprint(bp)
    ctx.add_menu_item("Admin Report", "my_plugin.report", admin_only=True)
```

Keep in mind that `admin_only` hides the menu link only. The page URL under
`/plugins/<your id>/` stays reachable by any logged-in user. The one page that
AudioMuse-AI gates to admins for you is the settings page.

## What works on each build

Nearly everything a plugin can do works the same on Docker, Kubernetes and the
Windows and macOS standalone builds. The only real difference is extra pip
packages.

| Capability | Docker / Kubernetes | Windows / macOS standalone |
|---|---|---|
| Pages, menu items, settings page | Yes | Yes |
| Read and write the database, own tables | Yes | Yes |
| Per-plugin settings | Yes | Yes |
| Create playlists on the media server | Yes | Yes |
| Cron tasks and worker tasks | Yes | Yes |
| Built-in libraries (Flask, numpy, psycopg2, onnxruntime, redis, rq, standard library) | Yes | Yes |
| Extra pip packages (`requirements`) | Yes, when `PLUGIN_ALLOW_PIP` is true (the default) | No, the plugin is marked "incompatible" |

If your plugin only uses built-in libraries, it runs everywhere. If it needs an
extra pip package, it runs on Docker and Kubernetes but is skipped on the
standalone builds.
