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
entry for it.

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

This works on Docker and Kubernetes. The Windows and macOS standalone builds
cannot install extra packages, so a plugin that lists `requirements` there is
marked as not compatible. Plugins that only use built-in libraries (Flask, numpy,
psycopg2, onnxruntime, redis, rq, and the standard library) work everywhere.
