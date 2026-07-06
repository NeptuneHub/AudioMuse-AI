# AudioMuse-AI Plugin System

The plugin system lets the community extend AudioMuse-AI without changing core
code. A plugin is a small pure-Python package with a manifest. Admins install,
update, enable/disable, and remove plugins from the web UI (Plugins in the
sidebar), Jellyfin-style, by pointing at a repository catalog.

Plugins run in-process with full application and database permissions. There is
no sandbox. Only install plugins you trust. This is the same trust model as
Jellyfin: the admin installs at their own risk.

## How it works

- The canonical store is the `plugins` table in Postgres (manifest, the code zip
  as bytes, checksum, requirements, enabled flag, and per-plugin settings). This
  is the only thing shared between the Flask (web) and worker containers, so no
  shared filesystem volume is required.
- At boot, every process (web and both workers) rebuilds a local cache under
  `PLUGINS_DIR` from that table, then imports each enabled plugin through the
  `audiomuse_plugins` namespace package and calls its `register(ctx)`.
- A plugin that fails to import or register is isolated: the error is recorded
  and shown in the UI, and the app still boots.
- Installing, enabling/disabling, or removing a plugin changes the DB and asks
  you to restart to apply. The restart is one click (Apply now) and reaches all
  containers via Redis.

## Package layout (what goes in the zip)

```
your_plugin/
  plugin.json          required manifest
  __init__.py          exposes register(ctx)
  tasks.py             optional worker/batch functions
  config.py            optional per-plugin default constants
  templates/           optional plugin-owned Jinja templates
  static/              optional plugin-owned static files
```

Zip the folder so that `plugin.json` is at the zip root (or inside a single
top-level folder). Absolute paths and `..` entries are rejected on extraction.

## plugin.json

```json
{
  "id": "your_plugin",
  "name": "Your Plugin",
  "version": "1.0.0",
  "min_core_version": "2.5.0",
  "author": "you",
  "description": "What it does.",
  "requirements": []
}
```

- `id` must match `^[a-z][a-z0-9_]{1,63}$` and is used for URLs, the namespace,
  and table prefixes.
- `min_core_version` is compared against the running app version; an
  incompatible plugin is skipped and flagged.
- `requirements` is a list of pip specs (see Dependencies).

## The register(ctx) contract

Your `__init__.py` must expose `def register(ctx):`. It is called on both the
web and worker processes; each activates only the parts it owns. You choose
where a component runs by which method you call.

Online (Flask / web container):
- `ctx.add_blueprint(bp)` - mounts your Flask Blueprint at `/plugins/<id>/`.
- `ctx.add_menu_item(label, endpoint, admin_only=False)` - adds a Plugins
  submenu link (endpoint is a Flask endpoint name, e.g. `your_plugin.home`).
- `ctx.on_flask_start(fn)` - runs once when the web process finishes loading.

Batch (worker container):
- `ctx.add_cron_task(name, fn, queue='default')` - schedulable from
  Administration > Scheduled Tasks with task_type `plugin.<id>.<name>`.
- `ctx.add_task(name, fn, queue='default')` - a task you enqueue yourself via
  `plugin.api.enqueue(fn, ...)`.
- `ctx.register_onnx_provider(name, options, position='before_cpu')` - adds an
  ONNX Runtime execution provider to the analysis session chain (see ONNX).
- `ctx.on_worker_start(fn)` - runs once when a worker finishes loading.

Setup:
- `ctx.on_install(fn)` - runs once at install/update, before the restart, with a
  live DB connection: `fn(db)`. Use it to create your tables.

## Using the app from a plugin

Import only from `plugin.api` (the stable surface):

- `get_db()` - the Postgres connection (inside a request or task context).
- `save_task_status(...)` and the `TASK_STATUS_*` constants for progress.
- `get_score_data_by_ids(ids)`, `get_tracks_by_ids(ids)` - read analyzed songs.
- `rq_queue_high`, `rq_queue_default`, `enqueue(fn, *args, queue=...)`.
- `config` - read-only access to the core app config.
- `logger` - a shared plugin logger.
- `get_setting(key, default)` / `set_setting(key, value)` - per-plugin settings
  stored in the DB (editable from the UI).
- `table(name)` - returns `plugin_<id>__<name>`, the namespaced table name you
  should use for any table your plugin creates.

## Owning your own database tables

Plugins may create and use their own tables freely. Always name them with
`api.table('...')` so they live under the `plugin_<id>__` namespace and never
collide with core or other plugins. Create them in an `on_install` hook with
`CREATE TABLE IF NOT EXISTS`. On uninstall the app keeps your data by default;
the admin can choose "also delete this plugin's data" to drop `plugin_<id>__*`.

## Configuration

- Core config: `from plugin.api import config` then read values (do not mutate).
- Your own defaults: ship a `config.py` with plain constants and import it
  relatively (`from . import config as plugin_config`).
- Runtime overrides: `get_setting('key', plugin_config.DEFAULT)`; the admin edits
  these from Plugins > Installed > Settings, stored in `plugins.settings`.

## Cron and background tasks

- Register a cron task with `ctx.add_cron_task('daily', fn)`. Then, in
  Administration > Scheduled Tasks, create a schedule whose task_type is
  `plugin.<id>.daily` with a standard 5-field cron expression.
- The scheduler (web process) enqueues it to the worker; the worker runs it
  inside a Flask app context, so `get_db()` and friends work.
- Report progress with `save_task_status(...)` if you want it visible in the UI.

## Dependencies

- Declare pip requirements in `requirements`. On Docker/Kubernetes they are
  installed into `PLUGINS_DIR/_lib` (added to `sys.path`) on each process at boot.
- Frozen standalone builds (Windows/macOS bundles) cannot pip-install; a plugin
  that declares requirements is flagged incompatible there. Pure-Python plugins
  that only use libraries already bundled with AudioMuse-AI (Flask, numpy,
  psycopg2, onnxruntime, redis, rq, ...) work everywhere.

## ONNX execution providers

`ctx.register_onnx_provider(name, options)` adds an execution provider to the
analysis ONNX session chain. It is only used if the installed onnxruntime build
actually exposes it (`onnxruntime.get_available_providers()`); a plugin cannot
swap the onnxruntime build itself (that is an image-level concern). For example,
shipping `onnxruntime-openvino` in an image and then registering
`OpenVINOExecutionProvider` from a plugin.

Note: swapping the analysis model itself (replacing musicnn) is not supported in
this version. The embedding dimension and mood labels are baked into the vector
index and clustering, so a real swap would require re-analyzing the whole library
and rebuilding the index. `ctx.register_analysis_provider` exists as a forward
seam only and currently does nothing.

## Publishing (repository catalog)

A repository is a static `manifest.json` hosted anywhere reachable over HTTPS
(GitHub raw works well). Admins add its URL under Plugins > Repositories. Format:

```json
{
  "plugins": [
    {
      "id": "your_plugin",
      "name": "Your Plugin",
      "description": "...",
      "author": "you",
      "imageUrl": "",
      "versions": [
        {
          "version": "1.0.0",
          "changelog": "...",
          "min_core_version": "2.5.0",
          "sourceUrl": "https://.../your_plugin_1.0.0.zip",
          "checksum": "<md5 of the zip>",
          "timestamp": "2026-07-06T00:00:00Z"
        }
      ]
    }
  ]
}
```

The `checksum` is the MD5 of the zip and is verified on download. `sourceUrl`
and the repository URL must be HTTPS and are SSRF-guarded (private IPs rejected).

## Example

See `examples/hello_world/` for a complete plugin (Flask page + menu item, an
install migration creating `plugin_hello_world__runs`, a daily cron task, an
ad-hoc task, and a setting) and `examples/hello_world_repo_manifest.json` for its
catalog entry.
