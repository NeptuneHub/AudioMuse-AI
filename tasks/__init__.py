# tasks/__init__.py
#
# Intentionally empty: this package does NOT eagerly import its submodules.
#
# Why: a previous version eagerly re-exported ``run_analysis_task``,
# ``run_clustering_task``, ``score_vector`` and ``sync_collections_task`` so
# they could be imported as ``from tasks import run_analysis_task``. No
# production code relied on that — RQ enqueues by string path and every other
# caller uses fully-qualified module paths (``tasks.analysis``,
# ``tasks.clustering``, ...).
#
# Those eager imports created a circular-import window during ``config.py``'s
# own load: ``config.py`` does ``from tasks.setup_manager import SetupManager``
# which first runs *this* package init; if that init pulls in ``tasks.analysis``
# etc., and any of those modules do ``from config import X`` for a value that
# ``config.py`` later overrides from the DB (line ~620), the importing module
# freezes its binding at the env default and silently drifts out of sync with
# user-saved settings (originally surfaced via issue #467).
#
# Keeping this file empty closes that window for every DB-overridable config
# key. Submodules must be imported by their full path.

