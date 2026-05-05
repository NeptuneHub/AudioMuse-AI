# Intentionally empty.
#
# This package previously eagerly re-imported submodules so they could be
# accessed as `from tasks import run_analysis_task`. That created a
# circular-import window during config.py's own load:
#
#   config.py -> from tasks.setup_manager import SetupManager
#     -> Python first runs tasks/__init__.py (package init)
#       -> from .analysis import run_analysis_task
#         -> tasks/analysis.py -> from ai import ...
#           -> ai.py -> from config import AI_MODEL_PROVIDER
#             -> config is partially loaded (env defaults set, DB
#                overlay not yet run); Python returns the partial
#                module, and ai.AI_MODEL_PROVIDER gets frozen at the
#                env default, permanently out-of-sync with the value
#                config.py settles on a moment later (issue #467
#                follow-up).
#
# Importing nothing here breaks the circular chain. All callers use
# fully-qualified paths (`from tasks.analysis import run_analysis_task`,
# `tasks.clustering.run_clustering_task` for RQ enqueue, etc.) so this
# is non-breaking.
