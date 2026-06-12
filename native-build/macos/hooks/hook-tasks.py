"""PyInstaller hook: collect every ``tasks.*`` submodule.

RQ deserializes job functions by dotted path (e.g. ``tasks.analysis.run_analysis_task``)
at run time, so static import analysis never sees them. Without this the frozen
workers import fine but every enqueued job fails with ModuleNotFoundError.
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("tasks")
