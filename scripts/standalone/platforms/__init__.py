"""Per-platform ``prepare(ctx)`` / ``package(ctx)`` hooks for the standalone build.

``build.py`` owns the shared steps (clean, version, PyInstaller); each module here
implements only the genuinely platform-specific parts: Windows zip, macOS
codesign + ditto, Linux nfpm staging + .deb/.rpm.
"""
