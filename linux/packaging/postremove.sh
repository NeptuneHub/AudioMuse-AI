#!/bin/sh
# Post-remove for the AudioMuse-AI native Linux package (deb + rpm).
# Refresh the desktop-entry / icon caches after our files are gone. We do NOT
# touch the user's data dir (~/.local/share/AudioMuse-AI -- the embedded
# Postgres cluster, Redis state, logs and backups); a package removal must never
# destroy a user's analysis database. Removing it is left to the user.
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications 2>/dev/null || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor 2>/dev/null || true
fi

exit 0
