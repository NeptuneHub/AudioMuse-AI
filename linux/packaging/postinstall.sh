#!/bin/sh
# Post-install for the AudioMuse-AI native Linux package (deb + rpm).
# Refresh the desktop-entry and icon caches so the launcher shows up
# immediately. All steps are best-effort: a missing tool must never fail the
# install (headless servers may have none of these).
set -e

# Make sure the bundled launcher is executable (cp -a preserves it, but be safe).
if [ -f /opt/AudioMuse-AI/AudioMuse-AI ]; then
    chmod +x /opt/AudioMuse-AI/AudioMuse-AI 2>/dev/null || true
fi

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications 2>/dev/null || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor 2>/dev/null || true
fi

echo "AudioMuse-AI installed. Launch it from your application menu, or run:"
echo "    audiomuse-ai start"
echo "Then open http://127.0.0.1:8000"

exit 0
