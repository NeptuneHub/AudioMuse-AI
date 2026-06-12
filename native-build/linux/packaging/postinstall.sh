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

echo "AudioMuse-AI installed."
echo
echo "It does not start on its own. Launch it from your application menu"
echo "(search for 'AudioMuse-AI'), or from a terminal run:"
echo "    audiomuse-ai start"
echo "then open http://127.0.0.1:8000 in your browser."
echo
echo "To start it automatically when you log in (optional):"
echo "    systemctl --user enable --now audiomuse-ai"
echo
echo "If the menu entry does not appear immediately, log out and back in."

exit 0
