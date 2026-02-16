#!/bin/bash
# Build macOS installer for AudioMuse-AI Standalone
# This creates a .dmg file that users can download and install

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   AudioMuse-AI macOS Installer Builder                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script must be run on macOS"
    exit 1
fi

# Variables
APP_NAME="AudioMuse-AI"
VERSION="${VERSION:-3.0.0}"  # Use VERSION env var, default to 3.0.0
BUILD_DIR="build_installer"
DIST_DIR="dist"

# Auto-detect Python from virtual environment or system
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    VENV_PYTHON="venv/bin/python"
else
    VENV_PYTHON="python3"
fi

echo "📦 Building $APP_NAME v$VERSION for macOS..."
echo "🐍 Using Python: $VENV_PYTHON ($($VENV_PYTHON --version 2>&1))"
echo ""

# Step 1: Install build dependencies
echo "1️⃣ Installing build dependencies..."
$VENV_PYTHON -m pip install -q --upgrade pip
$VENV_PYTHON -m pip install -q pyinstaller pillow rumps

# Step 1.5: Create icon from PNG
echo "1.5️⃣ Creating macOS icon..."
ICON_SOURCE="screenshot/audiomuseai.png"
ICON_OUTPUT="icon.icns"

if [ -f "$ICON_SOURCE" ]; then
    # Create iconset directory
    mkdir -p icon.iconset
    
    # Generate different icon sizes using sips (built-in macOS tool)
    sips -z 16 16     "$ICON_SOURCE" --out icon.iconset/icon_16x16.png > /dev/null 2>&1
    sips -z 32 32     "$ICON_SOURCE" --out icon.iconset/icon_16x16@2x.png > /dev/null 2>&1
    sips -z 32 32     "$ICON_SOURCE" --out icon.iconset/icon_32x32.png > /dev/null 2>&1
    sips -z 64 64     "$ICON_SOURCE" --out icon.iconset/icon_32x32@2x.png > /dev/null 2>&1
    sips -z 128 128   "$ICON_SOURCE" --out icon.iconset/icon_128x128.png > /dev/null 2>&1
    sips -z 256 256   "$ICON_SOURCE" --out icon.iconset/icon_128x128@2x.png > /dev/null 2>&1
    sips -z 256 256   "$ICON_SOURCE" --out icon.iconset/icon_256x256.png > /dev/null 2>&1
    sips -z 512 512   "$ICON_SOURCE" --out icon.iconset/icon_256x256@2x.png > /dev/null 2>&1
    sips -z 512 512   "$ICON_SOURCE" --out icon.iconset/icon_512x512.png > /dev/null 2>&1
    sips -z 1024 1024 "$ICON_SOURCE" --out icon.iconset/icon_512x512@2x.png > /dev/null 2>&1
    
    # Convert iconset to icns
    iconutil -c icns icon.iconset -o "$ICON_OUTPUT"
    rm -rf icon.iconset
    
    echo "✓ Icon created: $ICON_OUTPUT"
else
    echo "⚠️  Icon source not found: $ICON_SOURCE"
    echo "   Continuing without icon..."
    ICON_OUTPUT=""
fi
echo ""

# Step 2: Create PyInstaller spec file
echo "2️⃣ Creating PyInstaller spec file..."

# Determine icon path - properly quoted for Python
if [ -f "icon.icns" ]; then
    ICON_PATH="'icon.icns'"
else
    ICON_PATH="None"
fi

cat > audiomuse_macos.spec << EOF
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['selfcontained/launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('static/', 'static/'),
        ('templates/', 'templates/'),
        ('selfcontained/*.py', 'selfcontained/'),
        ('tasks/*.py', 'tasks/'),
        ('config.py', '.'),
        ('app.py', '.'),
        ('app_*.py', '.'),
        ('ai.py', '.'),
        # Don't bundle models - they'll be downloaded on first run
    ],
    hiddenimports=[
        'soundfile',
        'librosa',
        'resampy',
        'scipy',
        'scipy.signal',
        'scipy.fft',
        'scipy._lib.messagestream',
        'numba',
        'sklearn',
        'sklearn.cluster',
        'sklearn.mixture',
        'sklearn.preprocessing',
        'sklearn.metrics',
        'huey',
        'flask',
        'flask_cors',
        'onnxruntime',
        'numpy',
        'psycopg2',
        'duckdb',
        'voyager',
        'pydub',
        'mutagen',
        'PIL',
        'redis',
        'rq',
        'requests',
        'flask_socketio',
        'engineio',
        'socketio',
        'werkzeug',
        'werkzeug.security',
        'jinja2',
        'rumps',
        'objc',
        'PyObjCTools',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['selfcontained/pyinstaller_hook.py'],
    excludes=[
        'PyQt5',
        'PyQt6',
        'tkinter',
        'matplotlib',
        'IPython',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioMuse-AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console - logs go to system log
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=$ICON_PATH,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AudioMuse-AI',
)

app = BUNDLE(
    coll,
    name='AudioMuse-AI.app',
    icon=$ICON_PATH,
    bundle_identifier='com.audiomuse.ai',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '$VERSION',
        'CFBundleVersion': '$VERSION',
        'LSMinimumSystemVersion': '10.15.0',
        'NSHumanReadableCopyright': 'Copyright © 2026 AudioMuse-AI',
        'LSUIElement': False,  # Show in dock initially, then hide to menu bar
    },
)
EOF

echo "✓ Spec file created: audiomuse_macos.spec"
echo ""

# Step 3: Build with PyInstaller
echo "3️⃣ Building executable with PyInstaller..."
echo "   This may take 5-10 minutes..."
$VENV_PYTHON -m PyInstaller --clean --noconfirm audiomuse_macos.spec

if [ ! -d "dist/AudioMuse-AI.app" ]; then
    echo "❌ Error: Build failed - AudioMuse-AI.app not created"
    exit 1
fi

echo "✓ Executable built: dist/AudioMuse-AI.app"
echo ""

# Step 3.5: Create README for DMG
echo "3.5️⃣ Creating README for installer..."
cat > dist/README.txt << 'READMEEOF'
AudioMuse-AI for macOS
==============================

INSTALLATION:
1. Drag "AudioMuse-AI.app" to your Applications folder
2. Double-click AudioMuse-AI in Applications to launch

   If macOS prevents launching the app due to Gatekeeper/quarantine, run:
   sudo xattr -rd com.apple.quarantine "/Applications/AudioMuse-AI.app" && open "/Applications/AudioMuse-AI.app"

3. A menu bar icon (♪) will appear in the top-right menu bar
4. On first run, models will be downloaded (~1.7 GB)
5. Click the menu bar icon and select "Open Web Interface"

HOW TO USE:
- Launch: Double-click AudioMuse-AI.app
- Menu Bar: Click the ♪ icon in the top-right corner
- Web Interface: Click menu → "Open Web Interface"
- View Logs: Click menu → "View Logs"
- Configuration: Click menu → "Open Configuration"
- Quit: Click menu → "Quit AudioMuse-AI"

MENU BAR OPTIONS:
♪ AudioMuse-AI
  ├─ Open Web Interface (opens http://localhost:8000)
  ├─ View Logs (opens log file)
  ├─ Open Configuration (edit config.ini)
  ├─ Open Data Folder (opens ~/.audiomuse/)
  ├─ Server Status: Running
  └─ Quit AudioMuse-AI

REQUIREMENTS:
- macOS 10.15 (Catalina) or later
- 4 GB RAM minimum (8 GB recommended)
- 5 GB free disk space (for models and cache)
- Internet connection (for first-time setup)

MEDIA SERVER SETUP:
Configure your media server (Jellyfin, Emby, Plex, or Navidrome):
Click menu bar icon → "Open Web Interface" → Setup

LOG FILES:
Application logs: ~/.audiomuse/audiomuse.log
Database: ~/.audiomuse/audiomuse.duckdb
Models: ~/.audiomuse/model/
Config: ~/.audiomuse/config.ini

FEATURES:
- AI-powered music analysis and similarity search
- Automatic playlist generation
- Music collection visualization
- Artist similarity clustering
- Mood and genre detection

SUPPORT:
- GitHub: https://github.com/yourusername/AudioMuse-AI
- Issues: https://github.com/yourusername/AudioMuse-AI/issues

Copyright © 2026 AudioMuse-AI
READMEEOF

echo "✓ README created"
echo ""

# Step 4: Check if create-dmg is installed
echo "4️⃣ Creating DMG installer..."
if ! command -v create-dmg &> /dev/null; then
    echo "⚠️  create-dmg not found. Installing with Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Error: Homebrew not installed. Please install from https://brew.sh"
        echo ""
        echo "You can still use the .app file in dist/AudioMuse-AI.app"
        echo "Or manually create DMG with Disk Utility"
        exit 1
    fi
    brew install create-dmg
fi

# Step 5: Create DMG
DMG_NAME="AudioMuse-AI-${VERSION}-macOS.dmg"
rm -f "$DMG_NAME"  # Remove old DMG if exists

# Set icon parameter
if [ -f "icon.icns" ]; then
    ICON_PARAM="--volicon icon.icns"
else
    ICON_PARAM=""
fi

# Try create-dmg if available, otherwise use hdiutil
if command -v create-dmg &> /dev/null; then
    echo "Using create-dmg for professional DMG..."
    
    create-dmg \
      --volname "AudioMuse-AI" \
      $ICON_PARAM \
      --window-pos 200 120 \
      --window-size 800 450 \
      --icon-size 100 \
      --icon "AudioMuse-AI.app" 200 190 \
      --hide-extension "AudioMuse-AI.app" \
      --app-drop-link 600 185 \
      --text-size 12 \
      --add-file "README.txt" "dist/README.txt" 200 350 \
      "$DMG_NAME" \
      "dist/AudioMuse-AI.app" 2>&1 || {
        echo "⚠️  create-dmg advanced options failed, trying simpler version..."
        
        # Simplified create-dmg (without README)
        create-dmg \
          --volname "AudioMuse-AI" \
          --icon "AudioMuse-AI.app" 200 190 \
          --app-drop-link 600 185 \
          "$DMG_NAME" \
          "dist/AudioMuse-AI.app" 2>&1 || {
            echo "⚠️  create-dmg failed completely, falling back to hdiutil..."
            USE_HDIUTIL=1
        }
    }
else
    echo "create-dmg not available, using hdiutil..."
    USE_HDIUTIL=1
fi

# Fallback: Create simple DMG with hdiutil
if [ "$USE_HDIUTIL" = "1" ]; then
    echo "Creating DMG with hdiutil (simple method)..."
    TMP_DIR="dmg_contents"
    mkdir -p "$TMP_DIR"
    cp -R "dist/AudioMuse-AI.app" "$TMP_DIR/"
    cp "dist/README.txt" "$TMP_DIR/" 2>/dev/null || true
    
    # Create symbolic link to Applications
    ln -s /Applications "$TMP_DIR/Applications"
    
    hdiutil create -volname "AudioMuse-AI" -srcfolder "$TMP_DIR" -ov -format UDZO "$DMG_NAME"
    rm -rf "$TMP_DIR"
fi

if [ -f "$DMG_NAME" ]; then
    echo "✓ DMG created: $DMG_NAME"
    echo ""
    
    # Get file size
    SIZE=$(du -h "$DMG_NAME" | cut -f1)
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║   ✅ BUILD SUCCESSFUL!                                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "📦 Installer: $DMG_NAME"
    echo "💾 Size: $SIZE"
    echo ""
    echo "To test:"
    echo "  1. Open the DMG: open $DMG_NAME"
    echo "  2. Drag AudioMuse-AI.app to Applications"
    echo "  3. Run from Applications folder"
    echo "  4. Open browser to http://localhost:8000"
    echo ""
    echo "To distribute:"
    echo "  - Upload $DMG_NAME to GitHub releases"
    echo "  - Users download, mount DMG, drag to Applications"
    echo "  - First run downloads models (~1.7 GB)"
    echo ""
    
    # Cleanup
    echo "Cleaning up build artifacts..."
    rm -f audiomuse_macos.spec
    rm -rf icon.iconset
    echo "✓ Cleanup complete"
    echo ""
    echo "Build artifacts preserved in: dist/AudioMuse-AI.app"
    
else
    echo "❌ Error: DMG creation failed"
    echo ""
    echo "You can still use: dist/AudioMuse-AI.app"
    echo "Or create DMG manually with Disk Utility"
    exit 1
fi
