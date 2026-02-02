#!/bin/bash
# Prepare environment for building macOS installer
# Run this once before building

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   AudioMuse-AI macOS Build Preparation                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script must be run on macOS"
    exit 1
fi

# Check for Homebrew
echo "1️⃣ Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "⚠️  Homebrew not found."
    # In CI environment, skip Homebrew installation
    if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ]; then
        echo "ℹ️  Running in CI - will use hdiutil for DMG creation"
    else
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for current session
        if [[ $(uname -m) == 'arm64' ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
else
    echo "✓ Homebrew installed: $(brew --version | head -n1)"
fi
echo ""

# Check for create-dmg
echo "2️⃣ Checking create-dmg..."
if ! command -v create-dmg &> /dev/null; then
    if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ]; then
        echo "ℹ️  Running in CI - will use hdiutil fallback for DMG"
    elif command -v brew &> /dev/null; then
        echo "⚠️  create-dmg not found. Installing..."
        brew install create-dmg
    else
        echo "⚠️  create-dmg not available - will use hdiutil fallback"
    fi
else
    echo "✓ create-dmg installed"
fi
echo ""

# Check Python virtual environment
echo "3️⃣ Checking Python virtual environment..."
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "✓ Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
echo ""

# Install Python dependencies
echo "4️⃣ Installing Python dependencies..."
$PYTHON_CMD -m pip install -q --upgrade pip
$PYTHON_CMD -m pip install -q pyinstaller pillow rumps

# Check if main requirements are installed
echo "5️⃣ Checking main application dependencies..."
if ! $PYTHON_CMD -c "import flask" 2>/dev/null; then
    echo "⚠️  Main dependencies not installed. Installing..."
    
    # Detect if we should use GPU or CPU requirements
    if [ -d "requirements" ]; then
        if [[ $(uname -m) == 'arm64' ]]; then
            # Apple Silicon
            echo "   Detected Apple Silicon - installing GPU requirements..."
            if [ -f "requirements/standalone-gpu-macos.txt" ]; then
                $PYTHON_CMD -m pip install -r requirements/standalone-gpu-macos.txt
            elif [ -f "requirements/standalone.txt" ]; then
                $PYTHON_CMD -m pip install -r requirements/standalone.txt
            fi
        else
            # Intel Mac
            echo "   Detected Intel Mac - installing CPU requirements..."
            if [ -f "requirements/standalone.txt" ]; then
                $PYTHON_CMD -m pip install -r requirements/standalone.txt
            fi
        fi
    fi
else
    echo "✓ Main dependencies already installed"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅ PREPARATION COMPLETE!                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "You're ready to build the macOS installer:"
echo "  ./build_macos_installer.sh"
echo ""
echo "The build process will:"
echo "  1. Create an app icon from the logo"
echo "  2. Build AudioMuse-AI.app with PyInstaller"
echo "  3. Create a distributable .dmg installer"
echo ""
