#!/bin/bash

# Exit on error
set -e

# Detect OS
IS_MAC=false
IS_LINUX=false
IS_WINDOWS=false

case "$(uname -s)" in
    Darwin)
        IS_MAC=true
        ;;
    Linux)
        IS_LINUX=true
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        IS_WINDOWS=true
        ;;
    *)
        echo "❌ Unsupported OS: $(uname -s)"
        exit 1
        ;;
esac

# Set desired Python version
PYTHON_VERSION="3.10.9"
VENV_DIR=".venv"

# Use pyenv if available to ensure the right Python version
if command -v pyenv &>/dev/null; then
    echo "🔍 Using pyenv to select Python $PYTHON_VERSION"
    pyenv install -s $PYTHON_VERSION
    pyenv local $PYTHON_VERSION
fi

# Determine python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.x first."
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment in $VENV_DIR/"
$PYTHON_CMD -m venv "$VENV_DIR"

# Activate virtual environment
echo "🐍 Activating virtual environment"
if $IS_WINDOWS; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip properly
echo "⬆️  Upgrading pip"
if $IS_WINDOWS; then
    "$VENV_DIR/Scripts/python.exe" -m pip install --upgrade pip
else
    pip install --upgrade pip
fi

# Install PIXAL in editable mode with correct extras
if $IS_MAC; then
    echo "🍎 Installing PIXAL with macOS dependencies"
    pip install -e '.[mac]'
elif $IS_WINDOWS || $IS_LINUX; then
    echo "🖥️ Installing PIXAL with GPU dependencies"
    pip install -e '.[gpu]'
fi

echo "✅ PIXAL setup complete!"
echo "👉 To activate your environment later, run:"
if $IS_WINDOWS; then
    echo "   source $VENV_DIR/Scripts/activate"
else
    echo "   source $VENV_DIR/bin/activate"
fi
