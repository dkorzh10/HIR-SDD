#!/bin/bash
# Setup development environment using uv
# Usage: ./manage/setup_env.sh

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "Creating virtual environment with uv..."
uv venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "=========================================="
echo "Environment setup complete!"
echo "Activate it with: source .venv/bin/activate"
echo "=========================================="


