#!/bin/bash
set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   AI VIDEO AGENT — Local Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python version
PYTHON=$(which python3)
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "✓ Using $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment..."
  $PYTHON -m venv venv
  echo "✓ Virtual environment created"
else
  echo "✓ Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install dependencies
echo "→ Installing dependencies (this may take a minute)..."
pip install -r local/requirements.txt --quiet
echo "✓ All dependencies installed"

# Create .env if it doesn't exist
if [ ! -f "local/.env" ]; then
  cp local/.env.example local/.env
  echo "✓ Created local/.env from template"
  echo ""
  echo "⚠️  IMPORTANT: Add your API keys to local/.env before running"
else
  echo "✓ local/.env already exists"
fi

# Create persona directory
mkdir -p persona
if [ ! -f "persona/persona.jpg" ]; then
  echo ""
  echo "⚠️  Place your persona image at: persona/persona.jpg"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Setup Complete!"
echo ""
echo "   To start the dashboard:"
echo "   source venv/bin/activate"
echo "   python local/app.py"
echo ""
echo "   Then open: http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
