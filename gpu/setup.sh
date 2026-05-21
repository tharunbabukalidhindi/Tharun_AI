#!/bin/bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   AI VIDEO AGENT — GPU Instance Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Install requirements
echo "→ Installing system requirements..."
pip install -r gpu/requirements.txt --quiet
echo "✓ Python dependencies installed"

# Create checkpoint folders
echo "→ Creating models weight folders..."
mkdir -p checkpoints/musetalk
mkdir -p checkpoints/whisper
mkdir -p checkpoints/dwpose
mkdir -p checkpoints/emage

echo "✓ Folders created under checkpoints/"
echo ""
echo "💡 To configure MuseTalk, copy weights files to checkpoints/:"
echo "   - checkpoints/musetalk/musetalk.json"
echo "   - checkpoints/musetalk/pytorch_model.bin"
echo "   - checkpoints/whisper/tiny.pt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   GPU Setup Complete!"
echo "   Start the GPU Server with:"
echo "   python gpu/main.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
