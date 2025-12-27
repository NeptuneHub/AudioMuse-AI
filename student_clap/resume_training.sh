#!/bin/bash
# Quick resume script for Student CLAP training

cd /Users/guidocolangiuli/Music/AudioMuse-AI/student_clap
source venv/bin/activate

echo "🔍 Looking for latest checkpoint..."

# Check if latest checkpoint exists
if [ -f "checkpoints/latest.pth" ]; then
    echo "✅ Found latest checkpoint: checkpoints/latest.pth"
    python3 train_real.py --config config.yaml --resume checkpoints/latest.pth
elif [ -f "checkpoints/best_model_epoch_*.pth" ]; then
    # Find the latest best model
    latest_best=$(ls -t checkpoints/best_model_epoch_*.pth | head -1)
    echo "✅ Found latest best checkpoint: $latest_best"
    python3 train_real.py --config config.yaml --resume "$latest_best"
else
    echo "❌ No checkpoints found. Starting fresh training..."
    python3 train_real.py --config config.yaml
fi