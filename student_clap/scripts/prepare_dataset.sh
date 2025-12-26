#!/bin/bash
# Prepare Dataset for Student CLAP Training
# Downloads and caches all audio files from Jellyfin

set -e

echo "=============================================="
echo "Student CLAP Dataset Preparation"
echo "=============================================="
echo ""

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found"
    echo "Please create config.yaml from the template"
    exit 1
fi

# Create cache directory
echo "Creating cache directory..."
mkdir -p cache/audio

# Check database connection
echo "Checking database connection..."
python data/database_loader.py --config config.yaml --verify
if [ $? -ne 0 ]; then
    echo "Error: Database connection failed"
    exit 1
fi

# Check Jellyfin connection
echo ""
echo "Checking Jellyfin connection..."
python data/jellyfin_downloader.py --config config.yaml --test
if [ $? -ne 0 ]; then
    echo "Error: Jellyfin connection failed"
    exit 1
fi

# Test loading a few samples
echo ""
echo "Testing dataset loading..."
python data/dataset.py --config config.yaml --num-samples 3
if [ $? -ne 0 ]; then
    echo "Error: Dataset loading failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "✓ Dataset preparation complete"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Review cache/audio/ directory"
echo "  2. Start training with: python train.py --config config.yaml"
echo ""
