#!/bin/bash
set -e

echo "Home Credit data download script"
echo "================================"

# Check kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Check credentials
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set"
    echo "Copy .env.example to .env and fill in your credentials"
    exit 1
fi

mkdir -p data/raw

echo "Downloading Home Credit Default Risk dataset..."
kaggle competitions download -c home-credit-default-risk -p data/raw/

echo "Unzipping..."
unzip -o data/raw/home-credit-default-risk.zip -d data/raw/
rm data/raw/home-credit-default-risk.zip

echo "Registering with DVC..."
dvc add data/raw/*.csv

echo "Done. Files in data/raw/:"
ls -lh data/raw/*.csv
