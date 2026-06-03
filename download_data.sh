#!/bin/bash
set -e

# Create directories
mkdir -p data/ml-1m
mkdir -p data/processed
mkdir -p saved_models
mkdir -p src/data src/models src/training src/evaluation src/inference

# Download MovieLens 1M dataset if not exists
if [ ! -f "data/ml-1m/ratings.dat" ]; then
    echo "Downloading MovieLens 1M dataset..."
    curl -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
    echo "Unzipping dataset..."
    unzip -q ml-1m.zip
    # Move files to data/ml-1m/
    mv ml-1m/* data/ml-1m/
    # Clean up
    rm -rf ml-1m ml-1m.zip
    echo "Dataset downloaded and unzipped successfully."
else
    echo "Dataset already exists."
fi
