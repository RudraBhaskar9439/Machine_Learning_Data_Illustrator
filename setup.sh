#!/bin/bash

# Setup script for Streamlit Cloud deployment
# This script will be executed during the deployment process

# Install system dependencies
apt-get update
apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libhdf5-dev \
    libnetcdf-dev

# Create sample data directory if it doesn't exist
mkdir -p sample_data

# Generate sample datasets if they don't exist
if [ ! -f "sample_data/classification_sample.csv" ]; then
    python sample_data.py
fi

echo "Setup completed successfully!"
