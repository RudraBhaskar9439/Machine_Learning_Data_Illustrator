#!/bin/bash

# Setup script for ML Illustrator deployment on Streamlit Cloud

echo "Setting up ML Illustrator environment..."

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev

echo "System dependencies installed successfully!"

# Create sample data directory if it doesn't exist
mkdir -p sample_data

echo "Setup completed successfully!"
