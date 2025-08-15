#!/bin/bash

# ML Illustrator - Launch Script
# This script provides an easy way to run the ML Illustrator application

echo "🤖 ML Illustrator - Launch Script"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the ML-Statistics directory"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit, pandas, numpy, sklearn, matplotlib, seaborn, plotly" 2>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ All dependencies are already installed"
fi

# Create sample data if it doesn't exist
if [ ! -d "sample_data" ]; then
    echo "📊 Creating sample datasets..."
    python3 sample_data.py
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: Failed to create sample datasets"
    else
        echo "✅ Sample datasets created"
    fi
fi

# Run the application
echo ""
echo "🚀 Starting ML Illustrator..."
echo "📱 The application will open in your default web browser"
echo "🌐 URL: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

python3 -m streamlit run app.py
