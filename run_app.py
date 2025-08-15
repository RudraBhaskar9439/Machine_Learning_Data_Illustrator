#!/usr/bin/env python3
"""
ML Illustrator - Startup Script
This script helps you run the ML Illustrator application with proper setup.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"✅ {package} is installed")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Error installing packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def create_sample_data():
    """Create sample datasets if they don't exist"""
    if not os.path.exists('sample_data'):
        print("📊 Creating sample datasets...")
        try:
            from sample_data import create_sample_datasets
            create_sample_datasets()
            print("✅ Sample datasets created!")
        except ImportError:
            print("⚠️ Could not create sample datasets. You can create them manually by running: python sample_data.py")
    else:
        print("✅ Sample datasets already exist")

def main():
    """Main function to run the ML Illustrator application"""
    print("🤖 ML Illustrator - Starting up...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("🚀 Starting ML Illustrator application...")
    print("📱 The application will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Run the Streamlit application
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
