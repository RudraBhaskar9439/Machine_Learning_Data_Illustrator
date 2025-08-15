#!/usr/bin/env python3
"""
Simple test script to verify deployment setup
"""

import sys
import importlib

def test_imports():
    """Test all required imports"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly',
        'xgboost'
    ]
    
    print("Testing imports...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_deploy_app():
    """Test deploy_app.py import"""
    try:
        import deploy_app
        print("✅ deploy_app.py imports successfully")
        return True
    except Exception as e:
        print(f"❌ deploy_app.py import failed: {e}")
        return False

if __name__ == "__main__":
    print("=== ML Illustrator Deployment Test ===\n")
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test deploy app
    if not test_deploy_app():
        success = False
    
    print()
    
    if success:
        print("🎉 All tests passed! Ready for deployment.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        sys.exit(1)
