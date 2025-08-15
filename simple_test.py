#!/usr/bin/env python3
"""
Simple test script for ML Illustrator core functionality
"""

import sys
import os

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    return True

def test_ml_illustrator_basic():
    """Test basic MLIllustrator functionality"""
    print("\n🔍 Testing MLIllustrator basic functionality...")
    
    try:
        # Import the class
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from app import MLIllustrator
        
        # Create instance
        ml_illustrator = MLIllustrator()
        print("✅ MLIllustrator instance created successfully")
        
        # Test basic attributes
        if hasattr(ml_illustrator, 'data') and hasattr(ml_illustrator, 'model'):
            print("✅ MLIllustrator has required attributes")
        else:
            print("❌ MLIllustrator missing required attributes")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ MLIllustrator test failed: {e}")
        return False

def test_colab_integration_basic():
    """Test basic ColabIntegration functionality"""
    print("\n🔍 Testing ColabIntegration basic functionality...")
    
    try:
        from colab_integration import ColabIntegration
        
        # Create instance
        colab_integration = ColabIntegration()
        print("✅ ColabIntegration instance created successfully")
        
        # Test basic attributes
        if hasattr(colab_integration, 'colab_url') and hasattr(colab_integration, 'is_connected'):
            print("✅ ColabIntegration has required attributes")
        else:
            print("❌ ColabIntegration missing required attributes")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ColabIntegration test failed: {e}")
        return False

def test_config_basic():
    """Test basic configuration"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import config
        
        # Test basic config
        if config.APP_NAME == "ML Illustrator":
            print("✅ Basic config test passed")
        else:
            print("❌ Basic config test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("🧪 ML Illustrator - Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Import Test", test_basic_imports),
        ("MLIllustrator Basic Test", test_ml_illustrator_basic),
        ("ColabIntegration Basic Test", test_colab_integration_basic),
        ("Configuration Basic Test", test_config_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The application should work.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run app.py")
        print("   or")
        print("   python run_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
