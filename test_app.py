#!/usr/bin/env python3
"""
Test script for ML Illustrator application
This script tests the main components of the application.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
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
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Seaborn: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Plotly: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ XGBoost imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import XGBoost: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print("✅ LightGBM imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LightGBM: {e}")
        return False
    
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
        print("✅ CatBoost imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CatBoost: {e}")
        return False
    
    return True

def test_ml_illustrator_class():
    """Test the MLIllustrator class"""
    print("\n🔍 Testing MLIllustrator class...")
    
    try:
        # Import the class
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from app import MLIllustrator
        
        # Create instance
        ml_illustrator = MLIllustrator()
        print("✅ MLIllustrator instance created successfully")
        
        # Test data loading
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        test_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        test_data['target'] = y
        
        # Save test data temporarily
        test_file = 'test_data.csv'
        test_data.to_csv(test_file, index=False)
        
        # Test loading data
        with open(test_file, 'rb') as f:
            success = ml_illustrator.load_data(f)
            if success:
                print("✅ Data loading test passed")
            else:
                print("❌ Data loading test failed")
                return False
        
        # Test column info
        column_info = ml_illustrator.get_column_info()
        if column_info and 'columns' in column_info:
            print("✅ Column info test passed")
        else:
            print("❌ Column info test failed")
            return False
        
        # Test data preparation
        feature_columns = [col for col in test_data.columns if col != 'target']
        success = ml_illustrator.prepare_data('target', feature_columns)
        if success:
            print("✅ Data preparation test passed")
        else:
            print("❌ Data preparation test failed")
            return False
        
        # Test model options
        model_options = ml_illustrator.get_model_options()
        if model_options and len(model_options) > 0:
            print("✅ Model options test passed")
        else:
            print("❌ Model options test failed")
            return False
        
        # Test hyperparameters
        hyperparams = ml_illustrator.get_hyperparameters('Random Forest')
        if hyperparams is not None:
            print("✅ Hyperparameters test passed")
        else:
            print("❌ Hyperparameters test failed")
            return False
        
        # Test model training
        success = ml_illustrator.train_model('Random Forest', {'n_estimators': 10})
        if success:
            print("✅ Model training test passed")
        else:
            print("❌ Model training test failed")
            return False
        
        # Test model evaluation
        evaluation_results = ml_illustrator.evaluate_model()
        if evaluation_results and 'metrics' in evaluation_results:
            print("✅ Model evaluation test passed")
        else:
            print("❌ Model evaluation test failed")
            return False
        
        # Clean up
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ MLIllustrator test failed: {e}")
        return False

def test_colab_integration():
    """Test the ColabIntegration class"""
    print("\n🔍 Testing ColabIntegration class...")
    
    try:
        from colab_integration import ColabIntegration
        
        # Create instance
        colab_integration = ColabIntegration()
        print("✅ ColabIntegration instance created successfully")
        
        # Test connection status
        status = colab_integration.get_connection_status()
        if status and 'is_connected' in status:
            print("✅ Connection status test passed")
        else:
            print("❌ Connection status test failed")
            return False
        
        # Test connection (this will fail with invalid URL, but should handle gracefully)
        success = colab_integration.connect_to_colab("invalid_url")
        if not success:  # Expected to fail
            print("✅ Invalid URL handling test passed")
        else:
            print("❌ Invalid URL handling test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ColabIntegration test failed: {e}")
        return False

def test_config():
    """Test the configuration"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import config
        
        # Test basic config
        if config.APP_NAME == "ML Illustrator":
            print("✅ Basic config test passed")
        else:
            print("❌ Basic config test failed")
            return False
        
        # Test model options
        model_options = config.get_model_options()
        if 'classification' in model_options and 'regression' in model_options:
            print("✅ Model options config test passed")
        else:
            print("❌ Model options config test failed")
            return False
        
        # Test hyperparameter options
        hyperparams = config.get_hyperparameter_options('Random Forest')
        if hyperparams and 'n_estimators' in hyperparams:
            print("✅ Hyperparameter config test passed")
        else:
            print("❌ Hyperparameter config test failed")
            return False
        
        # Test metrics
        metrics = config.get_metrics('classification')
        if metrics and 'accuracy' in metrics:
            print("✅ Metrics config test passed")
        else:
            print("❌ Metrics config test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation"""
    print("\n🔍 Testing sample data creation...")
    
    try:
        from sample_data import create_sample_datasets
        
        # Create sample datasets
        create_sample_datasets()
        
        # Check if files were created
        expected_files = [
            'sample_data/classification_sample.csv',
            'sample_data/regression_sample.csv',
            'sample_data/binary_classification_sample.csv',
            'sample_data/customer_churn_sample.csv',
            'sample_data/iris_sample.csv'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} created successfully")
            else:
                print(f"❌ {file_path} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ML Illustrator - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("MLIllustrator Class Test", test_ml_illustrator_class),
        ("ColabIntegration Class Test", test_colab_integration),
        ("Configuration Test", test_config),
        ("Sample Data Creation Test", test_sample_data_creation)
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
        print("🎉 All tests passed! The application is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   python run_app.py")
        print("   or")
        print("   streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
