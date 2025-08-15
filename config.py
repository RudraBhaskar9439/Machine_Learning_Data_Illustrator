"""
Configuration file for ML Illustrator application
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for ML Illustrator"""
    
    # Application settings
    APP_NAME = "ML Illustrator"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "A comprehensive machine learning web application"
    
    # Page configuration
    PAGE_TITLE = "🤖 ML Illustrator"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Data settings
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    SUPPORTED_FILE_TYPES = ['csv']
    DEFAULT_TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Model settings
    DEFAULT_HYPERPARAMETERS = {
        'Logistic Regression': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear'
        },
        'Random Forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'XGBoost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
    }
    
    # Visualization settings
    PLOT_HEIGHT = 800
    PLOT_WIDTH = None  # Auto
    COLOR_SCHEME = "plotly"
    
    # Google Colab settings
    COLAB_TIMEOUT = 300  # 5 minutes
    COLAB_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Performance settings
    MAX_SAMPLES_FOR_VISUALIZATION = 10000
    ENABLE_CACHING = True
    CACHE_TTL = 3600  # 1 hour
    
    # UI settings
    THEME = {
        'primaryColor': '#1f77b4',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6',
        'textColor': '#262730'
    }
    
    # CSS styles
    CUSTOM_CSS = """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .model-section {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #c3e6cb;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #f5c6cb;
        }
        .warning-message {
            background-color: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #ffeaa7;
        }
    </style>
    """
    
    @classmethod
    def get_model_options(cls) -> Dict[str, Any]:
        """Get available model options"""
        return {
            'classification': [
                'Logistic Regression',
                'Random Forest',
                'SVM',
                'Decision Tree',
                'K-Nearest Neighbors',
                'Gradient Boosting',
                'XGBoost',
                'LightGBM',
                'CatBoost',
                'Naive Bayes'
            ],
            'regression': [
                'Linear Regression',
                'Ridge Regression',
                'Lasso Regression',
                'Random Forest',
                'SVR',
                'Decision Tree',
                'K-Nearest Neighbors',
                'Gradient Boosting',
                'XGBoost',
                'LightGBM',
                'CatBoost'
            ]
        }
    
    @classmethod
    def get_hyperparameter_options(cls, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter options for a specific model"""
        if model_name == 'Logistic Regression':
            return {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == 'Random Forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'SVM':
            return {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_name == 'Linear Regression':
            return {}
        else:
            return {}
    
    @classmethod
    def get_metrics(cls, problem_type: str) -> Dict[str, str]:
        """Get metrics for a specific problem type"""
        if problem_type == 'classification':
            return {
                'accuracy': 'Accuracy Score',
                'precision': 'Precision Score',
                'recall': 'Recall Score',
                'f1_score': 'F1 Score'
            }
        else:  # regression
            return {
                'mse': 'Mean Squared Error',
                'rmse': 'Root Mean Squared Error',
                'mae': 'Mean Absolute Error',
                'r2_score': 'R-squared Score'
            }
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'platform': os.sys.platform,
            'app_version': cls.APP_VERSION,
            'max_file_size': cls.MAX_FILE_SIZE,
            'supported_file_types': cls.SUPPORTED_FILE_TYPES
        }

# Create a global config instance
config = Config()
