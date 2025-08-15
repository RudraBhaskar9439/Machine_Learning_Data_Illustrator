# 📁 ML Illustrator - Project Structure

This document explains the organization and purpose of each file in the ML Illustrator project.

## 🗂️ File Organization

```
ML-Statistics/
├── 📄 app.py                    # Main Streamlit application
├── 📄 colab_integration.py      # Google Colab integration module
├── 📄 config.py                 # Configuration settings
├── 📄 sample_data.py            # Sample dataset generation
├── 📄 run_app.py                # Python startup script
├── 📄 run.sh                    # Shell startup script
├── 📄 test_app.py               # Test suite
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 PROJECT_STRUCTURE.md      # This file
└── 📁 sample_data/              # Generated sample datasets
    ├── classification_sample.csv
    ├── regression_sample.csv
    ├── binary_classification_sample.csv
    ├── customer_churn_sample.csv
    └── iris_sample.csv
```

## 📋 File Descriptions

### 🚀 Core Application Files

#### `app.py`
- **Purpose**: Main Streamlit application file
- **Contains**: 
  - `MLIllustrator` class with all ML functionality
  - Streamlit UI components and pages
  - Data processing, model training, and evaluation logic
  - Interactive visualizations
- **Key Features**:
  - Multi-page navigation (Home, Google Colab, Data Upload, Model Config, Results)
  - Automatic problem type detection (Classification/Regression)
  - Comprehensive model evaluation with metrics and plots
  - Interactive hyperparameter configuration

#### `colab_integration.py`
- **Purpose**: Google Colab integration for remote computation
- **Contains**:
  - `ColabIntegration` class for managing Colab connections
  - Remote code execution capabilities
  - File upload/download functionality
- **Key Features**:
  - Connect to Google Colab notebooks
  - Execute code remotely
  - Transfer data between local and remote environments
  - Access to GPU/TPU resources

#### `config.py`
- **Purpose**: Centralized configuration management
- **Contains**:
  - `Config` class with all application settings
  - Model options and hyperparameters
  - UI themes and styling
  - Performance and file size limits
- **Key Features**:
  - Centralized configuration for easy maintenance
  - Model-specific hyperparameter options
  - Customizable UI themes
  - Environment information

### 🛠️ Utility Files

#### `sample_data.py`
- **Purpose**: Generate sample datasets for testing
- **Contains**:
  - `create_sample_datasets()` function
  - Multiple dataset types (classification, regression, binary)
  - Real-world like datasets (customer churn)
- **Generated Datasets**:
  - `classification_sample.csv` (Multi-class)
  - `regression_sample.csv` (Regression)
  - `binary_classification_sample.csv` (Binary)
  - `customer_churn_sample.csv` (Real-world like)
  - `iris_sample.csv` (Simple classification)

#### `run_app.py`
- **Purpose**: Python startup script with dependency checking
- **Contains**:
  - Python version validation
  - Dependency installation
  - Sample data creation
  - Application launch
- **Key Features**:
  - Automatic dependency checking and installation
  - Environment validation
  - User-friendly startup process

#### `run.sh`
- **Purpose**: Shell script for easy application launch
- **Contains**:
  - Python environment checks
  - Dependency management
  - Application startup
- **Key Features**:
  - Cross-platform compatibility
  - Automatic dependency resolution
  - Simple one-command launch

#### `test_app.py`
- **Purpose**: Comprehensive test suite
- **Contains**:
  - Import testing for all dependencies
  - MLIllustrator class testing
  - ColabIntegration testing
  - Configuration testing
  - Sample data creation testing
- **Key Features**:
  - Automated testing of all components
  - Detailed error reporting
  - Validation of application functionality

### 📚 Documentation Files

#### `README.md`
- **Purpose**: Comprehensive project documentation
- **Contains**:
  - Project overview and features
  - Installation instructions
  - Usage guide with step-by-step instructions
  - Technical details and architecture
  - Troubleshooting guide
  - Use cases and examples

#### `requirements.txt`
- **Purpose**: Python dependency specification
- **Contains**:
  - All required Python packages with versions
  - Core ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)
  - Visualization libraries (Plotly, Matplotlib, Seaborn)
  - Web framework (Streamlit)

## 🔧 Key Components

### MLIllustrator Class (`app.py`)
The core class that handles all machine learning operations:

```python
class MLIllustrator:
    def __init__(self):
        # Initialize data and model storage
    
    def load_data(self, file):
        # Load and validate CSV data
    
    def prepare_data(self, target_column, feature_columns):
        # Preprocess data for training
    
    def train_model(self, model_name, hyperparameters):
        # Train selected model
    
    def evaluate_model(self):
        # Evaluate model performance
    
    def create_visualizations(self, results):
        # Generate interactive plots
```

### ColabIntegration Class (`colab_integration.py`)
Handles Google Colab integration:

```python
class ColabIntegration:
    def connect_to_colab(self, url):
        # Connect to Google Colab
    
    def execute_remote_computation(self, code):
        # Run code on Colab
    
    def upload_data_to_colab(self, data, filename):
        # Upload files to Colab
```

## 🎯 Usage Workflow

1. **Setup**: Run `./run.sh` or `python run_app.py`
2. **Data Upload**: Upload CSV file in Data Upload page
3. **Model Configuration**: Select target, features, and model
4. **Training**: Configure hyperparameters and train model
5. **Results**: View metrics and visualizations
6. **Optional**: Connect to Google Colab for remote computation

## 🔍 Testing

Run the test suite to verify everything works:

```bash
python test_app.py
```

This will test:
- ✅ All package imports
- ✅ MLIllustrator functionality
- ✅ ColabIntegration
- ✅ Configuration
- ✅ Sample data creation

## 🚀 Deployment

The application can be deployed using:

1. **Local Development**: `streamlit run app.py`
2. **Production**: Deploy to Streamlit Cloud or similar platforms
3. **Docker**: Create a Docker container (not included)

## 📊 Supported Models

### Classification Models
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Naive Bayes

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Support Vector Regression (SVR)
- Decision Tree
- K-Nearest Neighbors
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

## 🎨 Visualizations

The application generates comprehensive visualizations using Plotly:

- **Classification**: Confusion Matrix, ROC Curve, Feature Importance
- **Regression**: Actual vs Predicted, Residuals, Feature Importance
- **General**: Prediction Distributions, Model Performance Metrics

## 🔧 Configuration

All settings are centralized in `config.py`:

- Application settings (name, version, theme)
- Data processing limits
- Model hyperparameter options
- Visualization settings
- Google Colab integration settings

This modular structure makes the application easy to maintain, extend, and customize for different use cases.
