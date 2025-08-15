# 🤖 ML Illustrator

A comprehensive machine learning web application built with Python and Streamlit that allows users to build, train, and evaluate machine learning models with an intuitive interface.

## 🌟 Features

### 🔗 Google Colab Integration
- Connect to Google Colab for remote computation
- Upload data to Google Colab
- Execute code remotely
- Download results from Google Colab
- Access to GPU/TPU resources

### 📊 Data Management
- Upload CSV datasets
- Automatic data exploration and analysis
- Missing value detection and visualization
- Column information and statistics
- Data type detection

### 🔧 Model Selection & Configuration
- **Classification Models:**
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

- **Regression Models:**
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

### ⚙️ Hyperparameter Tuning
- Model-specific hyperparameter configuration
- Interactive parameter selection
- Automatic problem type detection (Classification/Regression)

### 📈 Comprehensive Evaluation
- **Classification Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report
  - ROC Curve (AUC)

- **Regression Metrics:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)

### 🎨 Beautiful Visualizations
- Interactive plots using Plotly
- Confusion Matrix Heatmaps
- ROC Curves
- Feature Importance Charts
- Actual vs Predicted Scatter Plots
- Residuals Analysis
- Prediction Distributions

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ML-Statistics
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Home Page
- Overview of the application features
- List of supported models
- Getting started instructions

### 2. Google Colab Integration (Optional)
1. Go to the **Google Colab** page
2. Enter your Google Colab notebook URL
3. Click "Connect to Colab"
4. Use remote computation capabilities

### 3. Data Upload
1. Navigate to **Data Upload** page
2. Upload your CSV file using the file uploader
3. Explore dataset information:
   - Dataset shape and memory usage
   - Column data types
   - Missing values analysis
   - First 5 rows preview

### 4. Model Configuration
1. Go to **Model Configuration** page
2. Select your target column (what you want to predict)
3. Choose feature columns (input variables)
4. Set test set size (default: 20%)
5. Click "Prepare Data"
6. Select your preferred ML model
7. Configure hyperparameters (if available)
8. Click "Train Model"

### 5. Results & Visualization
1. View **Results & Visualization** page
2. Examine model performance metrics
3. Analyze interactive visualizations:
   - Confusion Matrix (Classification)
   - ROC Curve (Classification)
   - Actual vs Predicted plots
   - Feature Importance charts
   - Residuals analysis (Regression)

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python with scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

### Key Components

#### MLIllustrator Class
- Core machine learning functionality
- Data preprocessing and preparation
- Model training and evaluation
- Visualization generation

#### ColabIntegration Class
- Google Colab connection management
- Remote computation execution
- File upload/download capabilities

### Supported File Formats
- **Input**: CSV files
- **Output**: Interactive visualizations and metrics

## 🔧 Configuration

### Environment Variables
No environment variables are required for basic functionality.

### Google Colab Integration
For full Google Colab integration, you may need to:
1. Set up Google Colab API access
2. Configure authentication tokens
3. Enable necessary permissions

## 📊 Example Datasets

The application works with any CSV dataset. Here are some examples:

### Classification Datasets
- Iris dataset (flower classification)
- Titanic dataset (survival prediction)
- Breast cancer dataset (malignant/benign)
- Credit card fraud detection

### Regression Datasets
- Boston housing prices
- California housing prices
- Diabetes progression
- Wine quality prediction

## 🎯 Use Cases

### Data Scientists
- Quick model prototyping
- Hyperparameter exploration
- Model comparison
- Results visualization

### Students & Learners
- Understanding ML concepts
- Experimenting with different algorithms
- Learning about model evaluation
- Visualizing model performance

### Business Analysts
- Predictive modeling
- Data exploration
- Model deployment preparation
- Stakeholder presentations

## 🔍 Troubleshooting

### Common Issues

#### 1. Installation Problems
```bash
# If you encounter dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### 2. Memory Issues
- Use Google Colab integration for large datasets
- Reduce test set size
- Select fewer features

#### 3. Model Training Errors
- Check data quality (missing values, data types)
- Ensure target column is properly selected
- Verify feature columns are numeric or categorical

#### 4. Google Colab Connection Issues
- Verify the URL is correct
- Check internet connection
- Ensure proper authentication

### Performance Tips
1. **For Large Datasets**: Use Google Colab integration
2. **For Faster Training**: Select fewer features or use simpler models
3. **For Better Results**: Experiment with different hyperparameters
4. **For Memory Efficiency**: Use data sampling for initial exploration

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing web framework
- Scikit-learn team for the comprehensive ML library
- Plotly team for interactive visualizations
- Google Colab team for cloud computing resources

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Happy Machine Learning! 🚀**
