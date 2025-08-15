import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

# Import Google Colab integration
from colab_integration import show_colab_connection_page, integrate_colab_with_ml_illustrator

# Set page config
st.set_page_config(
    page_title="ML Illustrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': '# ML Illustrator\nA comprehensive machine learning application'
    }
)

# Custom CSS
st.markdown("""
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
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Ensure radio buttons are fully visible */
    .stRadio > label {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.25rem;
        transition: background-color 0.2s;
    }
    
    .stRadio > label:hover {
        background-color: #f0f2f6;
    }
    
    /* Sidebar title styling */
    .css-1d391kg h1 {
        color: #1f77b4;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class MLIllustrator:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_classification = False
        
    def load_data(self, file):
        """Load and preprocess the uploaded dataset"""
        try:
            self.data = pd.read_csv(file)
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_column_info(self):
        """Get information about dataset columns"""
        if self.data is None:
            return None
        
        info = {
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'unique_counts': {col: self.data[col].nunique() for col in self.data.columns}
        }
        return info
    
    def prepare_data(self, target_column, feature_columns, test_size=0.2):
        """Prepare data for training"""
        try:
            # Select features and target
            X = self.data[feature_columns].copy()
            y = self.data[target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean() if X.dtypes.any() in ['float64', 'int64'] else X.mode().iloc[0])
            y = y.fillna(y.mean() if y.dtype in ['float64', 'int64'] else y.mode().iloc[0])
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X[col] = self.label_encoder.fit_transform(X[col].astype(str))
            
            # Determine if classification or regression
            if y.dtype == 'object' or y.nunique() < 10:
                self.is_classification = True
                y = self.label_encoder.fit_transform(y.astype(str))
            else:
                self.is_classification = False
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            return True
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return False
    
    def get_model_options(self):
        """Get available model options based on problem type"""
        if self.is_classification:
            return {
                'Logistic Regression': LogisticRegression,
                'Random Forest': RandomForestClassifier,
                'SVM': SVC,
                'Decision Tree': DecisionTreeClassifier,
                'K-Nearest Neighbors': KNeighborsClassifier,
                'Gradient Boosting': GradientBoostingClassifier,
                'XGBoost': xgb.XGBClassifier,
                'Naive Bayes': GaussianNB
            }
        else:
            return {
                'Linear Regression': LinearRegression,
                'Ridge Regression': Ridge,
                'Lasso Regression': Lasso,
                'Random Forest': RandomForestRegressor,
                'SVR': SVR,
                'Decision Tree': DecisionTreeRegressor,
                'K-Nearest Neighbors': KNeighborsRegressor,
                'Gradient Boosting': GradientBoostingRegressor,
                'XGBoost': xgb.XGBRegressor
            }
    
    def get_hyperparameters(self, model_name):
        """Get hyperparameter options for selected model"""
        if self.is_classification:
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
            else:
                return {}
        else:
            if model_name == 'Linear Regression':
                return {}
            elif model_name == 'Random Forest':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'XGBoost':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                return {}
    
    def train_model(self, model_name, hyperparameters):
        """Train the selected model with given hyperparameters"""
        try:
            model_options = self.get_model_options()
            model_class = model_options[model_name]
            
            # Initialize model with hyperparameters
            if model_name in ['XGBoost']:
                self.model = model_class(**hyperparameters, random_state=42, verbose=0)
            else:
                self.model = model_class(**hyperparameters, random_state=42)
            
            # Train model
            self.model.fit(self.X_train, self.y_train)
            
            return True
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def evaluate_model(self):
        """Evaluate the trained model and return metrics"""
        if self.model is None:
            return None
        
        try:
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = None
            
            # Check if model supports probability predictions
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_pred_proba = self.model.predict_proba(self.X_test)
                except Exception as e:
                    # Some models might not support predict_proba for certain configurations
                    y_pred_proba = None
            
            # Calculate metrics
            if self.is_classification:
                try:
                    metrics = {
                        'accuracy': accuracy_score(self.y_test, y_pred),
                        'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                    }
                except Exception as e:
                    # Fallback to basic accuracy if other metrics fail
                    metrics = {
                        'accuracy': accuracy_score(self.y_test, y_pred),
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0
                    }
            else:
                try:
                    metrics = {
                        'mse': mean_squared_error(self.y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        'mae': mean_absolute_error(self.y_test, y_pred),
                        'r2_score': r2_score(self.y_test, y_pred)
                    }
                except Exception as e:
                    # Fallback to basic metrics if calculation fails
                    metrics = {
                        'mse': 0.0,
                        'rmse': 0.0,
                        'mae': 0.0,
                        'r2_score': 0.0
                    }
            
            return {
                'metrics': metrics,
                'y_test': self.y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")
            return None
    
    def create_visualizations(self, evaluation_results):
        """Create comprehensive visualizations"""
        if evaluation_results is None:
            return
        
        y_test = evaluation_results['y_test']
        y_pred = evaluation_results['y_pred']
        y_pred_proba = evaluation_results['y_pred_proba']
        
        # Create subplots
        if self.is_classification:
            # Determine if binary or multiclass
            is_binary = len(np.unique(y_test)) == 2
            second_plot_title = 'ROC Curve' if is_binary else 'Class Distribution'
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Confusion Matrix', second_plot_title, 'Feature Importance', 'Prediction Distribution'),
                specs=[[{"type": "heatmap"}, {"type": "scatter" if is_binary else "bar"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            n_classes = len(np.unique(y_test))
            
            if n_classes == 2:
                # Binary classification
                x_labels = ['Predicted 0', 'Predicted 1']
                y_labels = ['Actual 0', 'Actual 1']
            else:
                # Multiclass classification
                x_labels = [f'Predicted {i}' for i in range(n_classes)]
                y_labels = [f'Actual {i}' for i in range(n_classes)]
            
            fig.add_trace(
                go.Heatmap(z=cm, x=x_labels, y=y_labels,
                          colorscale='Blues', showscale=True),
                row=1, col=1
            )
            
            # ROC Curve (only for binary classification)
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    auc_score = auc(fpr, tpr)
                    fig.add_trace(
                        go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')),
                        row=1, col=2
                    )
                except Exception as e:
                    # If ROC curve fails, show a placeholder
                    fig.add_trace(
                        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='ROC Curve (Not available)', 
                                 line=dict(dash='dot', color='gray')),
                        row=1, col=2
                    )
            else:
                # For multiclass, show class distribution instead
                class_counts = np.bincount(y_test)
                class_labels = [f'Class {i}' for i in range(len(class_counts))]
                fig.add_trace(
                    go.Bar(x=class_labels, y=class_counts, name='Class Distribution'),
                    row=1, col=2
                )
            
            # Feature Importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
                fig.add_trace(
                    go.Bar(x=feature_names, y=feature_importance, name='Feature Importance'),
                    row=2, col=1
                )
            
            # Prediction Distribution
            fig.add_trace(
                go.Histogram(x=y_pred, name='Predictions', nbinsx=20),
                row=2, col=2
            )
            
        else:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Actual vs Predicted', 'Residuals Plot', 'Feature Importance', 'Prediction Distribution'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # Actual vs Predicted
            fig.add_trace(
                go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                          mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
                row=1, col=1
            )
            
            # Residuals Plot
            residuals = y_test - y_pred
            fig.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
                row=1, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            # Feature Importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
                fig.add_trace(
                    go.Bar(x=feature_names, y=feature_importance, name='Feature Importance'),
                    row=2, col=1
                )
            
            # Prediction Distribution
            fig.add_trace(
                go.Histogram(x=y_pred, name='Predictions', nbinsx=20),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Model Evaluation Visualizations")
        return fig

def main():
    st.markdown('<h1 class="main-header">🤖 ML Illustrator</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'ml_illustrator' not in st.session_state:
        st.session_state.ml_illustrator = MLIllustrator()
    
    ml_illustrator = st.session_state.ml_illustrator
    
    # Sidebar for navigation
    st.sidebar.title("🤖 ML Illustrator")
    st.sidebar.markdown("---")
    
    # Use radio buttons for better visibility
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔗 Google Colab", "📊 Data Upload", "🔧 Model Configuration", "📈 Results & Visualization"],
        index=0
    )
    
    # Add some spacing and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Info:**")
    st.sidebar.markdown("• Upload CSV data")
    st.sidebar.markdown("• Train ML models")
    st.sidebar.markdown("• View results")
    st.sidebar.markdown("• Connect to Colab")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Powered by Streamlit*")
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔗 Google Colab":
        show_colab_connection_page()
    elif page == "📊 Data Upload":
        show_data_upload_page(ml_illustrator)
    elif page == "🔧 Model Configuration":
        show_model_config_page(ml_illustrator)
    elif page == "📈 Results & Visualization":
        show_results_page(ml_illustrator)

def show_home_page():
    st.markdown("""
    ## Welcome to ML Illustrator! 🚀
    
    This application helps you build, train, and evaluate machine learning models with ease.
    
    ### Features:
    - 🔗 **Google Colab Integration**: Connect to Google Colab for remote computation
    - 📊 **Data Upload & Exploration**: Upload your CSV files and explore the dataset
    - 🔧 **Model Selection**: Choose from various ML algorithms
    - ⚙️ **Hyperparameter Tuning**: Configure model parameters
    - 📈 **Comprehensive Evaluation**: Get detailed metrics and visualizations
    - 🎨 **Beautiful Visualizations**: Interactive plots with Plotly
    
    ### Supported Models:
    
    **Classification Models:**
    - Logistic Regression
    - Random Forest
    - Support Vector Machine
    - Decision Tree
    - K-Nearest Neighbors
    - Gradient Boosting
    - XGBoost
    - Naive Bayes
    
    **Regression Models:**
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest
    - Support Vector Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Gradient Boosting
    - XGBoost
    
    ### Getting Started:
    1. (Optional) Connect to **Google Colab** for remote computation
    2. Go to **Data Upload** to upload your dataset
    3. Navigate to **Model Configuration** to select and configure your model
    4. View **Results & Visualization** to see the model performance
    
    ---
    
    **Note**: This application automatically detects whether your problem is classification or regression based on your target variable.
    """)

def show_data_upload_page(ml_illustrator):
    st.header("📊 Data Upload & Exploration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your training or test dataset"
    )
    
    if uploaded_file is not None:
        if ml_illustrator.load_data(uploaded_file):
            st.success("✅ Data loaded successfully!")
            
            # Display dataset info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Overview")
                st.write(f"**Shape:** {ml_illustrator.data.shape}")
                st.write(f"**Columns:** {len(ml_illustrator.data.columns)}")
                st.write(f"**Memory Usage:** {ml_illustrator.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            with col2:
                st.subheader("Data Types")
                dtype_counts = ml_illustrator.data.dtypes.value_counts()
                st.write(dtype_counts)
            
            # Display first few rows
            st.subheader("First 5 Rows")
            st.dataframe(ml_illustrator.data.head())
            
            # Column information
            column_info = ml_illustrator.get_column_info()
            if column_info:
                st.subheader("Column Information")
                
                # Create a DataFrame for better display
                col_info_df = pd.DataFrame({
                    'Column': column_info['columns'],
                    'Data Type': [column_info['dtypes'][col] for col in column_info['columns']],
                    'Missing Values': [column_info['missing_values'][col] for col in column_info['columns']],
                    'Unique Values': [column_info['unique_counts'][col] for col in column_info['columns']]
                })
                
                st.dataframe(col_info_df)
                
                # Missing values visualization
                if col_info_df['Missing Values'].sum() > 0:
                    st.subheader("Missing Values Analysis")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    col_info_df.set_index('Column')['Missing Values'].plot(kind='bar', ax=ax)
                    plt.xticks(rotation=45)
                    plt.title('Missing Values per Column')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Store data in session state
            st.session_state.data_loaded = True
            st.session_state.column_info = column_info

def show_model_config_page(ml_illustrator):
    st.header("🔧 Model Configuration")
    
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload page.")
        return
    
    # Data preparation section
    st.subheader("Data Preparation")
    
    column_info = st.session_state.column_info
    all_columns = column_info['columns']
    
    # Target column selection
    target_column = st.selectbox(
        "Select Target Column",
        all_columns,
        help="Choose the column you want to predict"
    )
    
    # Feature columns selection
    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in all_columns if col != target_column],
        default=[col for col in all_columns if col != target_column],
        help="Choose the columns to use as features"
    )
    
    # Test size
    test_size = st.slider(
        "Test Set Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Percentage of data to use for testing"
    )
    
    if st.button("Prepare Data", type="primary"):
        with st.spinner("Preparing data..."):
            if ml_illustrator.prepare_data(target_column, feature_columns, test_size):
                st.success("✅ Data prepared successfully!")
                st.session_state.data_prepared = True
                
                # Show data info
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Training set size:** {ml_illustrator.X_train.shape[0]}")
                    st.write(f"**Test set size:** {ml_illustrator.X_test.shape[0]}")
                with col2:
                    st.write(f"**Number of features:** {ml_illustrator.X_train.shape[1]}")
                    st.write(f"**Problem type:** {'Classification' if ml_illustrator.is_classification else 'Regression'}")
    
    # Model selection and configuration
    if hasattr(st.session_state, 'data_prepared') and st.session_state.data_prepared:
        st.subheader("Model Selection & Configuration")
        
        # Get available models
        model_options = ml_illustrator.get_model_options()
        selected_model = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            help="Choose the machine learning algorithm"
        )
        
        # Get hyperparameters for selected model
        hyperparams = ml_illustrator.get_hyperparameters(selected_model)
        
        if hyperparams:
            st.subheader("Hyperparameter Configuration")
            selected_hyperparams = {}
            
            for param, options in hyperparams.items():
                if isinstance(options, list):
                    if param == 'penalty' or param == 'kernel' or param == 'gamma':
                        selected_hyperparams[param] = st.selectbox(f"{param}", options)
                    else:
                        selected_hyperparams[param] = st.selectbox(f"{param}", options)
                else:
                    selected_hyperparams[param] = options
            
            st.session_state.selected_hyperparams = selected_hyperparams
        else:
            st.info("This model doesn't require hyperparameter tuning.")
            st.session_state.selected_hyperparams = {}
        
        # Train model button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                if ml_illustrator.train_model(selected_model, st.session_state.selected_hyperparams):
                    st.success("✅ Model trained successfully!")
                    st.session_state.model_trained = True
                    st.session_state.selected_model_name = selected_model
                else:
                    st.error("❌ Model training failed!")

def show_results_page(ml_illustrator):
    st.header("📈 Results & Visualization")
    
    if not hasattr(st.session_state, 'model_trained') or not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Model Configuration page.")
        return
    
    # Additional safety check
    if ml_illustrator.model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    # Evaluate model
    evaluation_results = ml_illustrator.evaluate_model()
    
    if evaluation_results:
        # Display metrics
        st.subheader("Model Performance Metrics")
        
        metrics = evaluation_results['metrics']
        
        if ml_illustrator.is_classification:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MSE", f"{metrics['mse']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col3:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col4:
                st.metric("R² Score", f"{metrics['r2_score']:.4f}")
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        
        if ml_illustrator.is_classification:
            # Classification report
            st.text("Classification Report:")
            report = classification_report(evaluation_results['y_test'], evaluation_results['y_pred'])
            st.text(report)
        else:
            # Regression metrics
            st.write(f"**Mean Squared Error (MSE):** {metrics['mse']:.4f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {metrics['rmse']:.4f}")
            st.write(f"**Mean Absolute Error (MAE):** {metrics['mae']:.4f}")
            st.write(f"**R-squared (R²):** {metrics['r2_score']:.4f}")
        
        # Visualizations
        st.subheader("Model Visualizations")
        
        fig = ml_illustrator.create_visualizations(evaluation_results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        st.subheader("Additional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction vs Actual scatter plot
            fig_scatter = px.scatter(
                x=evaluation_results['y_test'],
                y=evaluation_results['y_pred'],
                title="Actual vs Predicted Values",
                labels={'x': 'Actual', 'y': 'Predicted'}
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[evaluation_results['y_test'].min(), evaluation_results['y_test'].max()],
                    y=[evaluation_results['y_test'].min(), evaluation_results['y_test'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Residuals histogram
            if not ml_illustrator.is_classification:
                residuals = evaluation_results['y_test'] - evaluation_results['y_pred']
                fig_hist = px.histogram(
                    x=residuals,
                    title="Residuals Distribution",
                    labels={'x': 'Residuals', 'y': 'Count'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                # Confusion matrix heatmap
                cm = confusion_matrix(evaluation_results['y_test'], evaluation_results['y_pred'])
                fig_cm = px.imshow(
                    cm,
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"),
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # Model summary
        st.subheader("Model Summary")
        st.write(f"**Model:** {st.session_state.selected_model_name}")
        st.write(f"**Problem Type:** {'Classification' if ml_illustrator.is_classification else 'Regression'}")
        st.write(f"**Training Set Size:** {ml_illustrator.X_train.shape[0]}")
        st.write(f"**Test Set Size:** {ml_illustrator.X_test.shape[0]}")
        st.write(f"**Number of Features:** {ml_illustrator.X_train.shape[1]}")
        
        if hasattr(st.session_state, 'selected_hyperparams'):
            st.write("**Hyperparameters:**")
            for param, value in st.session_state.selected_hyperparams.items():
                st.write(f"  - {param}: {value}")

if __name__ == "__main__":
    main()
