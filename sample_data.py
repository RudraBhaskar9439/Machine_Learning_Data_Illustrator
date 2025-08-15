import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris, fetch_california_housing
from sklearn.preprocessing import StandardScaler
import os

def create_sample_datasets():
    """Create sample datasets for testing the ML Illustrator application"""
    
    # Create output directory
    os.makedirs('sample_data', exist_ok=True)
    
    # 1. Classification Dataset - Iris-like
    print("Creating classification dataset...")
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X_class.shape[1])]
    
    # Create classification dataset
    class_df = pd.DataFrame(X_class, columns=feature_names)
    class_df['target'] = y_class
    
    # Add some categorical features
    class_df['category'] = np.random.choice(['A', 'B', 'C'], size=len(class_df))
    class_df['binary'] = np.random.choice([0, 1], size=len(class_df))
    
    # Add some missing values
    class_df.loc[np.random.choice(class_df.index, size=50, replace=False), 'feature_1'] = np.nan
    class_df.loc[np.random.choice(class_df.index, size=30, replace=False), 'feature_2'] = np.nan
    
    class_df.to_csv('sample_data/classification_sample.csv', index=False)
    print("✅ Classification dataset saved: sample_data/classification_sample.csv")
    
    # 2. Regression Dataset - Housing-like
    print("Creating regression dataset...")
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=12,
        n_informative=10,
        n_targets=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X_reg.shape[1])]
    
    # Create regression dataset
    reg_df = pd.DataFrame(X_reg, columns=feature_names)
    reg_df['target'] = y_reg
    
    # Add some categorical features
    reg_df['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(reg_df))
    reg_df['condition'] = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], size=len(reg_df))
    
    # Add some missing values
    reg_df.loc[np.random.choice(reg_df.index, size=40, replace=False), 'feature_1'] = np.nan
    reg_df.loc[np.random.choice(reg_df.index, size=25, replace=False), 'feature_3'] = np.nan
    
    reg_df.to_csv('sample_data/regression_sample.csv', index=False)
    print("✅ Regression dataset saved: sample_data/regression_sample.csv")
    
    # 3. Binary Classification Dataset
    print("Creating binary classification dataset...")
    X_binary, y_binary = make_classification(
        n_samples=800,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X_binary.shape[1])]
    
    # Create binary classification dataset
    binary_df = pd.DataFrame(X_binary, columns=feature_names)
    binary_df['target'] = y_binary
    
    # Add some categorical features
    binary_df['gender'] = np.random.choice(['Male', 'Female'], size=len(binary_df))
    binary_df['age_group'] = np.random.choice(['18-25', '26-35', '36-45', '46+'], size=len(binary_df))
    
    # Add some missing values
    binary_df.loc[np.random.choice(binary_df.index, size=30, replace=False), 'feature_1'] = np.nan
    
    binary_df.to_csv('sample_data/binary_classification_sample.csv', index=False)
    print("✅ Binary classification dataset saved: sample_data/binary_classification_sample.csv")
    
    # 4. Real-world like dataset (Customer Churn)
    print("Creating customer churn dataset...")
    np.random.seed(42)
    
    n_customers = 1000
    
    # Generate customer data
    customer_data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(45, 15, n_customers).astype(int),
        'tenure': np.random.exponential(5, n_customers).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_customers),
        'total_charges': np.random.normal(2000, 1000, n_customers),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
        'gender': np.random.choice(['Male', 'Female'], n_customers)
    }
    
    churn_df = pd.DataFrame(customer_data)
    
    # Create target variable based on features
    churn_prob = (
        (churn_df['tenure'] < 10) * 0.3 +
        (churn_df['monthly_charges'] > 80) * 0.2 +
        (churn_df['contract_type'] == 'Month-to-month') * 0.3 +
        (churn_df['payment_method'] == 'Electronic check') * 0.1 +
        (churn_df['internet_service'] == 'Fiber optic') * 0.1
    )
    
    churn_df['churn'] = np.random.binomial(1, churn_prob)
    
    # Add some missing values
    churn_df.loc[np.random.choice(churn_df.index, size=50, replace=False), 'monthly_charges'] = np.nan
    churn_df.loc[np.random.choice(churn_df.index, size=30, replace=False), 'total_charges'] = np.nan
    
    churn_df.to_csv('sample_data/customer_churn_sample.csv', index=False)
    print("✅ Customer churn dataset saved: sample_data/customer_churn_sample.csv")
    
    # 5. Simple Iris-like dataset
    print("Creating simple iris-like dataset...")
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = iris.target_names[iris.target]
    
    iris_df.to_csv('sample_data/iris_sample.csv', index=False)
    print("✅ Iris dataset saved: sample_data/iris_sample.csv")
    
    print("\n🎉 All sample datasets created successfully!")
    print("\n📁 Files created in 'sample_data/' directory:")
    print("   - classification_sample.csv (Multi-class classification)")
    print("   - regression_sample.csv (Regression)")
    print("   - binary_classification_sample.csv (Binary classification)")
    print("   - customer_churn_sample.csv (Real-world like)")
    print("   - iris_sample.csv (Simple classification)")
    
    print("\n💡 You can now use these datasets to test the ML Illustrator application!")
    print("   Upload any of these CSV files in the 'Data Upload' section.")

if __name__ == "__main__":
    create_sample_datasets()
