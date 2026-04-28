"""
Model Training Functions for House Price Prediction
====================================================
Pre-built functions for students to use. DO NOT MODIFY these functions!
Focus all your effort on DATA CLEANING and FEATURE ENGINEERING.

Usage:
------
1. Clean your data
2. Engineer features
3. Call train_and_predict(X_train, y_train, X_test)
4. Submit predictions to Kaggle

Author: Course Instructor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_predict(X_train, y_train, X_test, random_state=42):
    """
    Train a Random Forest model and generate predictions.

    Students should NOT modify this function.
    Focus all effort on data cleaning and feature engineering!

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features (cleaned and engineered)
        - Must be numeric (no strings, no NaN values)
        - Must have same columns as X_test
    y_train : pd.Series or np.array
        Training target (sale_price)
    X_test : pd.DataFrame
        Test features (must have same columns as X_train)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    predictions : np.array
        Predicted prices for test set
    model : RandomForestRegressor
        Trained model (for feature importance inspection)
    train_rmse : float
        Training set RMSE (for checking overfitting)

    Example:
    --------
    >>> # After cleaning and feature engineering:
    >>> X_train = train_df.drop(['sale_price', 'property_id', 'sale_id'], axis=1)
    >>> y_train = train_df['sale_price']
    >>> X_test = test_df.drop(['property_id', 'sale_id'], axis=1)
    >>>
    >>> # Ensure all columns are numeric
    >>> X_train = X_train.select_dtypes(include=[np.number])
    >>> X_test = X_test[X_train.columns]
    >>>
    >>> predictions, model, train_rmse = train_and_predict(X_train, y_train, X_test)
    """

    # Validation checks
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # Check for non-numeric columns
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"\n[ERROR] Non-numeric columns found: {non_numeric}")
        print("Please encode categorical variables before training.")
        print("Tips:")
        print("  - Use pd.get_dummies() for one-hot encoding")
        print("  - Use LabelEncoder for ordinal variables")
        raise ValueError(f"Non-numeric columns: {non_numeric}")

    # Check for missing values
    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()
    if train_missing > 0 or test_missing > 0:
        print(f"\n[ERROR] Missing values found!")
        print(f"  - Training set: {train_missing} missing values")
        print(f"  - Test set: {test_missing} missing values")
        print("Please handle missing values before training.")
        print("Tips:")
        print("  - Use df.fillna() to fill missing values")
        print("  - Use df.dropna() to remove rows with missing values")
        raise ValueError("Missing values in data")

    # Check column alignment
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    missing_in_train = set(X_test.columns) - set(X_train.columns)
    if missing_in_test or missing_in_train:
        print(f"\n[ERROR] Column mismatch!")
        if missing_in_test:
            print(f"  - Columns in train but not in test: {missing_in_test}")
        if missing_in_train:
            print(f"  - Columns in test but not in train: {missing_in_train}")
        raise ValueError("Column mismatch between train and test")

    # Ensure same column order
    X_test = X_test[X_train.columns]

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    model.fit(X_train, y_train)

    # Training RMSE (for diagnostics)
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Test predictions
    predictions = model.predict(X_test)

    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTraining Set Performance:")
    print(f"  - RMSE: ${train_rmse:,.2f}")
    print(f"  - MAE:  ${train_mae:,.2f}")
    print(f"  - R2:   {train_r2:.4f}")
    print(f"\nFeatures used: {X_train.shape[1]}")
    print(f"Predictions generated: {len(predictions)}")

    print(f"\n[TIP] If training RMSE is very low but Kaggle score is high,")
    print(f"      your model may be overfitting. Consider:")
    print(f"      - Removing features that might cause data leakage")
    print(f"      - Using cross-validation to evaluate locally")
    print(f"      - Adding regularization")

    return predictions, model, train_rmse


def show_feature_importance(model, X_train, top_n=15):
    """
    Display and plot the top N most important features.

    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model from train_and_predict()
    X_train : pd.DataFrame
        Training features (to get column names)
    top_n : int
        Number of top features to display (default: 15)

    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with feature names and importance scores

    Example:
    --------
    >>> predictions, model, train_rmse = train_and_predict(X_train, y_train, X_test)
    >>> importance_df = show_feature_importance(model, X_train)
    """

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 60)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance_df.head(top_n).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(top_n)

    bars = ax.barh(range(top_n), top_features['importance'])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    print("\n[TIP] Features with high importance are crucial for predictions.")
    print("      Consider engineering more features related to the top ones!")

    return importance_df


def create_submission(predictions, test_df, filename='predictions.csv'):
    """
    Create a Kaggle submission file.

    Parameters:
    -----------
    predictions : np.array
        Predicted prices from train_and_predict()
    test_df : pd.DataFrame
        Original test DataFrame (must have 'sale_id' column)
    filename : str
        Output filename (default: 'predictions.csv')

    Example:
    --------
    >>> predictions, model, train_rmse = train_and_predict(X_train, y_train, X_test)
    >>> create_submission(predictions, test_df)
    """

    submission = pd.DataFrame({
        'Id': test_df['sale_id'],
        'SalePrice': predictions
    })

    submission.to_csv(filename, index=False)

    print(f"\n" + "=" * 60)
    print("SUBMISSION FILE CREATED!")
    print("=" * 60)
    print(f"\nFile: {filename}")
    print(f"Rows: {len(submission)}")
    print(f"\nPreview:")
    print(submission.head())
    print(f"\nPrice Statistics:")
    print(f"  - Min: ${predictions.min():,.2f}")
    print(f"  - Max: ${predictions.max():,.2f}")
    print(f"  - Mean: ${predictions.mean():,.2f}")
    print(f"  - Median: ${np.median(predictions):,.2f}")

    # Sanity checks
    if predictions.min() < 0:
        print(f"\n[WARNING] Negative prices detected! Check your model.")
    if predictions.max() > 50000000:
        print(f"\n[WARNING] Very high prices detected! Check for outliers.")

    print(f"\nNext step: Upload {filename} to Kaggle!")

    return submission


def evaluate_local(y_true, y_pred):
    """
    Evaluate predictions locally using the same metrics as Kaggle.

    Parameters:
    -----------
    y_true : pd.Series or np.array
        Actual prices
    y_pred : np.array
        Predicted prices

    Returns:
    --------
    metrics : dict
        Dictionary with RMSE, MAE, and R2 scores

    Example:
    --------
    >>> # If you have validation data:
    >>> metrics = evaluate_local(y_val, y_pred_val)
    """

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("LOCAL EVALUATION")
    print("=" * 60)
    print(f"\nMetrics:")
    print(f"  - RMSE: ${rmse:,.2f}  <-- This is the Kaggle metric!")
    print(f"  - MAE:  ${mae:,.2f}")
    print(f"  - R2:   {r2:.4f}")

    # Baseline reference
    print(f"\nBaseline RMSE: ~$65,000")
    if rmse < 60000:
        print(f"[GOOD] You beat the baseline!")
    else:
        print(f"[WARNING] You haven't beat the baseline yet. Keep cleaning and engineering!")

    if rmse < 50000:
        print(f"[EXCELLENT] Your model is performing very well!")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# Usage example (commented out)
"""
# ============================================
# EXAMPLE WORKFLOW
# ============================================

import pandas as pd
from model_training import train_and_predict, show_feature_importance, create_submission

# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Clean and engineer features (YOUR WORK HERE!)
# ... your cleaning code ...
# ... your feature engineering code ...

# 3. Prepare for training
# Remove non-feature columns
drop_cols = ['sale_price', 'property_id', 'sale_id', 'address', 'neighborhood', 'sale_date']
X_train = train_df.drop(drop_cols, axis=1, errors='ignore')
y_train = train_df['sale_price']
X_test = test_df.drop([c for c in drop_cols if c != 'sale_price'], axis=1, errors='ignore')

# Ensure numeric only
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test[X_train.columns]

# Fill any remaining NaN
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# 4. Train and predict
predictions, model, train_rmse = train_and_predict(X_train, y_train, X_test)

# 5. Check feature importance
importance_df = show_feature_importance(model, X_train)

# 6. Create submission
create_submission(predictions, test_df)
"""
