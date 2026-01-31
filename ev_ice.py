```python
!pip install kagglehub -q

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("shubhamindulkar/ev-datasets-for-the-indian-market")
print(f"Dataset path: {path}")

files = os.listdir(path)
print("\nAvailable files:")
for file in files:
    if file.endswith('.csv'):
        print(f"  {file}")

print("\nLoading datasets...")
try:
    reg_df = pd.read_csv(os.path.join(path, "vehicle_registrations_detailed.csv"))
    adoption_df = pd.read_csv(os.path.join(path, "india_ev_ice_adoption_large.csv"))
    sales_df = pd.read_csv(os.path.join(path, "ev_ice_market_sales_india.csv"))
    charging_df = pd.read_csv(os.path.join(path, "ev_charging_infrastructure_india.csv"))
    battery_df = pd.read_csv(os.path.join(path, "ev_vehicle_battery_specs_india.csv"))
    policy_df = pd.read_csv(os.path.join(path, "ev_policy_incentives_india.csv"))
    
    print(f"✓ Registration data: {reg_df.shape}")
    print(f"✓ Adoption data: {adoption_df.shape}")
    print(f"✓ Sales data: {sales_df.shape}")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        reg_df = pd.read_csv(os.path.join(path, csv_files[0]))
        print(f"Loaded {csv_files[0]}: {reg_df.shape}")

print(f"\nAnalyzing registration dataset...")
print(f"Shape: {reg_df.shape}")
print(f"\nColumns: {reg_df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(reg_df.head(3))

reg_df_clean = reg_df.copy()

numeric_cols = reg_df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = reg_df_clean.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

if reg_df_clean.isnull().sum().sum() > 0:
    for col in numeric_cols:
        if reg_df_clean[col].isnull().sum() > 0:
            reg_df_clean[col].fillna(reg_df_clean[col].median(), inplace=True)
    
    for col in categorical_cols:
        if reg_df_clean[col].isnull().sum() > 0:
            reg_df_clean[col].fillna(reg_df_clean[col].mode()[0], inplace=True)

target_candidates = []
for col in reg_df_clean.columns:
    if pd.api.types.is_numeric_dtype(reg_df_clean[col]):
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['ev', 'electric', 'registration', 'sales']):
            target_candidates.append((col, reg_df_clean[col].nunique()))

if target_candidates:
    if any('ev' in col.lower() for col, _ in target_candidates):
        target_col = next(col for col, _ in target_candidates if 'ev' in col.lower())
    else:
        target_col = target_candidates[0][0]
else:
    target_col = numeric_cols[-1] if numeric_cols else reg_df_clean.columns[-1]

print(f"\nSelected target: '{target_col}'")

features = [col for col in reg_df_clean.columns if col != target_col]
X = reg_df_clean[features]
y = reg_df_clean[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

if categorical_cols:
    cat_cols_to_encode = [col for col in categorical_cols if col != target_col]
    if cat_cols_to_encode:
        le = LabelEncoder()
        for col in cat_cols_to_encode:
            if X[col].nunique() <= 10:
                X[col] = le.fit_transform(X[col])
            else:
                X[col] = le.fit_transform(X[col])

print(f"Final feature matrix shape: {X.shape}")

if len(X) > 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\nModel Performance:")
    print(f"Training R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Testing R²:  {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_train, y_pred_train, alpha=0.6)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Training (R² = {train_r2:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test, y_pred_test, alpha=0.6, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Test (R² = {test_r2:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    errors = y_test - y_pred_test
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nAnalysis complete! Model R²: {test_r2:.3f}")
else:
    print(f"\nNot enough samples. Only {len(X)} rows available.")

print("\nAdditional datasets loaded:")
datasets_to_check = ['sales_df', 'adoption_df', 'charging_df', 'battery_df', 'policy_df']
for ds_name in datasets_to_check:
    if ds_name in locals():
        df_check = locals()[ds_name]
        if df_check is not None and len(df_check) > 0:
            print(f"  {ds_name}: {df_check.shape}")
```
