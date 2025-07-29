import pandas as pd
import numpy as np
from datetime import datetime
import os
import xgboost as xgb

# Scikit-learn imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Data Preprocessing Functions (from your code) ---
def get_hour(dt):
    return dt.hour

def get_season(dt):
    month = dt.month
    if month in [12, 1, 2]:
        return 1  # Summer
    elif month in [3, 4, 5]:
        return 2  # Autumn
    elif month in [6, 7, 8]:
        return 3  # Winter
    else:
        return 4  # Spring

def is_peak_hour(dt):
    return 14 <= dt.hour <= 20 and dt.weekday() not in [5, 6]

def preprocess_data(df, target_region="NSW1"):
    numeric_columns = ['RRP', 'AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 'SEMISCHEDULE_CLEAREDMW']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['RRP']).copy()
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE']).dt.floor('30min')

    df['hour'] = df['SETTLEMENTDATE'].apply(get_hour)
    df['season'] = df['SETTLEMENTDATE'].apply(get_season)
    df['is_peak'] = df['SETTLEMENTDATE'].apply(is_peak_hour).astype(int)
    df['is_cap'] = (df['RRP'] >= 300).astype(int)

    # --- FIX IS HERE ---
    # Define aggregations: 'mean' for features, 'max' for the target to keep it binary.
    agg_dict = {
        'AVAILABLEGENERATION': 'mean',
        'AVAILABLELOAD': 'mean',
        'DEMANDFORECAST': 'mean',
        'SEMISCHEDULE_CLEAREDMW': 'mean',
        'RRP': 'mean',
        'is_cap': 'max' # This is the crucial change
    }
    # Filter out columns that might not be in the DataFrame to prevent errors
    cols_to_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    df = df.groupby(['SETTLEMENTDATE', 'hour', 'season', 'is_peak', 'REGIONID'], as_index=False).agg(cols_to_agg)
    # --- END FIX ---

    df = df.pivot_table(
        index=['SETTLEMENTDATE', 'hour', 'season', 'is_peak'],
        columns='REGIONID',
        values=['AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 'SEMISCHEDULE_CLEAREDMW', 'RRP', 'is_cap'],
        aggfunc='first'
    )

    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df = df.reset_index()

    target_cap_col = f'is_cap_{target_region}'
    if target_cap_col not in df.columns:
        print(f"Warning: Target region {target_region} not found in data")
        cap_cols = [col for col in df.columns if col.startswith('is_cap_')]
        if cap_cols:
            df[target_cap_col] = df[cap_cols[0]]
            print(f"Using {cap_cols[0]} as target, renamed to is_cap_{target_region}")
    
    cols_to_drop = [col for col in df.columns if col.startswith('is_cap_') and col != target_cap_col]
    cols_to_drop.extend([col for col in df.columns if col.startswith('RRP_')])
    
    df_processed = df.drop(columns=cols_to_drop)

    return df_processed

# --- Model Training Functions ---

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine classifier with hyperparameter tuning.
    """
    print("--- Training Support Vector Machine (SVM) ---")
    param_dist = {
        'C': np.logspace(-3, 2, 10),
        'gamma': np.logspace(-4, 1, 10),
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    svm = SVC(class_weight='balanced', probability=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_iter=300,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    print(f"Best SVM parameters: {random_search.best_params_}")
    print("-" * 40)
    return random_search.best_estimator_

def train_lr(X_train, y_train):
    """
    Trains a Logistic Regression classifier with hyperparameter tuning.
    """
    print("--- Training Logistic Regression (LR) ---")
    param_dist = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'] # 'liblinear' works well with l1 and l2
    }
    
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_dist,
        n_iter=300,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best LR parameters: {random_search.best_params_}")
    print("-" * 40)
    return random_search.best_estimator_

def train_xgb(X_train, y_train):
    """
    Trains an XGBoost classifier with hyperparameter tuning.
    """
    print("--- Training XGBoost (XGB) ---")
    # Calculate scale_pos_weight for handling class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5, 1],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=300,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best XGB parameters: {random_search.best_params_}")
    print("-" * 40)
    return random_search.best_estimator_

def train_knn(X_train, y_train):
    """
    Trains a K-Nearest Neighbors classifier with a scaling pipeline and hyperparameter tuning.
    """
    print("--- Training K-Nearest Neighbors (KNN) ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ])
    
    param_dist = {
        'knn__n_neighbors': np.arange(3, 31, 2),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2] # 1 for Manhattan distance, 2 for Euclidean
    }
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=300, # Will be capped by the number of combinations, which is less than 200
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best KNN parameters: {random_search.best_params_}")
    print("-" * 40)
    return random_search.best_estimator_

# --- Model Evaluation Function ---

def evaluate_model(model, X_test, y_test, test_df, model_name, target_region):
    """
    Evaluates the model, providing overall metrics and a detailed per-interval and daily breakdown.
    """
    print(f"--- Evaluating {model_name} ---")
    
    # 1. Make predictions
    y_pred = model.predict(X_test)
    
    # 2. Overall performance metrics
    print("Overall Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Capped (0)', 'Capped (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 20)
    
    # 3. Per-interval classification matrix (as a DataFrame)
    results_df = test_df.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred
    
    def get_classification_type(row):
        if row['actual'] == 1 and row['predicted'] == 1:
            return 'TP' # True Positive
        if row['actual'] == 0 and row['predicted'] == 0:
            return 'TN' # True Negative
        if row['actual'] == 0 and row['predicted'] == 1:
            return 'FP' # False Positive
        if row['actual'] == 1 and row['predicted'] == 0:
            return 'FN' # False Negative
            
    results_df['classification_type'] = results_df.apply(get_classification_type, axis=1)
    
    print("\nPer-Interval Classification Matrix (Sample):")
    print(results_df[['SETTLEMENTDATE', 'actual', 'predicted', 'classification_type']].head(20))
    print("-" * 20)
    
    # 4. Daily accuracy
    results_df['date'] = pd.to_datetime(results_df['SETTLEMENTDATE']).dt.date
    daily_accuracy = results_df.groupby('date').apply(
        lambda df_group: accuracy_score(df_group['actual'], df_group['predicted'])
    )
    
    print("\nAccuracy per Day:")
    print(daily_accuracy)
    print("-" * 40 + "\n")


# --- Main Execution ---

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv("merged_dispatch_price_2021-2025.csv")
    df = df.sort_values('SETTLEMENTDATE')
    
    target_region = "NSW1"
    nem_df = preprocess_data(df, target_region)

    # Time-based split (80/20)
    split_index = int(len(nem_df) * 0.8)
    train_df = nem_df.iloc[:split_index]
    test_df = nem_df.iloc[split_index:]

    # Separate features and target. Drop 'SETTLEMENTDATE' from features.
    X_train = train_df.drop([f'is_cap_{target_region}', 'SETTLEMENTDATE'], axis=1)
    y_train = train_df[f'is_cap_{target_region}']
    X_test = test_df.drop([f'is_cap_{target_region}', 'SETTLEMENTDATE'], axis=1)
    y_test = test_df[f'is_cap_{target_region}']

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Training class distribution: \n{y_train.value_counts(normalize=True)}\n")
    print(f"Test class distribution: \n{y_test.value_counts(normalize=True)}\n")
    print(f"Features: {X_train.columns.tolist()}")

    # --- Train and Evaluate Models ---
    
    # XGBoost
    xgb_model = train_xgb(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, test_df, "XGBoost", target_region)
    
    # Logistic Regression
    lr_model = train_lr(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, test_df, "Logistic Regression", target_region)
    
    # KNN
    knn_model = train_knn(X_train, y_train)
    evaluate_model(knn_model, X_test, y_test, test_df, "K-Nearest Neighbors", target_region)
    
    # SVM (Note: Can be very slow on large datasets)
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, test_df, "Support Vector Machine", target_region)
    
    print("Script finished.")
