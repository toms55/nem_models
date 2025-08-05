import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train_xgb(X_train, y_train, X_val, y_val):
    print("--- Training XGBoost (XGB) ---")
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

    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        early_stopping_rounds=50
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=1000,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    print(f"Best XGB parameters: {random_search.best_params_}")
    print("-" * 40)
    return random_search.best_estimator_

def train_lr(X_train, y_train, X_val, y_val):
    print("--- Training Logistic Regression (LR) ---")
    
    param_dist = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_dist,
        n_iter=1000,
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

def train_knn(X_train, y_train, X_val, y_val):
    print("--- Training K-Nearest Neighbors (KNN) ---")
    
    # Create validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ])

    param_dist = {
        'knn__n_neighbors': np.arange(3, 31, 2),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=1000,
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

def train_meta_model(base_models, X_val, y_val):
    """
    Trains a meta-model (Logistic Regression) on the outputs of base models.
    
    base_models: dict of trained base models, e.g. {'xgb': model, 'lr': model, 'knn': model}
    X_val: validation features
    y_val: validation labels
    """

    print("--- Training manual stacking meta-model ---")

    preds = []
    for name, model in base_models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_val)[:, 1].reshape(-1, 1)  # get prob of class 1
            preds.append(prob)
        else:
            raise ValueError(f"Model '{name}' does not support predict_proba")

    Z = np.hstack(preds)

    meta_model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
    meta_model.fit(Z, y_val)

    print("--- Meta-model training complete ---")
    print("-" * 40)
    return meta_model

def train_all_models(X_train_full, y_train_full):
    # Split into sub-train and validation (meta-model uses val)
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Base models trained only on subtrain
    xgb_model = train_xgb(X_subtrain, y_subtrain, X_val, y_val)
    lr_model = train_lr(X_subtrain, y_subtrain, X_val, y_val)
    knn_model = train_knn(X_subtrain, y_subtrain, X_val, y_val)

    base_models = {
        'xgb': xgb_model,
        'lr': lr_model,
        'knn': knn_model
    }

    # Meta-model trained on validation set predictions from base models
    meta_model = train_meta_model(base_models, X_val, y_val)

    return {
        'xgb': xgb_model,
        'lr': lr_model,
        'knn': knn_model,
        'meta': meta_model
    }
