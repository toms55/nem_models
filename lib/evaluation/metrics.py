from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


def evaluate_meta_model(meta_model, base_models, X_test_base, y_test, test_df, model_name, target_region):
    """
    Evaluates the manually stacked meta-model.
    """
    print(f"--- Evaluating {model_name} (Meta-Model) ---")

    # Step 1: Get predicted probabilities from base models
    Z_test = np.hstack([
        model.predict_proba(X_test_base)[:, 1].reshape(-1, 1)
        for model in base_models.values()
    ])

    # Step 2: Predict using the meta-model
    y_pred = meta_model.predict(Z_test)

    # Step 3: Evaluation (reuse your logic)
    print("Overall Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Capped (0)', 'Capped (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 20)

    results_df = test_df.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred

    def get_classification_type(row):
        if row['actual'] == 1 and row['predicted'] == 1:
            return 'TP' 
        if row['actual'] == 0 and row['predicted'] == 0:
            return 'TN'
        if row['actual'] == 0 and row['predicted'] == 1:
            return 'FP' 
        if row['actual'] == 1 and row['predicted'] == 0:
            return 'FN'

    results_df['classification_type'] = results_df.apply(get_classification_type, axis=1)

    print("\nPer-Interval Classification Matrix (Sample):")
    print(results_df[['SETTLEMENTDATE', 'actual', 'predicted', 'classification_type']].head(20))
    print("-" * 20)

    results_df['date'] = pd.to_datetime(results_df['SETTLEMENTDATE']).dt.date
    daily_accuracy = results_df.groupby('date').apply(
        lambda df_group: accuracy_score(df_group['actual'], df_group['predicted'])
    )

    print("\nAccuracy per Day:")
    print(daily_accuracy)
    print("-" * 40)


def evaluate_model(model, X_test, y_test, test_df, model_name, target_region):
    """
    Evaluates the model, providing overall metrics and a detailed per-interval and daily breakdown.
    """
    print(f"--- Evaluating {model_name} ---")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Overall performance metrics
    print("Overall Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Capped (0)', 'Capped (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 20)
    
    # Half hourly accuracy 
    results_df = test_df.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred
    
    def get_classification_type(row):
        if row['actual'] == 1 and row['predicted'] == 1:
            return 'TP' 
        if row['actual'] == 0 and row['predicted'] == 0:
            return 'TN'
        if row['actual'] == 0 and row['predicted'] == 1:
            return 'FP' 
        if row['actual'] == 1 and row['predicted'] == 0:
            return 'FN'
            
    results_df['classification_type'] = results_df.apply(get_classification_type, axis=1)
    
    print("\nPer-Interval Classification Matrix (Sample):")
    print(results_df[['SETTLEMENTDATE', 'actual', 'predicted', 'classification_type']].head(20))
    print("-" * 20)
    
    # Daily accuracy
    results_df['date'] = pd.to_datetime(results_df['SETTLEMENTDATE']).dt.date
    daily_accuracy = results_df.groupby('date').apply(
        lambda df_group: accuracy_score(df_group['actual'], df_group['predicted'])
    )
    
    print("\nAccuracy per Day:")
    print(daily_accuracy)
    print("-" * 40 + "\n")
