from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def print_daily_confusion_matrix(results_df):
    # A day is capped if it has any capped interval
    daily_preds = results_df.groupby('date')['predicted'].agg(lambda x: int(x.sum() > 0))
    daily_truths = results_df.groupby('date')['actual'].agg(lambda x: int(x.sum() > 0))

    print("\nConfusion Matrix (Daily - Any Capped Interval):")
    print(confusion_matrix(daily_truths, daily_preds, labels=[0, 1]))
    print("-" * 40)

def evaluate_meta_model(meta_model, base_models, X_test_base, y_test, test_df, model_name, target_region):
    print(f"--- Evaluating {model_name} (Meta-Model) ---")

    # Predicted probabilities from base models
    Z_test = np.hstack([
        model.predict_proba(X_test_base)[:, 1].reshape(-1, 1)
        for model in base_models.values()
    ])

    # Predict using meta-model
    y_pred = meta_model.predict(Z_test)

    print("Overall Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Capped (0)', 'Capped (1)']))
    print("\nConfusion Matrix (All Intervals):")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 20)

    results_df = test_df.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred
    results_df['date'] = pd.to_datetime(results_df['SETTLEMENTDATE']).dt.date

    daily_accuracy = results_df.groupby('date').apply(
        lambda df_group: accuracy_score(df_group['actual'], df_group['predicted'])
    )
    print("\nAccuracy per Day:")
    print(daily_accuracy)

    print_daily_confusion_matrix(results_df)


def evaluate_model(model, X_test, y_test, test_df, model_name, target_region):
    print(f"--- Evaluating {model_name} ---")
    
    # Make predictions
    y_pred = model.predict(X_test)

    print("Overall Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Capped (0)', 'Capped (1)']))
    print("\nConfusion Matrix (All Intervals):")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 20)

    results_df = test_df.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred
    results_df['date'] = pd.to_datetime(results_df['SETTLEMENTDATE']).dt.date

    daily_accuracy = results_df.groupby('date').apply(
        lambda df_group: accuracy_score(df_group['actual'], df_group['predicted'])
    )
    print("\nAccuracy per Day:")
    print(daily_accuracy)

    print_daily_confusion_matrix(results_df)

