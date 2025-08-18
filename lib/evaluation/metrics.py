from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np

def print_daily_confusion_matrix(results_df):
    daily_preds = results_df.groupby('date')['predicted'].agg(lambda x: int(x.sum() > 0))
    daily_truths = results_df.groupby('date')['actual'].agg(lambda x: int(x.sum() > 0))
    print("\nConfusion Matrix (Daily - Any Capped Interval):")
    print(confusion_matrix(daily_truths, daily_preds, labels=[0, 1]))
    print("-" * 40)

def print_evaluation_results(y_test, y_pred, test_df, model_name):
    print(f"--- Evaluating {model_name} ---")
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
    print_daily_confusion_matrix(results_df)

def store_model_results(model_name, y_test, y_pred, results_storage):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    results_storage[model_name] = {
        'Precision': f"{precision:.3f}" if precision > 0 else "0",
        'Recall': f"{recall:.3f}" if recall > 0 else "0", 
        'F1': f"{f1:.3f}" if f1 > 0 else "0",
        'Accuracy': f"{accuracy:.3f}"
    }

def print_results_table(results_storage):
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
    models = list(results_storage.keys())
    
    header = "Metric\t" + "\t".join(models)
    print("\n" + header)
    
    for metric in metrics:
        row = metric + "\t" + "\t".join([results_storage[model][metric] for model in models])
        print(row)

def evaluate_meta_model(meta_model, base_models, X_test_base, y_test, test_df, model_name, target_region, results_storage=None):
    Z_test = np.hstack([
        model.predict_proba(X_test_base)[:, 1].reshape(-1, 1)
        for model in base_models.values()
    ])
    y_pred = meta_model.predict(Z_test)
    print_evaluation_results(y_test, y_pred, test_df, f"{model_name} (Meta-Model)")
    
    if results_storage is not None:
        store_model_results(model_name, y_test, y_pred, results_storage)

def evaluate_model(model, X_test, y_test, test_df, model_name, target_region, results_storage=None):
    y_pred = model.predict(X_test)
    print_evaluation_results(y_test, y_pred, test_df, model_name)
    
    if results_storage is not None:
        store_model_results(model_name, y_test, y_pred, results_storage)
