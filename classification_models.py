import pandas as pd
from lib.preprocessor.preprocessor import preprocess_data
from lib.training.models import train_all_models
from lib.evaluation.metrics import evaluate_model, evaluate_meta_model
from lib.utils.io import save_model, load_model

def main():
    regions = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
    
    # --- Load and preprocess data ---
    df = pd.read_csv("merged_dispatch_price_2021-2025.csv")
    df = df.sort_values('SETTLEMENTDATE')

    target_region = "NSW1"
    nem_df = preprocess_data(df, target_region)

    # --- Train/test split ---
    split_index = int(len(nem_df) * 0.8)
    train_df = nem_df.iloc[:split_index]
    test_df = nem_df.iloc[split_index:]

    X_train = train_df.drop([f'is_cap_{target_region}', 'SETTLEMENTDATE'], axis=1)
    y_train = train_df[f'is_cap_{target_region}']
    X_test = test_df.drop([f'is_cap_{target_region}', 'SETTLEMENTDATE'], axis=1)
    y_test = test_df[f'is_cap_{target_region}']

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Training class distribution: \n{y_train.value_counts(normalize=True)}\n")
    print(f"Test class distribution: \n{y_test.value_counts(normalize=True)}\n")
    print(f"Features: {X_train.columns.tolist()}\n")

    # --- Ask user for action ---
    choice = input("Train new models or load existing ones? [t/l]: ").strip()

    model_names = ["xgb", "lr", "knn", "meta"]

    while True:
        if choice == "t":
            models = train_all_models(X_train, y_train, target_region)
            for name, model in models.items():
                save_model(model, name)
            break
        elif choice == "l":
            models = {name: load_model(name) for name in model_names}
            break
        else:
            print("Invalid choice")
            choice = input("Train new models or load existing ones? [t/l]: ").strip()
    
    for name, model in models.items():
        if name == "meta":
            # Collect base models (exclude 'stacked')
            base_models = {k: v for k, v in models.items() if k != "stacked"}
            evaluate_meta_model(
                meta_model=models["meta"],
                base_models={k: v for k, v in models.items() if k in ["xgb", "lr", "knn"]},
                X_test_base=X_test,
                y_test=y_test,
                test_df=test_df,
                model_name="meta",
                target_region=target_region
            )
        else:
            evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                test_df=test_df,
                model_name=name,
                target_region=target_region
            )

if __name__ == "__main__":
    main()

