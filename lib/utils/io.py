import os
import joblib

def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{name}.joblib")
    print(f"Saved model: models/{name}.joblib")

def load_model(name):
    path = f"models/{name}.joblib"
    if os.path.exists(path):
        print(f"Loaded model: {path}")
        return joblib.load(path)
    else:
        print(f"Model not found: {path}")
        return None
