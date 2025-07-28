import os
import pandas as pd
from datetime import datetime

def get_hour(dt):
    return dt.hour

def get_season(dt):
    month = dt.month

    if month in [12, 1, 2]:
        return 1 
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4

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

    df = df.groupby(['SETTLEMENTDATE', 'hour', 'season', 'is_peak', 'REGIONID'], as_index=False).mean(numeric_only=True)

    df = df.pivot_table(
        index=['SETTLEMENTDATE', 'hour', 'season', 'is_peak'],
        columns='REGIONID',
        values=['AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 'SEMISCHEDULE_CLEAREDMW', 'RRP', 'is_cap'],
        aggfunc='first'
    )

    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df = df.reset_index()

    target_cap_col = f'is_cap_{target_region}'
    if target_cap_col in df.columns:
        df[f'is_cap_{target_region}'] = df[target_cap_col]
    else:
        print(f"Warning: Target region {target_region} not found in data")
        cap_cols = [col for col in df.columns if col.startswith('is_cap_')]
        if cap_cols:
            df[f'is_cap_{target_region}'] = df[cap_cols[0]]
            print(f"Using {cap_cols[0]} as target, renamed to is_cap_{target_region}")

    cols_to_drop = [col for col in df.columns if col.startswith('is_cap_') and col != f'is_cap_{target_region}']
    cols_to_drop.extend([col for col in df.columns if col.startswith('RRP_')])
    cols_to_drop.append('SETTLEMENTDATE')

    df = df.drop(columns=cols_to_drop)

    return df



def train_svm():
    pass

def train_lr():
    pass

def train_xgb():
    pass

def train_knn():
    pass

def evaluate_model():
    pass


df = pd.read_csv("merged_dispatch_price_2021-2025.csv")
df = df.sort_values('SETTLEMENTDATE')

# Preprocess data

target_region = "NSW1"

nem_df = preprocess_data(df, target_region)

# Time-based split (80/20)
split_index = int(len(nem_df) * 0.8)
train_df = nem_df.iloc[:split_index]
test_df = nem_df.iloc[split_index:]

# Separate features and target
X_train = train_df.drop([f'is_cap_{target_region}'], axis=1)
y_train = train_df[f'is_cap_{target_region}']
X_test = test_df.drop([f'is_cap_{target_region}'], axis=1)
y_test = test_df[f'is_cap_{target_region}']

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")
print(f"Features: {X_train.columns.tolist()}")

os.makedirs('state_data', exist_ok=True)
nem_df.to_csv(f'state_data/nem_{target_region}_preprocessed.csv', index=False)

print(nem_df.columns.tolist())

print(nem_df.head(20))
