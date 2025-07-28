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
    return 2 <= dt.hour <= 8 and dt.weekday() not in [5, 6]
 

def preprocess_data(df, region_id="NSW1"):
    # Convert and clean data - convert all required columns to numeric first
    numeric_columns = ['RRP', 'AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 'SEMISCHEDULE_CLEAREDMW']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where RRP is NaN
    df = df.dropna(subset=['RRP']).copy()
    
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    # Floor to half-hour blocks
    df['SETTLEMENTDATE'] = df['SETTLEMENTDATE'].dt.floor('30min')
    
    # Add time features
    df['hour'] = df['SETTLEMENTDATE'].apply(get_hour)
    df['season'] = df['SETTLEMENTDATE'].apply(get_season)
    df['is_peak'] = df['SETTLEMENTDATE'].apply(is_peak_hour).astype(int)
    df['is_cap'] = (df['RRP'] >= 300).astype(int)
    
    # If single region requested, return original format
    if region_id:
        return df[df['REGIONID'] == region_id].copy()
    
    # Group to 30-minute intervals with categorical context
    # Now that all columns are numeric, this should work
    df = df.groupby(
        ['SETTLEMENTDATE', 'hour', 'season', 'is_peak', 'REGIONID'],
        as_index=False
    ).mean(numeric_only=True)
    
    print("Columns before pivot:", df.columns.tolist())
    
    # Pivot: make each region a column
    df = df.pivot_table(
        index=['SETTLEMENTDATE', 'hour', 'season', 'is_peak'],
        columns='REGIONID', 
        values=['AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 
                'SEMISCHEDULE_CLEAREDMW', 'RRP', 'is_cap'],
        aggfunc='first'
    )
    
    # Flatten column names: ('RRP', 'NSW1') becomes 'RRP_NSW1'
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df = df.reset_index()
    
    # Drop all is_cap and RRP columns except for the selected region
    keep_cap = f'is_cap_{region_id}' if region_id else None
    cap_cols = [col for col in df.columns if col.startswith('is_cap_') and col != keep_cap]
    rrp_cols = [col for col in df.columns if col.startswith('RRP_')]
    df = df.drop(columns=cap_cols + rrp_cols)
    
    # Drop timestamp
    df = df.drop(columns=['SETTLEMENTDATE'])
    
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
nem_df = preprocess_data(df, "NSW1")

# Time-based split (80/20)
split_index = int(len(nem_df) * 0.8)
train_df = nem_df.iloc[:split_index]
test_df = nem_df.iloc[split_index:]

# Separate features and target
X_train = train_df.drop(['is_cap'], axis=1)
y_train = train_df['is_cap']
X_test = test_df.drop(['is_cap'], axis=1)
y_test = test_df['is_cap']

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")
print(f"Features: {X_train.columns.tolist()}")


nem_df = preprocess_data(df, "NSW1")

print(nem_df.columns.tolist())

print(nem_df.head(20))
