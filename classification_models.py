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
 

def preprocess_data(df, region_id=None):
    # Convert and clean data
    df['RRP'] = pd.to_numeric(df['RRP'], errors='coerce')
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
    df = df.groupby(
        ['SETTLEMENTDATE', 'hour', 'season', 'is_peak', 'REGIONID'],
        as_index=False
    ).mean(numeric_only=True)

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

nem_df = preprocess_data(df, "")

print(nem_df.columns.tolist())

print(nem_df.head(20))
