import pandas as pd

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


    agg_dict = {
        'AVAILABLEGENERATION': 'mean',
        'AVAILABLELOAD': 'mean',
        'DEMANDFORECAST': 'mean',
        'SEMISCHEDULE_CLEAREDMW': 'mean',
        'RRP': 'mean',
        'is_cap': 'max'
    }
    cols_to_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    df = df.groupby(['SETTLEMENTDATE', 'hour', 'season', 'is_peak', 'REGIONID'], as_index=False).agg(cols_to_agg)

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
