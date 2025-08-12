import pandas as pd

# Load both CSVs without parsing dates
dispatch_df = pd.read_csv(
    'nem_data/dispatch_history_2021-2025.csv',
    on_bad_lines='skip'
)

price_df = pd.read_csv(
    'price_data/price_history_2021-2025.csv',
    on_bad_lines='skip'
)

print(dispatch_df.columns.tolist())
# Keep only selected columns
dispatch_df = dispatch_df[[
    'SETTLEMENTDATE',
    'TOTALDEMAND',
    'AVAILABLEGENERATION',
    'AVAILABLELOAD',
    'DEMANDFORECAST',
    'SEMISCHEDULE_CLEAREDMW'
]]

price_df = price_df[[
    'SETTLEMENTDATE',
    'REGIONID',
    'RRP'
]]

# Merge on SETTLEMENTDATE
merged_df = pd.merge(dispatch_df, price_df, on='SETTLEMENTDATE', how='inner')

# Save the merged result
merged_df.to_csv('merged_dispatch_price_2021-2025.csv', index=False)

