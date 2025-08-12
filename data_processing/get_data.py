import nemosis

print(nemosis.defaults.dynamic_tables) # check available dynamic_tables
print("\nCOLS:\n")
print(nemosis.defaults.table_columns['DISPATCH_UNIT_SCADA']) # check what columns a table has

start_time = '2021/01/01 00:00:00'
end_time = '2025/07/24 23:59:59'

price_table = 'DISPATCHPRICE'
load_table = 'DISPATCHLOAD'
misc_table = 'DISPATCHREGIONSUM'

price_data_cache = '/home/tom/Documents/nem_models/price_data/'
load_data_cache = '/home/tom/Documents/nem_models/load_data/'
misc_data_cache = '/home/tom/Documents/nem_models/nem_data/' # for checking table contents

price_data = nemosis.dynamic_data_compiler(start_time, end_time, price_table, price_data_cache, fformat='csv')

# load_data = nemosis.dynamic_data_compiler(start_time, end_time, load_table, load_data_cache, fformat='csv')


# price_data = nemosis.dynamic_data_compiler(start_time, end_time, table, price_data_cache, filter_cols=['REGIONID'], filter_values=(['SA1', 'NSW1'],), fformat='csv', rebuild=True) # filter based on REGIONID

misc_data = nemosis.dynamic_data_compiler(start_time, end_time, misc_table, misc_data_cache, fformat='csv', rebuild=True)

print(misc_data.head())
print(misc_data.info())
