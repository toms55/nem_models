import nemosis

print(nemosis.defaults.dynamic_tables) # check available dynamic_tables
print("\nCOLS:\n")
print(nemosis.defaults.table_columns['DISPATCHLOAD']) # check what columns a table has

start_time = '2021/03/01 00:00:00'
end_time = '2021/03/01 23:59:59'
price_table = 'DISPATCHPRICE'
load_table = 'DISPATCHLOAD'
price_data_cache = '/home/tom/Documents/nem_honours/price_data/'
load_data_cache = '/home/tom/Documents/nem_honours/load_data/'

# price_data = nemosis.dynamic_data_compiler(start_time, end_time, price_table, price_data_cache, fformat='csv')

load_data = nemosis.dynamic_data_compiler(start_time, end_time, load_table, load_data_cache, fformat='csv')


# price_data = nemosis.dynamic_data_compiler(start_time, end_time, table, raw_data_cache, filter_cols=['REGIONID'], filter_values=(['SA1', 'NSW1'],), fformat='csv', rebuild=True) # filter based on REGIONID

# print(load_data.head())
# print(load_data.info())
