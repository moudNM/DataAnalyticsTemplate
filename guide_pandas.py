"""
guide for pandas, numpy
"""

import pandas as pd
import numpy as np
import functions as func


"""
pandas data display
set number of columns/rows to display
"""
desired_width = 1000                             # width of data (increase to have all on one line)
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)        # no of columns
pd.set_option('display.max_rows', None)         # no of rows, works only for < 10, use None for all rows

"""
Read data files
Change directory and file names as necessary
"""
data_directory = "datasets/"
file_name = "DatafinitiElectronicsProductsPricingData.csv"
data = pd.read_csv(data_directory + file_name)      # replace with read_json etc for diff files


"""
print first or last few data points
uncomment as required
"""
# df = data.head()        # first 5
# df = data.tail()      # last 5
# df = data.head(4)     # first n
# df = data.tail(7)     # last n
# print(df)


"""
Create new dataframe or
create new series (one dimensional array, only 1 column)
"""
# df = pd.DataFrame(columns=['Col 1', 'Col 2'])                       # new dataframe
# df = pd.DataFrame([[1, 2], [3, 4]], columns=['Col 1', 'Col 2'])     # new dataframe with data
#
# some_data = np.array(['hello','bye','gracias'])     # new array of data
# series = pd.Series(some_data)                       # new series
#
# print(df)

"""
create series from dataframe
"""
# some_data = data['prices.merchant']
# series = pd.Series(some_data)
# print(series.value_counts())


"""
select col by name
"""
# df = data['id']
# df2 = data[['id', 'brand']]
#
# print(df2)

"""
get rows by labels (row and col headers)
left hand argument(s) - rows
right hand arguments(s) - column
"""
# df = data.loc[:, 'brand']                           # 1 specific column
# df2 = data.loc[:, ['brand', 'prices.amountMax']]    # multiple specific columns
# df3 = data.loc[5]                                   # row at one index
# df4 = data.loc[[1, 5], :]                           # rows at range of indexes
#
# df5 = data.loc[3, ['brand', 'prices.amountMax']]
# df6 = data.loc[0:5, ['brand', 'prices.amountMax']]
#
# print(df6)


"""
get rows by index
left hand argument(s) - rows
right hand arguments(s) - column
"""
# df = data.iloc[:, [0]]                           # 1 specific column
# df2 = data.iloc[:, [0, 2]]    # multiple specific columns
# df3 = data.iloc[5]                                   # row at one index
# df4 = data.iloc[[1, 5], :]                           # rows at range of indexes
#
# df5 = data.iloc[3, [0,2]]
# df6 = data.iloc[0:10, [0,2]]
#
# print(df6)


"""
sort, count etc
"""
# df = data
#
# df_sorted = df.sort_values(by='brand')     # sort values
# print(df_sorted.head(20))
#
# count = df['brand'].value_counts()   # count occurence
# print(count)


"""
filter rows,
based on multiple conditions
"""
# df = data[(data['brand'] == 'Yamaha') & (data['prices.amountMin'] > 200)]
# print(df)


