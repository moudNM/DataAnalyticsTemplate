"""
Use this file to select data
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


data_directory = "datasets/"
file_name = "DatafinitiElectronicsProductsPricingData.csv"
data = pd.read_csv(data_directory + file_name)

df = data['prices.amountMin']

print(df)