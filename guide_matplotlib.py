"""
guide for matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as func

"""
pandas data display
set number of columns/rows to display
"""
desired_width = 1000  # width of data (increase to have all on one line)
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)  # no of columns
pd.set_option('display.max_rows', None)  # no of rows, works only for < 10, use None for all rows

data_directory = "datasets/"
file_name = "DatafinitiElectronicsProductsPricingData.csv"
data = pd.read_csv(data_directory + file_name)

# limit to 1st 500 items, amountMin <= 150, sorted by brand
df = data
df = df[['prices.amountMin', 'brand']]
df = df[(data['prices.amountMin'] > 100)]
df = df.sort_values(by='brand')
df = df.reset_index()
df = df.head(500)

count = len(df['brand'].value_counts())
print(len(df))

func.write_to_file(df, 'def', 'csv')


# plot graph
fig = plt.figure()
plot = fig.add_subplot(111)

"""
Single
"""

# x_points = []
# y_points = []
#
# for index, row in df.iterrows():
#     x_points.append(index)
#     y_points.append(row['prices.amountMin'])
#
# p = plot.plot(x_points, y_points, linestyle='--', marker='.', label=('min prices'))
#
# plot.set_xlabel('Index no.')
# plot.set_ylabel('Min price')
# plot.set_title('Test')
#
# plot.legend(bbox_to_anchor=(0.99, 1.1))
#
# graph_file_name = 'figure.png'
# plt.savefig(graph_file_name)
#
# plt.show()
# fig.show()
#
#
# func.write_to_file(df, 'def', 'csv')


"""
multiple graphs
"""

brands = []

curr_brand = df.at[0, 'brand']

x_points = []
y_points = []


for index, row in df.iterrows():

    if curr_brand == row['brand']: # same brand
        x_points.append(index)
        y_points.append(row['prices.amountMin'])

    else: # different brand
        brands.append([curr_brand, x_points, y_points])
        curr_brand = row['brand']
        x_points = []
        y_points = []
        x_points.append(index)
        y_points.append(row['prices.amountMin'])

    if index == (len(df)-1):
        brands.append([curr_brand, x_points, y_points])

for i in brands:
    p = plot.plot(i[1], i[2], linestyle='--', marker='.', label=(i[0]))

plot.set_xlabel('Index no.')
plot.set_ylabel('Min Price')
plot.set_title('Graph')

plot.legend(bbox_to_anchor=(0.99, 1.1))

graph_file_name = 'figure.png'
plt.savefig(graph_file_name)

plt.show()
fig.show()








