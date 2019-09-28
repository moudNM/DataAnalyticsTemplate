#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

import re


# # Training Workshop Part IV - Portfolio analysis

# In[2]:

folder = "datasets/"
stock = pd.read_csv(folder + '1Y_Stock_Data.csv')
stock['Symbol'].unique()


# In[4]:


stock.head()


# In[5]:


stock.columns = stock.columns.map(lambda x: x.lower())
stock.rename(columns={'adj close':'adj_close'},inplace=True)
stock['date']=pd.to_datetime(stock['date'],format='%d/%m/%Y')


# In[6]:


stock.set_index(['date', 'symbol'], inplace=True)


# In[8]:


daily_price = stock['adj_close'].unstack()
daily_ret = daily_price.pct_change().dropna()
daily_ret.columns = daily_ret.columns.map(lambda x: x + '_return')


# In[9]:


daily_ret.head(20)


# In[10]:


import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.pairplot(daily_ret)


# In[11]:


# check worst drop in each stock
daily_ret.idxmin()


# In[12]:


# JD_return     2018-09-05
# Richard Liu rape allegation on 2018-09-02


# In[13]:


daily_ret.idxmax()


# In[17]:


# compare std of returns between FB and GS
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
sns.distplot(daily_ret.loc[:,'FB_return'],color='green',bins=100,label='FB return')
sns.distplot(daily_ret.loc[:,'GS_return'],color='red',bins=100, label='GS return')
ax.legend(loc=0)
ax.set_xlim(-0.1,0.1)
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
#import plotly


# In[20]:


daily_price.plot(figsize=(16,8),title='Daily Close Price')


# ## draw moving average

# In[21]:


# use FB as example
ma_days = [5,30]
fb = daily_price['FB'].to_frame()
for d in ma_days:
    column_name = 'MA_%s_days' % str(d)    
    fb[column_name] = daily_price.rolling(window=d, min_periods=1).mean()['FB']
    column_name = 'STD_%s_days' % str(d)    
    fb[column_name] = daily_price.rolling(window=d, min_periods=1).std()['FB']
fb.head(10)


# In[22]:


fb.loc[:,['FB', 'MA_5_days', 'MA_30_days']].plot(figsize=(16,8))
plt.legend(prop={'size': 20})
plt.show()


# ## correlation and risk

# In[23]:


# scatter plot to find correlation
# FB and BABA stock price are quite correlated
sns.jointplot('BABA', 'FB', daily_price, kind='scatter', height=8)
plt.show()


# In[24]:


# from correlation function we can see BABA and FB correlation is high
daily_price.corr()


# In[25]:


# std of returns
daily_ret.std()


# In[26]:


# add uppper and lower band (std)
fb['upper_band_30'] = fb['MA_30_days'] + (fb['STD_30_days'] * 2)
fb['lower_band_30'] = fb['MA_30_days'] - (fb['STD_30_days'] * 2)


# In[27]:


# set style, empty figure and axes
#plt.style.use('ggplot')
sns.set_style('white')

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)

# Get index values for the X axis for facebook DataFrame
x_axis = fb.index

# Plot shaded 21 Day Bollinger Band for Facebook
ax.fill_between(x_axis, fb['upper_band_30'], fb['lower_band_30'], color='aliceblue')

# Plot Adjust Closing Price and Moving Averages
ax.plot(x_axis, fb['FB'])#, color='blue', lw=2)
ax.plot(x_axis, fb['MA_30_days'])#, color='black', lw=2)

# Set Title & Show the Image
ax.set_title('30 Day Bollinger Band For Facebook')
ax.set_xlabel('Date (Year/Month)')
ax.set_ylabel('Price(USD)')
ax.set_xlim(fb.index.min(), fb.index.max())

ax.legend()

plt.legend(prop={'size': 20})
plt.show()


# In[452]:


fb.index.min().strftime('%Y-%d')


# In[ ]:




