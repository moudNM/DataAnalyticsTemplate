#!/usr/bin/env python
# coding: utf-8

# # Training Workshop Part III

# In[ ]:





# In[5]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime



# In[6]:


# plot s&p 500 stock price
folder = "datasets/"
sp = pd.read_csv(folder + 'GSPC.csv', index_col=0, parse_dates=True)


# In[7]:


# Date as index and converted to date type
sp.head()


# In[8]:


sp.index


# In[9]:


# load data directly and convert string to datetime
folder = "datasets/"
raw_sp = pd.read_csv(folder + 'GSPC.csv')
raw_sp.info()


# In[10]:


raw_sp['Date'] = pd.to_datetime(raw_sp['Date']) # replaced Date column
raw_sp.set_index('Date', inplace=True)
raw_sp.index


# In[11]:


# generate a datetime index
pd.date_range(pd.Timestamp("2018-03-10"), periods=21, freq='d')


# In[12]:


# convert datetime to string
raw_sp.index.strftime('%Y-%m-%d')


# In[13]:


# resample
# resample daily to fill all missing dates
raw_sp.resample('D').sum().head()


# In[14]:


# resample yearly to aggreate
raw_sp.resample('Y').mean()


# In[15]:


# date is still timestamp format, but we need year only
raw_sp.resample('Y',kind='period').mean()


# In[16]:


# now becomes period index
raw_sp.resample('Y',kind='period').mean().index


# ## start to plot

# In[17]:


spx = sp.loc['2007-01-01':'2009-12-31',['Adj Close']]
spx.sort_index()
# if plot only contains y values, then x values will automatically be the range start from 0
plt.plot(spx.values)
plt.show()


# ## plot refinement

# ### <font color='red'> issues - plot is too small. lots of information is missing </font>

# In[18]:


# some pre-defined
r_hex = '#dc2624'    # red, RGB = 220,38,36
dt_hex = '#2b4750'   # dark teal, RGB = 43,71,80
tl_hex = '#45a0a2'   # teal, RGB = 69,160,162
r1_hex = '#e87a59'   # red,  RGB = 232,122,89
tl1_hex = '#7dcaa9'  # teal, RGB = 125,202,169
g_hex = '#649E7D'    # green, RGB = 100,158,125
o_hex = '#dc8018'    # orange, RGB = 220,128,24
tn_hex = '#C89F91'   # tan, RGB = 200,159,145
g50_hex = '#6c6d6c'  # grey-50, RGB = 108,109,108 
bg_hex = '#4f6268'   # blue grey, RGB = 79,98,104
g25_hex = '#c7cccf'  # grey-25, RGB = 199,204,207


# In[19]:


# print some of the properties of the plot
print( 'figure size:', plt.rcParams['figure.figsize'] )
print( 'figure dpi:',plt.rcParams['figure.dpi'] )
print( 'line color:',plt.rcParams['lines.color'] ) 
print( 'line style:',plt.rcParams['lines.linestyle'] ) 
print( 'line width:',plt.rcParams['lines.linewidth'] )
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
ax.plot( spx['Adj Close'].values )
print( 'xticks:', ax.get_xticks() ) 
print( 'yticks:', ax.get_yticks() ) 
print( 'xlim:', ax.get_xlim() ) 
print( 'ylim:', ax.get_ylim() )


# ### change size

# In[20]:


# create a figure of size 16*6 points using 100 dots per inch
plt.figure(figsize=(16,6), dpi=100)
plt.plot(spx.values)
plt.show()


# ### change line color, width and style

# In[21]:


plt.figure( figsize=(16,6), dpi=100 )
plt.plot( spx.values, color=dt_hex, linewidth=2, linestyle='-' )
plt.show()


# ### set plot boundaries

# In[22]:


# a figure can contain multiple plots
# to control x/y axis, need to add an ax to the figure
fig = plt.figure( figsize=(16,6), dpi=100 )
# subplot (row, col, position)
ax = fig.add_subplot(1,1,1)
# x is an index object! make dataframe natrually tailored to plot
# think of using numpy.ndarray to plot - need to set up x and y separately
x = spx.index
y = spx.values
plt.plot(x, y, color=dt_hex, linewidth=2, linestyle='-' )
ax.set_xlim('2007-01-01','2009-12-31')
ax.set_ylim(y.min()*0.95, y.max()*1.05)
plt.show()


# ### <font color='red'> issues: x axis does not have day, and too sparse </font>

# ### set up ticks and tick labels

# In[23]:


# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y = spx.values
plt.plot(x, y, color=dt_hex, linewidth=2, linestyle='-' )

# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
ax.set_ylim(y.min()*0.95, y.max()*1.05)

# set up ticks and tick label
ax.set_xticks(x[range(0,len(x),20)])
# iteration to generate list (recall list generation from class 1)
# strftime convert a datetime format to string
ax.set_xticklabels([x[i].strftime('%Y-%m-%d') for i in range(0,len(x),10)], rotation=90)
plt.show()


# ### add legend

# In[25]:


# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y = spx.values
# need to give plot a label in order to display legend
plt.plot(x, y, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')

# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
ax.set_ylim(y.min()*0.95, y.max()*1.05)

# add legend
ax.legend(loc=0)

# set up ticks and tick label
ax.set_xticks(x[range(0,len(x),20)])
ax.set_xticklabels([x[i].strftime('%Y-%m-%d') for i in range(0,len(x),10)], rotation=90)
plt.show()


# ### add a second plot

# In[26]:


# import VIX data
folder = "datasets/"
data = pd.read_csv(folder + 'vixcurrent.csv', index_col=0, parse_dates=True)
vix = data.loc['2007-01-01':'2009-12-31', ['VIX Close']]


# In[27]:


vix.head()


# In[28]:


# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y1 = spx.values
y2 = vix.values
# need to give plot a label in order to display legend
plt.plot(x, y1, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')
plt.plot(x, y2, color=r_hex, linewidth=2, linestyle='-', label='VIX')


# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
# np.vstack stacks two ndarray vertically (recall from class 1)
ax.set_ylim(np.vstack([y1,y2]).min()*0.9, np.vstack([y1,y2]).max()*1.1)

# add legend
ax.legend(loc=0)

# set up ticks and tick label
ax.set_xticks(x[range(0,len(x),20)])
ax.set_xticklabels([x[i].strftime('%Y-%m-%d') for i in range(0,len(x),10)], rotation=90)
plt.show()


# ### <font color='red'> issue: the measure of VIX and S&P 500 is different </font>

# ### two axis

# In[29]:


# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y1 = spx.values
y2 = vix.values
# need to give plot a label in order to display legend
# ax.plot and plt.plot is exchangeable
ax.plot(x, y1, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')


# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
# np.vstack stacks two ndarray vertically (recall from class 1)
ax.set_ylim(np.vstack([y1,y2]).min()*0.9, np.vstack([y1,y2]).max()*1.1)

# add legend
ax.legend(loc='upper left')

# set up another axis
ax2 = ax.twinx()
ax2.plot(x, y2, color=r_hex, linewidth=2, linestyle='-', label='VIX')
ax2.legend(loc='upper right')

# set up ticks and tick label
xticks = x[range(0,len(x),20)]
ax.set_xticks(xticks)
ax.set_xticklabels([i.strftime('%Y-%m-%d') for i in xticks], rotation=90)

plt.show()


# ### two subplots

# In[30]:


# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )


# subplot 1, two rows 1 column
plt.subplot(2,1,1)
x = spx.index
y1 = spx.values
plt.plot(x, y1, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')

xticks = x[range(0,len(x),20)]
xlabels = [i.strftime('%Y-%m-%d') for i in xticks]

# axis and plt have slightly different grammar

plt.xlim('2007-01-01','2009-12-31')
plt.ylim(y1.min()*0.9, y1.max()*1.1)
plt.legend(loc='upper left')
plt.xticks(xticks, xlabels, rotation=45)


# subplot 2
plt.subplot(2,1,2)
y2 = vix.values
plt.plot(x, y2, color=r_hex, linewidth=2, linestyle='-', label='VIX')
# axis and plt have slightly different grammar
plt.xlim('2007-01-01','2009-12-31')
plt.ylim(y2.min()*0.9, y2.max()*1.1)
plt.legend(loc='upper left')
plt.xticks(xticks, xlabels, rotation=45)

# adjust space between two plots
plt.subplots_adjust(hspace=0.4)
plt.show()


# ### annotations

# In[31]:


# annotate following 5 incidents on the plot
# 2007-10-11: peak of bull market
# 2008-03-12: Bear Stearns collapse
# 2008-09-15: Lehman Brothers collapse
# 2009-01-20: RBS sell-off
# 2009-04-02: G20 summit


# In[32]:


# need to scatter plot on existing figure
# first let's test to make sure the axis is correct

# define incidents
# a list of tuples
crisis_data = [
    (datetime(2007,10,11), 'Peak of Bull Market'),
    (datetime(2008,3,12), 'Bear Stearns Collapse'),
    (datetime(2008,9,15), 'Lehman Brothers Collapse'),
    (datetime(2009,1,20), 'RBS Sell-off'),
    (datetime(2009,4,2), 'G20 Summit')
]


fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)

xticks = x[range(0,len(x),20)]
ax.set_xticks(xticks)
ax.set_xticklabels([i.strftime('%Y-%m-%d') for i in xticks], rotation=90)
ax.set_xlim('2007-01-01','2009-12-31')
ax.set_ylim(np.vstack([y1,y2]).min()*0.8, np.vstack([y1,y2]).max()*1.2)

for xi, label in crisis_data:
    yi = spx.loc[xi].values
    plt.scatter(xi, yi, 100, color = r_hex)
    plt.annotate(label, # this is the text
                 (xi,yi), # this is the point to label
                 #textcoords="offset points", # how to position the text
                 xytext=(xi,yi+260), # distance from text to points (x,y)
                 arrowprops=dict(facecolor='black', headwidth=4, width=1, headlength=6, shrink=0.2),
                 ha='left',va='top') # horizontal/vertical alignment can be left, right or center
plt.show()


# In[33]:


# Now let's combine them together

# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y1 = spx.values
y2 = vix.values
# need to give plot a label in order to display legend
# ax.plot and plt.plot is exchangeable
ax.plot(x, y1, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')


# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
# np.vstack stacks two ndarray vertically (recall from class 1)
ax.set_ylim(np.vstack([y1,y2]).min()*0.8, np.vstack([y1,y2]).max()*1.2)

# add legend
ax.legend(loc='upper left')



for xi, label in crisis_data:
    yi = spx.loc[xi].values
    plt.scatter(xi, yi, 100, color = r_hex)
    plt.annotate(label, # this is the text
                 (xi,yi), # this is the point to label
                 #textcoords="offset points", # how to position the text
                 xytext=(xi,yi+260), # distance from text to points (x,y)
                 arrowprops=dict(facecolor='black', headwidth=4, width=1, headlength=6, shrink=0.2),
                 ha='left',va='top') # horizontal/vertical alignment can be left, right or center



# set up another axis
ax2 = ax.twinx()
ax2.plot(x, y2, color=r_hex, linewidth=2, linestyle='-', label='VIX')
ax2.legend(loc='upper right')

# set up ticks and tick label
xticks = x[range(0,len(x),20)]
ax.set_xticks(xticks)
ax.set_xticklabels([i.strftime('%Y-%m-%d') for i in xticks], rotation=90)

plt.show()


# ### <font color='red'> issues: too messy with annotations </font>

# In[34]:


# make VIX plot opaque

# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y1 = spx.values
y2 = vix.values
# need to give plot a label in order to display legend
# ax.plot and plt.plot is exchangeable
ax.plot(x, y1, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')


# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
# np.vstack stacks two ndarray vertically (recall from class 1)
ax.set_ylim(np.vstack([y1,y2]).min()*0.8, np.vstack([y1,y2]).max()*1.2)

# add legend
ax.legend(loc='upper left')



for xi, label in crisis_data:
    yi = spx.loc[xi].values
    plt.scatter(xi, yi, 80, color = r_hex)
    plt.annotate(label, # this is the text
                 (xi,yi), # this is the point to label
                 #textcoords="offset points", # how to position the text
                 xytext=(xi,yi+280), # distance from text to points (x,y)
                 arrowprops=dict(facecolor='black', headwidth=4, width=1, headlength=6, shrink=0.25),
                 ha='left',va='top') # horizontal/vertical alignment can be left, right or center



# set up another axis
ax2 = ax.twinx()
# change VIX plot to opaque
ax2.plot(x, y2, color=r_hex, linewidth=2, linestyle='-', label='VIX', alpha = 0.3)
ax2.legend(loc='upper right')

# set up ticks and tick label
xticks = x[range(0,len(x),20)]
ax.set_xticks(xticks)
ax.set_xticklabels([i.strftime('%Y-%m-%d') for i in xticks], rotation=90)

plt.show()


# In[407]:


# add annotations' corresponding date

# set figure size and line style
fig = plt.figure( figsize=(16,6), dpi=100 )
ax = fig.add_subplot(1,1,1)
x = spx.index
y1 = spx.values
y2 = vix.values
# need to give plot a label in order to display legend
# ax.plot and plt.plot is exchangeable
ax.plot(x, y1, color=dt_hex, linewidth=2, linestyle='-', label='S&P 500')


# set up x/y limit
ax.set_xlim('2007-01-01','2009-12-31')
# np.vstack stacks two ndarray vertically (recall from class 1)
ax.set_ylim(np.vstack([y1,y2]).min()*0.8, np.vstack([y1,y2]).max()*1.2)

# add legend
ax.legend(loc='upper left')



for xi, label in crisis_data:
    yi = spx.loc[xi].values
    plt.scatter(xi, yi, 80, color = r_hex)
    plt.annotate(label, # this is the text
                 (xi,yi), # this is the point to label
                 #textcoords="offset points", # how to position the text
                 xytext=(xi,yi+280), # distance from text to points (x,y)
                 arrowprops=dict(facecolor='black', headwidth=4, width=1, headlength=6, shrink=0.25),
                 ha='left',va='top') # horizontal/vertical alignment can be left, right or center



# set up another axis
ax2 = ax.twinx()
# change VIX plot to opaque
ax2.plot(x, y2, color=r_hex, linewidth=2, linestyle='-', label='VIX', alpha = 0.3)
ax2.legend(loc='upper right')

# set up ticks and tick label
# need to store the string of dates in crisis_data
impt_tick_label = [x[0].strftime('%Y-%m-%d') for x in crisis_data]
impt_tick = pd.Index([x[0] for x in crisis_data])

# combine two index together
xticks = x[range(0,len(x),55)].union(impt_tick)
ax.set_xticks(xticks)
ax.set_xticklabels([i.strftime('%Y-%m-%d') for i in xticks], rotation=90)

for idx in ax.get_xticklabels():
    # idx is a matplotlib.text type
    if idx.get_text() in impt_tick_label:
        idx.set_color(r_hex)
        idx.set_fontweight('bold')

plt.show()


# ## types of plot

# In[35]:


# check stock price (adjusted close) distribution of S&P 500
fig = plt.figure(figsize=(8,4))
plt.hist(spx.values, bins=30, color=dt_hex)
plt.xlabel('S&P 500 Adjusted Close Price')
plt.ylabel('Number of Days Observed')
plt.title('Frequency Distribution of S&P 500, Jan-2007 to Dec-2009')
plt.show()


# In[36]:


# scatter plot
#sp['Year']
nsp = sp.reset_index()
nsp['Year'] = nsp['Date'].map(lambda x: x.strftime('%Y'))
avg_vol = nsp.groupby('Year')['Volume'].mean()
avg_price = nsp.groupby('Year')['Adj Close'].mean()


# In[37]:


# trend of volume and price for s&p 500 yearly
fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(14,6))
axes[0].scatter(avg_vol.index, avg_vol.values, color=dt_hex)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Averaged Volume')
axes[0].set_title('Yearly Averaged S&P 500 Transaction Volume from 2004 to 2019')

axes[1].scatter(avg_price.index, avg_price.values, color=r_hex)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Averaged Price')
axes[1].set_title('Yearly Averaged S&P 500 Close Price from 2004 to 2019')

plt.subplots_adjust(hspace=0.4)

plt.show()


# In[38]:


# trend of volume and price for s&p 500 yearly
# use resample instead of group by

rsp = sp.resample('Y',kind='period').mean()
avg_vol = rsp['Volume']
avg_price = rsp['Adj Close']


fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(14,6))
axes[0].scatter(avg_vol.index.astype(str), avg_vol.values, color=dt_hex)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Averaged Volume')
axes[0].set_title('Yearly Averaged S&P 500 Transaction Volume from 2004 to 2019')

axes[1].scatter(avg_price.index.astype(str), avg_price.values, color=r_hex)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Averaged Price')
axes[1].set_title('Yearly Averaged S&P 500 Close Price from 2004 to 2019')

plt.subplots_adjust(hspace=0.4)

plt.show()


# In[39]:


# Pie chart
# load 1Y stock data
folder = "datasets/"
stock = pd.read_csv(folder + '1Y_Stock_Data.csv')
stock.sort_values('Symbol',inplace=True)
stock.tail()


# In[40]:


# how many stocks are there
stock.Symbol.unique()


# In[41]:


# calculate stock price on 26/02/2019
stock_list = ['AAPL', 'BABA', 'FB', 'GS', 'JD']
share = np.array([100,20,50,10,50])

# assume a portfolio with share listed below on these 5 stocks 
price = stock.loc[(stock.Date=='26/02/2019'), 'Adj Close']


# In[42]:


total = np.array(price.values) * share
fig = plt.figure(figsize=(16,6))
ax = fig.add_subplot(1,1,1)
ax.pie(total, labels=stock_list, colors=[dt_hex,r_hex,g_hex,tn_hex,g25_hex], autopct='%.0f%%')
plt.show()


# In[45]:


# let's reorder it, put majority to right-hand side and order in clock-wise
idx = total.argsort()[::-1]
fig = plt.figure(figsize=(16,6))
ax = fig.add_subplot(1,1,1)
ax.pie(total[idx], labels=[stock_list[i] for i in idx], 
       colors=[dt_hex,r_hex,g_hex,tn_hex,g25_hex], 
       startangle=90, counterclock=False,
       autopct='%.0f%%')
plt.show()


# In[46]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
total_ordered = total[idx]
pct = total_ordered/np.sum(total_ordered)
stock_list_ordered = [stock_list[i] for i in idx]
xticks = np.arange(len(pct))
ax.bar(xticks, pct, facecolor=r_hex, edgecolor=dt_hex)
ax.set_xticks(xticks)
ax.set_xticklabels(stock_list_ordered)

for x,y in zip(xticks,pct):
    ax.text(x+0.04, y+0.05/100, '{0:.0%}'.format(y), ha='center', va='bottom')
plt.show()


# In[532]:


# adot ggplot style
plt.style.use('ggplot')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
total_ordered = total[idx]
pct = total_ordered/np.sum(total_ordered)
stock_list_ordered = [stock_list[i] for i in idx]
xticks = np.arange(len(pct))
ax.bar(xticks, pct)
ax.set_xticks(xticks)
ax.set_xticklabels(stock_list_ordered)

for x,y in zip(xticks,pct):
    ax.text(x+0.04, y+0.05/100, '{0:.0%}'.format(y), ha='center', va='bottom')
plt.show()


# In[533]:


# adot ggplot style
plt.style.use('ggplot')

idx = total.argsort()[::-1]
fig = plt.figure(figsize=(16,6))
ax = fig.add_subplot(1,1,1)
ax.pie(total[idx], labels=[stock_list[i] for i in idx], 
       startangle=90, counterclock=False,
       autopct='%.0f%%')
plt.show()


# In[ ]:


# skipped line chart, as we have intensively work on it previously

