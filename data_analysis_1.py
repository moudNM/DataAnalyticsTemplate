#!/usr/bin/env python
# coding: utf-8

# # Training Workshop Part I
# https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html

# ## Import Package paul

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


a = 3
b = 6
print(a)
print(b)


# In[4]:


# This is a Markdown
a = 5
a


# In[46]:


print("this is a test")


# In[6]:


# ## Python Fundamentals
# 
# In this part, we will recap some fundamental operations in Python. 

# In[8]:


# print('comment')
type('123')
#type(1 == 2)


# In[9]:


a = 2 * 5 # assign values to variables
a # jupyter notebook won't print variable value in assignment


# In[55]:


a * 199 # this is an expression, result is not stored


# In[10]:


# list
ages = [15,17,10,23,26,27,29,30]


# In[11]:


new_ages = [e * 10 for e in ages] # for loop construction
new_ages


# In[12]:


# list can actually hold different types of values
combined = [15, 'str', 3.0, 'blah..']
combined


# In[13]:


# dict
menu = {
    "Big Mac": 3.99,
    "McSpicy": 5.45,
    "McWings": 4.25
}
menu


# In[14]:


# iterate list values
for e in ages:
    print(e)


# In[15]:


# iterate list with index and value
for i, e in enumerate(ages):
    print(i, e)


# In[16]:


# key-value iteration over map
for k, v in menu.items():
    print(k, v)


# In[17]:


for k in menu:
    print(k)


# ### function and lambda function

# In[18]:


# function
def cube(x):
    return x * x * x
print(cube)
print(cube(3))


# In[19]:


# lambda function
g = lambda x: x * x * x
print(g)
print(g(3))
f=g
print(f)


# In[20]:


# filter
nums = [5, 7, 22, 97, 54, 62, 77, 23, 73, 61] 
even_nums = list(filter(lambda x: (x%2 == 0) , nums))
print(filter(lambda x: (x%2 == 0) , nums))
print(even_nums)


# In[21]:


# map
nums_2x = list(map(lambda x: x * 2, nums))
print(nums_2x)
#nums_3x = list(map(g, nums))
nums_3x = list(map(cube, nums))
print(nums_3x)


# In[22]:


# reduce
from functools import reduce
nums_sum = reduce(lambda x, y: x + y, nums)
print(nums_sum)


# ## Numpy

# In[22]:


import numpy as np


# ### Array Basics

# In[23]:


x = np.array(nums)
print(x)
print(type(x))


# In[24]:


y = np.array([[2,5,6],[5,8,-1]])
print(y)
print(y.shape)
print(type(y.shape))


# In[28]:


# arrange [) (start from left boundary, with given steps)
m = np.arange(1,100,5)
print(m)
print(type(m))
print(m.shape)


# In[26]:


# linspace [] (evenly partitioned both boundaries are included)
m2 = np.linspace(1,10,7)
print(m2)
print(type(m2))
print(m2.dtype)


# In[31]:


# reshape
print(m)
print(m.reshape(10, 2))
print(m.reshape(4, 5))
#print(m.reshape(10,3))
print(m)


# In[32]:


# resize
# resize changes the variable while reshape does not
print(x)
x.resize(2, 5)
print(x)


# In[33]:


print(np.ones((3,5)))
print(np.zeros(8))


# In[34]:


np.eye(10)


# In[35]:


np.diag([2,5,6])


# In[36]:


t = np.ones((2,3), int)
print(t)
print(t + 10)


# In[37]:


print(np.hstack([t, t+10]))
print((np.hstack([t, t+10])).shape)


# In[38]:


print(np.vstack([t, t*2]))
# same as np.concatenate(t, t*2)


# ### Array Indexing and Slicing

# In[39]:


x = np.array([[ 5,  7, 22, 97, 54],
    [62, 77, 23, 73, 61]])
print(x)
print('-----------------')
print(x[0,0])
print(x[1,2])
print(x[0:])
print(x[:,-3])
print(x[-1,::2])


# In[42]:


index_temp = x>10
print(index_temp.dtype)
print(x[x>10])


# In[46]:


# distinguish between slicing and slice assignment
x = np.array([[ 5,  7, 22, 97, 54],
    [62, 77, 23, 73, 61]])
print(x)
r = x[:,-3] # create a slice of x, assign to r
print(r)
#r[:] = 1 # as slice points to the original array, changing values affect original array

r = 1 # this is assignment, only changes value for r
print(r)
print(x)


# In[47]:


r = 100
print(x)
x[:,-3] = 100
print(x)


# ### Array Operation

# In[40]:


a = [1.2,3.4,1.5]
b = [3,4,2]
from operator import mul
print(list(map(mul, a, b)))


# In[50]:


import numpy as np
a = [1.2,3.4,1.5]
b = [3,4,2]
print(a)
print(b)
print(np.array(a) * np.array(b))


# In[51]:


np.array([2,5,8]).dot(np.array([0,1,2]))


# In[52]:


x = np.array([[  5, 7, 1, 97, 54], [ 62, 77, 1, 100, 61]])
print(x)
x.T


# In[53]:


x.dtype


# In[54]:


x.shape


# In[55]:


x.sum()


# In[56]:


x.max()


# In[103]:


x[1,3] = 100
print(x)
x.argmax()


# In[57]:


np.sort(x)


# In[58]:


np.ndarray.flatten(x)


# In[59]:


np.median(x)


# In[60]:


print(np.mean(x))
print(np.sum(x)/np.size(x))


# In[61]:


from scipy import stats

print(stats.mode(np.ndarray.flatten(x)))

##### exercise: how to implement mode with numpy functions #####


# In[63]:


unique, counts = np.unique(x, return_counts=True)
print(unique)
print(counts)
print(counts.argmax())
print(x)


# In[64]:


np.split(x,2)


# In[65]:


# arbitrary split
test_list = [2,3,6,7,1,23,22,3]
split_array = np.split(test_list,[2,6])
for e in split_array:
    print(e, len(e))


# In[66]:


old = np.array([[1, 1, 1],
                [1, 1, 1]])

new = old
new[0, :2] = 0

print(old)

######### what is the output ###########


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Introduction to Dataframe

# In[44]:


import pandas as pd
folder = "datasets/"
titanic = pd.read_csv(folder + "titanic.csv")


# In[45]:


pd.options.display.max_rows = 30


# In[46]:


titanic


# In[8]:


# first n rows
titanic.head(5)


# In[197]:


titanic.columns


# In[11]:


titanic.index


# In[7]:


titanic.info()


# In[201]:


titanic.shape
# print(type(titanic.shape))


# In[8]:


titanic.describe()


# In[12]:


min(titanic)
# returns the minimum column label


# In[13]:


titanic.min()


# ## Built-in Functions and Methods

# In[219]:


titanic.shape


# In[203]:


len(titanic)


# In[218]:


titanic.size


# In[47]:


# round(dataframe, #digit)

round(titanic, 2)


# In[5]:


titanic.sort_values(by = 'Age', ascending=False)


# In[ ]:





# ## Indexing DataFrame
# 

# ### select column

# In[9]:


# select column
print(type(titanic['Age'])) # -> this returns a series
print(type(titanic[['Age']])) # -> this returns a dataframe


# In[11]:


print(type(titanic[['Age', 'Sex']]))


# In[12]:


# select column with dot
print(titanic.Age)


# In[14]:


# select column by dot equals to series as well
print(titanic.Age.equals(titanic['Age']))


# ### select rows

# In[11]:


# 1st method: iloc
# position based index
titanic.iloc[2]


# In[12]:


# last row
titanic.iloc[-1]


# In[13]:


# select row 2-4
titanic.iloc[2:5]


# In[18]:


# select last 5 row
titanic.iloc[-5:]


# In[22]:


# select a selection of rows
titanic.iloc[[2,5,141]] # -> params is a list


# ### combination of selection rows and columns

# In[25]:


# iloc: first position is row, second position is column
titanic.iloc[0,3]


# In[26]:


# a range selection
titanic.iloc[[0,2,4], 3]


# In[29]:


titanic.iloc[[0,2,4], [3,5,8,9]]


# In[31]:


titanic.iloc[5,:]


# In[104]:


# second method: loc
# loc is label-based (the value of index)
titanic.loc[3]


# In[49]:


newdf = titanic.set_index('Name')
newdf.loc['Rice, Master. Eugene']


# In[52]:


newdf.loc['Rice, Master. Eugene', ['Age', 'Ticket']]


# In[53]:


# slicing rows with loc
# label must be unique for slice
newdf.loc[:'Rice, Master. Eugene', ['Age', 'Ticket']]


# In[54]:


newdf.loc[:'Rice, Master. Eugene', 'Age':'Ticket']


# # Pandas Series

# In[ ]:


age = titanic['Age']
age.dtype


# ### each row/column is a Series!!

# In[ ]:


print(titanic.iloc[5])
print(type(titanic.iloc[5]))


# In[17]:


print(age)
print(type(age))


# In[59]:


age.describe()


# ## analyzing series

# In[60]:


age.sum()


# In[61]:


# sum cannot handle missing values
sum(age)


# In[63]:


age.size


# In[64]:


len(age)


# In[65]:


age.mean()


# In[72]:


age.mean(skipna=False)


# In[74]:


# unique is only available for series
age.unique()


# In[75]:


age.nunique()


# In[76]:


# nunique by default does not count nan/null
age.nunique(dropna=False)


# In[80]:


age.value_counts()


# In[83]:


age.value_counts(normalize=True)


# In[85]:


age.value_counts(sort=True).head()


# In[89]:


age.value_counts(bins=10,sort=False)


# ## create pandas series

# ### from dataframe

# In[4]:


# a single row and a single column is a series
print(type(titanic.Age))
print(type(titanic.iloc[0]))


# ### create from list

# In[5]:


pd.Series([2,5,6])


# In[6]:


# pass index
pd.Series([2,5,6], index=['two','five','six'])


# ### create from dictionary

# In[8]:


d = {"two":2, "five":5, "six":6}
pd.Series(d)


# ### sort series

# In[21]:


age = titanic.Age

# sort_values return a new series, won't affect original series
sorted_age = age.sort_values()
print(sorted_age.head())
print(age.head())


# In[3]:


# use inplace to replace original series
tmp_num = pd.Series([2,3,4,1,100,15])
tmp_num.sort_values(inplace=True)
print(tmp_num)


# In[4]:


tmp_num.nlargest(3)


# In[23]:


tmp_num.idxmin()


# ### test: what if there are multiple max/min?

# In[48]:


pd.Series([1,5,5,5,2,5,5,5]).idxmax()


# In[25]:


sales = pd.Series([10,5,6,None,100], index=['Mon','Tue','Wed','Thu','Fri'])


# In[26]:


print(sales)


# In[28]:


sales.iloc[2]


# In[32]:


(sales*12.466).round(2)


# ## Pandas Index

# In[33]:


sales.index


# In[34]:


titanic.columns


# In[35]:


titanic.index


# In[39]:


titanic.axes


# In[46]:


print(type(titanic.index))
print(type(titanic.columns))
# to change row index to base.Index, simple assign index as one of the column when loading csv


# ### change index (row/column index of dataframe)

# In[50]:


titanic.set_index('PassengerId', inplace=True)


# In[51]:


titanic


# In[52]:


titanic.reset_index()


# In[57]:


test_titanic = titanic.loc[:10,['Name', 'Sex']]


# In[58]:


test_titanic


# In[59]:


test_titanic.columns = ['New_name', "New_sex"]


# In[61]:


test_titanic


# In[ ]:




