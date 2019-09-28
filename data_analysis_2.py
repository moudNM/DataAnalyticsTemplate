#!/usr/bin/env python
# coding: utf-8

# # Training Workshop Part II

# In[ ]:





# ## DataFrame filtering

# In[2]:


import pandas as pd
folder = "datasets/"
titanic = pd.read_csv(folder + 'titanic.csv')
titanic.head(5)


# In[3]:


titanic.Sex == 'male'


# In[4]:


titanic.loc[titanic.Sex == 'male'].head()


# In[5]:


# filter by multiple conditions
# filter out male with age > 20
titanic.loc[(titanic.Sex == 'male') & (titanic.Age > 20)].head() # brackets are required
#titanic.loc[titanic.Sex == 'male' && titanic.Age > 20].head()


# In[6]:


titanic.loc[(titanic.Sex == 'male') | (titanic.Age > 50)].head() # brackets are required


# In[7]:


titanic.loc[titanic.Age.between(20,25)].head()


# In[8]:


titanic.loc[titanic.Age.isin([20,25])].head()


# In[28]:


titanic.loc[~titanic.Age.isin([20,25])].head()


# In[9]:


# any and all
# is any passneger is 80-year old
(titanic.Age==80).any()


# In[10]:


# string contains
titanic[titanic.Name.str.contains('Miss')].head()


# ### insert/drop rows/columns

# In[11]:


test_titanic = titanic.loc[:,['Name','Age']]


# In[12]:


test_titanic['IsSenior']=test_titanic.Age > 55


# In[13]:


test_titanic.head()


# In[14]:


test_titanic.shape[0]


# In[15]:


# add single row
test_titanic.loc[test_titanic.shape[0]] = ['this is test', 10, False]


# In[16]:


test_titanic.shape[0]


# In[19]:


# rename column
test_titanic.rename(columns = {'Name': 'PassengerName'}, inplace=True)


# In[20]:


test_titanic.head()


# ### sort dataframe

# In[21]:


titanic.sort_values(['Age', 'Fare']).head(20)


# In[22]:


titanic['FareRank'] = titanic.Fare.rank(method='min')


# In[23]:


titanic.head()


# In[24]:


titanic.nunique()


# In[25]:


# count without missing values
titanic.count()


# In[26]:


titanic.nlargest(n=5, columns=['Fare']).head()


# In[27]:


# measure correlation
titanic.corr()


# In[28]:


# describe on descrete values
titanic['Embarked'].describe()


# In[29]:


# describe on continuous values
titanic['Age'].describe()


# In[ ]:





# 

# In[30]:


nt = titanic.iloc[1:50,:]
nt


# In[31]:


nt.set_index(['Pclass', 'Sex'], inplace=True)


# In[32]:


nt.head()


# In[13]:


nt.sort_index()


# In[33]:


nt.loc[1]


# In[34]:


nt.loc[[2,3]]


# In[35]:


nt.loc[(slice(None), ['female','male']), ['Name']]


# In[36]:


nt.index


# In[37]:


price = [190,32,196,192,200,189,31,30,199]
dates = ['2019-04-01']*3 + ['2019-04-02']*2 +['2019-04-03']*2 + ['2019-04-04']*2
codes = ['BABA','JD','GS','BABA','GS','BABA','JD','JD','GS']


# In[38]:


data = pd.Series(price,index=[ dates, codes ])


# In[39]:


data


# In[40]:


data.index


# In[41]:


titanic.set_index('PassengerId', inplace=True)
titanic.index


# In[42]:


nt.swaplevel(0,1)


# In[ ]:





# # data cleaning

# In[43]:


import pandas as pd

# load S&P 500 index data
folder = "datasets/"
sp = pd.read_csv(folder + 'GSPC.csv')


# In[46]:


sp.head(-10)


# ## missing values

# In[47]:


sp.info()


# In[48]:


sp.isna().sum()


# In[49]:


# print rows with missing values
print(sp.isna().any(axis=1).head())

sp[sp.isna().any(axis=1)]


# In[122]:


#remove na values
sp.dropna().info()


# In[50]:


# require each row to have at least 3 non-null values
sp.dropna(axis=0,thresh=3).info()


# In[52]:


# fill the missing values with mean
vol_mean = round(sp.Volume.mean(), 0)
print(vol_mean)


# In[53]:


sp.Volume.fillna(vol_mean, inplace=True)
sp.info()


# ## duplicates

# In[54]:


# check Volume
sp.duplicated(subset=['Date']).any()


# In[55]:


sp[sp.duplicated(subset=['Date'], keep=False)]


# In[56]:


tmp = sp.drop_duplicates()
tmp[tmp.duplicated(subset=['Date'], keep=False)]


# ## outliers

# In[ ]:





# In[137]:


sp.describe()
# negative values are outliers


# In[139]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[140]:


sp.Volume.plot()


# In[141]:


sp[sp.High<0]


# In[147]:


sp.loc[sp.High<0,['High']] = sp.loc[sp.High>0,['High']].mean()
sp[sp.High<0]


# In[ ]:




