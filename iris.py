'''
IRIS DATASET
'''

# required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import matplotlib as mpl
import matplotlib.pyplot as plt

# read the dataset
folder = 'datasets/'
data = pd.read_csv(folder + 'iris.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

"""
encode label with values btw 0 and n-1
(assign classes/label)
"""
# label encode the target variable
encode = LabelEncoder()
data.Species = encode.fit_transform(data.Species) # use species as classes

# print(data.head(60))

"""
Create training and testing data sets
"""
# train-test-split
train, test = train_test_split(data, test_size=0.2, random_state=0) # split it 80-20

# print('\n\ntest \n\n')
# print(test)

print('shape of training data : ', train.shape)
print('shape of testing data', test.shape)

# seperate the target and independent variable
train_x = train.drop(columns=['Species'], axis=1) # remove species label (input/training)
train_y = train['Species'] # (target)

print(train_x)
# print(train_y)

test_x = test.drop(columns=['Species'], axis=1) # remove species label
test_y = test['Species']

"""
Create model and train

Logistic regression 
- use when the dependent variable is dichotomous (binary).
- predictive analysis.
- Describe data and explain the relationship between one dependent binary variable
and one or more nominal, ordinal, interval or ratio-level independent variables.
"""

# create the object of the model
model = LogisticRegression()
model.fit(train_x, train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data', encode.inverse_transform(predict)) # change from int to values (species)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y, predict)) # compare prediction and label

spx = data.loc[:,['sepalwidth']]
spx.sort_index()
# if plot only contains y values, then x values will automatically be the range start from 0
plt.plot(spx.values)
plt.show()
