#!/usr/bin/env python
# coding: utf-8

# Predict the percentage of an student based on the no. of study hours. 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


path = "http://bit.ly/w-data"
data_load = pd.read_csv(path)
data_load


# # preprocessing

data_load.info()

data_load.describe()


data_load.plot(x='Hours', y='Scores', style='o')
plt.title("Study Hours VS Percentage Scored")
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()


data_load.isnull().sum()


data_load.mean()


# # spliting the dataset

X = data_load.iloc[:, :-1].values
y = data_load.iloc[:, 1].values
print("shape of x:", X.shape)
print("shape of y:", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

print("shape of X train:", X_train.shape)
print("shape of X_test:", X_test.shape)
print("shape of y:", y_train.shape)
print("shape of y:", y_test.shape)



# # selecting and training the model

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()

regressor.fit(X_train ,y_train)

regressor.coef_

regressor.intercept_

y_pred = regressor.predict(X_test)
y_pred


# # plotting


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


df = pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["Study Hours", "students_original_marks", "Students_predicted_marks"])
df


# # accuracy

regressor.score(X_test, y_test)


# # plotting

plt.scatter(X_test, y_test)

plt.scatter(X_test, y_test)
plt.plot(X_train, regressor.predict(X_train), color="red")


# # presenting solution

hours = [[2]]
own_pred = regressor.predict(hours)
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))





