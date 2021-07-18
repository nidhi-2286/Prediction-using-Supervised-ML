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

# In[34]:


data_load.info()


# In[ ]:


data_load.describe()


# In[22]:


data_load.plot(x='Hours', y='Scores', style='o')
plt.title("Study Hours VS Percentage Scored")
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()


# In[36]:


data_load.isnull().sum()


# In[37]:


data_load.mean()


# # spliting the dataset

# In[38]:


X = data_load.iloc[:, :-1].values
y = data_load.iloc[:, 1].values
print("shape of x:", X.shape)
print("shape of y:", y.shape)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# In[39]:


print("shape of X train:", X_train.shape)
print("shape of X_test:", X_test.shape)
print("shape of y:", y_train.shape)
print("shape of y:", y_test.shape)


# # selecting and training the model

# In[ ]:





# In[44]:


from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()


# In[45]:


regressor.fit(X_train ,y_train)


# In[48]:


regressor.coef_


# In[49]:


regressor.intercept_


# In[51]:


y_pred = regressor.predict(X_test)
y_pred


# # plotting

# In[26]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[54]:


df = pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["Study Hours", "students_original_marks", "Students_predicted_marks"])
df


# # accuracy

# In[55]:


regressor.score(X_test, y_test)


# # plotting

# In[56]:


plt.scatter(X_test, y_test)


# In[57]:


plt.scatter(X_test, y_test)
plt.plot(X_train, regressor.predict(X_train), color="red")


# # presenting solution

# In[58]:


hours = [[2]]
own_pred = regressor.predict(hours)
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[ ]:




