#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


# Importing dataset

data=pd.read_excel(r"C:\Users\Admin\Downloads\iris.xls")


# In[3]:


# Displaying data 
data.head()


# In[4]:


#Summary Statistics of data
data.describe()


# In[5]:


# Finding the datatypes of each column
data.dtypes


# In[6]:


#Checking null values if any
data.isna().sum()


# In[7]:


#Filling the null values
for i in ['SL','SW','PL']:
    data[i]=data[i].fillna(data[i].median())


# In[8]:


data.isna().sum()


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X=data.drop('Classification',axis=1)
y=data['Classification']


# In[11]:


# Splitting into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# In[12]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


# Decision tree model
dtclsf=DecisionTreeClassifier()
dtclsf=dtclsf.fit(X_train,y_train)
y_pred_dt=dtclsf.predict(X_test)


# In[15]:


#confusion matrix
confusion_matrix(y_test,y_pred_dt)


# In[16]:


#Finding accuracy
accuracy_score(y_test,y_pred_dt)


# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


# Random forest model
rndmf=RandomForestClassifier()
rndmf=rndmf.fit(X_train,y_train)
y_pred_rf=rndmf.predict(X_test)


# In[19]:


confusion_matrix(y_test,y_pred_rf)


# In[20]:


#Finding Accuracy
accuracy_score(y_test,y_pred_rf)


# In[21]:


filename='savedmodel.pkl'


# In[22]:


pickle.dump(dtclsf,open(filename,'wb'))


# In[23]:


load_model=pickle.load(open(filename,'rb'))


# In[24]:


X_test.head()


# In[25]:


load_model.predict([[6.1,2.8,4.7,1.2]])


# In[ ]:




