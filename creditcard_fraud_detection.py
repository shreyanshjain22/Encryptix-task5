#!/usr/bin/env python
# coding: utf-8

# # TASK: CREDIT CARD FRAUD DETECTION

# DOMAIN : DATA SCIENCE

# ## Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ## Data Collection and Processing

# In[2]:


# Creating a DataFrame using CSV file
data=pd.read_csv("creditcard.csv")


# In[3]:


# print first five rows of the DataFrame
data.head()


# In[4]:


# print the number of rows and columns
data.shape


# In[5]:


# checking the null values
data.isnull().sum().sum()


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


# Statistical analysis
data.describe()


# In[9]:


# Checking for duplicate data
data.duplicated().any()


# In[10]:


data.duplicated().sum()


# In[11]:


# Drop the dupliucate rows
data=data.drop_duplicates()
data.shape


# In[12]:


data.duplicated().any()


# In[13]:


# Categorise the legal and fraud Transactions 
data["Class"].value_counts()


#  The data is highly biased

# class=0 represents legal Transactions and 
# class=1 represents fraud Transactions

# In[14]:


print("percentage of legitimate data:",283253*100/283726)
print("percentage of fraud data:",473*100/283726)


# In[15]:


legal=data[data["Class"]==0]
fraud=data[data["Class"]==1]


# In[16]:


print(legal.shape)


# In[17]:


print(fraud.shape)


# In[18]:


legal["Amount"].describe()


# In[19]:


fraud["Amount"].describe()


# In[20]:


# Compare the Values for both transactions 
data.groupby("Class").mean()


# ## Under-Sampling

# In[21]:


# Build a Sample Dataset contain similar Distribution  for Legal Transactions and Fraud Transaction 
legal_sample=legal.sample(n=473)


# In[22]:


# concatenate the DataFrames
new_data = pd.concat([legal_sample, fraud], axis = 0)
new_data


# In[23]:


new_data["Class"].value_counts()


# In[24]:


new_data.groupby(["Class"]).mean()


# ## Splitting the new data into Features and Target

# In[25]:


X=new_data.drop("Class",axis=1)
Y=new_data["Class"]


# In[26]:


X.head()


# In[27]:


Y.head()


# ## Splitting the data into Testing and Traning data

# In[28]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)


# In[29]:


print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# ## Model Training

# In[30]:


model=LogisticRegression(max_iter=150)


# In[31]:


model.fit(X_train,Y_train)


# ## Model Evaluation

# In[32]:


training_predict=model.predict(X_train)
training_predict


# In[33]:


testing_predict=model.predict(X_test)
testing_predict


# In[34]:


#Precision score of training data
precision_train=metrics.precision_score(training_predict,Y_train)
print("Precision score of training data:",precision_train)

#Precision score of testing data
precision_test=metrics.precision_score(testing_predict,Y_test)
print("Precision score of testing data:",precision_test)


# In[35]:


# Recall score of training data
recall_train=metrics.recall_score(training_predict,Y_train)
print("Recall score of training data:",recall_train)

#Recall score of testing data
recall_test=metrics.recall_score(testing_predict,Y_test)
print("Recall score of testing data:",recall_test)


# In[36]:


# f1-score on training data
f1score_train =metrics.f1_score(training_predict, Y_train)
print('F1-score Score of Training data:',f1score_train)

# f1-score on test data
f1score_test = metrics.f1_score(testing_predict, Y_test)
print('F1-score Score of Testing data:',f1score_test)

