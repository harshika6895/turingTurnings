#!/usr/bin/env python
# coding: utf-8

# In[4]:


#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab as pl
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


''' Dataset preparation - The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).
The dataset provides the patients’ information. 
It includes over 4,000 records and 15 attributes.'''
ds = pd.read_csv("framingham.csv")
ds.head(10)


# In[11]:


#since education column is of no use let's drop it
ds.drop(['education'], inplace = True , axis =1)
ds.head(10)


# In[12]:


ds.info()


# In[14]:


#renaming the column male as sex_male for better understanding
ds.rename(columns = {'male':'sex_male'} , inplace = True)


# In[19]:


#removing null values
ds.dropna(axis = 0 , inplace = True)
print(ds.head() , ds.shape)


# In[20]:


ds.info()


# In[21]:


print(ds.TenYearCHD.value_counts())


# In[34]:


#splitting the dataset intp training dataset and testing dataset
X = np.asarray(ds[['age','sex_male' , 'cigsPerDay' , 'totChol' , 'sysBP' , 'glucose']])

y = np.asarray(ds['TenYearCHD'])


# In[35]:


#normalization of the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[36]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 4)


# In[37]:


#exploratory data analysis of heart disease dataset
#counting no. of patients with CHD 
plt.figure(figsize = (7,5))
sns.countplot(x='TenYearCHD', data=ds,
             palette="BuGn_r")
plt.show()


# In[38]:


laste = ds['TenYearCHD'].plot()
plt.show(laste)


# In[41]:


from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(X_train , y_train)
y_pred = Lr.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score
print("Accuracy of the model is " , accuracy_score(y_test , y_pred))


# In[46]:


#confusion matrix 
from sklearn.metrics import confusion_matrix , classification_report

cm = confusion_matrix(y_test , y_pred)
conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['Predicted:0', 'Predicted:1'], 
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")
 
plt.show()
print('The details for confusion matrix is =')
print (classification_report(y_test, y_pred))

