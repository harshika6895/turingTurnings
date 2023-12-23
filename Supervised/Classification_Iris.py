#!/usr/bin/env python
# coding: utf-8

# OUR FIRST CLASSIFICATION(Supervised learning) PROBLEM -
# 
# The standard lowers species classification problem where we are going to work on IRIS DATASET to classify Iris flowers into species by their morphology. The four features are : Sepal Length, Sepal Width, Petal Length, Petal width.
# 
# DATASET - Iris dataset is a small, 150 examples, 4 features dataset which can easily be visualized and manipulated.
# 
# Step 1 : Load the dataset
# 

# In[1]:


#import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#load the dataset
columns = ['Sepal Length' , 'Sepal Width' , 'Petal Length' , 'Petal Width', 'Class_Labels']

iris = pd.read_csv('iris.data' , names = columns)
iris.head()


# STEP 2 : ANALYZE AND VISUALIZE THE DATASET

# In[4]:


iris.describe()


# In[5]:


#visualize the data set

sns.pairplot(iris, hue='Class_Labels')


# After visualizing the whole dataset we can say that iris-setosa is well seperated from the other two flowers and also that iris virginica is the longest flower and iris setosa is the shortest.
# 
# Lets plot the average of each feature of each class.
# 

# In[7]:


#Seperate features and target 

data = iris.values

X= data[:,0:4]
Y = data[: ,4]


# In[13]:


#calculating the average for each feature for each classes 

Y_Data = np.array([np.average(X[:,i][Y==j].astype('float32'))
                   for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4,3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped,0,1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# In the above cell, np.average is used to calculate the average from an array. We defined Y_Data as the array in which 2 for loops were there this is known as list comprehension which reduces the lines of code. The Y_Data is a 1D array with average values of each feature for each class which is then reshaped into a (4,3) array using .reshape and then we changed the axis of the reshaped matrix.
# 
# Now We will use matplotlib to show averages in a bar plot

# In[15]:


#plot the avg
plt.bar(X_axis, Y_Data_reshaped[0], width, label = "Setosa")
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = "Versicolor")
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = "Virginica")

plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("VaLues in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# STEP 3 : TRAIN THE MODEL
# 
# Using the train_test_split we will split the whole data into training and testing dataset. Later testing dataset will be used to check the accuracy of the model.

# In[22]:


#SPLIT THE DATA TO TRAIN AND TEST DATASET.
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test = train_test_split(X,Y, test_size=0.2)


# In[23]:


#support vector machine algorithm (SVM)
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, Y_train)


# Here in the above cell we imported support vector classifier(SVC) from sklearn support vector machine and then we created an object and named it svn. We fed the training dataset to the algorithm by using svm.fit() method.
# 
# STEP 4 : MODEL EVALUATION: 
# Now we are going to predict the classes for our test dataset using our trained model and then we will check accuracy score of the predicted classes. 
# 
# Accuracy_score() takes the true values and predicted values and returns the percentage of accuracy

# In[24]:


#Predict from the test dataset 
predictions = svn.predict(X_test)


# In[25]:


#Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)


# The accuracy is 93%
# 
# DETAILED CLASSIFICATION REPORT based on test dataset

# In[26]:


#A detailed classification report 
from sklearn.metrics import classification_report 
print(classification_report(Y_test , predictions))


# The classification report gives a detailed report of the prediction.
# 
# Precision defines the ratio of true positives to the sum of true positive and false positives. 
# 
# Recall defines the ratio of true positive to the sum of true positive and false negative.
# 
# F1-score is the mean of precision and recall value.
# 
# Support is the number of actual occurences of the class in the specified dataset.
# 
# STEP 5 : TESTING THE MODEL:
# 
# To test the model we will take some random values based on the average plot to see if the model can predict accurately.
# 
# 

# In[27]:


X_new = np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])

#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of species: {}".format(prediction))


# STEP 6 : SAVING THE MODEL 
# 
# We can save the model using pickle format and load the model again in any other program using pickle and use it using classification0.predict to predict the iris data.

# In[30]:


#Save the model
import pickle 
with open('SVM.pickle','wb') as f:
    pickle.dump(svn, f)
    
#load the model
with open('SVM.pickle', 'rb') as f:
    classification0 = pickle.load(f)
classification0.predict(X_new)


# SUMMARY :
# 
# In this project, we learned to train our own supervised machine learning model using IRIS FLOWER CLASSIFICATION PROJECT with Machine Learning.
