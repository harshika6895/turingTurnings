#!/usr/bin/env python
# coding: utf-8

# CLASSIFICATION OF WHEAT SEEDS 
# 
# In this notebook I will perform classication of wheat seeds into 1)KAMA, 2)ROSA and 3)CANADIAN using KNN classification. 
# 
# About the dataset :
# 
# The dataset has been taken from UCI Machine Learning Repository (Eldem 2020, Yasar et al. 2016, Kayabasi et al. 2018).Margapuri et al. (2021). currently loaded from kaggle. 
# 
# The data set of wheat seeds containing 210 pieces of data with 7 features and labelled into 3 classes Canadian, Kama and Rosa. 
# 
# Importing the Libraries and Loading the Dataset:

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

seeds = pd.read_csv('seeds.csv')
seeds.describe()
#getting the mean, count of each feature


# In[4]:


seeds.head() #checking the columns 


# In[5]:


seeds.info() #no missing values and checking the datatypes


# Analysing using visualization tools 

# In[6]:


sns.pairplot(seeds)


# In[7]:


sns.pairplot(seeds, hue='seedType')


# In[8]:


#Heat Map

plt.figure(figsize=(8,8))
cor = seeds.corr()
sns.heatmap(cor, annot = True,cmap = 'coolwarm')
plt.ylim(8,0)


# Splitting dependent and independent variables: 

# In[9]:


X = seeds.drop("ID", axis=1,inplace=True)


# In[12]:


feature_cols = ["area","perimeter","compactness","lengthOfKernel","widthOfKernel", "asymmetryCoefficient","lengthOfKernelGroove"]
X = seeds[feature_cols]
Y = seeds['seedType']

seeds.area.astype(float)


# In[13]:


seeds.info()


# Standardization of the data:

# In[15]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
seeds_new = ss.fit_transform(seeds)


# In[16]:


X = seeds.drop('seedType', axis=1)
Y = seeds['seedType']


# Train-Test Split: 
# 

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 3 , test_size = 0.3)


# Training the Models:
#     
# LINEAR REGRESSION : Using the algorithm of linear regression we can predict the numerical values based on the input features. It believes a linear relationship between the features and the target variable. It remembers the best coefficients and uses it to predict the target variables for the given features

# In[18]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)


# In[19]:


print(f'Coefficients : {lin_reg.coef_}')
print(f'Intercepts : {lin_reg.intercept_}')
print(f'R^2 score : {lin_reg.score(X,Y)}')


# In[24]:


LIN_REG = LinearRegression()
LIN_REG.fit(X_train, Y_train)

print(f'R^2 score for train data: {lin_reg.score(X_train,Y_train)}')
print(f'R^2 score for test data: {lin_reg.score(X_test,Y_test)}')


# LINEAR REGRESSION OLS(ORDINARY LEAST SQUARE): ordinary least squares is a type of linear least squares method for choosing the unknown parameters in a linear regression model by the principle of least squares.

# In[26]:


import statsmodels.api as sm #python library for statistics
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE #Recursive feature elimination (RFE) is a feature selection method that 
#fits a model and removes the weakest feature (or features) until the specified number of features is reached.

lin_reg2 = LinearRegression()
X_constant = sm.add_constant(X)
model = sm.OLS(Y, X_constant).fit()

predictions = model.predict(X_constant)
model.summary()


# LOGISITIC REGRESSION : This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. In logistic regression, a logit transformation is applied on the odds—that is, the probability of success divided by the probability of failure. 

# In[34]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(fit_intercept = True, solver='liblinear' , multi_class='ovr')
log_reg.fit(X_train,Y_train)

Y_test_pred = log_reg.predict(X_test)#to predict the lables of testing data
Y_test_prob = log_reg.predict_proba(X_test)# to predict the probability of each label for the testing data
Y_test_pred


# In[39]:


predict = log_reg.predict(X_train)
predict


# In[45]:


from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score, roc_curve , classification_report

print(classification_report(Y_train,predict))


# In[46]:


print('The AUC score of the model :' , roc_auc_score(Y_test, Y_test_prob, multi_class = 'ovr'))


# DECISION TREE CLASSIFIER : Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions. It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.
# In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[48]:


dt.fit(X,Y)


# In[50]:


dt_train = DecisionTreeClassifier()
dt_train.fit(X_train , Y_train)


# In[54]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score , classification_report
Y_test_pred = dt.predict(X_test)
Y_test_prob = dt.predict_proba(X_test)


# In[55]:


print('Confusion-Matrix :','\n' ,confusion_matrix(Y_test,Y_test_pred))
print('Overall accuracy test:', accuracy_score(Y_test,Y_test_pred))
print('AUC SCORE: '  , roc_auc_score(Y_test,Y_test_prob, multi_class = 'ovr'))


# RANDOM FOREST CLASSIFIER : It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
# 
# As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
# 
# 

# In[56]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)
#n_estimators= The required number of trees in the Random Forest. 
#The default value is 10. We can choose any number but need to take care of the overfitting issue.
rfc.fit(X_train,Y_train)


# In[57]:


Y_test_pred = rfc.predict(X_test)
Y_test_prob = rfc.predict_proba(X_test)
print('Confusion-Matrix :','\n' ,confusion_matrix(Y_test,Y_test_pred))
print('Overall accuracy test:', accuracy_score(Y_test,Y_test_pred))
print('AUC SCORE: '  , roc_auc_score(Y_test,Y_test_prob, multi_class = 'ovr'))


# MULTINOMIAL NAIVE BAYES : Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.
# 
# 

# In[59]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()


# In[60]:


mnb.fit(X_train,Y_train)

Y_test_pred = mnb.predict(X_test)
Y_test_prob = mnb.predict_proba(X_test)


# In[61]:


print('Confusion matrix : ','\n' , confusion_matrix(Y_test,Y_test_pred))
print('Overall Acurracy score : ' , accuracy_score(Y_test,Y_test_pred))
print('AUC-test: ', roc_auc_score(Y_test,Y_test_prob, multi_class = 'ovr'))


# K-NN NEIGHBOUr: K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.
# K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
# K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data.
# It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.
# KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.
# 

# In[68]:


from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore

knn = KNeighborsClassifier()


# In[76]:


Y


# In[97]:


X_train_standardize = X_train.apply(zscore)
X_test_standardize = X_test.apply(zscore)


# In[98]:


knn.fit(X_train_standardize,Y_train)


# In[99]:


Y_test_pred = knn.predict(X_test)
Y_test_prob = knn.predict_proba(X_test)



# In[100]:


print('Confusion Matrix :' , '\n' , confusion_matrix(Y_test ,Y_test_pred))
print('Overall Accuracy Score : ' , accuracy_score(Y_test, Y_test_pred))
print('AUC-TEST: ' , roc_auc_score(Y_test, Y_test_prob, multi_class = 'ovr'))

