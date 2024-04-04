#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix ,classification_report,precision_score, recall_score ,f1_score , accuracy_score



# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
import warnings
warnings.filterwarnings('ignore')


# In[60]:


url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/heart_disease.csv"
dataset = pd.read_csv(url)
dataset.head()
# cp - chest pain
# trestbps - resting blood pressure 
# chol - cholestrol
# fbs - fasting blood sugar
# restecg - resting electrocardiographic 
# thalach - max heart rate achieved 
# exang - exercise induced angina
# ca - no. of major vessels
# thal - thalium stress test 


# In[27]:


dataset.describe()


# In[28]:


dataset.info()
#age - integer
# sex - 0 : male 1: female
# cp - 0: typical Angina,
#      1: Atypical Angina,
#      2: Non-Anginal Pain,
#      3: Asymptomatic
#trestbps -Integer in mm Hg
# chol - Integer in mg/dl
# fbs - True: 1, False: 0
# restecg - 0: normal,
#           1: ST-T wave abnomalitywith inversions and depression, 
#           2: left ventricular hypertrophy (probable diagnosis or confirmed also)
# thalach - 0: less chance, 1: more chance
# exang - Yes: 1, No: 0
#4 attributes, oldpeak, slope, number of major vessels, and output are the numeric values related to heart disease in 
#the dataset and were not included in the 10 variables of this study.


# In[29]:


dataset.describe().columns


# In[30]:


dataset_num = dataset[['age','trestbps','chol','thalach','oldpeak']]
dataset_cat =dataset[['sex','cp','fbs','restecg','exang']]


# In[31]:


for i in dataset_num.columns:
    plt.hist(dataset_num[i])
    plt.title(i)
    plt.show()


# In[32]:


pd.pivot_table(dataset, index='target', values=['age','trestbps','chol','thalach','oldpeak'])


# In[33]:


for i in dataset_cat.columns:
    sns.barplot(x = dataset_cat[i].value_counts().index, y = dataset_cat[i].value_counts()).set_title(i)
    plt.show()


# In[34]:


print(pd.pivot_table(dataset,index='target',columns='sex', values='age'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='cp', values='age'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='fbs', values='age'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='restecg', values='age'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='exang', values='age'))


# In[35]:


print(pd.pivot_table(dataset,index='target',columns='sex', values='chol'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='cp', values='chol'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='fbs', values='chol'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='restecg', values='chol'))
print("="*100)
print(pd.pivot_table(dataset,index='target',columns='exang', values='chol'))


# In[36]:


for i in dataset_num.columns:
    sns.boxplot(dataset_num[i])
    plt.title(i)
    plt.show()


# In[37]:


def outlinefree(datasetCol):
    # sorting column
    sorted(datasetCol)
    
    # getting percentile 25 and 27 that will help us for getting IQR (interquartile range)
    Q1,Q3 = np.percentile(datasetCol,[25,75])
    
    # getting IQR (interquartile range)
    IQR = Q3-Q1
    
    # getting Lower range error
    LowerRange = Q1-(1.5 * IQR)
    
    # getting upper range error
    UpperRange = Q3+(1.5 * IQR)
    
    # return Lower range and upper range.
    return LowerRange,UpperRange


# In[38]:


lwtrestbps,uptrestbps = outlinefree(dataset['trestbps'])
lwchol,upchol = outlinefree(dataset['chol'])
lwoldpeak,upoldpeak = outlinefree(dataset['oldpeak'])


# In[39]:


dataset['trestbps'].replace(list(dataset[dataset['trestbps'] > uptrestbps].trestbps) ,uptrestbps,inplace=True)
dataset['chol'].replace(list(dataset[dataset['chol'] > upchol].chol) ,upchol,inplace=True)
dataset['oldpeak'].replace(list(dataset[dataset['oldpeak'] > upoldpeak].oldpeak) ,upoldpeak,inplace=True)


# In[40]:


features = dataset.iloc[:,:-1].values
label = dataset.iloc[:,-1].values


# In[41]:


#------------------------LogisticRegression-----------------------
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=102)

classimodel= LogisticRegression()  
classimodel.fit(X_train, y_train)
trainscore =  classimodel.score(X_train,y_train)
testscore =  classimodel.score(X_test,y_test)  

print("test score: {} train score: {}".format(testscore,trainscore),'\n')

y_pred =  classimodel.predict(X_test)


# In[42]:


#from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(' Accuracy ',accuracy_score(y_test, y_pred),'\n')
print(' f1 score: ',f1_score(y_test, y_pred),'\n')
print(' precision score: ',precision_score(y_test, y_pred),'\n')
print(' recall score: ',recall_score(y_test, y_pred),'\n')
print(classification_report(y_test, y_pred))


# In[52]:


#--------------------------------------K-Nearest Neighbor(KNN)-----------------
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=193) 


classifier= KNeighborsClassifier()  
knnmodel =  classifier.fit(X_train, y_train) 

trainscore =  knnmodel.score(X_train,y_train)
testscore =  knnmodel.score(X_test,y_test)  

print("test score: {} train score: {}".format(testscore,trainscore),'\n')

y_predknn =  knnmodel.predict(X_test) 


# In[53]:


print(confusion_matrix(y_test, y_predknn))
print("Accuracy ",accuracy_score(y_test, y_predNB),'\n')
print("f1_score: ",f1_score(y_test, y_predknn),'\n')
print("precision_score: ",precision_score(y_test, y_predknn),'\n')
print("recall_score: ",recall_score(y_test, y_predknn),'\n')
print(classification_report(y_test, y_predknn))


# In[43]:


#------------------------------naive bayes---------------------------
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=34) 

NBmodel = GaussianNB()  
NBmodel.fit(X_train, y_train) 

trainscore =  NBmodel.score(X_train,y_train)
testscore =  NBmodel.score(X_test,y_test)  

print("test score: {} train score: {}".format(testscore,trainscore),'\n')
y_predNB =  NBmodel.predict(X_test)


# In[44]:


print(confusion_matrix(y_test, y_predNB))
print("Accuracy ",accuracy_score(y_test, y_predNB),'\n')
print("f1_score: ",f1_score(y_test, y_predNB),'\n')
print("precision_score: ",precision_score(y_test, y_predNB),'\n')
print("recall_score: ",recall_score(y_test, y_predNB),'\n')
print(classification_report(y_test, y_predNB))


# In[45]:


#-------------------------------- support vector classification -------------------------------------  
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=8) 

svcmodel = SVC(probability=True)  
svcmodel.fit(X_train, y_train) 

trainscore =  svcmodel.score(X_train,y_train)
testscore =  svcmodel.score(X_test,y_test)  

print("test score: {} train score: {}".format(testscore,trainscore),'\n')


# In[46]:


y_predsvc =  svcmodel.predict(X_test)

print(confusion_matrix(y_test, y_predsvc))
print("Accuracy ",accuracy_score(y_test, y_predNB),'\n')
print("f1_score: ",f1_score(y_test, y_predsvc),'\n')
print("precision_score: ",precision_score(y_test, y_predsvc),'\n')
print("recall_score: ",recall_score(y_test, y_predsvc),'\n')
print(classification_report(y_test, y_predsvc),'\n')


# In[47]:


#-------------------------------------- LogisticRegression -------------------------------------
probabilityValues = classimodel.predict_proba(features)[:,1]
#Calculate AUC
auc = roc_auc_score(label,probabilityValues)
print(auc)
#Calculate roc_curve
fpr,tpr, threshold =  roc_curve(label,probabilityValues)
plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr,tpr)


# In[54]:


#-------------------------------------- KNeighborsClassifier -------------------------------------
probabilityValues = knnmodel.predict_proba(features)[:,1]
#Calculate AUC
auc = roc_auc_score(label,probabilityValues)
print(auc)
#Calculate roc_curve
fpr,tpr, threshold =  roc_curve(label,probabilityValues)
plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr,tpr)


# In[49]:


#-------------------------------------- naive bayes -------------------------------------
probabilityValues = NBmodel.predict_proba(features)[:,1]
#Calculate AUC
auc = roc_auc_score(label,probabilityValues)
print(auc)
#Calculate roc_curve
fpr,tpr, threshold =  roc_curve(label,probabilityValues)
plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr,tpr)


# In[50]:


#-------------------------------------- SVC -------------------------------------
probabilityValues = svcmodel.predict_proba(features)[:,1]
#Calculate AUC
auc = roc_auc_score(label,probabilityValues)
print(auc)
#Calculate roc_curve
fpr,tpr, threshold =  roc_curve(label,probabilityValues)
plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr,tpr)


# In[61]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[62]:


X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=102)

XGmodel= xgb.XGBRFClassifier()
XGmodel.fit(X_train, y_train)
trainscore =  XGmodel.score(X_train,y_train)
testscore =  XGmodel.score(X_test,y_test)  

print("test score: {} train score: {}".format(testscore,trainscore),'\n')


# In[65]:


y_predXG =  XGmodel.predict(X_test)

#from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
print("Accuracy ",accuracy_score(y_test, y_predNB),'\n')

print("f1_score: ",f1_score(y_test, y_predXG),'\n')
print("precision_score: ",precision_score(y_test, y_predXG),'\n')
print("recall_score: ",recall_score(y_test, y_predXG),'\n')
print(classification_report(y_test, y_predXG),'\n')


# In[64]:


probabilityValues = XGmodel.predict_proba(features)[:,1]
#Calculate AUC
auc = roc_auc_score(label,probabilityValues)
print(auc)
#Calculate roc_curve
fpr,tpr, threshold =  roc_curve(label,probabilityValues)
plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr,tpr)

