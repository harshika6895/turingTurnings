#!/usr/bin/env python
# coding: utf-8

# In[15]:


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[16]:


passengers = pd.read_csv("AirPassengers.csv")
passengers.head()


# In[17]:


passengers = passengers.rename(columns={"#Passengers": "No. Of Passengers"}, inplace = False)
#It refers to a boolean value and checks whether to return the new DataFrame or not. If it is true, it makes the changes in the original DataFrame. The default value of the inplace is True.
passengers.head()                               


# In[18]:


passengers_columns = ['Month','No. Of Passengers']
passengers['Month'] = pd.to_datetime(passengers['Month'], format='%Y-%m')
passengers = passengers.set_index('Month') #gives the list, series or data_frame to set the index of the dataframe
passengers.head()


# PLOT TIME SERIES DATA

# In[25]:


passengers.plot(y = 'No. Of Passengers', figsize=(20,4), color = "red", linestyle = "-", marker='o', markersize=5, label = "Passenger Traffic")
plt.grid(True)
plt.legend(loc="best")
plt.title("Airline Passenger Traffic over time")
plt.xlabel("Date")
plt.ylabel("No. of Passengers")
plt.show(block = False)


# MISSING VALUE TREATMENT

# In[27]:


#MEAN IMPUTATION 
#Mean imputation (MI) is one such method in which the mean of the observed values for each variable is computed and the missing values for that variable are imputed by this mean. 
#This method can lead into severely biased estimates
passengers['Passenger_mean_imputation'] = passengers['No. Of Passengers'].fillna(passengers['No. Of Passengers'].mean())


# In[28]:


passengers[['Passenger_mean_imputation']].plot(figsize=(20,4), color = "blue", linestyle = "-", marker='o', markersize=5, legend = True, grid = True)
plt.title("Airline Passenger Traffic: Mean Imputation")
plt.xlabel("Date")
plt.ylabel("No. of Passengers")
plt.show(block = False)


# In[29]:


# Linear Interpolation 
#Linear interpolation is a method of curve fitting using linear polynomials to construct new data points within the range of a discrete set of known data points
passengers['Passengers_Linear_Interpolation'] = passengers['No. Of Passengers'].interpolate(method='linear')


# In[31]:


passengers[['Passengers_Linear_Interpolation']].plot(figsize=(20,4), grid=True, legend = True, color='green', linestyle='-', marker='o', markersize=5)
plt.title('Airline passenger traffic : Linear Interpolation')
plt.xlabel('Date')
plt.ylabel('No. of Passengers')
plt.show(block = False)


# In[32]:


passengers.head()


# In[36]:


passengers['No. Of Passengers'] = passengers['Passengers_Linear_Interpolation']
passengers.drop(columns=['Passenger_mean_imputation','Passengers_Linear_Interpolation'], inplace=True)
#can use the inplace parameter to modify the original DataFrame in place instead of returning a new DataFrame. 
passengers.head()


# OUTLIER DETECTION 

# In[41]:


#boxplot and interquartile range
plt.figure(figsize=(20,5))
sns.boxplot(x=passengers['No. Of Passengers'], color= 'blue', width=0.5 , notch = True)

plt.title("Boxplot of passengers with the outliers")
plt.xlabel("No. of Passengers")
plt.xticks(fontsize = 12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[43]:


plt.figure(figsize=(12,6))
plt.hist(passengers["No. Of Passengers"], bins=20, color="cyan" , edgecolor="black")
plt.title("Histogram of passengers")
plt.xlabel("No. of Passengers")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.show()


# TIME SERIES DECOMPOSITION 

# In[48]:


#ADDITIVE SEASONAL DECOMPOSTION
from statsmodels.tsa.seasonal import seasonal_decompose
#The seasonal_decompose() function returns a result object. The result object contains arrays to access four pieces of data from the decomposition.
#performing seasonal decompostion
result= seasonal_decompose(passengers["No. Of Passengers"], model = 'additive')
fig, (ax1, ax2, ax3, ax4 )= plt.subplots(4,1, figsize=(10,8), sharex = True)

result.observed.plot(ax = ax1,title = "observed" )
result.trend.plot(ax = ax2,title = "trend" )
result.seasonal.plot(ax = ax3, title = "seasonal")
result.resid.plot(ax = ax4, title="Resid" )
plt.show()



# In[50]:


#MULTIPLICATIVE SEASONAL DECOMPOSITION
from statsmodels.tsa.seasonal import seasonal_decompose
#The seasonal_decompose() function returns a result object. The result object contains arrays to access four pieces of data from the decomposition.
#performing seasonal decompostion
result= seasonal_decompose(passengers["No. Of Passengers"], model = 'multiplicative')
fig, (ax1, ax2, ax3, ax4 )= plt.subplots(4,1, figsize=(10,8), sharex = True)

result.observed.plot(ax = ax1,title = "Observed" )
result.trend.plot(ax = ax2,title = "Trend" )
result.seasonal.plot(ax = ax3, title = "Seasonal")
result.resid.plot(ax = ax4, title="Resid" )
plt.show()


# In[51]:


len(passengers)


# BUILD AND EVALUATE TEST SERIES FORECAST

# In[52]:


#split train test data 
train_len= 120
train = passengers[:train_len]
test = passengers[train_len:]


# AUTO REGRESSIVE METHODS

# In[53]:


#STATIONARY VS NON STATIONARY TIME SERIES
passengers.plot(y = 'No. Of Passengers', figsize=(20,4), color = "red", linestyle = "-", label = "Passenger Traffic")
plt.grid(True)
plt.legend(loc="best")
plt.title("Airline Passenger Traffic over time")
plt.xlabel("Date")
plt.ylabel("No. of Passengers")
plt.show(block = False)
#As visible this is a non stationary time series means a non-stationary time series has statistical properties that change over time, 
#which can make it difficult to draw reliable inferences or make accurate forecasts. As the statistical properties of the data keep changing, 
#any model or analysis based on a non-stationary time series may not provide reliable results.


# In[54]:


#AUGMENTED DICKEY-FULLER(ADF) TEST
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(passengers['No. Of Passengers'])

print("ADF TEST STATISTICS : %f" %adf_test[0])
print("Crtical Values @ 0.05 : %.2f" %adf_test[4]['5%'])
print("p-value : %f" %adf_test[1])

