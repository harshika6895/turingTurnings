#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import yfinance as yf


# In[4]:


sp500 = yf.Ticker("^GSPC")


# In[5]:


sp500= sp500.history(period="max")


# In[6]:


sp500


# In[7]:


sp500.index


# In[8]:


sp500.plot.line(y="Close" , use_index = True)


# In[9]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[10]:


sp500["Tommorrow"] = sp500["Close"].shift(-1)


# In[11]:


sp500


# In[12]:


sp500["Target"] = (sp500["Tommorrow"] > sp500["Close"]).astype(int)


# In[13]:


sp500


# In[14]:


sp500 = sp500.loc["1990-01-01":].copy()


# In[15]:


sp500


# In[16]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators =100 , min_samples_split = 100 , random_state = 1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close" , "Volume","Open" , "High" , "Low"]

model.fit(train[predictors] , train["Target"])


# In[17]:


from sklearn.metrics import precision_score
preds = model.predict(test[predictors])

preds


# In[18]:


preds = pd.Series(preds , index = test.index)
preds


# In[19]:


precision_score(test["Target"] , preds)


# In[20]:


combined = pd.concat([test["Target"],preds] , axis =1)


# In[21]:


combined.plot()


# In[27]:


def predict(train , test, predictors, model):
    model.fit(train[predictors] , train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index , name = "Predictions")
    combined = pd.concat([test["Target"],preds] , axis =1)
    return combined
    


# In[28]:


def backtest(data, model, predictors, start = 2500 , step =250):
    all_predictions = []
    
    for i in range(start, data.shape[0] , step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train , test , predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
        


# In[29]:


predictions = backtest(sp500 , model , predictors)


# In[30]:


predictions["Predictions"].value_counts()


# In[31]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[32]:


predictions["Target"].value_counts()/ predictions.shape[0]


# In[34]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"]/ rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column , trend_column]


# In[35]:


sp500


# In[36]:


sp500 = sp500.dropna()


# In[37]:


sp500


# In[38]:


model = RandomForestClassifier(n_estimators = 200 , min_samples_split = 50, random_state =1)


# In[39]:


def predict(train , test, predictors, model):
    model.fit(train[predictors] , train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds>= 0.6] = 1
    preds[preds<0.6] = 0
    preds = pd.Series(preds, index = test.index , name = "Predictions")
    combined = pd.concat([test["Target"],preds] , axis =1)
    return combined
    


# In[40]:


predictions = backtest(sp500 , model , new_predictors)


# In[42]:


predictions["Predictions"].value_counts()


# In[44]:


precision_score(predictions["Target"] , predictions["Predictions"])

