#!/usr/bin/env python
# coding: utf-8

# In[2]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import re
import numpy as np
import plotly.express as px
import matplotlib.ticker as mtick  
import pmdarima as pm
import pandas as pd
import warnings
import statsmodels.api as sm
from scipy import stats
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from math import sqrt
warnings.filterwarnings("ignore")
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa import seasonal
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product


# In[4]:


data = pd.read_excel('Spare_Parts.xlsx')


# In[5]:


data.head()


# In[6]:


data['Spare Parts Name'].unique()


# In[7]:


grouped = data.groupby(['Spare Parts Name'])


# In[8]:


df_new = grouped.get_group('Air Filter')
#df_new["Date"] =pd.to_datetime(df_new["Date"]).dt.strftime('%Y-%m-%d')


# In[9]:


df = df_new[["Date","Volume of Spare Parts Supplied"]]
df.head()


# In[10]:


df = df.set_index('Date')['Volume of Spare Parts Supplied']
df.shape


# In[11]:


df[:500].plot()


# In[12]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[13]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')


# In[14]:


decompose_ts.plot()


# In[15]:


dftest = adfuller(df) # Sending whole time series
dftest


# In[16]:


dfs = adfuller(pd.Series(decompose_ts.resid).dropna()) # sending only residual part
dfs


# In[17]:


ts_array = df.to_numpy()


# In[18]:


sm.graphics.tsa.plot_acf(ts_array, lags=100)


# In[19]:


decompose_ts.resid
residual_array = decompose_ts.resid


# In[20]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[21]:


X = pd.DataFrame(pd.Series(residual_array).dropna())
X.head()


# In[22]:


train= pd.DataFrame(X[:len(X)-87])
print(train.shape)
test= pd.DataFrame(X[len(X)-87:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
print(train.head())
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# In[23]:


train.index


# In[24]:


value = train.loc['2021-05-21']
value


# # (a) PMDARIMA - Time Series Analysis

# # PMDARIMA

# In[25]:


# Splitting the data into Train & Test set
import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)


# In[26]:


Date_Test_ts= pd.DataFrame(test.index)
print("The first 5 Dates in the Test Data " , "\n",Date_Test_ts.head())
print("The Shape of the Test Date Frame " , "\n",Date_Test_ts.shape)


# In[27]:


Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
print("The first 5 Actual Test Data " , "\n",Actual_Test_Data_ts.head())
print("The Shape of the Actual Test Data Frame " , "\n",Actual_Test_Data_ts.shape)


# In[28]:


model12 = pm.auto_arima(train, seasonal=True, m=52, suppress_warnings=False) 


# In[29]:


model12.summary()


# In[30]:


fcs_pmdarima = model12.predict(87)
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})

predicted_pmdarima_df


# In[31]:


f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.plot(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.plot(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# ### ALTERNATOR

# In[32]:


df_new = grouped.get_group('Alternator')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[33]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[34]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[35]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[36]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[37]:


X = pd.Series(residual_array).dropna()
X.shape


# In[38]:


train= pd.DataFrame(X[:len(X)-256])
print(train.shape)
test= pd.DataFrame(X[len(X)-256:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# ### PMDARIMA

# In[39]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[40]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# ## BATTERY

# In[41]:


df_new = grouped.get_group('Battery')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[42]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[43]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[44]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[45]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[46]:


X = pd.Series(residual_array).dropna()
X.shape


# In[47]:


train= pd.DataFrame(X[:len(X)-89])
print(train.shape)
test= pd.DataFrame(X[len(X)-89:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# In[48]:


train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# ## PMDARIMA 

# In[49]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[50]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.plot(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.plot(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# # Brake Disc

# In[51]:


df_new = grouped.get_group('Brake Disc')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[52]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[53]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[54]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[55]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[56]:


X = pd.Series(residual_array).dropna()
X.shape


# In[57]:


train= pd.DataFrame(X[:len(X)-88])
print(train.shape)
test= pd.DataFrame(X[len(X)-88:])
print(test.shape)


# In[58]:


train


# In[59]:


train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[60]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 =pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[61]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# ### Brake Pad

# In[62]:


df_new = grouped.get_group('Brake Pad')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[63]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[64]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[65]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[66]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[67]:


X = pd.Series(residual_array).dropna()
X.shape


# In[68]:


train= pd.DataFrame(X[:len(X)-388])
print(train.shape)
test= pd.DataFrame(X[len(X)-388:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[69]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[70]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# ### Oil Filter

# In[71]:


df_new = grouped.get_group('Oil Filter')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[72]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[73]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[74]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[75]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[76]:


X = pd.Series(residual_array).dropna()
X.shape


# In[77]:


train= pd.DataFrame(X[:len(X)-159])
print(train.shape)
test= pd.DataFrame(X[len(X)-159:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[78]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[79]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# ### Radiator

# In[80]:


df_new = grouped.get_group('Radiator')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[81]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[82]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[83]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[84]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[85]:


X = pd.Series(residual_array).dropna()
X.shape


# In[86]:


train= pd.DataFrame(X[:len(X)-192])
print(train.shape)
test= pd.DataFrame(X[len(X)-192:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[87]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[88]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# # Spark Plug

# In[89]:


df_new = grouped.get_group('Spark Plug')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[90]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[91]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[92]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[93]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[94]:


X = pd.Series(residual_array).dropna()
X.shape


# In[95]:


train= pd.DataFrame(X[:len(X)-49])
print(train.shape)
test= pd.DataFrame(X[len(X)-49:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[96]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[97]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# # Starter Motor

# In[98]:


df_new = grouped.get_group('Starter Motor')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[99]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[100]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[101]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[102]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[103]:


X = pd.Series(residual_array).dropna()
X.shape


# In[104]:


train= pd.DataFrame(X[:len(X)-189])
print(train.shape)
test= pd.DataFrame(X[len(X)-189:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[105]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"], trace=True,suppress_warnings=False)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima
predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[106]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# # Tyre

# In[107]:


df_new = grouped.get_group('Tyre')
df = df_new[["Date","Volume of Spare Parts Supplied"]]
df = pd.DataFrame(df.set_index('Date')['Volume of Spare Parts Supplied'])
df[:500].plot()


# In[108]:


plt.figure(figsize=(20,5))
a = sns.lineplot(x='Date',y='Volume of Spare Parts Supplied',data=df_new)

# Adjusting the xticks to make it more visible
for index, label in enumerate(a.get_xticklabels()):
   if index % 3 == 0:
      label.set_visible(True)
   else:
      label.set_visible(False)

plt.xticks(fontsize=15, rotation=90)
plt.show()


# In[109]:


decompose_ts = seasonal.seasonal_decompose(df, period=12, model='additive')
decompose_ts.plot()
dftest = adfuller(df) # Sending whole time series
dftest


# In[110]:


ts_array = df.to_numpy()
sm.graphics.tsa.plot_acf(ts_array, lags=100)
decompose_ts.resid
residual_array = decompose_ts.resid


# In[111]:


sm.graphics.tsa.plot_acf(pd.Series(residual_array).dropna(), lags=100)


# In[112]:


X = pd.Series(residual_array).dropna()


# In[113]:


train= pd.DataFrame(X[:len(X)-183])
print(train.shape)
test= pd.DataFrame(X[len(X)-183:])
print(test.shape)
train = train.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#train.head()
test = test.rename(columns={'resid':'Volume of Spare Parts Supplied'})
#test.head()


# # PMDARIMA

# In[114]:


import pmdarima as pm
#train, test = pm.model_selection.train_test_split(df, train_size=.75, test_size = .25)
Date_Test_ts= pd.DataFrame(test.index)
Actual_Test_Data_ts=pd.DataFrame(test).reset_index(drop=True)
model2 = pm.auto_arima(train["Volume of Spare Parts Supplied"],max_order=5)
fcs_pmdarima = model2.predict(len(test)) 
fcs_pmdarima

predicted_pmdarima_df = pd.DataFrame({"Predicted":(fcs_pmdarima)})


# In[115]:


New_data = pd.DataFrame()
New_data['Actual'] = Actual_Test_Data_ts['Volume of Spare Parts Supplied'].values
New_data['Predicted'] = predicted_pmdarima_df['Predicted'].values
New_data['Predicted']


# In[116]:


New_data[['Actual', 'Predicted']].head()


# In[117]:


mse = mean_squared_error(test, predicted_pmdarima_df)
rmse = sqrt(mse)
r2score = r2_score(test, predicted_pmdarima_df)
print('RMSE: %f' % rmse)
print('R^2 Score: %.2f'% r2score)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

plt.scatter(Date_Test_ts['Date'], Actual_Test_Data_ts['Volume of Spare Parts Supplied'], label = "Actual Spare Parts Supplied", linestyle="-")
plt.scatter(Date_Test_ts['Date'], predicted_pmdarima_df['Predicted'], label = "Predicted Spare Parts Supplied", linestyle="-")
plt.xticks(fontsize=10, rotation=90)

plt.xlabel("Date")
plt.ylabel("Predicted Spare Parts Supplied")  ## need to check the unit
plt.title("Actual Spare Parts Supplied Vs Predicted Spare Parts Supplied with respect to time using pmdarima")
plt.legend()
plt.show()


# # (b) Facebook Prophet Time Series Analysis

# ## (a) Alternator 

# In[118]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Alternator")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast1 = m.predict(future)
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#yhat: forecast
#yhat_lower, yhat_upper: uncertainty interval


# In[119]:


fig1 = m.plot(forecast1)
m.plot_components(forecast1);


# ## (b) Air Filter

# In[120]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Air Filter")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast2 = m.predict(future)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[121]:


fig1 = m.plot(forecast2)
m.plot_components(forecast2);


# ## (c) Battery

# In[122]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Battery")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast3 = m.predict(future)
forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[123]:


fig1 = m.plot(forecast3)
m.plot_components(forecast3);


# ## (d) Brake Disc

# In[124]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Brake Disc")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast4 = m.predict(future)
forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[125]:


fig1 = m.plot(forecast4)
m.plot_components(forecast4);


# ## (e) Brake Pad

# In[126]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Brake Pad")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast5 = m.predict(future)
forecast5[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[127]:


fig1 = m.plot(forecast5)
m.plot_components(forecast5);


# ## (f) Oil Filter

# In[128]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Oil Filter")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast6 = m.predict(future)
forecast6[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[129]:


fig1 = m.plot(forecast6)
m.plot_components(forecast6);


# ## (g) Radiator

# In[130]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Radiator")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast7 = m.predict(future)
forecast7[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[131]:


fig1 = m.plot(forecast7)
m.plot_components(forecast7);


# ## (h) Spark Plug

# In[132]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Spark Plug")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast8 = m.predict(future)
forecast8[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[133]:


fig1 = m.plot(forecast8)
m.plot_components(forecast8);


# ## (i) Starter Motor

# In[134]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Starter Motor")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast9 = m.predict(future)
forecast9[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[135]:


fig1 = m.plot(forecast9)
m.plot_components(forecast9);


# ## (j) Tyre

# In[136]:


data = pd.read_excel('Spare_Parts.xlsx')
data_new = data
groupby = data_new.groupby("Spare Parts Name")
df_new = groupby.get_group("Tyre")
data = df_new
data = pd.DataFrame(data[["Date","Volume of Spare Parts Supplied"]])
start = dt.datetime(2021, 5, 12)
end = dt.datetime(2024,1,1)
data.reset_index(inplace=True)
data=data[["Date","Volume of Spare Parts Supplied"]]
data=data.rename(columns={"Date": "ds", "Volume of Spare Parts Supplied": "y"})
df_train=data[0:6000]
df_test=data[6000:8000]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast10 = m.predict(future)
forecast10[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[137]:


fig1 = m.plot(forecast10)
m.plot_components(forecast10);


# # Multivariate Time Series

# In[138]:


data1= pd.read_excel('Spare_Parts.xlsx')
df=data1[["Date","Lead time ","Re-Order Level","Replacement Cycle","Volume of Spare Parts Supplied"]]
df=df.set_index("Date")


# In[139]:


fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[140]:


ad_fuller_result_1 = adfuller(df['Volume of Spare Parts Supplied'].diff()[1:])

print('realgdp')
print(f'ADF Statistic: {ad_fuller_result_1[0]}')
print(f'p-value: {ad_fuller_result_1[1]}')

print('\n---------------------\n')

ad_fuller_result_2 = adfuller(df['Lead time '].diff()[1:])

print('realcons')
print(f'ADF Statistic: {ad_fuller_result_2[0]}')
print(f'p-value: {ad_fuller_result_2[1]}')


# In[141]:


print('Volume of Spare Parts Supplied causes Lead Time?\n')
print('------------------')
granger_1 = grangercausalitytests(df[['Lead time ', 'Volume of Spare Parts Supplied']], 4)
print('\Replacement Cycle causes Volume of Spare Parts Supplied?\n')
print('------------------')
granger_2 = grangercausalitytests(df[["Replacement Cycle","Volume of Spare Parts Supplied"]], 4)                                      
print('\Re-Order Level causes Volume of Spare Parts Supplied?\n')
print('------------------')
granger_3 = grangercausalitytests(df[['Re-Order Level', 'Volume of Spare Parts Supplied']], 4)                                     


# In[142]:


df


# In[143]:


train_df=df[:-2000]
test_df=df[-2000:]
print(test_df.shape)


# In[144]:


train_df.diff()


# In[145]:


model = VAR(train_df.diff()[1:])


# In[146]:


sorted_order=model.select_order(maxlags=20)
print(sorted_order.summary())


# In[147]:


var_model = VARMAX(train_df.astype('float64'), order=(4,0),enforce_stationarity= True)
fitted_model = var_model.fit(disp=False)
print(fitted_model.summary())


# In[148]:


len(train_df)


# In[149]:


n_forecast = 12
predict = fitted_model.get_prediction(start=len(train_df),end=len(train_df) + n_forecast-1)#start="1989-07-01",end='1999-01-01')

predictions=predict.predicted_mean


# In[150]:


predictions


# In[151]:


predictions.columns=['Lead_Time_predicted',"Re-Orer_Level_Predicted","Replacement_Cycle",'Volume_of_Spare_Parts_Supplied_predicted',]
predictions


# In[152]:


test_vs_pred=pd.concat([test_df,predictions])


# In[153]:


test_vs_pred[:100].plot(figsize=(12,5))


# # Machine Learning Model

# ## 1) Random Forest 

# In[154]:


data = pd.read_excel('Spare_Parts.xlsx')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Vehicle_Type_Encoded']=le.fit_transform(data['Vehicle Type'])
data['Spare_Parts_Name_Encoded']=le.fit_transform(data['Spare Parts Name'])
data['Region']=le.fit_transform(data['Region'])
data['Weather_encoded']=le.fit_transform(data['Weather Condition'])
data['City']=le.fit_transform(data['City'])
data.head(200)


# In[155]:


X  = data.drop(['Date','Volume of Spare Parts Supplied','Spare Parts Name', 'Weather Condition', 'Vehicle Type'],axis = 'columns')
Y  = data["Volume of Spare Parts Supplied"]
X


# In[156]:


from sklearn.model_selection import train_test_split
x_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)


# In[157]:


from sklearn.ensemble import RandomForestRegressor
regressor =  RandomForestRegressor(n_estimators =20, random_state = 0)
regressor.fit(x_train, Y_train)
y_pred = regressor.predict(X_test)


# In[158]:


y_pred


# In[159]:


df = pd.DataFrame()
df['Actual'] = Y_test
df['predicted'] = y_pred


# In[160]:


df


# In[161]:


plt.scatter(x=df['Actual'],y=df['predicted'],edgecolors='r')


# In[162]:


regressor.predict([[4,139,50,130,4,0,0,0]])


# In[163]:


X = regressor.predict(X_test)


# In[164]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, X)


# ## 2) Decision Tree

# In[165]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[166]:


features = ['Region','City','Replacement Cycle','Re-Order Level', 'Lead time ','Vehicle_Type_Encoded','Spare_Parts_Name_Encoded','Weather_encoded']


# In[167]:


X = data[features]
y = data['Volume of Spare Parts Supplied']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)


# In[168]:


X = dtree.predict(X_test)


# In[169]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, X)


# ## 3)MultiLinear Regressiom

# In[170]:


from sklearn import linear_model


# In[171]:


X  = data.drop(['Date','Volume of Spare Parts Supplied','Spare Parts Name', 'Weather Condition', 'Vehicle Type'],axis = 'columns')
Y  = data["Volume of Spare Parts Supplied"]


# In[172]:


from sklearn.model_selection import train_test_split
x_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)


# In[173]:


regr = linear_model.LinearRegression()
regr.fit(x_train, Y_train)


# In[174]:


predictedSale = regr.predict(X_test)


# In[175]:


predictedSale


# In[176]:


df = pd.DataFrame()
df['Actual'] = Y_test
df['predictedSale'] = predictedSale
df


# In[177]:


plt.plot(df[:10])


# In[178]:


plt.scatter(df.index[:10],df['Actual'][:10], linestyle="-")
plt.scatter(df.index[:10],df['predictedSale'][:10], linestyle="-")


# In[179]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test,predictedSale)


# ### Summary:
# 
# Decision Tree Shows less Means Sqaure Error as compare to Multilinear Regression and Random Forest.

# In[181]:


import joblib
joblib.dump(model,"model.pkl")


# In[ ]:




