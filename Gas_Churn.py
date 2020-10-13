#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")


# # **Data Import**

# **Coteq**

# In[41]:


data0= pd.read_csv('/content/drive/My Drive/Gas/coteq_gas_2015.csv')
data0['Year'] = dt.date(2015, rd.randint(11)+1,rd.randint(30)+1)
data1= pd.read_csv('/content/drive/My Drive/Gas/coteq_gas_2016.csv')
data1['Year'] = dt.date(2016, rd.randint(11)+1,rd.randint(30)+1)
data2= pd.read_csv('/content/drive/My Drive/Gas/coteq_gas_2017.csv')
data2['Year'] = dt.date(2017, rd.randint(11)+1,rd.randint(30)+1)
data3= pd.read_csv('/content/drive/My Drive/Gas/coteq_gas_2018.csv')
data3['Year'] = dt.date(2018, rd.randint(11)+1,rd.randint(30)+1)
data4= pd.read_csv('/content/drive/My Drive/Gas/coteq_gas_2019.csv')
data4['Year'] = dt.date(2019, rd.randint(11)+1,rd.randint(30)+1)
data5= pd.read_csv('/content/drive/My Drive/Gas/coteq_gas_2014.csv')
data5['Year'] = dt.date(2014, rd.randint(11)+1,rd.randint(30)+1)
listd= [data5,data0,data1,data2,data3,data4]
coteq= pd.concat (listd, axis=0, sort=False)
del [data0,data1,data2,data3,data4,data5]
coteq.index=coteq.Year
coteq=coteq.drop(['Year'],axis=1)
coteq


# **Rendo**

# In[42]:


data0= pd.read_csv('/content/drive/My Drive/Gas/rendo_gas_2015.csv')
data0['Year'] = dt.date(2015, rd.randint(11)+1,rd.randint(30)+1)
data1= pd.read_csv('/content/drive/My Drive/Gas/rendo_gas_2016.csv')
data1['Year'] = dt.date(2016, rd.randint(11)+1,rd.randint(30)+1)
data2= pd.read_csv('/content/drive/My Drive/Gas/rendo_gas_2017.csv')
data2['Year'] = dt.date(2017, rd.randint(11)+1,rd.randint(30)+1)
data3= pd.read_csv('/content/drive/My Drive/Gas/rendo_gas_2017.csv')
data3['Year'] = dt.date(2018, rd.randint(11)+1,rd.randint(30)+1)
data4= pd.read_csv('/content/drive/My Drive/Gas/rendo_gas_2019.csv')
data4['Year'] = dt.date(2019, rd.randint(11)+1,rd.randint(30)+1)
data5= pd.read_csv('/content/drive/My Drive/Gas/rendo_gas_2014.csv')
data5['Year'] = dt.date(2014, rd.randint(11)+1,rd.randint(30)+1)
listd= [data5,data0,data1,data2,data3,data4]
rendo= pd.concat (listd, axis=0, sort=False)
del [data0,data1,data2,data3,data4,data5]
rendo.index=rendo.Year
rendo=rendo.drop(['Year'],axis=1)
rendo


# **Westland-infra**

# In[43]:


data0= pd.read_csv('/content/drive/My Drive/Gas/westland-infra_gas_2015.csv')
data0['Year'] = dt.date(2015, rd.randint(11)+1,rd.randint(30)+1)
data1= pd.read_csv('/content/drive/My Drive/Gas/westland-infra_gas_2016.csv')
data1['Year'] = dt.date(2016, rd.randint(11)+1,rd.randint(30)+1)
data2= pd.read_csv('/content/drive/My Drive/Gas/westland-infra_gas_2017.csv')
data2['Year'] = dt.date(2017, rd.randint(11)+1,rd.randint(30)+1)
data3= pd.read_csv('/content/drive/My Drive/Gas/westland-infra_gas_2018.csv')
data3['Year'] = dt.date(2018, rd.randint(11)+1,rd.randint(30)+1)
data4= pd.read_csv('/content/drive/My Drive/Gas/westland-infra_gas_2019.csv')
data4['Year'] = dt.date(2019, rd.randint(11)+1,rd.randint(30)+1)
data5= pd.read_csv('/content/drive/My Drive/Gas/westland-infra_gas_2014.csv')
data5['Year'] = dt.date(2014, rd.randint(11)+1,rd.randint(30)+1)
listd= [data5,data0,data1,data2,data3,data4]
westland= pd.concat (listd, axis=0, sort=False)
del [data0,data1,data2,data3,data4,data5]
westland.index=westland.Year
westland=westland.drop(['Year'],axis=1)
westland


# # **Exploratory Data Analysis and Data Preprocessing**

# In[44]:


coteq.info()


# In[45]:


rendo.info()


# In[46]:


westland.info()


# In[47]:


coteq=coteq.drop(['annual_consume_lowtarif_perc','smartmeter_perc'], axis=1)
westland=westland.drop(['annual_consume_lowtarif_perc','smartmeter_perc','%Defintieve aansl (NRM)'], axis=1)


# In[48]:


coteq.isnull().any()


# In[49]:


rendo.isnull().any()


# In[50]:


rendo.isnull().sum()


# In[51]:


rendo = rendo.dropna()


# In[52]:


westland.isnull().any()


# **Label Encoding**

# In[53]:


le = LabelEncoder()


# In[54]:


objList = coteq.select_dtypes(include = "object").columns
print (objList)
for feat in objList:
    coteq[feat] = le.fit_transform(coteq[feat].astype(str))
print (coteq.info())


# In[55]:


# Converting categorical values to numeric and saving into new dataframe and hence removing the old to clean memory
objList = rendo.select_dtypes(include = "object").columns
print (objList)
for feat in objList:
    rendo[feat] = le.fit_transform(rendo[feat].astype(str))

print (rendo.info())


# In[56]:


# Converting categorical values to numeric and saving into new dataframe and hence removing the old to clean memory
objList = westland.select_dtypes(include = "object").columns
print (objList)
for feat in objList:
    westland[feat] = le.fit_transform(westland[feat].astype(str))

print (westland.info())


# **Data Preparation**

# In[57]:


trainX = coteq.drop(['perc_of_active_connections'],axis=1)
trainY = coteq['perc_of_active_connections']
testX= rendo.drop(['perc_of_active_connections'],axis=1)
testY = rendo['perc_of_active_connections']
validX = westland.drop(['perc_of_active_connections'],axis=1)
validY = westland['perc_of_active_connections']


# **Data Normalization**

# In[58]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
testX = scaler.fit_transform(testX)
validX = scaler.fit_transform(validX)


# ## **LSTM**

# In[59]:


# Initializing parameters
input_dim = 12
timesteps = 1


# In[60]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (len(trainX), timesteps, len(trainX[0])))
testX = np.reshape(testX, (len(testX), timesteps, len(testX[0])))
validX = np.reshape(validX, (len(validX), timesteps, len(validX[0])))


# In[61]:


# Initiliazing the sequential model
model = Sequential()
# Configuring the parameters
model.add(LSTM(32,input_shape=(1,11)))
# Adding a dropout layer
model.add(Dropout(.5))
# Adding a dense output layer with sigmoid activation
model.add(Dense(units = 1, activation='sigmoid'))
model.summary()


# In[62]:


# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[63]:


# Training the model
model.fit(trainX, trainY,batch_size=50,epochs=10)


# In[64]:


# make predictions
trainPred = model.predict(trainX)
testPred = model.predict(testX)
validPred = model.predict(validX) 


# In[65]:


# calculate root mean absolute percentage error
trainMAPE = (mean_absolute_error(trainY, trainPred))/100
print('Train Score: %.2f MAPE' % (trainMAPE))
testMAPE =(mean_absolute_error(testY, testPred))/100
print('Test Score: %.2f MAPE' % (testMAPE))
valMAPE = (mean_absolute_error(validY, validPred))/100
print('Valid Score: %.2f MAPE' % (valMAPE))


# ## **ARIMA**

# Rendo

# In[66]:


# Create Training and Test
rendo = rendo.perc_of_active_connections


# In[67]:


# fit model
model = ARIMA(rendo, order=(1, 1, 1))  
fitted = model.fit(disp=-1) 
print(fitted.summary())


# In[68]:


# Forecast
fc, se, conf = fitted.forecast(len(rendo), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=rendo.index)
lower_series = pd.Series(conf[:, 0], index=rendo.index)
upper_series = pd.Series(conf[:, 1], index=rendo.index)


# In[69]:


# calculate root mean absolute percentage error
mape = (mean_absolute_error(rendo, conf[:, 1]))/100
print('Train Score: %.2f MAPE' % (mape))


# In[70]:


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(rendo, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# Coteq

# In[71]:


coteq=coteq['perc_of_active_connections']

# Forecast
fc, se, conf = fitted.forecast(len(coteq), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=coteq.index)
lower_series = pd.Series(conf[:, 0], index=coteq.index)
upper_series = pd.Series(conf[:, 1], index=coteq.index)


# In[72]:


# calculate root mean absolute percentage error
mape = (mean_absolute_error(coteq, conf[:, 1]))/100
print('Test Score: %.2f MAPE' % (mape))


# In[73]:


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(coteq, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# Westland-Infra

# In[74]:


westland = westland['perc_of_active_connections']


# In[75]:


# Forecast
fc, se, conf = fitted.forecast(len(westland), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=westland.index)
lower_series = pd.Series(conf[:, 0], index=westland.index)
upper_series = pd.Series(conf[:, 1], index=westland.index)


# In[76]:


# calculate root mean absolute percentage error
mape = (mean_absolute_error(westland, conf[:, 1]))/100
print('Valid Score: %.2f MAPE' % (mape))


# In[77]:


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(westland, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# # **Future Prediction**

# In[78]:


# Get forecast 30 steps ahead in future
fc, se, conf = fitted.forecast(steps=30, alpha=0.05)  # 95% conf
index=np.arange(2020,2050)
# Make as pandas series
fc_series = pd.Series(fc, index=index)
lower_series = pd.Series(conf[:, 0], index=index)
upper_series = pd.Series(conf[:, 1], index=index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast of next 30 years')
plt.legend(loc='upper left', fontsize=8)
plt.show()

