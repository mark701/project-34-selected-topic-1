#!/usr/bin/env python
# coding: utf-8

# In[9]:


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 20:32:24 2021

@author: Mohamed Moubarak Re-pc
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score , auc , roc_curve
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt



dataset = pd.read_csv("Churn.csv")
#print(dataset.head())
#print(dataset.shape)
#print(dataset.info())

dataset = dataset.drop(['customerID'] , axis = 1)


encode_x = LabelEncoder()
encode_one = OneHotEncoder()
scaler= StandardScaler()


dataset['TotalCharges'] = pd.to_numeric(dataset.TotalCharges , errors = 'coerce' )
#print(dataset.isnull().sum())
# there is only 11 columns from 7280 has space or null value
dataset.dropna(inplace = True)
#print(dataset.isnull().sum())
#print(dataset.describe())

dataset['Churn'] = dataset['Churn'].map({'Yes':1 , 'No':0})
#print(dataset['Churn'])

#One_Hot Encoding
OneHotEncoding_columns = ['gender', 'StreamingTV', 'DeviceProtection', 'OnlineSecurity', 'StreamingMovies', 'PaperlessBilling', 'MultipleLines', 'TechSupport', 'OnlineBackup', 'Partner', 'Churn', 'SeniorCitizen', 'Dependents', 'PhoneService']
for i in  OneHotEncoding_columns:
    encode_x.fit(dataset[i])
    dataset[i]  = encode_x.transform(dataset[i])

#print(dataset.columns)


#LabelEncoding 
temp2 = ['PaymentMethod', 'Contract', 'InternetService']
for i in temp2:
    dataset = pd.concat([dataset,pd.get_dummies(dataset[i],prefix=i,drop_first=True)],axis=1)
    dataset.drop([i],axis = 1 , inplace = True)
    

#print(dataset.info())

#print(dataset.tail())
#print(dataset.shape)
X = dataset.drop(columns = ['Churn'])
y = dataset['Churn'].values

X_train , X_test , Y_train , Y_test = train_test_split(X, y, test_size = 0.20 , random_state = 40)

#scaleing
temp = ['MonthlyCharges','TotalCharges' , 'tenure']
dataset[temp] = scaler.fit_transform(dataset[temp])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


###ssss



# # Build the model
model = Sequential()
model.add(Dense(21, activation='linear'))
#model.add(Dropout(0.001))
model.add(Dense(50))
#model.add(Dropout(0.001))
model.add(Dense(1, activation='sigmoid'))

# # # Configure a model for mean-squared error regression.
model.compile(loss='mse', optimizer='adam',metrics = ['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)

# # # Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[checkpoint])
model.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# # # Evaluate

print(model.evaluate(X_test, Y_test))
print(model.metrics_names)

# # # Predict
# print(model.predict(X_test), Y_test)
# print(Y_test)

# # for i in range(len(labels)):
# #  print(model.predict(X_test)[i][0], Y_test[i])

# # model.predict(np.array([[1.2,2.3]]))

# #round(model.predict(np.array([[1, 1]]))[0][0])


#from mlxtend.plotting import plot_decision_regions

#plot_decision_regions(X_train, np.array( Y_train,dtype='int64'), clf=model, zoom_factor=1.)
#plt.show()






