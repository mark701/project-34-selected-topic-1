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

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.40 , random_state = 1)

#scaleing
temp = ['MonthlyCharges','TotalCharges' , 'tenure']
dataset[temp] = scaler.fit_transform(dataset[temp])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#svc
svc_model = SVC(kernel = 'linear', random_state = 1)
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print(accuracy_score(y_test, y_pred))
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)

