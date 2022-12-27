# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 21:54:02 2022

@author: waghm
"""

import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')
big_train=pd.read_csv('train_bigmart_sales.csv')
big_train.isnull().sum()/ big_train.shape[0] * 100.00
big_test=pd.read_csv('test_BigMartSales.csv')
big_test.isnull().sum()/big_test.shape[0]*100.00
big_train.Item_Weight.plot.box()
big_train.Item_Weight=big_train.Item_Weight.fillna(big_train.Item_Weight.mean())
big_train.isnull().sum()
big_train.Outlet_Size=big_train.Outlet_Size.fillna(big_train.Outlet_Size.mode()[0])
big_train.isnull().sum()
big_train.Item_Weight.plot.box()
big_test.Item_Weight=big_test.Item_Weight.fillna(big_train.Item_Weight.mean())
big_test.isnull().sum()
big_test.Outlet_Size=big_test.Outlet_Size.fillna(big_train.Outlet_Size.mode()[0])
big_test.isnull().sum()
plt.bar(big_train.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean().index.tolist(),big_train.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean().tolist(),color=('white'),edgecolor=('green','blue','red'))
plt.xlabel('Outlet_Location_Type')
plt.ylabel('Item_Outlet_Sales')
plt.show()
sns.boxplot(data=big_train,x='Outlet_Type',y='Item_Outlet_Sales')
def Correction_1(data,n):
    q1,q3=np.percentile(data[n],[25,75])
    low=q1-(1.5)*data[n].std()
    high=q3+(1.5)*data[n].std()
    for i in range(0,len(data)):
        if data[n][i]<low:
            data[n][i]=low
        elif data[n][i]>high:
            data[n][i]=high
        else:
            data[n][i]=data[n][i]
            
    return data[n]
big_train.Item_Visibility.plot.box()
big_train.Item_Visibility.describe()
Correction_1(big_train,'Item_Visibility')
big_train.Item_Visibility.plot.box()
big_train.select_dtypes(include=[np.number]).head(5)
oh=OneHotEncoder()
enc=oh.fit_transform(big_train[['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']]).toarray()
new_data=pd.DataFrame(enc)
big_train=big_train.join(new_data)
big_train=big_train.drop(columns=['Item_Identifier','Outlet_Identifier','Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type'])

big_test.Item_Visibility.plot.box()
Correction_1(big_test,'Item_Visibility')
big_test.Item_Visibility.plot.box()
#TestData_Encoding
enc_t=oh.fit_transform(big_test[['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']]).toarray()
new_data_test=pd.DataFrame(enc_t)
big_test=big_test.join(new_data_test)
big_test=big_test.drop(columns=['Item_Identifier','Outlet_Identifier','Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type'])
x=big_train.drop(columns='Item_Outlet_Sales')
y=big_train['Item_Outlet_Sales']
mx=MinMaxScaler()
pd.DataFrame(mx.fit_transform(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
m1=LinearRegression()
m1.fit(x_train,y_train)
pred_m1=m1.predict(x_test)
pred_m1
m2=SVR()
m2.fit(x_train,y_train)
pred_m2=m2.predict(x_test)
pred_m2
m3=DecisionTreeRegressor()
m3.fit(x_train,y_train)
pred_m3=m3.predict(x_test)
pred_m3
m4=RandomForestRegressor()
m4=RandomForestRegressor(n_estimators=100,max_depth=3,max_features=5,criterion='squared_error')
m4.fit(x_train,y_train)
pred_m4=m4.predict(x_test)
pred_m4
print(r2(m1.predict(x_test),y_test))
print(mse(m1.predict(x_test),y_test))
print(mae(m1.predict(x_test),y_test))
print(mape(m1.predict(x_test),y_test))
#Ploting Error by Linear Regression Model
plt.scatter(m1.predict(x_test), m1.predict(x_test) - y_test,
            color = "brown", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()
print(mae(m2.predict(x_test),y_test))
print(mse(m2.predict(x_test),y_test))
print(mape(m2.predict(x_test),y_test))
print(r2(m2.predict(x_test),y_test))
#Ploting Error by SVM Model
plt.scatter(m2.predict(x_test), m2.predict(x_test) - y_test,
            color = "red", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()
print(mae(m3.predict(x_test),y_test))
print(mse(m3.predict(x_test),y_test))
print(mape(m3.predict(x_test),y_test))
print(r2(m3.predict(x_test),y_test))
#Ploting Error by DecisionTree Model
plt.scatter(m3.predict(x_test), m3.predict(x_test) - y_test,
            color = "green", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()
print(mae(m4.predict(x_test),y_test))
print(mse(m4.predict(x_test),y_test))
print(mape(m4.predict(x_test),y_test))
print(r2(m4.predict(x_test),y_test))
#Ploting Error by RandomForest Model
plt.scatter(m4.predict(x_test), m4.predict(x_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()
#Model M3 i.e Decision Tree is Selected
#Final_Training
m2.fit(x,y)
BigPred_m3=m3.predict(big_test)
BigPred_m3
big_test_2=pd.read_csv('test_BigMartSales.csv')
big_test_2['Item_Outlet_Sales']=BigPred_m3
submission=big_test_2[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('submission_1.csv')