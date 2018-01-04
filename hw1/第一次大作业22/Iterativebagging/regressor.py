# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:32:09 2017

@author: LPZ
"""

#设置训练数据比例
train_percentage = 1


##使用boston数据集  
#from sklearn import datasets
#boston=datasets.load_boston()
#train_size = (int)(boston.data.shape[0]*train_percentage)
#X=boston.data[:train_size]  
#y=boston.target[:train_size]  
#X_test=boston.data[train_size:]  
#y_test=boston.target[train_size:]


#使用protein数据集
import numpy as np
import pandas
data_input = pandas.read_csv("CASP.csv", header = None)
#transform pandas data to numpy data
data = data_input.as_matrix()
train_size = (int)(data.shape[0]*train_percentage)
#get the train data and target data
data = data[1:,:]
np.random.shuffle(data)
X = data[:train_size,1:]
y = data[:train_size,:1]
y = y.reshape(y.size)
y = y.astype(float)


##使用friedman3数据集
#from sklearn.datasets import make_friedman3
#X,y = make_friedman3(n_samples=800,noise=0.0111,random_state=None)


#使用Bagging和IterativeBagging算法进行预测并输出均方差
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

#使用Bagging算法进行回归预测
br = BaggingRegressor(n_estimators=80,oob_score=True)
br.fit(X, y)
print("BaggingRegressor:train")
#包内回归测试
predict_train = br.predict(X)
print(mean_squared_error(y,predict_train))
#包外回归测试
predict_train = br._lpz_predict(X,y)
print(mean_squared_error(y,predict_train))
#print(bc.oob_score_)
#print("BaggingRegressor:test")
#predict = br.predict(X_test)
#print(mean_squared_error(y_test,predict))
y1 = y

err = mean_squared_error(y,predict_train)
min_err = err
#使用IterativeBagging算法进行回归预测
print("IterativeBagging")
for i in range(1):
    #predict test data
    y1 = y1 - br._lpz_predict(X,y1)
    br.fit(X, y1)
    predict_train += br._lpz_predict(X,y1)
    err = mean_squared_error(y,predict_train)
    print(err)
    if(err>1.2*min_err):
        break
    if(err<min_err):
        min_err = err
    #predict += br.predict(X_test)
print("m:\n",min_err)
#print(mean_squared_error(y_test,predict))

