# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:00:06 2017

@author: LPZ
"""

import pandas
import numpy as np
from sklearn import preprocessing,tree
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.metrics import accuracy_score

#read csv data by pandas.read_csv
data_input = pandas.read_csv("krkopt.data", header = None)

#transform pandas data to numpy data
data = data_input.as_matrix()
train_percentage = 0.8
train_size = (int)(data.shape[0]*train_percentage)
np.random.shuffle(data)

#get the train data and target data
X = data[:,:-1]
y = data[:,-1:]

#reshape the target data to one dimension
y = y.reshape(y.size)

#preprocess to transform the a,b,c.. to number
le = preprocessing.LabelEncoder()
le.fit(X[:,0])
le.fit(X[:,2])
le.fit(X[:,4])
#print(list(le.classes_))
X[:,0] = le.transform(X[:,0])
X[:,2] = le.transform(X[:,2])
X[:,4] = le.transform(X[:,4])

#to do Regressions
le.fit(y)
y = le.transform(y)

#get the train and test data
X_test = X[train_size:,:]
y_test = y[train_size:]
X = X[:train_size,:]
y = y[:train_size]

#use BaggingClassifier with oob_score = True
bc = BaggingClassifier(n_estimators = 60, oob_score=True)
bc.fit(X, y)
print("BaggingClassifier:train")
predict_train = bc.predict(X)
print(accuracy_score(y,predict_train))
predict_train = bc._lpz_predict(X,y)
print(accuracy_score(y,predict_train))
#print(bc.oob_score_)
print("BaggingClassifier:test")
predict = bc.predict(X_test)
acc = accuracy_score(y_test,predict)
print(acc)
max_acc = acc
max_acc_i = 0
y1 = y

acc = accuracy_score(y,predict_train)
#use IterativeBagging for classifier
print("IterativeBagging")
for i in range(20):
    #predict test data
    predict = bc.predict(X_test)
    acc2 = accuracy_score(y_test,predict)
    #print("test:",e2)
    if(max_acc<acc2):
        max_acc = acc2
        max_acc_i = i
    y1 = y - predict_train
    n = X.shape[0]
    X_train = np.column_stack((X,y))
    #repeat the wrong predict train data
    for j in range(n):
        if(y1[j] != 0):
            temp = np.append(X[i],y[i])
            X_train = np.row_stack((X_train,temp))
    np.random.shuffle(X_train)
    X = X_train[:,:-1]
    y = X_train[:,-1:]
    y = y.reshape(y.size)
    #transform the dtype of y from object to int64
    le.fit(y)
    y = le.transform(y)
    bc.fit(X, y)
    predict_train = bc._lpz_predict(X,y)
    acc1 = acc
    acc = accuracy_score(y,predict_train)
    #predict = bc.predict(X_test)
    #err = accuracy_score(y_test,predict)
    print("train:",acc)
    if(acc1*1.005>acc):
        print(i)
        break

print("max:",max_acc_i,":",max_acc)
print("BaggingClassifier:test")
err = accuracy_score(y_test,predict)
print(err)    


#use RandomForestClassifier
#rfc = RandomForestClassifier(n_estimators = 10)
#rfc.fit(X, y)
#print("RandomForestClassifier:train")
#predict = rfc.predict(X)
#print(accuracy_score(y,predict))
#print("RandomForestClassifier:test")
#predict = rfc.predict(X_test)
#print(accuracy_score(y_test,predict))

#DecisionTreeClassifier
#tr = tree.DecisionTreeClassifier()
#tr = tr.fit(X, y)
#predict = tr.predict(X)
#print(accuracy_score(y,predict))
#predict = tr.predict(X_test)
#err = accuracy_score(y_test,predict)
#print(err)

#use BaggingClassifier
#bc = BaggingClassifier(n_estimators = 50)
#bc.fit(X, y)
#print("BaggingClassifier:train")
#predict = bc.predict(X)
#print(accuracy_score(y,predict))
#print("BaggingClassifier:test")
#predict = bc.predict(X_test)
#err = accuracy_score(y_test,predict)
#print(err)

#use AdaBossterClassifier
#abc = AdaBoostClassifier()
#abc.fit(X, y)
#print("AdaBoostClassifier:train")
#predict = abc.predict(X)
#print(accuracy_score(y,predict))
#print("AdaBoostClassifier:test")
#predict = abc.predict(X_test)
#print(accuracy_score(y_test,predict))


#to use RandomForestRegressor, not fit on this question
#le.fit(y)
#print(list(le.classes_))
#y = le.transform(y)
#use RandomForestRegressor
#rfr = RandomForestRegressor(n_estimators = 10,oob_score = True)
#rfr.fit(X,y)
#predict = rfr.predict(X)
#predict = predict.astype(int)
#print(accuracy_score(y,predict))

#use Support Vector Machine
#clf = svm.SVC(gamma=0.001,C=100.)
#clf.fit(X,y)
#clf.predict(X[-1])

