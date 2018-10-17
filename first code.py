# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:50:34 2018

@author: Srinivas
"""

import pandas as pd
import numpy as np

traindata = pd.read_csv("train.csv",header = 0)
testdata = pd.read_csv("test.csv",header = 0)
sample= pd.read_csv("sample_submission.csv",header = 0)

traindata.dtypes

len(traindata['Tag'].unique())
len(testdata['Tag'].unique())

user_train = traindata.Username.unique().tolist()

user_test = testdata.Username.unique().tolist()

unlist = [x for x in user_test if x not in user_train]

X = traindata.drop(["ID","Username","Upvotes"],axis = 1)
X = pd.get_dummies(X)
Y = traindata[["Upvotes"]]

import random
random.seed(500)

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators = 100,max_depth = 62,oob_score=True,random_state = 500,bootstrap = True)
reg.fit(X,Y)
reg.oob_score_

testdata1 = testdata.drop(["ID","Username"],axis = 1)
testdata1 = pd.get_dummies(testdata1)

prediction = reg.predict(testdata1)
sample["Upvotes"] = prediction
sample.to_csv('prediction_file_1.csv',index=False)