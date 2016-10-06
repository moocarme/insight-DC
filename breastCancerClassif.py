#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:23:33 2016

@author: matt-666
"""

import pandas as pd
import requests
import io
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression

# Helper Functions =================================================
def simpleaxis(ax):
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.get_xaxis().tick_bottom()
   ax.get_yaxis().tick_left()
   ax.xaxis.set_tick_params(size=6)
   ax.yaxis.set_tick_params(size=6)


colnames = ['id_number' ,'Clump_Thickness','Uniformity_of_Cell_Size', 
            'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 
            'Bare_Nuclei', 'Bland_Chromatin' ,'Normal_Nucleoli',
            'Mitoses','Class']#: (2 for benign, 4 for malignant)
            
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
s = requests.get(url).content
all_data = pd.read_csv(io.StringIO(s.decode('utf-8')), header = None)

target = all_data[10]
features = all_data.drop([0, 10], 1)

target.unique()
trueTarget = pd.Series({'Result':['benign' if result==2 else 'malignent' for result in target]})

numBelign = sum(target == 2)
numMalig = sum(target == 4)

X_train

crossVal = 10
C = np.logspace(-5,5,31)
LRCV = [LogisticRegressionCV(cv = crossVal, Cs = C, penalty = 'l1', solver = 'liblinear'),
        LogisticRegressionCV(cv = crossVal, Cs = C, penalty = 'l2', solver = 'liblinear')]
   
for LR in LRCV: 
    LR.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    classif_rate = np.mean(y_pred.ravel() == y_test.ravel()) * 100
    classif_rates.append(classif_rate)

classifiers = [
               
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth = 5),
    RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features=1),
    #BernoulliRBM(learning_rate=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

