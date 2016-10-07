#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:23:33 2016

@author: matt-666
"""
# Import libraries

import pandas as pd
import requests
import io
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import tree
import pydotplus
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

# Helper Functions =================================================
def simpleaxis(ax):
    """
    This function removes spines for a cleaner plot - Thanks Hugo! 
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(); plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 25)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 20)
    plt.yticks(tick_marks, classes, size = 20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size = 20)

    plt.tight_layout()
    plt.ylabel('True label', size= 20)
    plt.xlabel('Predicted label', size= 20)

def plotROC(y_probs, classes):
    """
    This function plots the ROC of given probabilities
    """
    fig = plt.figure(668);plt.clf()
    ax = fig.add_subplot(111)
    ax = simpleaxis(ax)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    if len(classes) <= 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_probs[:,1], pos_label = classes[1])
        roc_auc = auc(fpr, tpr)    
        plt.plot(fpr, tpr, lw=2, label='Positive: %s (area = %0.4f)' % (classes[1], roc_auc))
    else:    
        for i, class_ in enumerate(classes):
            y_class_probs = [x[i] for x in y_probs]
            fpr, tpr, thresholds = roc_curve(y_test, y_class_probs, pos_label = classes[i])
            roc_auc = auc(fpr, tpr)    
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (classes[i], roc_auc))
        
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', size = 20); plt.xticks(size = 20)
    plt.ylabel('True Positive Rate', size = 20); plt.yticks(size = 20)
    plt.title('ROC Curve', size = 25)
    plt.legend(loc = "lower right", fontsize = 20); 


def plot_Learning_Curve(X_train, X_test, y_train, y_test, clf, res = 20):
    """
    Plots the training, test, and validation accuracies of a model as a
    function of the 
    """
    train_len = X_train.shape[0]
    samp_size = range(train_len//res, train_len, train_len//res)
    
    mean_scores, std_scores = [], []
    scores_train, scores_test = [], []
    
    for sample in samp_size:
        score = cross_val_score(clf, X_train.iloc[:int(sample)], y_train.iloc[:int(sample)], cv = 5)
        mean_scores.append(np.mean(score) * 100)
        std_scores.append(np.std(score) * 100)
        clf.fit(X_train.iloc[:int(sample)], y_train.iloc[:int(sample)])
        scores_train.append(np.mean(clf.predict(X_train.iloc[:int(sample)]).ravel() == y_train.iloc[:int(sample)].ravel()) * 100)
        scores_test.append(np.mean(clf.predict(X_test.iloc[:int(sample)]).ravel() == y_test.iloc[:int(sample)].ravel()) * 100)
    
    fig = plt.figure(); plt.clf()
    ax = fig.add_subplot(111)
    ax = simpleaxis(ax)
    plt.fill_between(samp_size, np.asarray(mean_scores) - np.asarray(std_scores),
                         np.asarray(mean_scores) + np.asarray(std_scores), alpha=0.1,
                         color="r", label = 'Validation')
    plt.plot(samp_size, mean_scores, color = 'r', linewidth = 2, label = 'Validation')                     
    plt.plot(samp_size, scores_train, 'b', linewidth = 2, label = 'Train')
    plt.plot(samp_size, scores_test, 'g', linewidth = 2, label = 'Test')
    plt.title('Learning Curve', size = 25)
    plt.xticks(size = 20); plt.xlabel('Sample Size', size = 20)
    plt.yticks(size = 20); plt.ylabel('Accuracy (%)', size = 20)
    plt.legend(loc = 4, fontsize = 20)

def plot_correlation_matrix(features, title = 'Correlation Matrix', cmap = plt.cm.RdBu_r):
    """
    This function plots the confusion matrix, which measures
    the correlation between all the features of a dataframe.
    """
    plt.figure(); plt.clf()
    plt.imshow(np.corrcoef(features.T), interpolation='nearest', cmap=cmap)
    plt.title(title, size = 25)
    plt.colorbar()
    
def plot_Features_Box_Plot(features, target, colnames = None):
    """
    This function makes a Boxplot from the various features, separated by
    target column
    """
    assert (features.shape[0] == target.shape[0]), "Feature set and target set have different observations"
    if colnames:
        assert ((len(colnames) - 1) == features.shape[1]), "Colnames should have same length as number of columns of features and target combined"
    else:
        colnames = range(features.shape[1] + 1)
    plotData = pd.concat([features, target], axis = 1, ignore_index = True)
    plotData.columns = colnames[:-1]+['Result']
    df_long = pd.melt(plotData, 'Result', var_name="Features", value_name="Count")
    sns.set(font_scale = 1.5)
    g = sns.factorplot("Features", hue = 'Result', y="Count", data=df_long, kind="box")
    g.set_xticklabels(rotation=30)
    
def print_Eval_Metrics(y_pred, y_test):
    """
    Prints the accuracy, precision, recall and f1-score
    """
    prf = precision_recall_fscore_support(y_test, y_pred)
    print('Accuracy: %s' % np.mean(y_pred.ravel() == y_test.ravel()))
    print('Precision: %s' % prf[0][0])
    print('Recall: %s' % prf[1][0])
    print('f1-score: %s' % prf[2][0])

def plot_Feature_Importances(clf, featureNames):
    """
    Plots the sorted feaure importances
    """
    sorted_coeffs = [y for (x,y) in sorted(zip(clf.coef_[0], featureNames))]
    sorted_coeff_vals = sorted(clf.coef_[0])
    
    plt.barh(range(len(sorted_coeff_vals)),sorted_coeff_vals, color = '#34495e')
    plt.yticks(np.asarray(range(len(sorted_coeffs)))+0.5, sorted_coeffs)
    
# =====================================================================    
    
colnames = ['id_number' ,'Clump_Thickness','Uniformity_of_Cell_Size', 
            'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 
            'Bare_Nuclei', 'Bland_Chromatin' ,'Normal_Nucleoli',
            'Mitoses','Class']#: (2 for benign, 4 for malignant)

# url containing the data            
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
s = requests.get(url).content

#grab the dats in csv format
all_data = pd.read_csv(io.StringIO(s.decode('utf-8')), header = None)

# exploratory data analysis
all_data.shape

all_data.dtypes
all_data[6].unique()

# how many values of ?
all_data[all_data[6] == '?'].shape[0]/float(all_data.shape[0])*100

# Not that many (~2.3%)so lets drop them and reformat
all_data_noNAs = all_data.drop(all_data.index[all_data[6] == '?'], 0)
all_data_noNAs[6] = all_data_noNAs[6].astype(int)

# count num benign and malignant
numBelign = sum(all_data_noNAs[10] == 2)/float(all_data_noNAs.shape[0])*100
numMalig = sum(all_data_noNAs[10] == 4)/float(all_data_noNAs.shape[0])*100

# create target and features dataset
target = all_data_noNAs[10]
features = all_data_noNAs.drop([0, 10], 1)

#check for consistency
target.unique()

# convert to interpretable results (and to check we dont treat as numerical values)
trueTarget = pd.Series(['benign' if result==2 else 'malignent' for result in target], index = features.index)

# Plot correlation between features 
plot_correlation_matrix(features)

# Plot correlation between features 
plot_Features_Box_Plot(features, trueTarget, colnames[1:])

# Split into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(features, trueTarget, test_size = 0.2, random_state = 42)

# 10 fold CV logistic regression
crossVal = 10
C = np.logspace(0,5,51) # sweep over regularization params
LRCV = [LogisticRegressionCV(cv = crossVal, Cs = C, penalty = 'l1', solver = 'liblinear'),
        LogisticRegressionCV(cv = crossVal, Cs = C, penalty = 'l2', solver = 'liblinear')]

classif_rates = []
for LR in LRCV: 
    LR.fit(X_train, y_train)     # fit model
    y_pred = LR.predict(X_test)  # predict on test set
    # find accuracy
    classif_rate = np.mean(y_pred.ravel() == y_test.ravel()) * 100
    classif_rates.append(classif_rate)
    
# Since both the same lets use l2 since all coeffs are preserved
    
# Logistic Regression Diagnosis ==============================================

# Retrain
L2 = LogisticRegression(penalty = 'l2', C = LR.C_[0])
L2.fit(X_train, y_train)
y_pred = L2.predict(X_test)

# Evaluation metrics
print_Eval_Metrics(y_pred, y_test)

# confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, L2.classes_)

# ROC curve =
y_probs = L2.predict_proba(X_test)
plotROC(y_probs[:,1], L2.classes_[1])

# Learning Curve =
plot_Learning_Curve(X_train, X_test, y_train, y_test, clf = L2, res = 40)

# Plot coeffs =
plot_Feature_Importances(L2, colnames[1:-1])


# Decision tree ==============================================================

DTC2 = tree.DecisionTreeClassifier(max_depth = 3, max_features = 9)
DTC2.fit(X_train, y_train)
y_pred_dtc = DTC2.predict(X_test)

# Print eval metrics =
print_Eval_Metrics(y_pred_dtc, y_test)

# Plot confusion matrix
cnf_matrix_dtc = confusion_matrix(y_test, y_pred_dtc)
plot_confusion_matrix(cnf_matrix_dtc, DTC2.classes_)

# Plot ROC
y_probs_dtc = DTC2.predict_proba(X_test)
plotROC(y_probs_dtc, L2.classes_)

# Plot decision tree
dot_data = tree.export_graphviz(DTC2, out_file=None, feature_names = colnames[1:-1],
                                class_names = DTC2.classes_, filled = True,
                                rounded = True, special_characters = True) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("breastCancer.png")

# SVM ====================================================

# set clsssifiers
classifiers = [
    SVC(kernel="linear", C = 0.025),
    SVC(gamma=2, C=1)]

# perform cross-validation
scores = []
for clf in classifiers:
    score = cross_val_score(clf, X_train, y_train, cv = 5)
    scores.append(score)
    print("Model has accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std() * 2))

# fit models    
clf_lin = classifiers[0]
clf_rbf = classifiers[1]
clf_lin.fit(X_train, y_train)
clf_rbf.fit(X_train, y_train)

# evaluate on test set
y_pred_lin = clf_lin.predict(X_test)
y_pred_rbf = clf_rbf.predict(X_test)

# Print evaluation metrics
print_Eval_Metrics(y_pred_lin, y_test)
print_Eval_Metrics(y_pred_rbf, y_test)

# Plot confusion matrices
cnf_matrix_lin = confusion_matrix(y_test, y_pred_lin)
plot_confusion_matrix(cnf_matrix_lin, clf_lin.classes_)

cnf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
plot_confusion_matrix(cnf_matrix_rbf, clf_rbf.classes_)

#y_probs_lin = clf_lin.predict_proba(X_test)
#plotROC(y_probs_lin, L2.classes_)
