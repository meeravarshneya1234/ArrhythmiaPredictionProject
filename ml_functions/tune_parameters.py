import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

def lr_param_selection(x_train, y_train, nfolds,seed):
    """ X is the training set
        y is the validation set
        nfolds is the number of folds chosen for cross validation
    """
    scoring = {'AUC': 'roc_auc'}
    grid_values = {'C':[0.001,0.01,0.1,1,10,100]}
    grid_search = GridSearchCV(LogisticRegression(random_state=seed), param_grid = grid_values,cv=nfolds,scoring=scoring,return_train_score=True,refit='AUC')
    grid_search.fit(x_train, y_train)
    print('LR best params: ')
    print(grid_search.best_params_)
    return grid_search


def svc_param_selection(x_train, y_train, nfolds,seed):
    """ X is the training set
        y is the validation set
        nfolds is the number of folds chosen for cross validation
    """
    param_grid = {'C': [0.1,1,10,30,50,70, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'linear']}    
    grid_search = GridSearchCV(SVC(probability=True, random_state=seed), param_grid, cv=nfolds,verbose = 2,n_jobs = -1)
    grid_search.fit(x_train, y_train)
    print('SVM best params: ')
    print(grid_search.best_params_)
    return grid_search

def knn_param_selection(x_train, y_train, nfolds,seed):
    """ X is the training set
        y is the validation set
        nfolds is the number of folds chosen for cross validation
    """
    scoring = {'AUC': 'roc_auc'}
    param_grid = {'n_neighbors': [3,5,11,19]}    
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid,cv=nfolds,scoring=scoring,return_train_score=True,refit='AUC')
    grid_search.fit(x_train, y_train)
    print('KNN best params: ')
    print(grid_search.best_params_)
    return grid_search

def random_forest_selection(x_train, y_train, nfolds,seed):
    scoring = {'AUC': 'roc_auc'}
    param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 200],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8],
    'n_estimators': [100, 200],
    'criterion': ['entropy','gini']
    }
    # Create a based model
    rf = RandomForestClassifier(random_state=seed)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = nfolds, n_jobs = -1, verbose = 2,scoring=scoring,return_train_score=True,refit='AUC')
    grid_search.fit(x_train, y_train)
    print('Random forest best params:') 
    print(grid_search.best_params_)
    return grid_search  

def gradientboosting_selection(x_train, y_train, nfolds,seed):
    param_grid = {
    'learning_rate': [1, 0.5],
    'max_depth': [100, 200],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8],
    'n_estimators': [100, 200]
    }
    # Create a based model
    gb = GradientBoostingClassifier(random_state=seed)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, 
                              cv = nfolds,n_jobs = -1)
    grid_search.fit(x_train, y_train)
    print('Gradient Boosting best params:')
    print(grid_search.best_params_)
    return grid_search

def xgboost_selection(x_train, y_train, nfolds,seed):
    param_grid = {
        'learning_rate': [0.1,0.01],
    'gamma': [0, 0.1],
    'subsample': [0.8],
    'min_child_weight': [8],
    'max_depth': [100, 200],
    #'max_features': [2, 3],
    #'min_samples_leaf': [3],
    #'min_samples_split': [8],
    'n_estimators': [100, 200]
    }
    # Create a based model
    xgb = XGBClassifier(random_state=seed)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = xgb, param_grid = param_grid, 
                              cv = nfolds,n_jobs = -1)
    grid_search.fit(x_train, y_train)
    print('XGBoost best params: ')
    print(grid_search.best_params_)
    return grid_search

def ann_selection(x_train, y_train, nfolds,seed):
    param_grid = {
    'hidden_layer_sizes': [(32,), (32,32), (32,32,64)],
    'solver': ['lbfgs','adam'],
    'learning_rate_init':[0.001, 0.01, 0.1, 0.2, 0.3],
    'activation':['relu','tanh']
    }
    # Create a based model
    ann = MLPClassifier(alpha=1e-5,random_state=seed)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = ann, param_grid = param_grid, 
                              cv = nfolds,n_jobs = -1)
    grid_search.fit(x_train, y_train)
    print('ANN Best Params: ')
    print(grid_search.best_params_)
    return grid_search

def nb_selection(x_train, y_train, nfolds,seed):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    return gnb