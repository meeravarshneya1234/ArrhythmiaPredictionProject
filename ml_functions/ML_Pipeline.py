#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

class ML_Pipeline(object):
    
    flag_save = True
    
    def __init__(self,data,label,features,seed=134556):
        self.data = data
        self.label = label
        self.features = features
        self.seed = seed
        
    def convert_df(self):
        df = pd.read_csv(self.data)
        df = df.dropna()
        return df  
    
    def evaluate(self,classifer, x_test, y_test,optimal_threshold):
        prob = classifer.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, prob[:,1])
#         optimal_idx = np.argmax(tpr-fpr)
#         optimal_threshold = 0.5 #threshold[optimal_idx]
        accuracy = accuracy_score(y_test,prob[:,1]>=optimal_threshold)
        tn, fp, fn, tp = confusion_matrix(y_test,prob[:,1]>=optimal_threshold).ravel()
        sen = tp/(tp + fn)
        spe = tn/(tn + fp)
        npv = tn/(tn + fn)
        ppv = tp/(tp + fp) 
        auc_score = auc(fpr, tpr)
        f1score = f1_score(y_test, prob[:,1]>=optimal_threshold)
        mcc = matthews_corrcoef(y_test, prob[:,1]>=optimal_threshold)
        metrics = [accuracy, optimal_threshold, sen, spe, npv, ppv, f1score, mcc, auc_score]
        rocs = [fpr, tpr]
        prediction = np.multiply(prob[:,1]>optimal_threshold,1)
        conf_matrix = confusion_matrix(y_test,prob[:,1]>=optimal_threshold)
        return metrics, rocs, prob, prediction, conf_matrix
    
    def get_threshold(self,classifier,x,y):
        prob = classifier.predict_proba(x)
        fpr, tpr, threshold = roc_curve(y, prob[:,1])
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = threshold[optimal_idx]
        return optimal_threshold
    
    
    def run_LR(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]
        
        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        
        LR = LogisticRegression(random_state=self.seed)
        
        scoring = {'AUC': 'roc_auc'}
        grid_values = {'C':[0.001,0.01,0.1,1,10,100]}
        grid_search = GridSearchCV(LR, param_grid = grid_values,cv=nfolds,
                                   scoring=scoring,return_train_score=True,refit='AUC')
        grid_search.fit(x_train, y_train)
        print('------------------------------------------')
        print('LR Best Params: ')
        print(grid_search.best_params_)        
        print(grid_search.best_score_)  
        
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)                
        return grid_search,result
    
    def run_SVM(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)

        param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'linear']}    
        grid_search = GridSearchCV(SVC(probability=True, random_state=self.seed), param_grid, cv=nfolds,verbose = 0,n_jobs = -1)
        grid_search.fit(x_train, y_train)
        print('------------------------------------------')
        print('SVM Best Params: ')
        print(grid_search.best_params_)
        print(grid_search.best_score_)       
        
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)   
        return grid_search,result
        
    def run_KNN(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)

        scoring = {'AUC': 'roc_auc'}
        param_grid = {'n_neighbors': [3,5,11,19]}    
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid,cv=nfolds,scoring=scoring,return_train_score=True,refit='AUC')
        grid_search.fit(x_train, y_train)
        print('------------------------------------------')
        print('KNN Best Params: ')
        print(grid_search.best_params_)
        print(grid_search.best_score_)       
        
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)   
        return grid_search,result
    
    def run_RF(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        
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

        rf = RandomForestClassifier(random_state=self.seed)
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                  cv = nfolds, n_jobs = -1, verbose = 0,scoring=scoring,return_train_score=True,refit='AUC')
        grid_search.fit(x_train, y_train)
        print('------------------------------------------')
        print('Random forest Best Params:') 
        print(grid_search.best_params_)
        print(grid_search.best_score_)       
        
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)   
        return grid_search,result

    def run_GB(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        
        param_grid = {
            'learning_rate': [1, 0.5, 0.01],
            'max_depth': [100, 200],
            'max_features': list(range(1,x_train.shape[1])),
            'n_estimators': [100, 200]
        }

        gb = GradientBoostingClassifier()
        grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, 
                                  cv = nfolds,n_jobs = -1)
        grid_search.fit(x_train, y_train)
        print('------------------------------------------')
        print('Gradient Boosting Best Params:')
        print(grid_search.best_params_)
        print(grid_search.best_score_)       
                
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)   
        return grid_search,result
    
    def run_XGB(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        
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
        xgb = XGBClassifier(random_state=self.seed)
        grid_search = GridSearchCV(estimator = xgb, param_grid = param_grid, 
                                  cv = nfolds,n_jobs = -1)
        grid_search.fit(x_train, y_train)          
        print('------------------------------------------')
        print('XGBoost Best Params: ')
        print(grid_search.best_params_)
        print(grid_search.best_score_) 
        
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)           
        return grid_search,result

    def run_MLP(self,nfolds=5,max_iter=300):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)
        
        param_grid = {
        'hidden_layer_sizes': [(32,), (32,32), (32,32,64)],
        'solver': ['lbfgs','adam'],
        'learning_rate_init':[0.001, 0.01, 0.1, 0.2, 0.3],
        'activation':['relu','tanh']
        }
        # Create a based model
        MLP = MLPClassifier(alpha=1e-5,random_state=self.seed)
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = MLP, param_grid = param_grid, 
                                  cv = nfolds,n_jobs = -1)
        grid_search.fit(x_train, y_train)
        print('------------------------------------------')
        print('MLP Best Params: ')
        print(grid_search.best_params_)
        print(grid_search.best_score_)       
        
        thres = self.get_threshold(grid_search, x_train, y_train)            
        result = self.evaluate(grid_search, x_test, y_test, thres)   
        return grid_search,result

    def run_NB(self,nfolds=5):
        
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)        
        thres = self.get_threshold(gnb, x_train, y_train)            
        result = self.evaluate(gnb, x_test, y_test, thres)   
        return result
    
    def create_table(self,classifiers,keys):    
        metrics =[]
        rocs = []
        prob =[]
        prediction =[]
        conf_matrix = []

        for i in range(len(classifiers)): 
            metrics.append(classifiers[i][0])
            rocs.append(classifiers[i][1])
            prob.append(classifiers[i][2])
            prediction.append(classifiers[i][3])
            conf_matrix.append(classifiers[i][4])


        result_table = dict(zip(keys, metrics)) 
        columns = ["Accuracy","Threshold","Sensitivity", "Specificity", "NPV", "PPV", "F1score", "MatthewCoef","AUC"]

        result_table = pd.DataFrame.from_dict(result_table, orient='index')
        result_table.columns = columns
        return result_table,metrics,rocs,prob,prediction,conf_matrix
    
    def save_data(self,filename,result,keys): 
        
        pred = result[4]
        prob = result[3]
        rocs = result[2]
        
        # save predictions
        preds = pd.DataFrame(pred).T
        preds.index = self.train_test_data[3].index
        preds.columns = keys
        preds["Label"] = self.train_test_data[3]   
        
        # save the rocs and probs 
        final_roc = pd.DataFrame()
        probs = pd.DataFrame()
        for i in range(len(keys)):
            fpr = pd.Series(rocs[i][0])
            tpr = pd.Series(rocs[i][1])
            roc = pd.concat([fpr, tpr], axis =1)
            roc.columns = [keys[i] + '_FPR',keys[i] + '_TPR']

            final_roc = pd.concat([final_roc, roc], axis =1)  
            
            probs[keys[i]] = pd.Series(prob[i][:,1])  

        probs.index = preds.index
        probs['Label'] = preds["Label"]    
        
        save_filename = filename + '.xlsx'
        exists = os.path.isfile(save_filename)
        if self.flag_save:
            if exists:
                print('File already exists')
            else:    
                writer = pd.ExcelWriter(save_filename, engine='xlsxwriter')
                result_table.to_excel(writer, sheet_name = 'Results')
                final_roc.to_excel(writer, sheet_name='ROCs',index=False)
                preds.to_excel(writer, sheet_name='Prediction')
                probs.to_excel(writer, sheet_name='Probability')
                writer.save()  
    
            
    def plotting(self,fpr_tpr,overall_result): 
        
        sns.set_context("notebook", font_scale=2.2, rc={"lines.linewidth": 5})
        sns.set_style("dark")
        sns.despine()
        sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 5})
    
        AUC = list(overall_result['AUC']) # get best algorithm 
        idx = AUC.index(max(AUC))
        algs = overall_result.index

        fig = plt.figure(figsize=(20,10) )
        plt.subplot2grid((2,4),(0,0), colspan = 2, rowspan=2)
        cmap = plt.cm.get_cmap('Set1')
        for ii in range(len(algs)):
            if ii != idx:
                ax = sns.lineplot(fpr_tpr[ii][0], fpr_tpr[ii][1],color = cmap(ii))
                ax.lines[ii].set_linestyle(":")
            else:
                ax = sns.lineplot(fpr_tpr[ii][0], fpr_tpr[ii][1],color = cmap(ii),linewidth = 7)

        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.legend(algs,loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3,borderaxespad=0, frameon=False)
        plt.plot([0,1], [0,1], 'k--')

        columns = ["Accuracy", "AUC"]
        for ii, col in enumerate(columns): 
            plt.subplot2grid((2,4), (ii,2),colspan = 2)
            bars = sns.barplot(x="index", y=col, data=overall_result.reset_index(),palette="Set1",edgecolor='black',linewidth = 3)
            bars.patches[idx].set_hatch('\\')
            plt.title(col)
            plt.ylim(0.5,1)
            if ii == 1:
                plt.xticks(np.arange(8),algs,rotation=90)
                cur_axes = plt.gca()
                cur_axes.set_xlabel('')
                cur_axes.set_ylabel('')
            else:
                cur_axes = plt.gca()
                cur_axes.axes.get_xaxis().set_visible(False)
                cur_axes.set_xlabel('')
                cur_axes.set_ylabel('')

        fig.tight_layout()

     

