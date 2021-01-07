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

def evaluate(classifer, x_test, y_test):
    prob = classifer.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, prob[:,1])
    optimal_idx = np.argmax(tpr-fpr)
    optimal_threshold = threshold[optimal_idx]
    accuracy = accuracy_score(y_test,prob[:,1]>=optimal_threshold)
    tn, fp, fn, tp = confusion_matrix(y_test,prob[:,1]>=optimal_threshold).ravel()
    sen = tp/(tp + fn)
    spe = tn/(tn + fp)
    npv = tn/(tn + fn)
    ppv = tp/(tp + fp) 
    auc_score = auc(fpr, tpr)
    metrics = [accuracy, optimal_threshold, sen, spe, npv, ppv, auc_score]
    rocs = [fpr, tpr]
    prediction = np.multiply(prob[:,1]>optimal_threshold,1)
    conf_matrix = confusion_matrix(y_test,prob[:,1]>=optimal_threshold)
    return metrics, rocs, prob, prediction, conf_matrix


def create_table(x_train_ann,x_test_ann,y_train_ann,y_test_ann,x_train,x_test,y_train,y_test,classifiers,keys):    
    metrics =[]
    rocs = []
    prob =[]
    prediction =[]
    conf_matrix = []

    for k in keys:
        idx = keys.index(k)
        if k == 'ANN':
            result = evaluate(classifiers[idx], x_test_ann, y_test_ann)
            metrics.append(result[0])
            rocs.append(result[1])
            prob.append(result[2])
            prediction.append(result[3])
            conf_matrix.append(result[4])
            print([idx])

        else:
            result = evaluate(classifiers[idx], x_test, y_test)
            metrics.append(result[0])
            rocs.append(result[1])
            prob.append(result[2])
            prediction.append(result[3])
            conf_matrix.append(result[4])

    result_table = dict(zip(keys, metrics)) 
    columns = ["Accuracy","Threshold","Sensitivity", "Specificity", "NPV", "PPV", "AUC"]

    result_table = pd.DataFrame.from_dict(result_table, orient='index')
    result_table.columns = columns
    result_table
    return result_table,metrics,rocs,prob,prediction,conf_matrix


def plot_algs(fpr_tpr,overall_result): 
    
    AUC = list(overall_result['AUC'])
    idx = AUC.index(max(AUC))

    algs = overall_result.index

    sns.set_context("notebook", font_scale=2.2, rc={"lines.linewidth": 5})
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
    
def save_prob(prob,ylabel,keys):
    p = pd.DataFrame()
    for i in range(8):
        p[keys[i]] = pd.Series(prob[i][:,1])  

    p.index = ylabel.index
    p['Label'] = ylabel  
    return p

def save_pred(pred,ylabel,keys):
    p = pd.DataFrame(pred).T
    p.index = ylabel.index
    p.columns = keys
    p["Label"] = ylabel
    return p

def save_ROCs(rocs,keys): 
    final_roc = pd.DataFrame()
    for i in range(8):
        fpr = pd.Series(rocs[i][0])
        tpr = pd.Series(rocs[i][1])
        roc = pd.concat([fpr, tpr], axis =1)
        roc.columns = [keys[i] + '_FPR',keys[i] + '_TPR']
    
        final_roc = pd.concat([final_roc, roc], axis =1)        
    return final_roc