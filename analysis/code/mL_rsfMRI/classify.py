#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
import itertools
#import other python scripts for further anlaysis
import reshape
import plotFW
import results
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/'
# Subjects and tasks
taskList=['mixed', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
#DS combination
DSvars=list(itertools.product(list(subsComb),list(taskList)))
##SS combination
SSvars=list(itertools.product(list(subList),list(tasksComb)))
#BS combination
BSvars=list(itertools.product(list(subsComb),list(tasksComb)))


""" run_prediction initializes what type of analysis you would like to do
and what classifier you would like to use. For now classifier options are svm:linear svm, logreg: logistic
regression, and ridge:ridge regression. Analysis is the type of analysis you wanted
to run. DS--different subject same task; SS--same subject different task;
BS--different subject different task. Each analysis will concatenate across
subjects and make a dataframe. If CVscores=True calculate accuracy of folds for subject and task.
If FW is true will collect all necessary feature weights and plot or save then
into the appropriate format. If plotACC=True will plot the accuracy as heatmaps.
If statsACC=True will make tables of the mean and sd for each task."""
def run_prediction(classifier, analysis, FW=False, CVscores=False, plotACC=False, statsACC=False):
    if CVscores is True:
        classifyCV(classifier)
    elif analysis=='DS':
        classifyDS(classifier, analysis, FW, plotACC, statsACC)
    elif analysis=='SS':
        classifySS(classifier, analysis, FW, plotACC, statsACC)
    elif analysis=='BS':
        classifyBS(classifier, analysis, FW, plotACC, statsACC)
    else:
        print('Error: You didnt specify what analysis')
def classifyDS(classifier, analysis, FW, plotACC, statsACC):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model(classifier, analysis,FW, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    if plotACC is True:
        results.plotACC(dfDS, classifier, analysis)
    else:
        print('skipping over heatmaps for accuracy')
    if statsACC is True:
        results.statsACC(dfDS, classifier, analysis)
    else:
        print('skipping over making stat tables for accuracy, saving accuracy as csv')
    #save accuracy
    dfDS.to_csv(outDir+'results/'+classifier+'/acc/DS/acc.csv')
def classifySS(classifier,analysis, FW, plotACC, statsACC):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        score=model(classifier, analysis,FW, train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfSS['acc']=acc_scores_per_task
    if plotACC is True:
        results.plotACC(dfSS, classifier, analysis)
    else:
        print('skipping over heatmaps for accuracy')
    if statsACC is True:
        results.statsACC(dfSS, classifier, analysis)
    else:
        print('skipping over making stat tables for accuracy, saving accuracy as csv')
    #save accuracy
    dfSS.to_csv(outDir+'results/'+classifier+'/acc/SS/acc.csv')
def classifyBS(classifier, analysis, FW, plotACC, statsACC):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        score=model(classifier, analysis,FW, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    if plotACC is True:
        results.plotACC(dfBS, classifier, analysis)
    else:
        print('skipping over heatmaps for accuracy')
    if statsACC is True:
        results.statsACC(dfBS, classifier, analysis)
    else:
        print('skipping over making stat tables for accuracy, saving accuracy as csv')
    #save accuracy
    dfBS.to_csv(outDir+'results/'+classifier+'/acc/BS/acc.csv')
def model(classifier, analysis,FW, train_sub, test_sub, train_task, test_task):
    if classifier=='SVC':
        clf=LinearSVC()
    elif classifier=='logReg':
        clf=LogisticRegression(solver = 'lbfgs')
    elif classifier=='ridge':
        clf=RidgeClassifier()
    else:
        print('Error: You didnt specify what classifier')
    taskFC=reshape.matFiles(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
    x_train, y_train=reshape.concateFC(taskFC, restFC)
    #if your subs are the same dont include rest...avoid overfitting!!
    if train_sub==test_sub:
        x_test=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        taskSize=x_test.shape[0]
        y_test=np.ones(taskSize, dtype=int)
    else:
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat')
        x_test, y_test=reshape.concateFC(test_taskFC, test_restFC)
    clf.fit(x_train,y_train)
    folds=taskFC.shape[0]
    CVscores=cross_val_score(clf,x_train,y_train, cv=folds)
    if FW is True:
        coef=clf.coef_
        plotFW.feature_plots(coef, classifier, analysis, train_task, train_sub)
    else:
        pass
    predict=clf.predict(x_test)
    #Get accuracy of model
    ACCscores=clf.score(x_test,y_test)
    return ACCscores
#Calculate acc of cross validation within sub within task
def classifyCV(classifier):
    avg_CV=[]
    if classifier=='SVC':
        clf=LinearSVC()
    elif classifier=='logReg':
        clf=LogisticRegression(solver = 'lbfgs')
    elif classifier=='ridge':
        clf=RidgeClassifier()
    else:
        print('Error: You didnt specify what classifier')
    for task in taskList:
        acc_scores_per_task=[]
        cvTable=[]
        for sub in subList:
            taskFC=reshape.matFiles(dataDir+task+'/'+sub+'_parcel_corrmat.mat')
            restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
            folds=taskFC.shape[0]
            x_train, y_train=reshape.concateFC(taskFC, restFC)
            CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
            mu=CVscores.mean()
            acc_scores_per_task.append(mu)
            cv_tmp_df=pd.DataFrame({sub:CVscores})
            cvTable.append(cv_tmp_df)
    #acc per fold per sub
        tmp_df=pd.DataFrame({'sub':subList, task:acc_scores_per_task}).set_index('sub')
        avg_CV.append(tmp_df)
        cvTable=pd.concat(cvTable, axis=1)
    #saving cv per folds if debugging
        cvTable.to_csv(outDir+'results/'+classifier+'/acc/DS/'+sub+'_cvTable_folds.csv')
    #average acc per sub per tasks
    avg_CVTable=pd.concat(avg_CV, axis=1)
    #plot as heatmaps
    results.plotACC(avg_CVTable, classifier, 'CV')
    results.statsACC(avgTable, classifier, 'CV')
    avg_CVTable.to_csv(outDir+'results/'+classifier+'/acc/DS/cvTable_avg.csv')
