#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import reshape
import os
import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/permutation/'
subsComb=(list(itertools.permutations(subList, 2)))
taskList=['glass','semantic', 'motor','mem']

def classifyAll():
    """
    Classifying different subjects along available data rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    #comparison of days analysis
    allDay=pd.DataFrame()
    days=39 #zero based indexing
    while days>4:
        idx=np.random.randint(39, size=(days))
        acc_scores_per_sub=[]
        acc_scores_cv=[]
        tmpdf=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
        for index, row in tmpdf.iterrows():
            diff_score, same_score=model(idx, train_sub=row['train_sub'], test_sub=row['test_sub'])
            acc_scores_per_sub.append(diff_score)
            acc_scores_cv.append(same_score)
        tmpdf['Within']=acc_scores_cv
        tmpdf['Between']=acc_scores_per_sub
        tmpdf['Days']=days
        allDay=pd.concat([allDay, tmpdf])
        days=days-1
    #allDay.to_csv(outDir+'results/ridge/acc/ALL/days/acc.csv',index=False)
    return allDay
def model(idx, train_sub, test_sub):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """

    clf=RidgeClassifier()
    #train sub
    memFC=reshape.matFiles(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.matFiles(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.matFiles(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.matFiles(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat')
    taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #test sub
    test_memFC=reshape.matFiles(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.matFiles(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    diff_score, same_score=CV_folds(idx,train_sub, clf, taskFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score

def CV_folds(idx, train_sub, clf, taskFC, restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
    taskFC, restFC, test_taskFC, test_restFC : array_like
        Input arrays, training and testing set of task and rest FC
    Returns
    -----------
    total_score : float
        Average accuracy across folds
    acc_score : list
        List of accuracy for each outer fold
    """
    kf = KFold(n_splits=5)
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    task_X=taskFC[idx][:]
    rest_X=restFC[idx][:]
    task_y=t[idx]
    rest_y=r[idx]
    X=np.concatenate((task_X,rest_X))
    Y=np.concatenate((task_y,rest_y))
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    CVacc=[]
    CVdf=pd.DataFrame()
    df=pd.DataFrame()
    acc_score=[]
    #fold each training set
    for train_index, test_index in kf.split(X):
        X_tr,X_val=X[train_index],X[test_index]
        y_tr, y_val=Y[train_index],Y[test_index]
        clf.fit(X_tr,y_tr)
        #cross validation
        clf.predict(X_val)
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        #fold each testing set
        for t_index, te_index in kf.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            ytest_task=testT[te_index]
            ytest_rest=testR[te_index]
            y_te=np.concatenate((ytest_task, ytest_rest))
            #test set
            clf.predict(X_te)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
    CVdf['acc']=CVacc
    df['cv']=CVacc
    df['outer_fold']=acc_score
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    return diff_sub_score, same_sub_score


def runDays():
    iterDays=pd.DataFrame()
    for i in range(125):
        df=classifyAll()
        iterDays=pd.concat([iterDays,df])
    iterDays.to_csv(outDir+'manDays.csv',index=False)
