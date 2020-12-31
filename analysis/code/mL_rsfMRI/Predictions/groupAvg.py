#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import itertools
#import other python scripts for further anlaysis
import reshape
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/groupAvg/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
#Only using subs with full 10 sessions
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))

def groupApp():
    """
    Classifying using all subs testing unseen sub on same task and diff task

    Parameters
    -------------


    Returns
    -------------
    dfGroup : DataFrame
        Dataframe consisting of group average accuracy training with subs instead of individualized classifier

    """
    cv_acc_per_task=[]
    cv_sen_per_task=[]
    cv_spec_per_task=[]
    SS_acc_per_task=[]
    DS_acc_per_task=[]
    dfGroup=pd.DataFrame(tasksComb, columns=['train_task','test_task'])
    for index, row in dfGroup.iterrows():
        cv_score, cv_sensitivity, cv_specificity, SS_score, DS_score=model(train_task=row['train_task'], test_task=row['test_task'])
        cv_acc_per_task.append(cv_score)
        cv_spec_per_task.append(cv_specificity)
        cv_sen_per_task.append(cv_sensitivity)
        SS_acc_per_task.append(SS_score)
        DS_acc_per_task.append(DS_score)
    dfGroup['CV_acc']=cv_acc_per_task
    dfGroup['SS_acc']=SS_acc_per_task
    dfGroup['DS_acc']=DS_acc_per_task
    dfGroup['CV_sen']=cv_sen_per_task
    dfGroup['CV_spec']=cv_spec_per_task
    #save accuracy
    dfGroup.to_csv(outDir+'allSess_acc.csv',index=False)

def model(train_task, test_task):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    CV_score : float
            Average accuracy of all folds leave one sub out of a given task
    CV_sens : float
            Sensitivity score for model within task
    CV_spec : float
            Specificity score for model within task
    SS_score : float
            Average accuracy of all folds same subject but tested on a new task
    DS_score : float
            Average accuracy of all folds of a left out subject but tested on a new task

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    df=pd.DataFrame()
    #nsess x fc x nsub
    ds_T=np.empty((10,55278,7))
    ds_R=np.empty((10,55278,7))
    ds_Test=np.empty((10,55278,7))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_taskFC=reshape.matFiles(dataDir+train_task+'/'+sub+'_parcel_corrmat.mat')
        tmp_restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        #reshape 2d into 3d nsessxfcxnsubs
        ds_T[:,:,count]=tmp_taskFC
        ds_R[:,:,count]=tmp_restFC
        #testing task
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+sub+'_parcel_corrmat.mat')
        ds_Test[:,:,count]=test_taskFC
        count=count+1
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    #split up by subs not sess
    sub_splits=np.empty((7,55278))
    df=pd.DataFrame()
    #left out sub
    cv_scoreList=[]
    sen_CV=[]
    spec_CV=[]
    #new task same subjects
    wtn_scoreList=[]
    #new task new subject
    btn_scoreList=[]
    for train_index, test_index in loo.split(sub_splits):
        #train on all sessions 1-6 subs
        #test on all sessions of one sub
        Xtrain_task, Xtest_task=ds_T[:,:,train_index], ds_T[:,:,test_index[0]]
        Xtrain_rest, Xtest_rest=ds_R[:,:,train_index], ds_R[:,:,test_index[0]]
        #test on a new task of 1-6 subs
        #test on a new task of left out sub
        SS_newTask, DS_newTask=ds_Test[:,:,train_index], ds_Test[:,:,test_index[0]]
        #reshape data into a useable format for mL
        Xtrain_task=Xtrain_task.reshape(60,55278)
        Xtrain_rest=Xtrain_rest.reshape(60,55278)
        SS_newTask=SS_newTask.reshape(60,55278)
        #training set
        taskSize=Xtrain_task.shape[0]
        restSize=Xtrain_rest.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)
        x_train=np.concatenate((Xtrain_task,Xtrain_rest))
        y_train=np.concatenate((t,r))
        #testing set (left out sub CV)
        testSize=Xtest_task.shape[0]
        test_restSize=Xtest_rest.shape[0]
        test_t = np.ones(testSize, dtype = int)
        test_r=np.zeros(test_restSize, dtype=int)
        x_test=np.concatenate((Xtest_task, Xtest_rest))
        y_test=np.concatenate((test_t,test_r))
        #testing set of new task using same subs SS
        SS_taskSize=SS_newTask.shape[0]
        SS_y = np.ones(SS_taskSize, dtype = int)
        #testing set of new task using a diff sub DS
        DS_taskSize=DS_newTask.shape[0]
        DS_y = np.ones(DS_taskSize, dtype = int)
        #test left out sub 10 sessions
        clf.fit(x_train,y_train)
        y_pre=clf.predict(x_test)
        tn, fp, fn, tp=confusion_matrix(y_test, y_pre).ravel()
        specificity= tn/(tn+fp)
        sensitivity= tp/(tp+fn)
        sen_CV.append(sensitivity)
        spec_CV.append(specificity)
        ACCscores=clf.score(x_test,y_test)
        cv_scoreList.append(ACCscores)
        #test new task of same subjects
        clf.predict(SS_newTask)
        wtn_ACCscores=clf.score(SS_newTask, SS_y)
        wtn_scoreList.append(wtn_ACCscores)
        #test new task of new sub
        clf.predict(DS_newTask)
        btn_ACCscores=clf.score(DS_newTask, DS_y)
        btn_scoreList.append(btn_ACCscores)
    df['cv_fold']=cv_scoreList
    df['cv_sens']=sen_CV
    df['cv_spec']=spec_CV
    cv_score=df['cv_fold'].mean()
    cv_sensitivity=df['cv_sens'].mean()
    cv_specificity=df['cv_spec'].mean()
    df['SS_fold']=wtn_scoreList
    SS_score=df['SS_fold'].mean()
    df['DS_fold']=btn_scoreList
    DS_score=df['DS_fold'].mean()
    return cv_score, cv_sensitivity, cv_specificity, SS_score, DS_score