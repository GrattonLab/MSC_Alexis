#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import itertools
import scipy.io
import random
from sklearn.model_selection import KFold
import os
import sys
import reshape
from statistics import mean
#import other python scripts for further anlaysis
# Initialization of directory information:
#thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/acc/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
"""
Each function declares the type of analysis you wanted to run. DS--different subject same task; SS--same subject different task; BS--different subject different task.
Each analysis will concatenate across subjects and make a dataframe.
"""

def SSmodel():
    """
    Different sub different task

    Parameters
    -------------
    analysis : string
            The type of analysis to be conducted
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    taskData=np.array(['semantic','glass', 'motor','mem'], dtype='<U61')
    clf=RidgeClassifier()
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for testTask, trainTask in loo.split(taskData):
        testT=taskData[testTask]
        trainT=taskData[trainTask]
        for test, train in loo.split(data): #train on one sub test on the res
            tmp=pd.DataFrame()
            train_sub=data[train]
        #train sub
            taskFC=reshape.matFiles(dataDir+trainT[0]+'/'+train_sub[0]+'_parcel_corrmat.mat')
            restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_SS(train_sub,testT)
            same_Tsub, same_Rsub, diff_Tsub, diff_Rsub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=trainT
            tmp['same_subT']=same_Tsub
            tmp['same_subR']=same_Rsub
            tmp['diffT']=diff_Tsub
            #tmp['diff_subR']=diff_Rsub
            master_df=pd.concat([master_df,tmp])
    master_df.to_csv(outDir+'SS/separate_acc.csv',index=False)
def BSmodel():
    """
    Different sub different task

    Parameters
    -------------
    analysis : string
            The type of analysis to be conducted
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    taskData=np.array(['semantic','glass', 'motor','mem'], dtype='<U61')
    clf=RidgeClassifier()
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for testTask, trainTask in loo.split(taskData):
        testT=taskData[testTask]
        trainT=taskData[trainTask]
        for test, train in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
        #train sub
            taskFC=reshape.matFiles(dataDir+trainT[0]+'/'+train_sub[0]+'_parcel_corrmat.mat')
            restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_BS(test_sub,testT)
            same_Tsub, same_Rsub, diff_Tsub, diff_Rsub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=trainT
            tmp['same_subT']=same_Tsub
            tmp['same_subR']=same_Rsub
            tmp['diff_subT']=diff_Tsub
            tmp['diff_subR']=diff_Rsub

            master_df=pd.concat([master_df,tmp])
    master_df.to_csv(outDir+'BS/separate_acc.csv',index=False)

def DSmodel():
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    analysis : string
            The type of analysis to be conducted
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    clf=RidgeClassifier()
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for t in taskList:
        for  test, train in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
        #train sub
            taskFC=reshape.matFiles(dataDir+t+'/'+train_sub[0]+'_parcel_corrmat.mat')
            restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_DS(test_sub,t)
            same_Tsub, same_Rsub, diff_Tsub, diff_Rsub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=t
            tmp['same_subT']=same_Tsub
            tmp['same_subR']=same_Rsub
            tmp['diff_subT']=diff_Tsub
            tmp['diff_subR']=diff_Rsub
            master_df=pd.concat([master_df,tmp])
    master_df.to_csv(outDir+'DS/separate_acc.csv',index=False)
def folds(clf,taskFC, restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
    analysis : str
        Analysis type
    taskFC, restFC, test_taskFC, test_restFC : array_like
        Input arrays, training and testing set of task and rest FC
    Returns
    -----------
    total_score : float
        Average accuracy across folds
    acc_score : list
        List of accuracy for each outer fold
    """

    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    ttest = np.ones(test_taskSize, dtype = int)
    rtest=np.zeros(test_restSize, dtype=int)
    CVTacc=[]
    CVRacc=[]
    DSTacc=[]
    DSRacc=[]
    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index],taskFC[test_index]
        ytrain_rest,yval_rest=r[train_index],r[test_index]
        ytrain_task,yval_task=t[train_index],t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        clf.fit(X_tr,y_tr)
        sameT=clf.score(Xval_task,yval_task)
        sameR=clf.score(Xval_rest,yval_rest)
        diffT=clf.score(test_taskFC,ttest)
        diffR=clf.score(test_restFC,rtest)
        CVTacc.append(sameT)
        CVRacc.append(sameR)
        DSTacc.append(diffT)
        DSRacc.append(diffR)
    same_Tsub=mean(CVTacc)
    same_Rsub=mean(CVRacc)
    diff_Tsub=mean(DSTacc)
    diff_Rsub=mean(DSRacc)
    return same_Tsub, same_Rsub, diff_Tsub, diff_Rsub

def AllSubFiles_SS(train_sub,testT):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.matFiles(dataDir+testT[0]+'/'+train_sub[0]+'_parcel_corrmat.mat')
    a_semFC=reshape.matFiles(dataDir+testT[1]+'/'+train_sub[0]+'_parcel_corrmat.mat')
    a_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+train_sub[0]+'_parcel_corrmat.mat')
    restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat')
    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC))
    return taskFC, restFC

def AllSubFiles_BS(test_sub,testT):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.matFiles(dataDir+testT[0]+'/'+test_sub[0]+'_parcel_corrmat.mat')
    a_semFC=reshape.matFiles(dataDir+testT[1]+'/'+test_sub[0]+'_parcel_corrmat.mat')
    a_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')

    b_memFC=reshape.matFiles(dataDir+testT[0]+'/'+test_sub[1]+'_parcel_corrmat.mat')
    b_semFC=reshape.matFiles(dataDir+testT[1]+'/'+test_sub[1]+'_parcel_corrmat.mat')
    b_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat')

    c_memFC=reshape.matFiles(dataDir+testT[0]+'/'+test_sub[2]+'_parcel_corrmat.mat')
    c_semFC=reshape.matFiles(dataDir+testT[1]+'/'+test_sub[2]+'_parcel_corrmat.mat')
    c_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat')

    d_memFC=reshape.matFiles(dataDir+testT[0]+'/'+test_sub[3]+'_parcel_corrmat.mat')
    d_semFC=reshape.matFiles(dataDir+testT[1]+'/'+test_sub[3]+'_parcel_corrmat.mat')
    d_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat')

    e_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat')
    e_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat')
    e_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat')
    e_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat')

    f_memFC=reshape.matFiles(dataDir+testT[0]+'/'+test_sub[5]+'_parcel_corrmat.mat')
    f_semFC=reshape.matFiles(dataDir+testT[1]+'/'+test_sub[5]+'_parcel_corrmat.mat')
    f_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat')

    g_memFC=reshape.matFiles(dataDir+testT[0]+'/'+test_sub[6]+'_parcel_corrmat.mat')
    g_semFC=reshape.matFiles(dataDir+testT[1]+'/'+test_sub[6]+'_parcel_corrmat.mat')
    g_glassFC=reshape.matFiles(dataDir+testT[2]+'/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat')


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,b_memFC,b_semFC,b_glassFC,c_memFC,c_semFC,c_glassFC,d_memFC,d_semFC,d_glassFC,e_memFC,e_semFC,e_glassFC,f_memFC,f_semFC,f_glassFC,g_memFC,g_semFC,g_glassFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


def AllSubFiles_DS(test_sub,task):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')

    b_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat')

    c_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat')

    d_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat')

    e_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat')

    f_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat')

    g_taskFC=reshape.matFiles(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat')


    taskFC=np.concatenate((a_taskFC,b_taskFC,c_taskFC,d_taskFC,e_taskFC,f_taskFC,g_taskFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC

def AllSubFiles(test_sub):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat')
    a_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat')
    a_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat')
    a_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')

    b_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat')
    b_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat')
    b_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat')
    b_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat')

    c_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat')
    c_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat')
    c_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat')
    c_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat')

    d_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat')
    d_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat')
    d_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat')
    d_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat')

    e_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat')
    e_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat')
    e_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat')
    e_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat')

    f_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat')
    f_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat')
    f_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat')
    f_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat')

    g_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat')
    g_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat')
    g_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat')
    g_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat')


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


def modelAll():
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
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    clf=RidgeClassifier()
    #train sub
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.matFiles(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat')
        semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat')
        glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat')
        motFC=reshape.matFiles(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
        nsize=restFC.shape[1]
        restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        test_taskFC,test_restFC=AllSubFiles(test_sub)
        same_Tsub, same_Rsub, diff_Tsub, diff_Rsub=K_folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
        tmp['train']=train_sub
        tmp['same_subT']=same_Tsub
        tmp['diff_subT']=diff_Tsub
        tmp['same_subR']=same_Rsub
        tmp['diff_subR']=diff_Rsub
        master_df=pd.concat([master_df,tmp])
    master_df.to_csv(outDir+'ALL/separate_acc.csv',index=False)

def K_folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC):
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

    #kf = KFold(n_splits=5,shuffle=True)
    loo = LeaveOneOut()
    number=memFC.shape[1]
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    CVTacc=[]
    CVRacc=[]
    df=pd.DataFrame()
    DSTacc=[]
    DSRacc=[]
    #fold each training set

    if train_sub=='MSC03':
        split=np.empty((8,number))
        #xtrainSize=24
        #xtestSize=4
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,number))
    else:
        split=np.empty((10,number)

    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,number))
        Xval_rest=np.reshape(Xval_rest,(-1,number))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        clf.fit(X_tr,y_tr)
        CV_Tscore=clf.score(Xval_task, yval_task)
        CVTacc.append(CV_Tscore)
        CV_Rscore=clf.score(Xval_rest, yval_rest)
        CVRacc.append(CV_Rscore)
        scoreT=clf.score(test_taskFC, testT)
        DSTacc.append(scoreT)
        scoreR=clf.score(test_restFC, testR)
        DSRacc.append(scoreR)

    same_Tsub=mean(CVTacc)
    same_Rsub=mean(CVRacc)
    diff_Tsub=mean(DSTacc)
    diff_Rsub=mean(DSRacc)
    return same_Tsub, same_Rsub, diff_Tsub, diff_Rsub
