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
from sklearn.svm import LinearSVC
import scipy.io
import random
from sklearn.model_selection import KFold
import os
import sys
import reshape
from statistics import mean
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
#import other python scripts for further anlaysis
# Initialization of directory information:
#thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/acc/'
# Subjects and tasks
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])

taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
SSvars=list(itertools.product(list(subList),list(tasksComb)))
"""
Each function declares the type of analysis you wanted to run. DS--different subject same task; SS--same subject different task; BS--different subject different task.
Each analysis will concatenate across subjects and make a dataframe.
"""
def classifySS():
    """
    Classifying the same subject (SS) along a different task

    Parameters
    -------------


    Returns
    -------------
    dfSS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()
    same_task=[]
    diff_task=[]
    same_rest=[]
    diff_rest=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        taskFC=reshape.matFiles(dataDir+row['train_task']+'/'+row['sub']+'_parcel_corrmat.mat')
        tmp_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/half/'+row['sub']+'_parcel_corrmat.mat')
        restFC=tmp_restFC[:10]
        test_restFC=tmp_restFC[10:]
        testFC=reshape.matFiles(dataDir+row['test_task']+'/'+row['sub']+'_parcel_corrmat.mat')
        ytest=np.ones(testFC.shape[0])
        #same, diff=SS_folds(clf,taskFC,restFC,testFC,ytest)
        same_Tsub, same_Rsub, diff_Tsub, diff_Rsub=folds(clf, taskFC,restFC, testFC,test_restFC)
        same_task.append(same_Tsub)
        diff_task.append(diff_Tsub)
        same_rest.append(same_Rsub)
        diff_rest.append(diff_Rsub)
    dfSS['Same Task']=same_task
    dfSS['Different Task']=diff_task
    dfSS['Same Rest']=same_rest
    dfSS['Different Rest']=diff_rest
    dfSS.to_csv(outDir+'SS/separate_broken_acc.csv',index=False)
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
    #clf=LinearSVC()
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
            #ytask=np.ones(taskFC.shape[0])
            #restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
            tmp_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/half/'+train_sub[0]+'_parcel_corrmat.mat')
            #Split rest into a test and training set 10 test 10 train
            restFC=tmp_restFC[:10]
            test_restFC=tmp_restFC[10:]
            test_taskFC, ytask=AllSubFiles_SS(train_sub,testT)
            #same_Tsub, diff_Tsub=SS_folds(clf, taskFC,restFC, test_taskFC,ytask)
            same_Tsub, same_Rsub, diff_Tsub, diff_Rsub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['train_task']=trainT
            tmp['Same Rest']=same_Rsub
            tmp['Different Rest']=diff_Rsub
            tmp['Same Task']=same_Tsub
            tmp['Different Task']=diff_Tsub
            master_df=pd.concat([master_df,tmp])
    master_df.to_csv(outDir+'SS/sep_acc.csv',index=False)
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
    clf=LinearSVC()
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
    clf=LinearSVC()
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


def SS_folds(clf,taskFC, restFC, test_taskFC, ytest):
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
    CVTacc=[]
    DSTacc=[]
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
        diffT=clf.score(test_taskFC,ytest)
        CVTacc.append(sameT)
        DSTacc.append(diffT)
    same_Tsub=mean(CVTacc)
    diff_Tsub=mean(DSTacc)
    return same_Tsub, diff_Tsub

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

    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC))
    y=np.ones(taskFC.shape[0])
    return taskFC, y

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

    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))
    test_restSize=restFC.shape[0]
    testR= np.zeros(test_restSize, dtype = int)
    memFC=np.concatenate((a_memFC,b_memFC,c_memFC,d_memFC,e_memFC,f_memFC,g_memFC))
    ymem=np.ones(memFC.shape[0])

    semFC=np.concatenate((a_semFC,b_semFC,c_semFC,d_semFC,e_semFC,f_semFC,g_semFC))
    ysem=np.full(semFC.shape[0],2)

    glassFC=np.concatenate((a_glassFC,b_glassFC,c_glassFC,d_glassFC,e_glassFC,f_glassFC,g_glassFC))
    yglass=np.full(glassFC.shape[0],3)

    motFC=np.concatenate((a_motFC,b_motFC,c_motFC,d_motFC,e_motFC,f_motFC,g_motFC))
    ymot=np.full(motFC.shape[0],4)
    #rest, mem, sem, mot, glass

    testFC=np.concatenate((restFC,memFC,semFC,motFC, glassFC))
    ytest=np.concatenate((testR,ymem,ysem,ymot,yglass))
    #taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))

    #testFC=np.concatenate((taskFC,restFC))
    #ytest=np.concatenate((ytask,testR))
    return testFC,ytest
    #return taskFC, restFC,ytask, testR


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
    #each iteration is a subject training
    DS_conf=np.empty((8,2,2))
    #each cm for each sub
    CV_conf=np.empty((8,2,2))
    cm=0
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
        testFC,ytest=AllSubFiles(test_sub)
        same_Tsub, diff_Tsub, CV_cm,DS_cm=K_folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC,ytest)
        plt.figure()
        ax=ConfusionMatrixDisplay(CV_cm,display_labels=['Rest','Task']).plot(cmap=plt.cm.Blues)
        plt.savefig(outDir+'ALL/same/'+train_sub[0]+'.png', bbox_inches='tight')
        plt.figure()
        ax=ConfusionMatrixDisplay(DS_cm,display_labels=['Rest','Task']).plot(cmap=plt.cm.Blues)
        plt.savefig(outDir+'ALL/diff/'+train_sub[0]+'.png', bbox_inches='tight')
        tmp['train']=train_sub
        tmp['same_sub']=same_Tsub
        tmp['diff_sub']=diff_Tsub
        CV_conf[cm]=CV_cm
        DS_conf[cm]=DS_cm
        cm=cm+1
        #tmp['same_subR']=same_Rsub
        #tmp['diff_subR']=diff_Rsub
        master_df=pd.concat([master_df,tmp])
    #return CV_conf, DS_conf
    #master_df.to_csv(outDir+'ALL/separate_acc.csv',index=False)
    CV_conf.tofile('/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/ALL/CV_confusionmatrix.csv', sep = ',')
    DS_conf.tofile('/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/ALL/DS_confusionmatrix.csv', sep = ',')


def K_folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC,ytest):
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
    #for same subject take sum of each fold
    #for different subject take average of each fold
    number=memFC.shape[1]
    loo = LeaveOneOut()
    ymem=np.ones(memFC.shape[0])
    ysem=np.full(semFC.shape[0],1)
    yglass=np.full(glassFC.shape[0],1)
    ymot=np.full(motFC.shape[0],1)
    CVTacc=[]
    CVRacc=[]
    df=pd.DataFrame()
    DSTacc=[]
    DSRacc=[]
    #fold each training set

    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))
    same_sub_CM=np.array([[0,0],[0,0]])
    diff_sub_CM=np.empty((session,2,2))
    diff_cm=0
    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        ymemtrain, ymemval=ymem[train_index], ymem[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        ysemtrain, ysemval=ysem[train_index],ysem[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        ymottrain, ymotval=ymot[train_index],ymot[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        yglatrain,yglaval=yglass[train_index],yglass[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        ytrain_task=np.concatenate((ymemtrain,ysemtrain,ymottrain,yglatrain))
        yval_task=np.concatenate((ymemval,ysemval,ymotval,yglaval))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,number))
        Xval_rest=np.reshape(Xval_rest,(-1,number))

        #ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        #yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_val = np.concatenate((yval_task,yval_rest))
        clf.fit(X_tr,y_tr)
        score=clf.score(X_val, y_val)
        CVTacc.append(score)
        y_predict=clf.predict(X_val)
        CV_cm=confusion_matrix(y_val, y_predict)
        same_sub_CM=same_sub_CM+CV_cm
        #CV_Tscore=clf.score(Xval_task, yval_task)
        #CVTacc.append(CV_Tscore)
        #CV_Rscore=clf.score(Xval_rest, yval_rest)
        #CVRacc.append(CV_Rscore)
        scoreT=clf.score(testFC,ytest)
        DSTacc.append(scoreT)
        y_pre=clf.predict(testFC)
        DS_cm=confusion_matrix(ytest, y_pre)
        diff_sub_CM[diff_cm]=DS_cm
        diff_cm=diff_cm+1
        #scoreR=clf.score(test_restFC, testR)
        #DSRacc.append(scoreR)

    same_Tsub=mean(CVTacc)
    #same_Rsub=mean(CVRacc)
    diff_Tsub=mean(DSTacc)
    #diff_Rsub=mean(DSRacc)
    DS_cm=diff_sub_CM.mean(axis=0,dtype=int)
    return same_Tsub, diff_Tsub, same_sub_CM, DS_cm#diff_Tsub, diff_Rsub





def multiclassAll():
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
    d=pd.DataFrame()
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
        restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
        #nsize=restFC.shape[1]
        #restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        testFC,ytest=AllSubFiles(test_sub)
        same_Tsub, diff_Tsub,sameF,diffF,sameMCC, diffMCC, same_sub_CM, DS_cm=K_folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC, ytest)
        plt.figure()
        ax=ConfusionMatrixDisplay(same_sub_CM,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues)
        plt.savefig(outDir+'ALL/MC/same/'+train_sub[0]+'.png', bbox_inches='tight')
        plt.figure()
        ax=ConfusionMatrixDisplay(DS_cm,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues)
        plt.savefig(outDir+'ALL/MC/diff/'+train_sub[0]+'.png', bbox_inches='tight')
        tmp['train']=train_sub
        tmp['same_sub']=same_Tsub
        tmp['diff_sub']=diff_Tsub
        tmp['MCC_CV']=sameMCC
        tmp['MCC_DS']=diffMCC
        tmp[['rest_CV', 'mem_CV', 'sem_CV', 'mot_CV', 'glass_CV']]=sameF.reshape(-1,len(sameF))
        tmp[['rest_DS', 'mem_DS', 'sem_DS', 'mot_DS', 'glass_DS']]=diffF.reshape(-1,len(diffF))
        master_df=pd.concat([master_df,tmp])
    #return same_sub_CM, DS_cm
    #master_df.to_csv(outDir+'ALL/multiclass_acc.csv',index=False)

def K_folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC,ytest):
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

    loo = LeaveOneOut()
    number=memFC.shape[1]
    #test_taskSize=test_taskFC.shape[0]
    #test_restSize=test_restFC.shape[0]
    #testT= np.ones(test_taskSize, dtype = int)
    #testR= np.zeros(test_restSize, dtype = int)
    yrest=np.zeros(restFC.shape[0])
    ymem=np.ones(memFC.shape[0])
    ysem=np.full(semFC.shape[0],2)
    yglass=np.full(glassFC.shape[0],3)
    ymot=np.full(motFC.shape[0],4)
    CVTacc=[]

    df=pd.DataFrame()
    DSTacc=[]


    count=0
    #fold each training set
    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))
    sameF1=np.empty((session,5))
    diffF1=np.empty((session,5))
    same_sub_CM=np.zeros((5,5))
    diff_sub_CM=np.empty((session,5,5))
    diff_count=0
    CV_MCC=[]
    DS_MCC=[]

    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        ymemtrain, ymemval=ymem[train_index], ymem[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        ysemtrain, ysemval=ysem[train_index],ysem[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        ymottrain, ymotval=ymot[train_index],ymot[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        yglatrain,yglaval=yglass[train_index],yglass[test_index]
        resttrain, restval=restFC[train_index], restFC[test_index]
        yresttrain, yrestval=yrest[train_index],yrest[test_index]
        Xtrain=np.concatenate((resttrain,memtrain,semtrain,mottrain,glatrain))
        ytrain=np.concatenate((yresttrain,ymemtrain,ysemtrain,ymottrain,yglatrain))
        yval=np.concatenate((yrestval,ymemval,ysemval,ymotval,yglaval))
        Xval=np.concatenate((restval,memval,semval,motval,glaval))
        clf.fit(Xtrain,ytrain)
        score=clf.score(Xval, yval)
        CVTacc.append(score)
        y_predict=clf.predict(Xval)
        score_same = f1_score(yval, y_predict, average=None)
        cm_same = confusion_matrix(yval, y_predict)
        same_sub_CM=same_sub_CM+cm_same
        MCC_same = matthews_corrcoef(yval, y_predict)
        CV_MCC.append(MCC_same)
        #restFC,memFC,semFC,motFC, glassFC
        #order is position rest, mem, sem, mot, glass
        sameF1[count]=score_same
        scoreT=clf.score(testFC,ytest)
        DSTacc.append(scoreT)
        y_pre=clf.predict(testFC)
        score_diff = f1_score(ytest, y_pre, average=None)
        MCC_diff = matthews_corrcoef(ytest, y_pre)
        cm_diff = confusion_matrix(ytest, y_pre)
        diff_sub_CM[diff_count]=cm_diff
        diff_count=diff_count+1
        DS_MCC.append(MCC_diff)
        #order is position rest, mem, sem, mot, glass
        diffF1[count]=score_diff
        count=count+1
    same_f=sameF1.mean(axis=0)
    same_Tsub=mean(CVTacc)
    sameMCC=mean(CV_MCC)
    diff_Tsub=mean(DSTacc)
    diff_f=diffF1.mean(axis=0)
    diffMCC=mean(DS_MCC)
    DS_cm=diff_sub_CM.mean(axis=0,dtype=int)
    return same_Tsub,diff_Tsub,same_f, diff_f, sameMCC, diffMCC, same_sub_CM, DS_cm


def SS_inverse():
    taskData=np.array(['semantic','glass', 'motor','mem'], dtype='<U61')
    clf=RidgeClassifier()
    #clf=LinearSVC()
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for testTask, trainTask in loo.split(taskData):
        trainT=taskData[testTask]
        testT=taskData[trainTask]
        for test, train in loo.split(data): #train on one sub test on the res
            tmp=pd.DataFrame()
            train_sub=data[train]
        #train sub
            testFC=reshape.matFiles(dataDir+testT[0]+'/'+train_sub[0]+'_parcel_corrmat.mat')
            ytask=np.ones(testFC.shape[0])
            t1FC=reshape.matFiles(dataDir+trainT[0]+'/'+train_sub[0]+'_parcel_corrmat.mat')
            t2FC=reshape.matFiles(dataDir+trainT[1]+'/'+train_sub[0]+'_parcel_corrmat.mat')
            t3FC=reshape.matFiles(dataDir+trainT[2]+'/'+train_sub[0]+'_parcel_corrmat.mat')
            restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
            same_Tsub, diff_Tsub=SS_folds_inverse(train_sub,clf, t1FC,t2FC,t3FC,restFC, testFC,ytask)
            tmp['train']=train_sub
            tmp['test_task']=testT
            tmp['same_subT']=same_Tsub
            #tmp['same_subR']=same_Rsub
            tmp['diffT']=diff_Tsub
            #tmp['diff_subR']=diff_Rsub
            master_df=pd.concat([master_df,tmp])
    return master_df
    #master_df.to_csv(outDir+'SS/allTrain_acc.csv',index=False)


def SS_folds_inverse(train_sub, clf, memFC,semFC,glassFC, restFC, testFC,ytest):
    loo = LeaveOneOut()
    ymem=np.ones(memFC.shape[0])
    ysem=np.ones(semFC.shape[0])
    yglass=np.ones(glassFC.shape[0])
    #ymot=np.ones(motFC.shape[0],4)
    CVTacc=[]
    df=pd.DataFrame()
    DSTacc=[]
    #fold each training set
    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))
    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        ymemtrain, ymemval=ymem[train_index], ymem[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        ysemtrain, ysemval=ysem[train_index],ysem[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        yglatrain,yglaval=yglass[train_index],yglass[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,glatrain))
        ytrain_task=np.concatenate((ymemtrain,ysemtrain,yglatrain))
        yval_task=np.concatenate((ymemval,ysemval,yglaval))
        Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
        Xval_task=np.concatenate((memval,semval,glaval))
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_val = np.concatenate((yval_task,yval_rest))
        clf.fit(X_tr,y_tr)
        score=clf.score(X_val, y_val)
        CVTacc.append(score)
        t=testFC[test_index]
        y=ytest[test_index]
        scoreT=clf.score(t,y)
        DSTacc.append(scoreT)
    same_Tsub=mean(CVTacc)
    diff_Tsub=mean(DSTacc)
    return same_Tsub,diff_Tsub
