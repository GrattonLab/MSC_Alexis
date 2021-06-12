#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from statistics import mean
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import itertools
#import other python scripts for further anlaysis
import reshape
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/subNetwork/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
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
#CV combinations
CVvars=list(itertools.product(list(subList),list(taskList)))

#all combination of network to network
networks=['unassign','default','visual','fp','dan','van','salience','co','sm','sm-lat','auditory','pmn','pon']
#no repeats
netComb=(list(itertools.combinations(networks, 2)))
def Net2Net():
#only take within network for model
    finalDF=pd.DataFrame()
    netDF=pd.DataFrame(netComb, columns=['Network_A','Network_B'])
    for i in networks:
        tmp_df=classifyAll(network=i,subnetwork=i)
        tmp_df['Network_A']=i
        tmp_df['Network_B']=i
        finalDF=pd.concat([finalDF, tmp_df])
    for index, row in netDF.iterrows():
        tmp_df=classifyAll(network=row['Network_A'], subnetwork=row['Network_B'])
        tmp_df['Network_A']=row['Network_A']
        tmp_df['Network_B']=row['Network_B']
        finalDF=pd.concat([finalDF, tmp_df])
    return finalDF
    #finalDF.to_csv(thisDir+'output/results/acc/ALL/Net2Net_acc.csv')

def classifyAll(network,subnetwork=None):
    """
    Classifying different subjects along available data rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier(max_iter=10000)
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.network_to_network(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
        semFC=reshape.network_to_network(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
        glassFC=reshape.network_to_network(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
        motFC=reshape.network_to_network(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
        restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork) #keep tasks seperated in order to collect the right amount of days
        nsize=restFC.shape[1]
        restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        test_taskFC,test_restFC=AllSubFiles(test_sub,network, subnetwork)
        diff_score, same_score=K_folds(nsize,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
        tmp['train']=train_sub
        tmp['same_sub']=same_score
        tmp['diff_sub']=diff_score
        master_df=pd.concat([master_df,tmp])
    return master_df


def K_folds(netSize, train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC):
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
    #kf = KFold(n_splits=5,shuffle=True)
    """
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    """
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    ytest=np.concatenate((testT,testR))
    Xtest=np.concatenate((test_taskFC,test_restFC))
    CVacc=[]
    DSacc=[]
    #fold each training set
    if train_sub=='MSC03':
        split=np.empty((8,netSize))
        #xtrainSize=24
        #xtestSize=4
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,netSize))
    else:
        split=np.empty((10,netSize))
    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,netSize))
        Xval_rest=np.reshape(Xval_rest,(-1,netSize))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        scaler.transform(X_val)
        y_tr=np.random.permutation(y_tr)
        clf.fit(X_tr,y_tr)
        #cross validation
        y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        #tmpdf=pd.DataFrame()
        #acc_scores_per_fold=[]
        #fold each testing set
        scaler.transform(Xtest)
        DS_score=clf.score(Xtest,ytest)
        DSacc.append(DS_score)
    diff_sub_score=mean(DSacc)
    same_sub_score=mean(CVacc)
    return diff_sub_score, same_sub_score


def DSNet2Net():
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------


    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    netDF=pd.DataFrame(netComb, columns=['Network_A','Network_B'])
    finalDF=pd.DataFrame()
    for i in networks:
        tmp_df=classifyDS_Net2Net(network=i,subnetwork=i)
        tmp_df['Network_A']=i
        tmp_df['Network_B']=i
        finalDF=pd.concat([finalDF, tmp_df])
    for index, row in netDF.iterrows():
        tmp_df=classifyDS_Net2Net(network=row['Network_A'], subnetwork=row['Network_B'])
        tmp_df['Network_A']=row['Network_A']
        tmp_df['Network_B']=row['Network_B']
        finalDF=pd.concat([finalDF, tmp_df])
        #print(i)
    return finalDF
    #finalDF.to_csv(thisDir+'output/results/acc/DS/Net2Net.csv')

def classifyDS_Net2Net(network,subnetwork=None):
    clf=RidgeClassifier()
    DS=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for t in taskList:
        for  test, train in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
        #train sub
            taskFC=reshape.network_to_network(dataDir+t+'/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
            restFC=reshape.network_to_network(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork) #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_DS(test_sub,t,network, subnetwork)
            same_sub, diff_sub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=t
            tmp['same_sub']=same_sub
            tmp['diff_sub']=diff_sub
            DS=pd.concat([DS,tmp])
    return DS


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
    X_test=np.concatenate((test_taskFC, test_restFC))
    y_test = np.concatenate((ttest,rtest))
    df=pd.DataFrame()
    CVacc=[]
    DSacc=[]

    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index],taskFC[test_index]
        ytrain_rest,yval_rest=r[train_index],r[test_index]
        ytrain_task,yval_task=t[train_index],t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        Xval=np.concatenate((Xval_task,Xval_rest))
        yval=np.concatenate((yval_task,yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        y_tr=np.random.permutation(y_tr)
        clf.fit(X_tr,y_tr)
        scaler.transform(Xval)
        scaler.transform(X_test)
        same=clf.score(Xval,yval)
        diff=clf.score(X_test,y_test)

        CVacc.append(same)
        DSacc.append(diff)
    same_sub=mean(CVacc)
    diff_sub=mean(DSacc)

    return same_sub, diff_sub


def AllSubFiles_DS(test_sub,task,network, subnetwork):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
    a_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)

    b_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)
    b_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)

    c_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)
    c_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)

    d_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)
    d_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)

    e_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)
    e_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)

    f_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)
    f_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)

    g_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)
    g_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)


    taskFC=np.concatenate((a_taskFC,b_taskFC,c_taskFC,d_taskFC,e_taskFC,f_taskFC,g_taskFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC
def AllSubFiles(test_sub,network, subnetwork):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
    a_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
    a_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
    a_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
    a_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)

    b_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)
    b_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)
    b_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)
    b_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)
    b_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)

    c_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)
    c_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)
    c_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)
    c_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)
    c_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)

    d_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)
    d_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)
    d_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)
    d_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)
    d_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)

    e_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)
    e_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)
    e_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)
    e_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)
    e_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)

    f_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)
    f_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)
    f_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)
    f_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)
    f_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)

    g_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)
    g_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)
    g_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)
    g_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)
    g_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC
def permAll():
    final=pd.DataFrame()
    for i in range(1000):
        tmp=Net2Net()
        final=pd.concat([final,tmp])
        print(i)
    final.to_csv(thisDir+'output/results/permutation/ALL/Net2Net_acc.csv')

def permDS():
    final=pd.DataFrame()
    for i in range(1000):
        tmp=DSNet2Net()
        final=pd.concat([final,tmp])
        print(i)
    final.to_csv(thisDir+'output/results/permutation/DS/Net2Net_acc.csv')
