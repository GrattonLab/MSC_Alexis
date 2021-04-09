#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
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

netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 494),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])

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

def classifyDS(network):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model('DS', network, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    return dfDS
    #dfDS.to_csv(outDir+network+'/DS/acc.csv')
def classifySS(network):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        score=model('SS', network, train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfSS['acc']=acc_scores_per_task
    #save accuracy
    return dfSS
    #dfSS.to_csv(outDir+network+'/SS/acc.csv')
def classifyBS(network):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        score=model('BS', network, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    return dfBS
    #dfBS.to_csv(outDir+network+'/BS/acc.csv')

def model(analysis, network, train_sub, test_sub, train_task, test_task):
    clf=RidgeClassifier(max_iter=10000)
    taskFC=reshape.subNets(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat', network)
    #if your subs are the same split rest
    if train_sub==test_sub:
        tmp_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat', network)
        #Split rest into a test and training set 10 test 10 train
        restFC=tmp_restFC[:10]
        test_restFC=tmp_restFC[10:]
        test_taskFC=reshape.subNets(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',network)
        ACCscores=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC=reshape.subNets(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat', network)
        test_taskFC=reshape.subNets(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',network)
        test_restFC=reshape.subNets(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat',network)
        ACCscores=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    return ACCscores
#Calculate acc of cross validation within sub within task
def classifyCV(network):
    dfCV=pd.DataFrame(CVvars, columns=['sub','task'])
    clf=RidgeClassifier()
    acc_scores_per_task=[]
    for index, row in dfCV.iterrows():
        taskFC=reshape.subNets(dataDir+row['task']+'/'+row['sub']+'_parcel_corrmat.mat',network)
        restFC=reshape.subNets(dataDir+'rest/'+row['sub']+'_parcel_corrmat.mat',network)
        folds=taskFC.shape[0]
        x_train, y_train=reshape.concateFC(taskFC, restFC)
        CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
        mu=CVscores.mean()
        acc_scores_per_task.append(mu)
    #average acc per sub per tasks
    dfCV['acc']=acc_scores_per_task
    return dfCV
    #dfCV.to_csv(outDir+network+'/CV/acc.csv')
def CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC):
    loo = LeaveOneOut()
    t = np.ones(taskFC.shape[0], dtype = int)
    r=np.zeros(restFC.shape[0], dtype=int)
    testT= np.ones(test_taskFC.shape[0], dtype = int)
    testR= np.zeros(test_restFC.shape[0], dtype = int)
    X_te=np.concatenate((test_taskFC, test_restFC))
    y_te=np.concatenate((testT, testR))
    df=pd.DataFrame()
    acc_score=[]
    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest=restFC[train_index]
        Xtrain_task=taskFC[train_index]
        ytrain_rest=r[train_index]
        ytrain_task=t[train_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        #implement standardization
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        clf.fit(X_tr,y_tr)
        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        scaler.transform(X_te)
        score=clf.score(X_te,y_te)
        acc_score.append(score)
    df['outer_fold']=acc_score
    total_score=df['outer_fold'].mean()
    return total_score


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
    allDF=pd.DataFrame()
    for network in netRoi:
        tmp=modelAll(network)
        allDF=pd.concat([allDF,tmp])
    allDF.to_csv(outDir+'ALL/acc.csv',index=False)
def AllSubFiles(test_sub,network):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.subNets(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat',network)
    a_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat',network)
    a_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat',network)
    a_motFC=reshape.subNets(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat',network)
    a_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[0]+'_parcel_corrmat.mat',network)

    b_memFC=reshape.subNets(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat',network)
    b_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat',network)
    b_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat',network)
    b_motFC=reshape.subNets(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat',network)
    b_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[1]+'_parcel_corrmat.mat',network)

    c_memFC=reshape.subNets(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat',network)
    c_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat',network)
    c_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat',network)
    c_motFC=reshape.subNets(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat',network)
    c_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[2]+'_parcel_corrmat.mat',network)

    d_memFC=reshape.subNets(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat',network)
    d_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat',network)
    d_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat',network)
    d_motFC=reshape.subNets(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat',network)
    d_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[3]+'_parcel_corrmat.mat',network)

    e_memFC=reshape.subNets(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat',network)
    e_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat',network)
    e_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat',network)
    e_motFC=reshape.subNets(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat',network)
    e_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat',network)

    f_memFC=reshape.subNets(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat',network)
    f_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat',network)
    f_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat',network)
    f_motFC=reshape.subNets(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat',network)
    f_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[5]+'_parcel_corrmat.mat',network)

    g_memFC=reshape.subNets(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat',network)
    g_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat',network)
    g_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat',network)
    g_motFC=reshape.subNets(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat',network)
    g_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[6]+'_parcel_corrmat.mat',network)


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC
def modelAll(network):
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
    clf=RidgeClassifier(max_iter=10000)
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.subNets(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat',network)
        semFC=reshape.subNets(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat',network)
        glassFC=reshape.subNets(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat',network)
        motFC=reshape.subNets(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat',network)
        restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat',network) #keep tasks seperated in order to collect the right amount of days
        nsize=restFC.shape[1]
        restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        test_taskFC,test_restFC=AllSubFiles(test_sub,network)

        diff_score, same_score=K_folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
        tmp['train']=train_sub
        tmp['same_sub']=same_score
        tmp['diff_sub']=diff_score
        tmp['network']=network
        tmp['feature']=netRoi[network]
        master_df=pd.concat([master_df,tmp])
    return master_df

def K_folds(train_sub, clf, memFC,semFC,glassFC,motFC,restFC, test_taskFC, test_restFC):
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
    kf = KFold(n_splits=5,shuffle=True)
    testT= np.ones(test_taskFC.shape[0], dtype = int)
    testR= np.zeros(test_restFC.shape[0], dtype = int)
    X_te=np.concatenate((test_taskFC, test_restFC))
    y_te=np.concatenate((testT, testR))
    CVacc=[]
    df=pd.DataFrame()
    DSacc=[]
    nsize=restFC.shape[2]
    #fold each training set
    if train_sub=='MSC03':
        split=np.empty((8,55278))#doesnt matter what size second dim just splitting on num sessions
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,55278))
    else:
        split=np.empty((10,55278))
    for train_index, test_index in kf.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,nsize))
        Xval_rest=np.reshape(Xval_rest,(-1,nsize))
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
        clf.fit(X_tr,y_tr)
        #get accuracy
        scaler.transform(X_val)
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        scaler.transform(X_te)
        score=clf.score(X_te,y_te)
        DSacc.append(score)
    #same sub
    df['cv']=CVacc
    #Different sub
    df['ds']=DSacc
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['ds'].mean()
    return diff_sub_score, same_sub_score
