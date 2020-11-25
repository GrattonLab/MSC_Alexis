#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#comparison Same sub same task - same sub diff task etc etc
import reshape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import classification
from sklearn.model_selection import cross_val_score
import itertools
import reshape
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/permutation/'

# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
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

def DSmBS():
    DSmBS_perms=pd.DataFrame()
    for i in range(1000):
        diff=classify_DSmBS()
        DSmBS_perms=pd.concat([DSmBS_perms, diff])
    DSmBS_perms.to_csv(outDir+'STmDT_DSmBS_acc.csv',index=False)
def classify_DSmBS():
    BS=classifyBS()
    DS=classifyDS()
    diff_task=BS.merge(DS,how='left',on=['train_task','train_sub','test_sub'],suffixes=('','_DS'))
    diff_task['diff']=diff_task['acc_DS']-diff_task['acc']
    #diff sub same task - diff sub diff task
    STmDT=diff_task[['train_task','test_task','diff']]
    #take average
    diff=STmDT.groupby(['train_task','test_task']).mean()
    diff.reset_index(inplace=True)
    return diff
def CVmSS():
    CVmSS_perms=pd.DataFrame()
    for i in range(1000):
        dfSS=classifySS()
        #take average
        CVmSS_perms=pd.concat([CVmSS_perms,dfSS])
    CVmSS_perms.to_csv(outDir+'STmDT_CVmSS_acc.csv',index=False)
def classifyDS():
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects
    """
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['train_task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model('DS', train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['train_task'])
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    return dfDS

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
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        score=model('SS', train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfSS['diff']=acc_scores_per_task
    #subset so we only average the SS-OS per train/test tasks
    STmDT=dfSS[['train_task','test_task','diff']]
    #take average
    diff=STmDT.groupby(['train_task','test_task']).mean()
    diff.reset_index(inplace=True)
    return diff

def classifyBS():
    """
    Classifying different subjects (BS) along different tasks

    Parameters
    -------------

    Returns
    -------------
    dfBS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        score=model('BS', train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    return dfBS



def model(analysis, train_sub, test_sub, train_task, test_task):
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

    clf=RidgeClassifier()
    taskFC=classification.matFiles(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat')

    #if your subs are the same
    if train_sub==test_sub:
        restFC=classification.matFiles(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat')
        restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=classification.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC=classification.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
        test_taskFC=classification.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        test_restFC=classification.matFiles(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    return total_score



def CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC):
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
    t_tmp= np.ones(taskSize, dtype = int)
    r_tmp=np.zeros(restSize, dtype=int)
    #Concatenate rest and task labels
    Y=np.concatenate((t_tmp,r_tmp))
    #Permute the data
    Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    t, r =np.array_split(Y_perm, 2)
    if analysis=='SS':
        SS_ST_acc_scores=[]
        ST_tmp=pd.DataFrame()
        df=pd.DataFrame()
        acc_score=[]
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
            Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
            ytrain_rest=r[train_index]
            ytrain_task=t[train_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))

            #same sub
            X_val=np.concatenate((Xval_task,Xval_rest))
            y_val=np.array([1,0])

            clf.fit(X_tr,y_tr)
            clf.predict(X_val)
            #same sub same task:ST
            SS=clf.score(X_val,y_val)
            SS_ST_acc_scores.append(SS)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_task=test_taskFC[te_index]
                Xtest_rest=test_restFC[te_index]
                X_Test = np.concatenate((Xtest_task, Xtest_rest))
                #This way we are including the correct rest and task labels
                y_Test = np.array([1, 0])
                #same sub diff task DT
                clf.predict(X_Test)
                #Get accuracy of model
                ACCscores=clf.score(X_Test,y_Test)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)

        #Same sub same task
        ST_tmp['outer_fold']=SS_ST_acc_scores
        SS_ST=ST_tmp['outer_fold'].mean()

        #same sub diff task
        df['outer_fold']=acc_score
        SS_DT=df['outer_fold'].mean()

        #take the difference
        total_score=SS_ST-SS_DT
    elif analysis=='BS':
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
            clf.fit(X_tr,y_tr)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            #fold each testing set
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_rest=test_restFC[te_index]
                Xtest_task=test_taskFC[te_index]
                X_te=np.concatenate((Xtest_task, Xtest_rest))
                y_te=np.array([1, 0])
                #test set
                clf.predict(X_te)
                #Get accuracy of model
                ACCscores=clf.score(X_te,y_te)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
        df['outer_fold']=acc_score
        total_score=df['outer_fold'].mean()
    elif analysis=='DS':
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
            clf.fit(X_tr,y_tr)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            #fold each testing set
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_rest=test_restFC[te_index]
                Xtest_task=test_taskFC[te_index]
                X_te=np.concatenate((Xtest_task, Xtest_rest))
                y_te=np.array([1, 0])
                #test set
                clf.predict(X_te)
                #Get accuracy of model
                ACCscores=clf.score(X_te,y_te)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
        df['outer_fold']=acc_score
        total_score=df['outer_fold'].mean()
    else:
        print('Error')
    return total_score
