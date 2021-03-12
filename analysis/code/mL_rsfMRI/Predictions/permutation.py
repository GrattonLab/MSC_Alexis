#permutation testing for each analysis no difference scoring
#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import itertools
from sklearn.preprocessing import StandardScaler #data scaling
from sklearn import decomposition #PCA
#import other python scripts for further anlaysis
import reshape
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
#outDir = thisDir + 'output/results/'
outDir = thisDir + 'output/results/permutation/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
#omitting MSC06 for classify All
#subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC07','MSC10']
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
#CV combination
CVvars=list(itertools.product(list(subList),list(taskList)))

"""
Each function declares the type of analysis you wanted to run. DS--different subject same task; SS--same subject different task; BS--different subject different task.
Each analysis will concatenate across subjects and make a dataframe.
"""
def permuteProcess():
    #CVtotal=pd.DataFrame()
    DStotal=pd.DataFrame()
    #SStotal=pd.DataFrame()
    BStotal=pd.DataFrame()
    #ALLtotal=pd.DataFrame()
    for i in range(18):
        #CV=classifyCV()
        #CVtotal=pd.concat([CVtotal,CV])
        DS=classifyDS()
        DStotal=pd.concat([DStotal,DS])
        #SS=classifySS()
        #SStotal=pd.concat([SStotal,SS])
        BS=classifyBS()
        BStotal=pd.concat([BStotal,BS])
        #ALL=classifyAll()
        #ALLtotal=pd.concat([ALLtotal,ALL])
    DStotal.to_csv(outDir+'DS/acc.csv',index=False)
    #CVtotal.to_csv(outDir+'CV/acc.csv',index=False)
    #SStotal.to_csv(outDir+'SS/acc.csv',index=False)
    BStotal.to_csv(outDir+'BS/acc.csv',index=False)
    #ALLtotal.to_csv(outDir+'ALL/acc.csv',index=False)
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
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        total_score=model(train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(total_score)
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
        total_score=model(train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(total_score)
    dfSS['acc']=acc_scores_per_task
    #save accuracy
    return dfSS
def classifyBS():
    """
    Classifying between subjects and task (BS)

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
        total_score=model(train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(total_score)
    dfBS['acc']=acc_scores_per_task
    #save accuracy
    return dfBS
def classifyCV():
    """
    Classifying same subjects (CV) along the same task

    Parameters
    -------------

    Returns
    -------------
    dfCV : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    dfCV=pd.DataFrame(CVvars, columns=['sub','task'])
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    clf=RidgeClassifier()
    acc_scores_per_task=[]
    for index, row in dfCV.iterrows():
        taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/'+row['sub']+'_parcel_corrmat.mat')
        folds=taskFC.shape[0]
        x_train, y_train=reshape.concateFC(taskFC, restFC)
        y_train=np.random.permutation(y_train)
        CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
        #get accuracy
        mu=CVscores.mean()
        acc_scores_per_task.append(mu)
        #Get specificity/sensitivity measures
    #average acc per sub per tasks
    dfCV['acc']=acc_scores_per_task
    return dfCV

def model(train_sub, test_sub, train_task, test_task):
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
    df=pd.DataFrame()
    taskFC=reshape.matFiles(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat')
    #if your subs are the same
    if train_sub==test_sub:
        tmp_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat')
        #Split rest into a test and training set 10 test 10 train
        restFC=tmp_restFC[:10]
        test_restFC=tmp_restFC[10:]
        #restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)

    else:
        restFC=reshape.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return total_score

def CV_folds(clf,taskFC, restFC, test_taskFC, test_restFC):
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
    Y=np.concatenate((t,r))
    #Permute the data
    Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    t, r =np.array_split(Y_perm, 2)
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    test_t = np.ones(test_taskSize, dtype = int)
    test_r=np.zeros(test_restSize, dtype=int)
    Xtest=np.concatenate((test_taskFC,test_restFC))
    ytest=np.concatenate((test_t,test_r))
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
        ACCscore=clf.score(Xtest, ytest)
        acc_score.append(ACCscore)
        #fold each testing set
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
    acc_scores_per_sub=[]
    acc_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])

    for index, row in df.iterrows():
        diff_score, same_score=modelAll(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
    df['cv_acc']=acc_scores_cv
    df['acc']=acc_scores_per_sub
    return df

def modelAll(train_sub, test_sub):
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
    df=pd.DataFrame()
    #train sub
    memFC=reshape.matFiles(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.matFiles(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.matFiles(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.matFiles(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
    taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #to have more control over sessions
    #taskFC=np.dstack((memFC,semFC,glassFC,motFC))#10x55278x4
    #restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
    #test sub
    test_memFC=reshape.matFiles(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.matFiles(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    #return taskFC,restFC, test_taskFC,test_restFC
    diff_score, same_score=K_folds(train_sub, clf, taskFC, restFC, test_taskFC,test_restFC)
    return diff_score, same_score

def K_folds(train_sub, clf,taskFC,restFC, test_taskFC, test_restFC):
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
    #reshape rest to get specific days

    #having more control over sessions will require labeling within loop

#    taskSize=taskFC.shape[0]
#    restSize=restFC.shape[0]
#    t = np.ones(taskSize, dtype = int)
#    r=np.zeros(restSize, dtype=int)
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    Y=np.concatenate((t,r))
    #Permute the data
    Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    t, r =np.array_split(Y_perm, 2)
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    test_t = np.ones(test_taskSize, dtype = int)
    test_r=np.zeros(test_restSize, dtype=int)
    Xtest=np.concatenate((test_taskFC,test_restFC))
    ytest=np.concatenate((test_t,test_r))
    CVacc=[]
    df=pd.DataFrame()
    acc_score=[]
    #fold each training set
    for train_index, test_index in kf.split(taskFC):
        Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
        Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
        ytrain_rest, yval_rest=r[train_index], r[test_index]
        ytrain_task, yval_task=t[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        clf.fit(X_tr,y_tr)
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        DS_score=clf.score(Xtest,ytest)
        acc_score.append(DS_score)
    df['cv']=CVacc
    df['outer_fold']=acc_score
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    return diff_sub_score, same_sub_score
