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
    for i in networks:
        tmp_df=classifyAll(network=i,subnetwork=i)
        tmp_df=tmp_df.groupby(['train_sub']).mean()
        tmp_df.rename(columns={'cv_acc':'Same Subject','acc':'Different Subject'},inplace=True)
        tmp_df.reset_index(inplace=True)
        tmp_df=pd.melt(tmp_df, id_vars=['train_sub'], value_vars=['Same Subject','Different Subject'],var_name='Analysis',value_name='acc')
        tmp_df['Network_A']=i
        tmp_df['Network_B']=i
        finalDF=pd.concat([finalDF, tmp_df])
    finalDF.to_csv(thisDir+'output/results/acc/ALL/Net2Net_acc.csv')

def subNetAll():
    netDF=pd.DataFrame(netComb, columns=['Network_A','Network_B'])
    finalDF=pd.DataFrame()
    for index, row in netDF.iterrows():
        tmp_df=classifyAll(network=row['Network_A'], subnetwork=row['Network_B'])
        tmp_df=tmp_df.groupby(['train_sub']).mean()
        tmp_df.rename(columns={'cv_acc':'Same Subject','acc':'Different Subject'},inplace=True)
        tmp_df.reset_index(inplace=True)
        tmp_df=pd.melt(tmp_df, id_vars=['train_sub'], value_vars=['Same Subject','Different Subject'],var_name='Analysis',value_name='acc')
        tmp_df['Network_A']=row['Network_A']
        tmp_df['Network_B']=row['Network_B']
        finalDF=pd.concat([finalDF, tmp_df])
    finalDF.to_csv(thisDir+'output/results/acc/ALL/subNetwork_acc.csv')
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
    acc_scores_per_sub=[]
    acc_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        diff_score, same_score=modelAll(network,subnetwork,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
    df['cv_acc']=acc_scores_cv
    df['acc']=acc_scores_per_sub
    return df
    #df.to_csv(outDir+'acc/ALL/precision_acc.csv',index=False)

def modelAll(network,subnetwork,train_sub, test_sub):
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
    df=pd.DataFrame()
    #train sub
    memFC=reshape.network_to_network(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    semFC=reshape.network_to_network(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    glassFC=reshape.network_to_network(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    motFC=reshape.network_to_network(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #test sub
    test_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    #return taskFC,restFC, test_taskFC,test_restFC
    diff_score, same_score=K_folds(train_sub, clf, taskFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score

def K_folds(train_sub, clf, taskFC, restFC, test_taskFC, test_restFC):
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
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
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
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        scaler.transform(X_val)
        clf.fit(X_tr,y_tr)
        #cross validation
        y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
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
            scaler.transform(X_te)
            #test set
            y_pred_testset=clf.predict(X_te)
            #Test labels and predicted labels to calculate sensitivity specificity
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
    df['cv']=CVacc
    #Different sub outer acc
    df['outer_fold']=acc_score
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    return diff_sub_score, same_sub_score

def classifyDS(network,subnetwork=None):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model('DS', network,subnetwork, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    dfDS.to_csv(outDir+network+'_'+subnetwork+'/DS/acc.csv')
def classifySS(network,subnetwork=None):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        score=model('SS', network,subnetwork, train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfSS['acc']=acc_scores_per_task
    #save accuracy
    dfSS.to_csv(outDir+network+'_'+subnetwork+'/SS/acc.csv')
def classifyBS(network,subnetwork=None):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        score=model('BS', network,subnetwork, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    dfBS.to_csv(outDir+network+'_'+subnetwork+'/BS/acc.csv')

def model(analysis, network,subnetwork, train_sub, test_sub, train_task, test_task):
    clf=RidgeClassifier(max_iter=10000)
    taskFC=reshape.subNets(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat', network,subnetwork)
    #if your subs are the same split rest
    if train_sub==test_sub:
        restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat', network,subnetwork)
        #Split rest into a test and training set 10 test 10 train
        restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=reshape.subNets(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
        ACCscores=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC=reshape.subNets(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat', network,subnetwork)
        test_taskFC=reshape.subNets(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
        test_restFC=reshape.subNets(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
        ACCscores=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    return ACCscores
#Calculate acc of cross validation within sub within task
def classifyCV(network,subnetwork=None):
    dfCV=pd.DataFrame(CVvars, columns=['sub','task'])
    clf=RidgeClassifier()
    acc_scores_per_task=[]
    for index, row in dfCV.iterrows():
        taskFC=reshape.subNets(dataDir+row['task']+'/'+row['sub']+'_parcel_corrmat.mat',network,subnetwork)
        restFC=reshape.subNets(dataDir+'rest/'+row['sub']+'_parcel_corrmat.mat',network,subnetwork)
        folds=taskFC.shape[0]
        x_train, y_train=reshape.concateFC(taskFC, restFC)
        CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
        mu=CVscores.mean()
        acc_scores_per_task.append(mu)
    #average acc per sub per tasks
    dfCV['acc']=acc_scores_per_task
    dfCV.to_csv(outDir+network+'_'+subnetwork+'/CV/acc.csv')
def CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC):
    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
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
        #fold each testing set
        for t_index, te_index in loo.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            y_te=np.array([1, 0])
            #test set
            #standardization
            scaler.transform(X_te)
            clf.predict(X_te)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
    df['outer_fold']=acc_score
    total_score=df['outer_fold'].mean()

    return total_score
