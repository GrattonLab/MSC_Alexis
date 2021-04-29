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
outDir = thisDir + 'output/results/'
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
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])


"""
Each function declares the type of analysis you wanted to run. DS--different subject same task; SS--same subject different task; BS--different subject different task.
Each analysis will concatenate across subjects and make a dataframe.
"""
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
    sen_per_task=[]
    spec_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        total_score, total_sen, total_spec=model(train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(total_score)
        sen_per_task.append(total_sen)
        spec_per_task.append(total_spec)
    dfDS['acc']=acc_scores_per_task
    dfDS['spec']=spec_per_task
    dfDS['sen']=sen_per_task
    dfDS.to_csv(outDir+'acc/DS/acc.csv',index=False)

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
    sen_per_task=[]
    spec_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        total_score, total_sen, total_spec=model(train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(total_score)
        sen_per_task.append(total_sen)
        spec_per_task.append(total_spec)
    dfSS['acc']=acc_scores_per_task
    #dfSS['spec']=spec_per_task
    #dfSS['sen']=sen_per_task
    #save accuracy
    #dfSS.to_csv(outDir+'acc/SS/acc.csv',index=False)
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
    sen_per_task=[]
    spec_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        total_score, total_sen, total_spec=model(train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(total_score)
        sen_per_task.append(total_sen)
        spec_per_task.append(total_spec)
    dfBS['acc']=acc_scores_per_task
    dfBS['spec']=spec_per_task
    dfBS['sen']=sen_per_task
    #save accuracy
    dfBS.to_csv(outDir+'acc/BS/acc.csv',index=False)
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
    CVspec=[]
    CVsen=[]
    for index, row in dfCV.iterrows():
        taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/'+row['sub']+'_parcel_corrmat.mat')
        folds=taskFC.shape[0]
        x_train, y_train=reshape.concateFC(taskFC, restFC)
        CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
        #Get sensitivity/specificity measures
        y_pred=cross_val_predict(clf, x_train, y_train,cv=folds)
        tn, fp, fn, tp=confusion_matrix(y_train, y_pred).ravel()
        CV_specificity= tn/(tn+fp)
        CV_sensitivity= tp/(tp+fn)
        #get accuracy
        mu=CVscores.mean()
        acc_scores_per_task.append(mu)
        CVspec.append(CV_specificity)
        CVsen.append(CV_sensitivity)
        #Get specificity/sensitivity measures
    #average acc per sub per tasks
    dfCV['acc']=acc_scores_per_task
    dfCV['spec']=CVspec
    dfCV['sen']=CVsen
    dfCV.to_csv(outDir+'acc/CV/acc.csv', index=False)
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
        restFC=reshape.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
        #tmp_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat')
        #Split rest into a test and training set 10 test 10 train
        #restFC=tmp_restFC[:10]
        #test_restFC=tmp_restFC[10:]

        #restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        total_score, total_sen, total_spec=SS_folds(clf, taskFC, restFC, test_taskFC)

    else:
        restFC=reshape.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat')
        total_score, total_sen, total_spec=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return total_score, total_sen, total_spec

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
    df=pd.DataFrame()
    acc_score=[]
    spec_score=[]
    sen_score=[]
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
        #sensitivity and specificity per fold
        sen_per_fold=[]
        spec_per_fold=[]
        #fold each testing set
        for t_index, te_index in loo.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            y_te=np.array([1, 0])
            #test set
            y_pre=clf.predict(X_te)
            #calculate sensitivity/specificity
            tn, fp, fn, tp=confusion_matrix(y_te, y_pre).ravel()
            specificity= tn/(tn+fp)
            sensitivity= tp/(tp+fn)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
            sen_per_fold.append(sensitivity)
            spec_per_fold.append(specificity)
        tmpdf['inner_sens']=sen_per_fold
        sens=tmpdf['inner_sens'].mean()
        tmpdf['inner_spec']=spec_per_fold
        spec=tmpdf['inner_spec'].mean()
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
        spec_score.append(spec)
        sen_score.append(sens)
    df['outer_fold']=acc_score
    total_score=df['outer_fold'].mean()
    df['outer_sens']=sen_score
    total_sen=df['outer_sens'].mean()
    df['outer_spec']=spec_score
    total_spec=df['outer_spec'].mean()
    return total_score, total_sen, total_spec
def SS_folds(clf,taskFC, restFC, test_taskFC):
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
    df=pd.DataFrame()
    acc_score=[]
    spec_score=[]
    sen_score=[]
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
        #sensitivity and specificity per fold
        sen_per_fold=[]
        spec_per_fold=[]
        #fold each testing set

        for t_index, te_index in loo.split(test_taskFC):
            Xtest=test_taskFC[te_index]
            y_te=np.array([1])
            #test set
            #Get accuracy of model
            ACCscores=clf.score(Xtest,y_te)
            acc_scores_per_fold.append(ACCscores)
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
    df['outer_fold']=acc_score
    total_score=df['outer_fold'].mean()

    return total_score, sen_score, spec_score

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
    sen_scores_per_sub=[]
    spec_scores_per_sub=[]
    acc_scores_cv=[]
    sen_scores_cv=[]
    spec_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])

    for index, row in df.iterrows():

        diff_score, same_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score=modelAll(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
        sen_scores_cv.append(CV_sens_score)
        spec_scores_cv.append(CV_spec_score)
        sen_scores_per_sub.append(DS_sens_score)
        spec_scores_per_sub.append(DS_spec_score)

    df['cv_acc']=acc_scores_cv
    df['cv_sen']=sen_scores_cv
    df['cv_spec']=spec_scores_cv
    df['acc']=acc_scores_per_sub
    df['ds_sen']=sen_scores_per_sub
    df['ds_spec']=spec_scores_per_sub
    #df.to_csv(outDir+'acc/ALL/shufflekFold_acc.csv',index=False)
    df.to_csv(outDir+'acc/ALL/LOO_acc.csv',index=False)


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
    restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
    #test sub
    test_memFC=reshape.matFiles(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.matFiles(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))

    diff_score, same_score,CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score=K_folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
    return diff_score, same_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score

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
    loo = LeaveOneOut()
    #kf = KFold(n_splits=5)
    #reshape rest to get specific days

    #having more control over sessions will require labeling within loop

#    taskSize=taskFC.shape[0]
#    restSize=restFC.shape[0]
#    t = np.ones(taskSize, dtype = int)
#    r=np.zeros(restSize, dtype=int)

    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)

    taskScore=[]
    restScore=[]
    testTScore=[]
    testRScore=[]
    CVacc=[]
    CVspec=[]
    CVsen=[]
    df=pd.DataFrame()
    acc_score=[]
    DSspec=[]
    DSsen=[]
    #fold each training set
    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))

    for train_index, test_index in loo.split(split):

        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,55278))
        Xval_rest=np.reshape(Xval_rest,(-1,55278))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)

        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        clf.fit(X_tr,y_tr)
        ts=clf.score(Xval_task, yval_task)
        rs=clf.score(Xval_rest, yval_rest)
        taskScore.append(ts)
        restScore.append(rs)
        testTask=clf.score(test_taskFC, testT)
        testRest=clf.score(test_restFC, testR)
        testTScore.append(testTask)
        testRScore.append(testRest)
        #cross validation

        y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
        tn, fp, fn, tp=confusion_matrix(y_val, y_pred).ravel()
        CV_specificity= tn/(tn+fp)
        CV_sensitivity= tp/(tp+fn)
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        CVspec.append(CV_specificity)
        CVsen.append(CV_sensitivity)
        #fold each testing set

        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        sen_scores_per_fold=[]
        spec_scores_per_fold=[]

        for t_index, te_index in kf.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            ytest_task=testT[te_index]
            ytest_rest=testR[te_index]
            y_te=np.concatenate((ytest_task, ytest_rest))
            #test set
            y_pred_testset=clf.predict(X_te)
            #Test labels and predicted labels to calculate sensitivity specificity
            DStn, DSfp, DSfn, DStp=confusion_matrix(y_te, y_pred_testset).ravel()
            DS_specificity= DStn/(DStn+DSfp)
            DS_sensitivity= DStp/(DStp+DSfn)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
            sen_scores_per_fold.append(DS_sensitivity)
            spec_scores_per_fold.append(DS_specificity)

        tmpdf['inner_fold']=acc_scores_per_fold
        tmpdf['DS_sen']=sen_scores_per_fold
        tmpdf['DS_spec']=spec_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        sen=tmpdf['DS_sen'].mean()
        spec=tmpdf['DS_spec'].mean()
        acc_score.append(score)
        DSspec.append(spec)
        DSsen.append(sen)

    df['cv']=CVacc
    df['CV_sen']=CVsen
    df['CV_spec']=CVspec
    #Different sub outer acc
    df['outer_fold']=acc_score
    df['DS_sen']=DSsen
    df['DS_spec']=DSspec
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    CV_sens_score=df['CV_sen'].mean()
    CV_spec_score=df['CV_spec'].mean()
    DS_sens_score=df['DS_sen'].mean()
    DS_spec_score=df['DS_spec'].mean()


    return diff_sub_score, same_sub_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score


def classifyAll_wPCA(num,pca_var_explained=.5):
    """
    Classifying different subjects along available data rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    #for simplicity only take accuracy scores
    acc_scores_per_sub=[]
    #sen_scores_per_sub=[]
    #spec_scores_per_sub=[]
    acc_scores_cv=[]
    #sen_scores_cv=[]
    #spec_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        diff_score, same_score=modelAll_wPCA(num,pca_var_explained,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
        #sen_scores_cv.append(CV_sens_score)
        #spec_scores_cv.append(CV_spec_score)
        #sen_scores_per_sub.append(DS_sens_score)
        #spec_scores_per_sub.append(DS_spec_score)
    df['cv_acc']=acc_scores_cv
    #df['cv_sen']=sen_scores_cv
    #df['cv_spec']=spec_scores_cv
    df['acc']=acc_scores_per_sub
    #df['ds_sen']=sen_scores_per_sub
    #df['ds_spec']=spec_scores_per_sub
    return df
    #df.to_csv(outDir+'acc/ALL/pca05_acc.csv',index=False)

def modelAll_wPCA(num,pca_var_explained,train_sub, test_sub):
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
    restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat')
    #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    restFC=np.reshape(restFC,(10,4,55278)) #controlling for days
    #test sub
    test_memFC=reshape.matFiles(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.matFiles(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    diff_score, same_score=K_folds_wPCA(num,pca_var_explained,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score#, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score




def K_folds_wPCA(num,pca_var_explained,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using 5k fold
    Uses all data and fits PCA 50% of variance only on the training data

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
    #have to standardize data in order to use PCA
    scaler=StandardScaler()
    #PCA 50% variance explained based on Marek paper
    pca=decomposition.PCA(num,pca_var_explained)
    kf = KFold(n_splits=5,shuffle=True)
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
    CVacc=[]
    #CVspec=[]
    #CVsen=[]
    df=pd.DataFrame()
    acc_score=[]
    #DSspec=[]
    #DSsen=[]
    #fold each training set
    if train_sub=='MSC03':
        split=np.empty((8,55527))
        #xtrainSize=24
        #xtestSize=4
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
        if train_sub=='MSC06' or train_sub=='MSC07':
            if train_index.shape[0]==7:#getting around MSC06 missing one day of glass task
                Xtrain_rest=np.reshape(Xtrain_rest,(28,55278))
                Xval_rest=np.reshape(Xval_rest,(8,55278))
            else:
                Xtrain_rest=np.reshape(Xtrain_rest,(32,55278))
                Xval_rest=np.reshape(Xval_rest,(4,55278))
        elif train_sub=='MSC03':
            if train_index.shape[0]==6:
                Xtrain_rest=np.reshape(Xtrain_rest,(24,55278))
                Xval_rest=np.reshape(Xval_rest,(8,55278))
            else:
                Xtrain_rest=np.reshape(Xtrain_rest,(28,55278))
                Xval_rest=np.reshape(Xval_rest,(4,55278))
        else:
            Xtrain_rest=np.reshape(Xtrain_rest,(32,55278))
            Xval_rest=np.reshape(Xval_rest,(8,55278))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        #Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
        #ytrain_rest, yval_rest=r[train_index], r[test_index]
        #ytrain_task, yval_task=t[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        #PCA
        scaler.fit(X_tr) #scale and fit on training set only
        X_tr=scaler.transform(X_tr)
        X_val=scaler.transform(X_val)
        #fit only to the training set
        pca.fit(X_tr)
        #transform training and testing data
        X_tr=pca.transform(X_tr)
        X_val=pca.transform(X_val)
        #now we fit to classifier
        clf.fit(X_tr,y_tr)
        #cross validation
        #y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
        #tn, fp, fn, tp=confusion_matrix(y_val, y_pred).ravel()
        #CV_specificity= tn/(tn+fp)
        #CV_sensitivity= tp/(tp+fn)
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        #CVspec.append(CV_specificity)
        #CVsen.append(CV_sensitivity)
        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        #sen_scores_per_fold=[]
        #spec_scores_per_fold=[]
        #fold each testing set
        for t_index, te_index in kf.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            ytest_task=testT[te_index]
            ytest_rest=testR[te_index]
            y_te=np.concatenate((ytest_task, ytest_rest))
            #scale and PCA
            X_te=scaler.transform(X_te)
            #apply pca to test set
            X_te=pca.transform(X_te)
            #test set
            #y_pred_testset=clf.predict(X_te)
            #Test labels and predicted labels to calculate sensitivity specificity
            #DStn, DSfp, DSfn, DStp=confusion_matrix(y_te, y_pred_testset).ravel()
            #DS_specificity= DStn/(DStn+DSfp)
            #DS_sensitivity= DStp/(DStp+DSfn)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
            #sen_scores_per_fold.append(DS_sensitivity)
            #spec_scores_per_fold.append(DS_specificity)
        tmpdf['inner_fold']=acc_scores_per_fold
        #tmpdf['DS_sen']=sen_scores_per_fold
        #tmpdf['DS_spec']=spec_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        #sen=tmpdf['DS_sen'].mean()
        #spec=tmpdf['DS_spec'].mean()
        acc_score.append(score)
        #DSspec.append(spec)
        #DSsen.append(sen)
    df['cv']=CVacc
    #df['CV_sen']=CVsen
    #df['CV_spec']=CVspec
    #Different sub outer acc
    df['outer_fold']=acc_score
    #df['DS_sen']=DSsen
    #df['DS_spec']=DSspec
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    #CV_sens_score=df['CV_sen'].mean()
    #CV_spec_score=df['CV_spec'].mean()
    #DS_sens_score=df['DS_sen'].mean()
    #DS_spec_score=df['DS_spec'].mean()
    return diff_sub_score, same_sub_score#, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score
