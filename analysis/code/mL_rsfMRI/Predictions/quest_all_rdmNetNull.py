#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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
#import other python scripts for further anlaysis
# Initialization of directory information:
#thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = '/projects/p31240/'
outDir = '/projects/p31240/rdmNetwork/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
subsComb=(list(itertools.permutations(subList, 2)))


netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 484),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21),('co_co',780),('fp_co',960),('default_co',1640)])

#generate log sample
#1000 points for log selection
#loop through 125 times to generate 8*125=1000 samples per log point



def matFiles(df='path'):
    """
    Convert matlab files into upper triangle np.arrays
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #Consistent parameters to use for editing datasets
    nrois=333
    #Load FC file
    fileFC=scipy.io.loadmat(df)

    #Convert to numpy array
    fileFC=np.array(fileFC['parcel_corrmat'])
    #Replace nans and infs with zero
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0]
    df = ds[~mask,:]
    return df

def randFeats(df, idx):
    """
    Random feature selection based on random indexing

    Parameters
    ----------
    df : str
        path to file
    idx : int
        number to index from
    Returns
    ----------
    featDS : Array of task or rest with random features selected
    """
    data=matFiles(df)
    feat=idx.shape[0]
    nsess=data.shape[0]
    featDS=np.empty((nsess, feat))
    for sess in range(nsess):
        f=data[sess][idx]
        featDS[sess]=f
    return featDS

def classifyAll(feat,number):
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
        diff_score, same_score =modelAll(feat, number,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
    df['cv_acc']=acc_scores_cv
    df['acc']=acc_scores_per_sub
    df['features']=number
    return df

def modelAll(feat,number,train_sub, test_sub):
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
    memFC=randFeats(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat',feat)
    semFC=randFeats(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat',feat)
    glassFC=randFeats(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat',feat)
    motFC=randFeats(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat',feat)
    restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat',feat)
    restFC=np.reshape(restFC,(10,4,number))
    #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #test sub
    test_memFC=randFeats(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat',feat)
    test_semFC=randFeats(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat',feat)
    test_glassFC=randFeats(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat',feat)
    test_motFC=randFeats(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat',feat)
    test_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat',feat)
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    #return taskFC,restFC, test_taskFC,test_restFC
    diff_score, same_score=K_folds(train_sub,number, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score
def K_folds(train_sub, number,clf, memFC,semFC,glassFC,motFC,restFC, test_taskFC, test_restFC):
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
    df=pd.DataFrame()
    acc_score=[]
    #fold each training set
    if train_sub=='MSC03':
        split=np.empty((8,number))
        #xtrainSize=24
        #xtestSize=4
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,number))
    else:
        split=np.empty((10,number))
    for train_index, test_index in kf.split(split):
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
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
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

ALL_df=pd.DataFrame()
for nullNet in netRoi:
    number=netRoi[nullNet]
    for n in range(10):
        #generate a new index
        idx=np.random.randint(55278, size=(number))
        ALL=classifyAll(idx, number)
        ALL['Null_Network']=nullNet
        ALL_df=pd.concat([ALL_df,ALL])
ALL_df.to_csv(outDir+'ALL/nullNet_acc.csv', index=False)
