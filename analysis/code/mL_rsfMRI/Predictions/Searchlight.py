
#import relevant packages
#this script will just start off with doing a simple CV calculation
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import itertools
from sklearn.model_selection import cross_val_score
import nibabel as nib
import glob
#import time
#using for now; because dconns are all the same size
vertex_size=7320

#vertex_size=59412
scores=np.empty((1,vertex_size))
clf=RidgeClassifier()
#iterate through each index
#will have to determine if there is a way to declare size based on files found in glob function
#lets time for just one subject CV

for vertex in range(vertex_size):
#reshape dconns
    task_ds=np.empty((10,vertex_size))
    rest_ds=np.empty((10,vertex_size))
    task_count=0
    rest_count=0
    count=0
    for task_fname in glob.glob('/projects/p31240/downsampled_4k_surfs/mem/MSC01/*.dconn.nii'):
    #for task_fname in glob.glob('/projects/b1081/member_directories/aporter/Mem/*.dconn.nii'):
        data_file=nib.load(task_fname)
        task=data_file.get_data()
        #take out vertex append to new df
        task_ds[task_count]=task[vertex]
        task_count=task_count+1
    for rest_fname in glob.glob('/projects/p31240/downsampled_4k_surfs/rest/MSC01/*.dconn.nii'):
    #for rest_fname in glob.glob('/projects/b1081/member_directories/aporter/Rest/*.dconn.nii'):
        data_file==nib.load(rest_fname)
        rest=data_file.get_data()
        rest_ds[rest_count]=rest[vertex]
        rest_count=rest_count+1
    taskSize=task_ds.shape[0]
    restSize=rest_ds.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y = np.concatenate((t,r))
    x=np.concatenate((task_ds, rest_ds))
    CVscores=cross_val_score(clf, x, y, cv=10)
    acc=CVscores.mean()
    scores[count][vertex]=acc
    count=count+1
# save to csv file
np.savetxt('/projects/b1081/member_directories/aporter/CV_mem_MSC01.csv', scores, delimiter=',')

##########################################################################
#code that I couldn't get to work because not enough mem storage on quest#
##########################################################################
"""
allTask=np.empty((vertex_size,vertex_size,10))
allRest=np.empty((vertex_size,vertex_size,10))
task_dim=0
rest_dim=0
#start_time=time.time()
for task_fname in glob.glob('/projects/b1081/member_directories/aporter/Mem/*.dconn.nii'):
    data_file=nib.load(task_fname)
    task=data_file.get_data()
    allTask[:,:,task_dim]=task
    task_dim=task_dim+1
for rest_fname in glob.glob('/projects/b1081/member_directories/aporter/Rest/*.dconn.nii'):
    rest_file=nib.load(rest_fname)
    rest=rest_file.get_data()
    allRest[:,:,rest_dim]=rest
    rest_dim=rest_dim+1
#print("--- %s seconds to load all data ---" % (time.time() - start_time))
#Now go through vertex by vertex
#for vertex in range(vertex_size):
vertex=0
#reshape dconns
#Take row from all dimensions and store into new variable
task_ds=allTask[vertex,:,:]
rest_ds=allRest[vertex,:,:]
#reshape to fit the CV format
task_ds=task_ds.T
rest_ds=rest_ds.T
taskSize=task_ds.shape[0]
restSize=rest_ds.shape[0]
t = np.ones(taskSize, dtype = int)
r=np.zeros(restSize, dtype=int)
y = np.concatenate((t,r))
x=np.concatenate((task_ds, rest_ds))
CVscores=cross_val_score(clf, x, y, cv=10)
acc=CVscores.mean()
scores[0][vertex]=acc
# save to csv file
np.savetxt('/projects/b1081/member_directories/aporter/CV_mem_MSC01.csv', scores, delimiter=',')

"""
