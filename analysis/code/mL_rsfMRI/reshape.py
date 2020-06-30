#!/usr/bin/env python
# coding: utf-8

# In[2]:

import scipy.io
import pandas as pd
import numpy as np
import sys
import os
import pandas as pd
import scipy.io
import plotFW
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
def matFiles(df='path'):
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
    return ds
def concateFC(taskFC, restFC):
    x=np.concatenate((taskFC, restFC))
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y = np.concatenate((t,r))
    return x, y

def subNets(df='path', networkLabel='networklabel', otherNets=None):
    """options for networks ['unassign',
 'default',
 'visual',
 'fp',
 'dan',
 'van',
 'salience',
 'co',
 'sm',
 'sm-lat',
 'auditory',
 'pmn',
 'pon'] """
 #roi count for building arrays
    netRoi=dict([('default', 13653),('visual',12987),('fp', 7992),('dan',10656),('van',7659),('salience', 1332),('co', 13320),('sm', 12654),('sm-lat', 2664),('auditory', 7992),('pmn',1665),('pon',2331)])
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    if otherNets is None:
        dsNet=np.empty((nsess, netRoi[networkLabel]))
    else:
        netLength=netRoi[networkLabel]+netRoi[otherNets]
        dsNet=np.empty((nsess, netLength))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = plotFW.loadParcelParams('Gordon333',thisDir+'data/Parcel_info/')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        #avoid duplicates by taking upper triangle k=1 so we don't take the first value
        df_ut = df.where(np.triu(np.ones(df.shape)).astype(np.bool),1)
        if otherNets is None:
            df_new=df_ut.loc[[networkLabel]]
        else:
            df_new=df_ut.loc[[networkLabel, otherNets]]
        #convert to array
        array=df_new.to_numpy()
        #remove nans
        clean_array = array[~np.isnan(array)]
        dsNet[dsNet_count]=clean_array
        dsNet_count=dsNet_count+1
    return dsNet
# In[ ]:





# In[ ]:
