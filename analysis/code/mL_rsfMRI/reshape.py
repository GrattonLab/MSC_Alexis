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
    netRoi=dict([('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 484),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])
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
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        if otherNets is None:
            df_new=df_ut.loc[[networkLabel]]
        else:
            df_new=df_ut.loc[[networkLabel, otherNets]]
        #convert to array
        array=df_new.values
        #remove nans
        clean_array = array[~np.isnan(array)]
        dsNet[dsNet_count]=clean_array
        dsNet_count=dsNet_count+1
    return dsNet

#looking at the blocks of networks ex: default to default connections. This scripts grabs all network blocks
def subBlock(df='path'):
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
    netRoi=dict([('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 484),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    dsNet=np.empty((nsess, 4410))
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
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        #initlize empty
        allBlocks=np.array([])
        for network in netRoi:
            df_new=df_ut.loc[[network]]
            tmp=df_new[network].values
            clean_array = tmp[~np.isnan(tmp)]
            #stack all blocks horizontally
            allBlocks=np.append(allBlocks,clean_array)
        dsNet[dsNet_count]=allBlocks
        dsNet_count=dsNet_count+1
    return dsNet



# In[ ]:





# In[ ]:
