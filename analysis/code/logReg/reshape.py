#!/usr/bin/env python
# coding: utf-8

# In[2]:

import scipy.io
import pandas as pd
import numpy as np
def matFiles(df='path'):
    #Consistent parameters to use for editing datasets
    nrois=333
    nsess=10
    #Load FC file
    fileFC=scipy.io.loadmat(df)
    #Convert to numpy array
    fileFC=np.array(fileFC['parcel_corrmat'])
    #Replace nans and infs with zero
    fileFC=np.nan_to_num(fileFC)
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

# In[ ]:





# In[ ]:
