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

thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
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
    return ds
def concateFC(taskFC, restFC):
    """
    Concatenates task and rest FC arrays and creates labels
    Parameters
    -----------
    taskFC, restFC : array_like
        Numpy arrays of FC upper triangle for rest and task
    Returns
    -----------
    x, y : array_like
        Arrays containing task and restFC concatenated together and labels for each
    """
    x=np.concatenate((taskFC, restFC))
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y = np.concatenate((t,r))
    return x, y

def subNets(df='path', networkLabel='networklabel', otherNets=None):
    """
    Same as reshape but subset by network
    str options for networks ['unassign',
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
    'pon']
    Parameters
    -----------
    df : str
        Path to file
    networkLabel : str
        String to indicate which network to subset
    otherNets : str; optional
        If looking at specific network to network connection include other network
    Returns
    ----------
    dsNet : Array of task or rest FC containing only subnetworks
    """
 #roi count for building arrays
    netRoi=dict([('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 494),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])
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
        Parcel_params = loadParcelParams('Gordon333',thisDir+'data/Parcel_info/')
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
    """
    Same as subNets but subset by block level
    str options for networks ['unassign',
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
    'pon']
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
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
        Parcel_params = loadParcelParams('Gordon333',thisDir+'data/Parcel_info/')
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


def feature_plots(coef, classifier='which', analysis='type', train_task='task', train_sub='i'):
    #convert to 2d symmetrical matrix
    nrois=333
    ds=np.zeros((nrois, nrois))
    ds[np.triu_indices(ds.shape[0], k = 1)] = coef
    ds = ds + ds.T
    #get atlas you want to use
    Parcel_params = loadParcelParams('Gordon333',thisDir+'data/Parcel_info/')
    #make pretty fig
    #vmin=np.amin(ds)
    #vmax=np.amax(ds)
    fig = figure_corrmat(ds,Parcel_params, clims=(-.002,.002))
    fig.savefig(outDir+"images/"+classifier+"/fw/" +analysis+ "/" +task+ "_" +sub+ ".png", bbox_inches='tight')

def loadParcelParams(roiset,datadir):
    """ This function loads information about the ROIs and networks.
    For now, this is only set up to work with 333 Gordon 2014 Cerebral Cortex regions
    Inputs:
    roiset = string naming roi type to get parameters for (e.g. 'Gordon333')
    datadir = string path to the location where ROI files are stored
    Returns:
    Parcel_params: a dictionary with ROI information stored in it
    """
    import scipy.io as spio
    #initialize a dictionary where info will be stored
    Parcel_params = {}

    # put some info into the dict that will work for all roi sets
    Parcel_params['roiset'] = roiset
    dataIn_types = {'dmat','mods_array','roi_sort','net_colors'}
    for dI in dataIn_types:
          dataIn = spio.loadmat(datadir + roiset + '_' + dI + '.mat')
          Parcel_params[dI] = np.array(dataIn[dI])
    Parcel_params['roi_sort'] = Parcel_params['roi_sort'] - 1 #orig indexing in matlab, need to subtract 1

    #transition points and centers for plotting
    transitions,centers = compute_trans_centers(Parcel_params['mods_array'],Parcel_params['roi_sort'])
    Parcel_params['transitions'] = transitions
    Parcel_params['centers'] = centers

    # some ROI specific info that needs to be added by hand
    # add to this if you have a new ROI set that you're using
    if roiset == 'Gordon333':
        Parcel_params['dist_thresh'] = 20 #exclusion distance to not consider in metrics
        Parcel_params['num_rois'] = 333
        Parcel_params['networks'] = ['unassign','default','visual','fp','dan','van','salience',
                                         'co','sm','sm-lat','auditory','pmn','pon']
    else:
        raise ValueError("roiset input is recognized.")

    return Parcel_params
def compute_trans_centers(mods_array,roi_sort):
    """ Function that computes transitions and centers of networks for plotting names
    Inputs:
    mods_array: a numpy vector with the network assignment for each ROI (indexed as a number)
    roi_sort: ROI sorting ordered to show each network in sequence
    Returns:
    transitions: a vector with transition points between networks
    centers: a vector with center points for each network
    """

    mods_sorted = np.squeeze(mods_array[roi_sort])
    transitions = np.nonzero((np.diff(mods_sorted,axis=0)))[0]+1 #transition happens 1 after

    trans_plusends = np.hstack((0,transitions,mods_array.size)) #add ends
    centers = trans_plusends[:-1] + ((trans_plusends[1:] - trans_plusends[:-1])/2)

    return transitions,centers

def figure_corrmat(corrmat,Parcel_params, clims=(-1,1)):
    """ This function will make a nice looking plot of a correlation matrix for a given parcellation,
    labeling and demarkating networks.
    Inputs:
    corrmat: an roi X roi matrix for plotting
    Parcel_params: a dictionary with ROI information
    clims: (optional) limits to place on corrmat colormap
    Returns:
    fig: a figure handle for figure that was made
    """

    # some variables for ease
    roi_sort = np.squeeze(Parcel_params['roi_sort'])

    # main figure plotting
    fig, ax = plt.subplots()
    im = ax.imshow(corrmat[roi_sort,:][:,roi_sort],cmap='seismic',vmin=clims[0],vmax=clims[1], interpolation='none')
    plt.colorbar(im)

    # add some lines between networks
    for tr in Parcel_params['transitions']:
        ax.axhline(tr,0,Parcel_params['num_rois'],color='k')
        ax.axvline(tr,0,Parcel_params['num_rois'],color='k')

    # alter how the tick marks are shown to plot network names
    ax.set_xticks(Parcel_params['centers'])
    ax.set_yticks(Parcel_params['centers'])
    ax.set_xticklabels(Parcel_params['networks'],fontsize=8)
    ax.set_yticklabels(Parcel_params['networks'],fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

    plt.show()

    return fig
def saveFW(coef):
    #convert to 2d symmetrical matrix
    nrois=333
    ds=np.zeros((nrois, nrois))
    ds[np.triu_indices(ds.shape[0], k = 1)] = coef
    ds = ds + ds.T
    Parcel_params = loadParcelParams('Gordon333',thisDir+'data/Parcel_info/')
    roi_sort = np.squeeze(Parcel_params['roi_sort'])
    #rearrange roi's to be together
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
    df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
    return df
    #df.to_csv(outDir+"results/"+classifier+"/fw/" +analysis+ "/" +task+ "_" +sub+ ".csv")

    #you'll have to specify that it is a tuple pd.read_csv('test.csv',index_col=[0,1])

#standard deviation of feature weights between folds
#did not initialize for classify.py since this is for debugging
def fwFolds(folds, classifier, analysis, task, sub):
    #concate into useable form
    fw=np.empty([10,55278])
    count=0
    for model in folds['estimator']:
        i=model.coef_
        fw[count]=i
        count=count+1
    fwSD=np.std(fw, axis=0)
    nrois=333
    ds=np.zeros((nrois, nrois))
    ds[np.triu_indices(ds.shape[0], k = 1)] = fwSD
    ds = ds + ds.T
    #get atlas you want to use
    Parcel_params = loadParcelParams('Gordon333',thisDir+'data/Parcel_info/')
    #make pretty fig
    vmin=np.amin(ds)
    vmax=np.amax(ds)
    fig = figure_corrmat(ds,Parcel_params, clims=(0, .0002))
    fig.savefig(outDir+"images/"+classifier+"/fw/" +analysis+ "/fwSD_Folds/" +task+ "_" +sub+ ".png", bbox_inches='tight')




# In[ ]:





# In[ ]:
