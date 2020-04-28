#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Imports
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd
def feature_plots(coef, f='DS', task='task', sub='i'):
    #convert to 2d symmetrical matrix
    nrois=333
    ds=np.zeros((nrois, nrois))
    ds[np.triu_indices(ds.shape[0], k = 1)] = coef
    ds = ds + ds.T
    #get atlas you want to use
    Parcel_params = loadParcelParams('Gordon333','/Users/Alexis/Desktop/MSC_Alexis/analysis/data/Parcel_info/')
    #make pretty fig
    #vmin=np.amin(ds)
    #vmax=np.amax(ds)
    fig = figure_corrmat(ds,Parcel_params, clims=(-.002,.002))
    fig.savefig("/Users/Alexis/Desktop/MSC_Alexis/analysis/output/images/SVC/fw/" +f+ "/" +task+ "_" +sub+ ".png", bbox_inches='tight')

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
def saveFW(coef, f, task, sub):
    #convert to 2d symmetrical matrix
    nrois=333
    ds=np.zeros((nrois, nrois))
    ds[np.triu_indices(ds.shape[0], k = 1)] = coef
    ds = ds + ds.T
    Parcel_params = loadParcelParams('Gordon333','/Users/Alexis/Desktop/MSC_Alexis/analysis/data/Parcel_info/')
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
    df.to_csv("/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/SVC/fw/" +f+ "/" +task+ "_" +sub+ ".csv")

    #you'll have to specify that it is a tuple pd.read_csv('test.csv',index_col=[0,1])

#standard deviation of feature weights between folds
def fwFolds(folds, f, task, sub):
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
    Parcel_params = loadParcelParams('Gordon333','/Users/Alexis/Desktop/MSC_Alexis/analysis/data/Parcel_info/')
    #make pretty fig
    vmin=np.amin(ds)
    vmax=np.amax(ds)
    fig = figure_corrmat(ds,Parcel_params, clims=(0, .0002))
    fig.savefig("/Users/Alexis/Desktop/MSC_Alexis/analysis/output/images/SVC/fw/" +f+ "/fwSD_Folds/" +task+ "_" +sub+ ".png", bbox_inches='tight')



# In[1]:


"""
import reshape
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
#Load task FC
taskFC=reshape.matFiles('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/MSC05_parcel_corrmat.mat')
#Load rest
restFC=reshape.matFiles('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/MSC05_parcel_corrmat.mat')
#Create a training dataset targets 1/0 ==task/rest, chunk=#days
x_train=np.concatenate((taskFC, restFC))
taskSize=taskFC.shape[0]
restSize=restFC.shape[0]
t = np.ones(taskSize, dtype = int)
r=np.zeros(restSize, dtype=int)
y_train = np.concatenate((t,r))

svm = LinearSVC()
svm.fit(x_train, y_train)
coef = svm.coef_
#convert to 2d symmetrical matrix
#nrois=333
#ds=np.zeros((nrois, nrois))
#ds[np.triu_indices(ds.shape[0], k = 1)] = coef
#ds = ds + ds.T
#Parcel_params = loadParcelParams('Gordon333','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/Parcel_info/')
saveFW(coef,'DS', 'test','foo')
"""


# In[ ]:
