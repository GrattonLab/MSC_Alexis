addpath /Users/Alexis/Desktop/BarchSZ/cifti-matlab-master

addpath /Applications/gifti-master

addpath /Users/Alexis/Applications/general_plotting
addpath /Users/Alexis/Applications/read_write_cifti/utilities


parcel='/Users/Alexis/Desktop/MSC_Alexis/analysis/code/matScripts/Parcels_LR.dtseries.nii';
dataF=['/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/permutation/ALL/MSC06_Row.csv'];
data=load(dataF);

assign_data_to_parcel_cifti_V2(data,parcel,'/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/permutation/ALL/', 'MSC06')
