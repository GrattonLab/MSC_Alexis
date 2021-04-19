addpath /Users/Alexis/Desktop/BarchSZ/cifti-matlab-master

addpath /Applications/gifti-master

addpath /Users/Alexis/Applications/general_plotting
addpath /Users/Alexis/Applications/read_write_cifti/utilities


parcel='/Users/Alexis/Desktop/MSC_Alexis/analysis/code/matScripts/Parcels_LR.dtseries.nii';
subs={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
folds={'0','1','2','3','4','5','6','7','8','9'}
for i=1:length(folds);
    dataF=['/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/ALL/foldMSC05/fold' folds{i} '.csv'];
    data=load(dataF);
    assign_data_to_parcel_cifti_V2(data,parcel,'/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/ALL/foldMSC05/', folds{i})
    clear dataF
    clear data 
end
