clear;
clear all;
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/mem/MSC02_parcel_corrmat.mat');
MSC02_mem=parcel_corrmat;
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/rest/MSC02_parcel_corrmat.mat');
MSC02_rest=parcel_corrmat;
%concatenating memory and rest along third dimension first 10 days are task
%then rest
%leaving a day out to use for testing later
MSC02_rest_train=MSC02_rest(:,:,2:end)
MSC02_mem_train=MSC02_mem(:,:,2:end)
MSC02_rest_test=MSC02_rest(:,:,1)
MSC02_mem_test=MSC02_mem(:,:,1)

MSC02_train=cat(3, MSC02_mem_train, MSC02_rest_train)
MSC02_test=cat(3, MSC02_mem_test, MSC02_rest_test)
%training across 9 days testing on one day
results=svm_scripts_beta(MSC02_train, [ones(9,1); -ones(9,1)],0,MSC02_test,[ones(1,1); -ones(1,1)],0)

mean((sum(results.predictedTestLabels(1,:)==1)+sum(results.predictedTestLabels(2,:)==-1))./2)

%MSC02_test,[ones(1,1); -ones(1,1)]

