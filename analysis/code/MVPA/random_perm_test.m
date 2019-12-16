load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/mem/MSC02_parcel_corrmat.mat');
MSC02_mem=parcel_corrmat;
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/rest/MSC02_parcel_corrmat.mat');
MSC02_rest=parcel_corrmat;
%MSC02_mem(isinf(MSC02_mem)|isnan(MSC02_mem)) = 0; % Replace NaNs and infinite values with zeros
%MSC02_rest(isinf(MSC02_rest)|isnan(MSC02_rest)) = 0; % Replace NaNs and infinite values with zeros
%concatenating memory and rest along third dimension first 10 days are task
%then rest
MSC02_mem_rest=cat(3, MSC02_mem, MSC02_rest)
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/motor/MSC02_parcel_corrmat.mat');
MSC02_motor=parcel_corrmat;
%MSC02_motor(isinf(MSC02_motor)|isnan(MSC02_motor)) = 0; % Replace NaNs and infinite values with zeros
idx_rand = randperm(20);
labels = [ones(10,1);-ones(10,1)];
%results=svm_scripts_beta(MSC02_mem_rest, labels(idx_rand),0,0,0,0)


results=svm_scripts_beta(MSC02_mem_rest, labels(idx_rand),0,MSC02_motor,[ones(10,1)],0)

mean((sum(results.predictedTestLabels(1:10,:)==1))./10)
