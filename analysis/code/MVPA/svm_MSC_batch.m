load('/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/mem/MSC02_parcel_corrmat.mat');
MSC02_mem=parcel_corrmat;
load('/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/rest/MSC02_parcel_corrmat.mat');
MSC02_rest=parcel_corrmat;
%MSC02_mem(isinf(MSC02_mem)|isnan(MSC02_mem)) = 0; % Replace NaNs and infinite values with zeros
%MSC02_rest(isinf(MSC02_rest)|isnan(MSC02_rest)) = 0; % Replace NaNs and infinite values with zeros
%concatenating memory and rest along third dimension first 10 days are task
%then rest
MSC02_mem_rest=cat(3, MSC02_mem, MSC02_rest)

%test along different subject
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/mem/MSC06_parcel_corrmat.mat');
MSC03_mem=parcel_corrmat;
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/rest/MSC06_parcel_corrmat.mat');
MSC03_rest=parcel_corrmat;
%MSC03_mem(isinf(MSC03_mem)|isnan(MSC03_mem)) = 0; % Replace NaNs and infinite values with zeros
%MSC03_rest(isinf(MSC03_rest)|isnan(MSC03_rest)) = 0; % Replace NaNs and infinite values with zeros
%concatenating memory and rest along third dimension first 10 days are task
%then rest
MSC03_mem_rest=cat(3, MSC03_mem, MSC03_rest)


%svm script
results=svm_scripts_beta(MSC02_mem_rest, [ones(10,1); -ones(10,1)],0,0,0,0)
%svm script with test set being another sub
results=svm_scripts_beta(MSC02_mem_rest, [ones(10,1); -ones(10,1)],0,MSC03_mem_rest,[ones(10,1); -ones(10,1)],0)


%msc03
results=svm_scripts_beta(MSC03_mem_rest, [ones(10,1); -ones(10,1)],0,0,0,0)

%svm script with test set being another task
load('/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/motor/MSC02_parcel_corrmat.mat');
MSC02_motor=parcel_corrmat;
%MSC02_motor(isinf(MSC02_motor)|isnan(MSC02_motor)) = 0; % Replace NaNs and infinite values with zeros

results=svm_scripts_beta(MSC02_mem_rest, [ones(10,1); -ones(10,1)],0,MSC02_motor,[ones(10,1)],0)



idx_rand = randperm(20);
results=svm_scripts_beta(MSC03_mem_rest, labels(idx_rand),0,0,0,0)
results=svm_scripts_beta(MSC03_mem_rest, [ones(10,1); -ones(10,1)],[200:200:1000],0,0,0)
mean((sum(results.predictedTestLabels(1:10,:)==1)+sum(results.predictedTestLabels(11:20,:)==-1))./20)
