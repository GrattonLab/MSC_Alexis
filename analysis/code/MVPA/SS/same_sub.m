%training 
load('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/MSC02_parcel_corrmat.mat');
MSC02_mem=parcel_corrmat;
load('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/MSC02_parcel_corrmat.mat');
MSC02_rest=parcel_corrmat;
%concatenating memory and rest along third dimension first 10 days are task
%then rest
MSC02_mem_rest=cat(3, MSC02_mem, MSC02_rest);

%svm script with test set using all tasks 
load('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/MSC02_parcel_corrmat.mat');
MSC02_motor=parcel_corrmat;
load('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/MSC02_parcel_corrmat.mat');
MSC02_mixed=parcel_corrmat;

test=cat(3,MSC02_motor, MSC02_mixed);
results=svm_scripts_beta(MSC02_mem_rest, [ones(10,1); -ones(10,1)],0,test,[ones(20,1)],0)



mean((sum(results.predictedTestLabels(1:20,:)==1))./20)

save('foo.mat', 'results')
