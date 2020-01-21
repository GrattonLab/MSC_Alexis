function LO1DAY(sub)
    taskList={'mem','motor','mixed'}
    myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' %defining working directory
    for i=1:3
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' taskList{i} '/' sub '_parcel_corrmat.mat']
        load(taskFC)
        t=parcel_corrmat;
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat']
        load(restFC)
        r=parcel_corrmat;
%concatenating memory and rest along third dimension first 10 days are task
%then rest
%leaving a day out to use for testing later
    rest_train=r(:,:,2:end)
    task_train=t(:,:,2:end)
    rest_test=r(:,:,1)
    task_test=t(:,:,1)
    train=cat(3, task_train, rest_train)
    test=cat(3, task_test, rest_test)
%training across 9 days testing on one day
    results=svm_scripts_beta(train, [ones(9,1); -ones(9,1)],0,test,[ones(1,1); -ones(1,1)],0)
    mean((sum(results.predictedTestLabels(1,:)==1)+sum(results.predictedTestLabels(2,:)==-1))./2)
    saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/results_mat/' taskList{i} sub '.mat']
    save(saveName, 'results')
    end 
end 
