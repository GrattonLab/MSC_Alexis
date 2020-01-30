function LO1DAY(sub)
    %open all the relevant files
    %motor
    motorFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_corrmat.mat'];
    motFile=load(motorFC);
    motor=motFile.parcel_corrmat;
    %memory
    memoryFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_corrmat.mat'];
    memFile=load(memoryFC);
    mem=memFile.parcel_corrmat;
    %mixed
    mixedFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_corrmat.mat'];
    mixFile=load(mixedFC);
    mixed=mixFile.parcel_corrmat;
    %rest
    restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat'];
    restFile=load(restFC);
    rest=restFile.parcel_corrmat;
    taskList={motor,mem,mixed};
    myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/'; %defining working directory
    
    for i=1:length(taskList)
        taskFC=taskList{i};
        restFC=rest;
        %select good days
        good_task = ~isnan(squeeze(sum(sum(taskFC,2),1)))
        good_rest = ~isnan(squeeze(sum(sum(restFC,2),1)))
        
        %select train and test indices
        nsamples = size(taskFC,3);
        sample_inds = [1:nsamples];
        for t = 1:nsamples
            test_ind = t;
            train_ind = setdiff(set(sample_inds),set(test_ind));
            %leaving a day out to use for testing later
            rest_train=r(:,:,train_ind);
            task_train=t(:,:,train_ind);
            rest_test=r(:,:,test_ind);
            task_test=t(:,:,test_ind);
            train=cat(3, task_train, rest_train);
            test=cat(3, task_test, rest_test);
            %training across 9 days testing on one day
            results=svm_scripts_beta(train, [ones(9,1); -ones(9,1)],0,test,[ones(length(test_set),1); -ones(length(test_set),1)],0);
            mean((sum(results.predictedTestLabels(1,:)==1)+sum(results.predictedTestLabels(2,:)==-1))./2)
            saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/results_mat/' taskList{i} sub '.mat']
            save(saveName, 'results')
    end
end 
