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
    taskListNames = {'motor', 'mem', 'mixed'}
    %myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/'; %defining working directory
    
    for i=1:length(taskList)
        taskFC=motor;
        restFC=rest;
        %select good days
        good_task = ~isnan(squeeze(sum(sum(taskFC,2),1)));
        good_rest = ~isnan(squeeze(sum(sum(restFC,2),1)));
        only_good = logical(good_task .* good_rest);
        taskFC_clean = taskFC(:,:,only_good);
        restFC_clean= restFC(:,:, only_good);
        %subjects 2, 5, 6, then look at 1 or 4 based on nans, 7, 8, 9
        %exclude, 3, 10 weird
        %select train and test indices
        nsamples = size(taskFC_clean, 3)
        sample_inds = [1:nsamples];
        %labels = eye(10) would need 10 columns but 40 rows to use all task
        %data 
        %labels(labels==0)=-1
        %labels = [labels;labels]
        %results.predictedLabels==[[1:10]';[1:10]']
        %results=svm_scripts_beta(t, labels,0,0,0,0);
        for t = 1:nsamples
            test_ind = t;
            train_ind = setdiff((sample_inds),(test_ind));
            %leaving a day out to use for testing later
            rest_train=restFC_clean(:,:,train_ind);
            task_train=taskFC_clean(:,:,train_ind);
            rest_test=restFC_clean(:,:,test_ind);
            task_test=taskFC_clean(:,:,test_ind);
            train=cat(3, task_train, rest_train); 
            test=cat(3, task_test, rest_test);
            %training across 9 days testing on one day
            results=svm_scripts_beta(train, [ones(size(task_train,3),1); -ones(size(rest_train,3),1)],0,test,[ones(size(task_test,3),1); -ones(size(rest_test,3),1)],0); %to arrange in pairs options=1
            (sum(results.predictedTestLabels(1,:)==1)+sum(results.predictedTestLabels(2,:)==-1))./numel(results.predictedTestLabels);   
            saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/results_mat/day_', num2str(test_ind), taskListNames{i}, sub, '.mat')]
            save(saveName, 'results')
        end 
    end
end 
