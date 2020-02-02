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
        train=cat(3, taskFC_clean, restFC_clean); 
        results=svm_scripts_beta(train, [ones(size(taskFC_clean,3),1); -ones(size(restFC_clean,3),1)],0,0,0,1); %to arrange in pairs options=1
        saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/results_mat/', taskListNames{i}, sub, '.mat')]
        save(saveName, 'results')
    end
end 
