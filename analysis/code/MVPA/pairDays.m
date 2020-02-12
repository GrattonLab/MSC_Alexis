function pairDays(sub)
    %open all the relevant files
    %motor
    %motorFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_corrmat.mat'];
    motorFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/corrmats_timesplit/' sub '_parcel_corrmat.mat'];
    motFile=load(motorFC);
    motor=motFile.motorFC_all;
    %memory
    %memoryFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_corrmat.mat'];
    memoryFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/corrmats_timesplit/' sub '_parcel_corrmat.mat'];
    memFile=load(memoryFC);
    mem=memFile.memFC_all;
    %mixed
    %mixedFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_corrmat.mat'];
    %mixFile=load(mixedFC);
    %mixed=mixFile.parcel_corrmat;
    
    GlassmixedFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/corrmats_timesplit/' sub '_AllGlass_parcel_corrmat.mat'];
    GlassmixFile=load(GlassmixedFC);
    glass=GlassmixFile.glassFC_all;
    
    SemmixedFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/corrmats_timesplit/' sub '_AllSemantic_parcel_corrmat.mat'];
    SemmixFile=load(SemmixedFC);
    sem=SemmixFile.semFC_all;
    
    %rest
    %restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat'];
    restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/corrmats_timesplit/' sub '_parcel_corrmat.mat'];
    restFile=load(restFC);
    rest=restFile.restFC_all;
    taskList={motor,mem, glass, sem};
    taskListNames = {'motor', 'mem', 'glass', 'sem'}
    %myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/'; %defining working directory
    
    for i=1:length(taskList)
        taskFC=taskList{i};
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
        saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/timesplit_mat/', taskListNames{i}, sub, '.mat')]
        save(saveName, 'results')
end 
