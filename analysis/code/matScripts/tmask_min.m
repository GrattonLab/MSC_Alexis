function tmask_min(sub)
filePath='/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/results/frames';
%load mem timeseries
    memFile=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);

%load motor timeseries
    motorFile=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_timecourse.mat'];
    motorFC=load(motorFile);

%load mixed timeseries
    mixedFile=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_timecourse.mat'];
    mixedFC=load(mixedFile);

%load rest timeseries
    restFile=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_timecourse.mat'];
    restFC=load(restFile);
    %loop through all days
    %memory
    nsamples = size(memFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        task_min=task(1:54, :);
        t=corr(task_min);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/tmask_min/mem/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task task_min t zt;
    
    %mixed
    nsamples = size(mixedFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = mixedFC.parcel_time{day}(logical(mixedFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        task_min=task(1:54, :);
        t=corr(task_min);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/tmask_min/mixed/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task task_min t zt;

    %motor
    nsamples = size(motorFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = motorFC.parcel_time{day}(logical(motorFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        task_min=task(1:54, :);
        t=corr(task_min);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/tmask_min/motor/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task task_min t zt;
    
    %rest
    nsamples = size(restFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        task_min=task(1:54, :);
        t=corr(task_min);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/tmask_min/rest/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
   
        
  
end
