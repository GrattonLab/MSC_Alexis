function tmask_all(sub)
filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/frames';
%load mem timeseries
    memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);

%load motor timeseries
    motorFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_timecourse.mat'];
    motorFC=load(motorFile);

%load mixed timeseries
    mixedFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_timecourse.mat'];
    mixedFC=load(mixedFile);

%load rest timeseries
    restFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_timecourse.mat'];
    restFC=load(restFile);
    %loop through all days
    %samples = 10;
    %motor=[];
    %mem=[];
    %rest=[];
    %mixed=[];
    
    %memory
    nsamples = size(memFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/tmask_all/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task t zt;
    
    %mixed
    nsamples = size(mixedFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = mixedFC.parcel_time{day}(logical(mixedFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/tmask_all/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task t zt;

    %motor
    nsamples = size(motorFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = motorFC.parcel_time{day}(logical(motorFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/tmask_all/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task t zt;
    
    %rest
    nsamples = size(restFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/tmask_all/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    %{
    %make list of times per day
    for day=1:samples
        mem_time = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        memtimeSlice=round(size(mem_time,1));
        mem=cat(1, mem, memtimeSlice);
        
        mixed_time = mixedFC.parcel_time{day}(logical(mixedFC.tmask_all{day}),:);
        mixedtimeSlice=round(size(mixed_time,1));
        mixed=cat(1, mixed, mixedtimeSlice);
        
        rest_time = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        resttimeSlice=round(size(rest_time,1));
        rest=cat(1, rest, resttimeSlice);
        
        mot_time = motorFC.parcel_time{day}(logical(motorFC.tmask_all{day}),:);
        mottimeSlice=round(size(mot_time,1));
        motor=cat(1, motor, mottimeSlice);
    end
    concat=[mem, mixed, rest, motor];
    T = array2table(concat, 'VariableNames',{'mem','mixed', 'rest','motor'});
    task_file=strcat(sub, '.csv');
    sname=fullfile(filePath, task_file);
    writetable(T, sname, 'WriteRowNames', true)
    %}
end
