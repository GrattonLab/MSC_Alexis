function tmask_time(sub)
%determine frames for setting the minimum
    filePath='/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/time/tmask_all';
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
    samples = 10;
    motor=[];
    mem=[];
    rest=[];
    mixed=[]
    %make list of times per day
    for day=1:samples
        mem_time = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        memtimeSlice=round(size(mem_time,1));
        mem=cat(1, mem, memtimeSlice);

        mix_time= mixedFC.parcel_time{day}(logical(mixedFC.tmask_all{day}),:);
        mixtimeSlice=round(size(mix_time,1));
        mixed=cat(1, mixed, mixtimeSlice);
        
        rest_time = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        resttimeSlice=round(size(rest_time,1));
        rest=cat(1, rest, resttimeSlice);
        
        mot_time = motorFC.parcel_time{day}(logical(motorFC.tmask_all{day}),:);
        mottimeSlice=round(size(mot_time,1));
        motor=cat(1, motor, mottimeSlice);
    end
    concat=[mem, mixed, rest, motor];
    T = array2table(concat, 'VariableNames',{'mem','mixed', 'rest','motor'});
    task_file=strcat(sub, '.csv')
    sname=fullfile(filePath, task_file)
    writetable(T, sname, 'WriteRowNames', true)
end

