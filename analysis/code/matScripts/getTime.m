function getTime(sub)
filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/output/'
%load mem timeseries
    memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);
%load tmask
    mem_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mem_pass2/condindices.mat'];
    memTmask=load(mem_tmaskFile);
%.Allmem will be different for Mixed
%load motor timeseries
    motorFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_timecourse.mat'];
    motorFC=load(motorFile);
%load tmask
    motor_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_motor_pass2/condindices.mat'];
    motorTmask=load(motor_tmaskFile);
%load mixed timeseries
    mixedFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_timecourse.mat'];
    mixedFC=load(mixedFile);
%load tmask
    mixed_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mixed_pass2/condindices.mat'];
    mixedTmask=load(mixed_tmaskFile);
%load rest timeseries
    restFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_timecourse.mat'];
    restFC=load(restFile);
    %loop through all days
    samples = 10;
    motor=[];
    mem=[];
    rest=[];
    AllGlass=[];
    AllSemantic=[];
    %make list of times per day
    for day=1:samples
        mem_time = memFC.parcel_time{day}(logical(memTmask.TIndFin(day).AllMem),:);
        memtimeSlice=round(size(mem_time,1));
        mem=cat(1, mem, memtimeSlice);
        
        Glass_time = mixedFC.parcel_time{day}(logical(mixedTmask.TIndFin(day).AllGlass),:);
        glasstimeSlice=round(size(Glass_time,1));
        AllGlass=cat(1, AllGlass, glasstimeSlice);
        
        Semantic_time= mixedFC.parcel_time{day}(logical(mixedTmask.TIndFin(day).AllSemantic),:);
        semtimeSlice=round(size(Semantic_time,1));
        AllSemantic=cat(1, AllSemantic, semtimeSlice);
        
        rest_time = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        resttimeSlice=round(size(rest_time,1));
        rest=cat(1, rest, resttimeSlice);
        
        mot_time = motorFC.parcel_time{day}(logical(motorTmask.TIndFin(day).AllMotor),:);
        mottimeSlice=round(size(mot_time,1));
        motor=cat(1, motor, mottimeSlice);
    end
    concat=[mem, AllGlass, AllSemantic, rest, motor];
    T = array2table(concat, 'VariableNames',{'mem','AllGlass','AllSemantic', 'rest','motor'});
    task_file=strcat(sub, '.csv')
    sname=fullfile(filePath, task_file)
    writetable(T, sname, 'WriteRowNames', true)
end
