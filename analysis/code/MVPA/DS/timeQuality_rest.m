function timeQuality_rest(sub)
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
%after this masked time find time and match to rest
%loop through all days
    rest_memFC=[]
    nsamples = size(memFC.parcel_time, 2);
    for i=1:nsamples
        masked_time = memFC.parcel_time{i}(logical(memTmask.TIndFin(i).AllMem),:);
        restmasked_time = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
%cut time in half
        timeSlice=round(size(masked_time,1));
        time1=restmasked_time(1:timeSlice, :);
        t1=corr(time1);
        zt1=atanh(t1);
        rest_memFC=cat(3, rest_memFC, zt1);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/timeQuality_rest/', sub, '_mem_parcel_corrmat.mat')]
    save(saveName, 'rest_memFC')
%now for mixed
%all Glass
%loop through all days
    rest_glassFC=[]
    mixsamples = size(mixedFC.parcel_time, 2);
    for i=1:mixsamples
        Glassmasked_time = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllGlass),:);
        restmasked_time = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
        if isempty(Glassmasked_time)
            continue
        end 
%cut time in half
        GtimeSlice=round(size(Glassmasked_time,1));
        Gtime1=restmasked_time(1:GtimeSlice, :);
        Gt1=corr(Gtime1);
        zGt1=atanh(Gt1);
        rest_glassFC=cat(3, rest_glassFC, zGt1);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/timeQuality_rest/', sub, '_AllGlass_parcel_corrmat.mat')]
    save(saveName, 'rest_glassFC')
%All semantic
%loop through all days
    rest_semFC=[]
    for i=1:mixsamples
        Semmasked_time = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllSemantic),:);
        restmasked_time = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
        %cut time in half
        SemtimeSlice=round(size(Semmasked_time,1));
        Stime1=restmasked_time(1:SemtimeSlice, :);
        St1=corr(Stime1);
        zSt1=atanh(St1);
        rest_semFC=cat(3, rest_semFC, zSt1);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/timeQuality_rest/', sub, '_AllSemantic_parcel_corrmat.mat')]
    save(saveName, 'rest_semFC')
%now for motor
%loop through all days
    rest_motorFC=[]
    motorsamples = size(motorFC.parcel_time, 2);
    for i=1:motorsamples
        motmasked_time = motorFC.parcel_time{i}(logical(motorTmask.TIndFin(i).AllMotor),:);
        restmasked_time = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
        if isempty(motmasked_time)
            continue
        end 
%cut time in half
        mottimeSlice=round(size(motmasked_time,1));
        Mtime1=restmasked_time(1:mottimeSlice, :);
        Mt1=corr(Mtime1);
        zMt1=atanh(Mt1);
        rest_motorFC=cat(3, rest_motorFC, zMt1);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/timeQuality_rest/', sub, '_motor_parcel_corrmat.mat')]
    save(saveName, 'rest_motorFC')
end 