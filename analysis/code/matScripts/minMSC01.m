%load mem timeseries
memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/MSC05_parcel_timecourse.mat'];
memFC=load(memFile);
%load tmask
mem_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_MSC05_mem_pass2/condindices.mat'];
memTmask=load(mem_tmaskFile);
%.Allmem will be different for Mixed
%load motor timeseries
motorFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/MSC05_parcel_timecourse.mat'];
motorFC=load(motorFile);
%load tmask
motor_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_MSC05_motor_pass2/condindices.mat'];
motorTmask=load(motor_tmaskFile);

%load rest timeseries
restFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/MSC05_parcel_timecourse.mat'];
restFC=load(restFile);

parcel_corrmat=[];
nsamples = size(memFC.parcel_time, 2);
%memory
for i=1:nsamples
    task = memFC.parcel_time{i}(logical(memTmask.TIndFin(i).AllMem),:);
    if isempty(task)==1
        continue;
    end 
    timeMin=task(1:62, :);
    t=corr(timeMin);
    zt=atanh(t);
    parcel_corrmat=cat(3, parcel_corrmat, zt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/min/MSC05_parcel_corrmat.mat')]
save(saveName, 'parcel_corrmat')

clear parcel_corrmat nsamples task

parcel_corrmat=[];
nsamples = size(motorFC.parcel_time, 2);
%motor
for i=1:nsamples
    task = motorFC.parcel_time{i}(logical(motorTmask.TIndFin(i).AllMotor),:);
    if isempty(task)==1
        continue;
    end 
    timeMin=task(1:62, :);
    t=corr(timeMin);
    zt=atanh(t);
    parcel_corrmat=cat(3, parcel_corrmat, zt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/min/MSC05_parcel_corrmat.mat')]
save(saveName, 'parcel_corrmat')
clear parcel_corrmat nsamples task

parcel_corrmat=[];
nsamples = size(restFC.parcel_time, 2);
%%%%rest
for i=1:nsamples
    task = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
    if isempty(task)==1
        continue;
    end 
    timeMin=task(1:62, :);
    t=corr(timeMin);
    zt=atanh(t);
    parcel_corrmat=cat(3, parcel_corrmat, zt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/min/MSC05_parcel_corrmat.mat')]
save(saveName, 'parcel_corrmat')

