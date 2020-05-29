function minTime(sub)
%load meåm timeseries
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
    motorsamples = size(motorFC.parcel_time, 2);
    min=[];
    %make list of times per day
    for day=1:motorsamples
        motmasked_time = motorFC.parcel_time{day}(logical(motorTmask.TIndFin(day).AllMotor),:);
        if isempty(motmasked_time)
            continue
        end 
        mottimeSlice=round(size(motmasked_time,1));
        min=cat(1, min, mottimeSlice);
    end
    parcel_corrmat=[];
    minsamples = size(min, 1);
    %minimizing renaming functions
    %%%memory
    %{
    for i=1:minsamples
        task = memFC.parcel_time{i}(logical(memTmask.TIndFin(i).AllMem),:);
        if isempty(task)==1
            continue;
        end 
        timeMin=task(1:min(i), :);
        t=corr(timeMin);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/minQuality/mem/', sub, '_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
    
    %%%%motor
    
    for i=1:minsamples
        task = motorFC.parcel_time{i}(logical(motorTmask.TIndFin(i).AllMotor),:);
        if isempty(task)
            continue;
        end 
        %timeMin=task(1:min(i), :);
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/minQuality/motor/', sub, '_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
   
    
    %%%%rest
    for i=1:minsamples
        task = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
        if isempty(task)==1
            continue;
        end 
        timeMin=task(1:min(i), :);
        t=corr(timeMin);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/minQuality/rest/', sub, '_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
    
    %}
    %%%%mixed
    
    for i=1:minsamples
        Glass = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllGlass),:);
        if isempty(Glass)==1
            continue;
        end 
        Gmin=round(size(Glass,1));
        if Gmin<min(i)
            timeMin=Glass;
        else 
            timeMin=Glass(1:min(i), :);
        end
        C_Glass=corr(timeMin);
        Semantic= mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllSemantic),:);
        Smin=round(size(Semantic,1));
   
        if Smin<min(i)
            timeMin2=Semantic;
        else
            timeMin2=Semantic(1:min(i), :);
        end
        C_Semantic=corr(timeMin2);
        t=corr(C_Glass, C_Semantic);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/minQuality/mixed/', sub, '_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
    %}
end 