function timeseries_efficiency(sub, task, howSplit, AllTask)
%load task timeseries
    filePath='/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data';
    timeFile='_parcel_timecourse.mat';
    saveFile=strcat(sub, '_parcel_corrmat.mat');
    subFile=strcat(sub, timeFile);
    taskFC=fullfile(filePath, task, subFile);
    t=load(taskFC);
%load tmask
    tmaskPath='/Users/aporter1350/Box/Quest_Backup/MSC/TaskFC/';
    sub_tmask=strcat('FCProc_', sub, '_', task, '_pass2');
    condition='condindices.mat';
    tmaskFile=fullfile(tmaskPath, sub_tmask, condition);
    tmask=load(tmaskFile);
    parcel_corrmat=[];
    nsamples = size(t.parcel_time, 2);
    %how to split depending on amount of split
    if contains('half', howSplit)
        for i=1:nsamples
            masked_time = t.parcel_time{i}(logical(tmask.TIndFin(i).(AllTask)),:);
%cut time in half
            timeSlice=round(size(masked_time,1)/2);
            time1=masked_time(1:timeSlice, :);
            time2=masked_time(timeSlice:end, :);
            t1=corr(time1);
            zt1=atanh(t1);
            t2=corr(time2);
            zt2=atanh(t2);
            corrs=cat(3, zt1, zt2);
            parcel_corrmat=cat(3, parcel_corrmat, corrs);
        end
        if contains('mixed', task)
            saveName=fullfile(filePath, task, 'corrmats_timesplit', 'half', AllTask, saveFile);
            save(saveName, 'parcel_corrmat')
        else 
            saveName=fullfile(filePath, task, 'corrmats_timesplit', 'half', saveFile);
            save(saveName, 'parcel_corrmat');
        end
    end
    if contains('thirds', howSplit)
        for i=1:nsamples
            masked_time = t.parcel_time{i}(logical(tmask.TIndFin(i).(AllTask)),:);
            timeSlice=round(size(masked_time,1)/3);
            time1=masked_time(1:timeSlice, :);
            time2=masked_time(timeSlice:timeSlice*2, :);
            time3=masked_time(timeSlice*2:end, :);
            t1=corr(time1);
            zt1=atanh(t1);
            t2=corr(time2);
            zt2=atanh(t2);
            t3=corr(time3);
            zt3=atanh(t3);
            corrs=cat(3, zt1, zt2, zt3);
            parcel_corrmat=cat(3, parcel_corrmat, corrs);
        end
            if contains('mixed', task)
            saveName=fullfile(filePath, task, 'corrmats_timesplit', howSplit, AllTask, saveFile);
            save(saveName, 'parcel_corrmat')
            else 
            saveName=fullfile(filePath, task, 'corrmats_timesplit', howSplit, saveFile);
            save(saveName, 'parcel_corrmat');
            end 
    end
end 