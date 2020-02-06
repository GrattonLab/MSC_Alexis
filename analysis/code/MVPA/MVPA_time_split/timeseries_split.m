function timeseries_split(sub)
%load timeseries
     memFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
     memFC=load(memFile);
%load tmask
    mem_tmaskFile=['~/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mem_pass2/condindices.mat'];
    memTmask=load(mem_tmaskFile);
%.Allmem will be different for Mixed
%masked time for day 1
%after this masked time split up into pieces then run as corr
%loop through all days
    nsamples = size(memFC.parcel_time, 2);
    for i=1:nsamples
        masked_time = memFC.parcel_time{i}(logical(memTmask.TIndFin(i).AllMem),:);
%cut time in half
        timeSlice=round(size(masked_time,1)/2);
        time1=masked_time(1:timeSlice, :);
        time2=masked_time(timeSlice:end, :);
end 