function memPass(sub)
%load meåm timeseries
    memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);
%load tmask
    mem_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mem_pass2_FDfilt/condindices.mat'];
    memTmask=load(mem_tmaskFile);
%.Allmem will be different for Mixed
    
    %%%memory
    nsamples = size(memFC.parcel_time, 2);
    parcel_corrmat=[];
    for i=1:nsamples
        task = memFC.parcel_time{i}(logical(memTmask.TIndFin(i).pres1),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/pres1/', sub, '_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
  
end 