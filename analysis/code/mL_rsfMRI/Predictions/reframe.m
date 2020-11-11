function parcel_corrmat=reframe(sub,frame,task)
    memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);    
    nsamples = size(memFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        elseif round(size(task,1))<frame
            continue; 
        end 
        %task_min=task(1:frames{f}, :);
        %this will have to change to some combination of the two where its
        %in chunks but randomely 
        task_min=datasample(task,frame,'Replace',false);
        t=corr(task_min);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
     end 
end
