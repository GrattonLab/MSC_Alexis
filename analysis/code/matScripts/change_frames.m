function change_frames(sub)
    memFile=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);
    frames={5,10,15,20,25,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,325,350};    
    nsamples = size(memFC.parcel_time, 2);
    for f=1:length(frames)
        parcel_corrmat=[];
        for day=1:nsamples
            task = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
            if isempty(task)==1
                continue;
            elseif round(size(task,1))<frames{f}
                continue; 
            end 
            task_min=task(1:frames{f}, :);
            t=corr(task_min);
            zt=atanh(t);
            parcel_corrmat=cat(3, parcel_corrmat, zt);
        end 
    
    saveName=[strcat('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/tmask_frames/mem/',num2str(frames{f}),'/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear parcel_corrmat task task_min t zt;
    end
    