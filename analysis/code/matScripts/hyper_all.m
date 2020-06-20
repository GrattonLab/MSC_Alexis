function hyper_all()    


trainList={'MSC02','MSC04','MSC05','MSC06','MSC07'};


 

filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data';
parcelFile='_parcel_timecourse.mat';
for sub=1:length(trainList)
    memFile=strcat('mem/', trainList{sub}, parcelFile);
    memFC=fullfile(filePath,memFile);
    mem=load(memFC);
    nsamples=size(mem.parcel_time,2);
    for day=1:nsamples
        time=mem.parcel_time{day}(logical(mem.tmask_all{day}),:);
        if ~isempty(time)
            cell{day}=time;
        end   
    end
    mem_cell=cell(~cellfun('isempty',cell));
    clear nsamples cell time
    mixFile=strcat('mixed/', trainList{sub}, parcelFile);
    mixFC=fullfile(filePath,mixFile);
    mixed=load(mixFC);
    nsamples=size(mixed.parcel_time,2);
    for day=1:nsamples
        time=mixed.parcel_time{day}(logical(mixed.tmask_all{day}),:);
        if ~isempty(time)
            cell{day}=time;
        end   
    end
    mix_cell=cell(~cellfun('isempty',cell));
%{
    clear nsamples cell time
    motorFile=strcat('motor/', trainList{sub}, parcelFile);
    motorFC=fullfile(filePath,motorFile);
    motor=load(motorFC);
    nsamples=size(motor.parcel_time,2);
    for day=1:nsamples
        time=motor.parcel_time{day}(logical(motor.tmask_all{day}),:);
        if ~isempty(time)
            cell{day}=time;
        end   
    end
    motor_cell=cell(~cellfun('isempty',cell));
%}         
    
    clear nsamples cell time
    restFile=strcat('rest/', trainList{sub}, parcelFile);
    restFC=fullfile(filePath,restFile);
    rest=load(restFC);
    nsamples=size(rest.parcel_time,2);
    for day=1:nsamples
        time=rest.parcel_time{day}(logical(rest.tmask_all{day}),:);
        if ~isempty(time)
            cell{day}=time;
        end   
    end
    rest_cell=cell(~cellfun('isempty',cell));
%now that you have all your time series 
    memSize=size(mem_cell,2);
    mixSize=size(mix_cell,2);
    %motorSize=size(motor_cell,2);
    restSize=size(rest_cell,2);
    aligned=hyperalign(mem_cell{:},mix_cell{:},rest_cell{:});%,motor_cell{:},rest_cell{:});
    
    memT=aligned(1,1:memSize);
    mixT=aligned(1, memSize+1:memSize+mixSize);
    motorT=aligned(1,memSize+mixSize+1:memSize+mixSize+motorSize);
    restT=aligned(1, memSize+mixSize+motorSize+1:memSize+mixSize+motorSize+restSize);
    
    parcel_corrmat=[];
    for d=1:memSize
        mem_reformat=memT{1,d};
        if isempty(mem_reformat)==1
            continue;
        end
        rt=corr(mem_reformat);
        zt=atanh(rt);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_task_within_sub_hyperalign/mem/', trainList{sub}, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear  parcel_corrmat rt;
    
    parcel_corrmat=[];
    for d=1:mixSize
        mix_reformat=mixT{1,d};
        if isempty(mix_reformat)==1
            continue;
        end
        rt=corr(mix_reformat);
        zt=atanh(rt);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_task_within_sub_hyperalign/mixed/', trainList{sub}, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear  parcel_corrmat rt;
    
    parcel_corrmat=[];
    for d=1:motorSize
        motor_reformat=motorT{1,d};
        if isempty(motor_reformat)==1
            continue;
        end
        rt=corr(motor_reformat);
        zt=atanh(rt);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_task_within_sub_hyperalign/motor/', trainList{sub}, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear  parcel_corrmat rt;
    
    parcel_corrmat=[];
    for d=1:restSize
        rest_reformat=restT{1,d};
        if isempty(rest_reformat)==1
            continue;
        end
        rt=corr(rest_reformat);
        zt=atanh(rt);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_task_within_sub_hyperalign/rest/', trainList{sub}, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear  parcel_corrmat rt;
end 
end 
