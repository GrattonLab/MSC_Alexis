function hyper_all()    

%
    trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    taskList={'mem','mixed','motor','rest'};
    % load the data into a struct containing all subjest task and rest
 
task=1
i=1
filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data';
parcelFile='_parcel_timecourse.mat';
subFile=strcat(taskList{task}, '/', trainList{i}, parcelFile);
taskFC=fullfile(filePath,subFile);
t=load(taskFC);
nsamples=size(t.parcel_time,2);
for day=1:nsamples
    time=t.parcel_time{day}(logical(t.tmask_all{day}),:);
    if ~isempty(time)
        t_cell{day}=time;
    end   
end
mem=t_cell(~cellfun('isempty',t_cell));
clear task
task=4

subFile=strcat(taskList{task}, '/', trainList{i}, parcelFile);
taskFC=fullfile(filePath,subFile);
t=load(taskFC);
nsamples=size(t.parcel_time,2);
for day=1:nsamples
    time=t.parcel_time{day}(logical(t.tmask_all{day}),:);
    if ~isempty(time)
        t_cell{day}=time;
    end   
end
rest=t_cell(~cellfun('isempty',t_cell));        
            
aligned=hyperalign(mem{:});
asample=size(aligned,2);
parcel_corrmat=[];
for d=1:asample
    reformated_task=aligned{1,d};
    if isempty(reformated_task)==1
        continue;
    end
    rt=corr(reformated_task);
    zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, zt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/within_sub_hyperalign/',taskList{task}, '/', trainList{i}, '_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear nsamples asample t_cell parcel_corrmat t zt;
end 
    %end 
 
      %{  
    %aligned, transforms =hyperalign(t);
    memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/MSC01_parcel_timecourse.mat'];
    memFC=load(memFile);
    %memory
    nsamples = size(memFC.parcel_time, 2);
    for day=1:nsamples
        task = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        t_cell{day}=task;
        if isempty(task)==1
            continue;
        end 
        
        
    end 
    aligned=hyperalign(t_cell{:});
    parcel_corrmat=[]
    for d=1:nsamples
        reformated_task=aligned{:,:,d};
        rt=corr(reformated_task);
        zt=atanh(rt);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/within_sub_hyperalign/',task, '/', i, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task t zt;
    %}
%end