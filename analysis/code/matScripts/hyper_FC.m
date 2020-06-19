function hyper_FC()    

%only subs with all 10 days and above 109 frames of data
%not using atanh for this analysis
%trainList={'MSC02','MSC04','MSC05','MSC07'};


 

filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data';
parcelFile='_parcel_corrmat.mat';
%mem
memFile=strcat('mem/MSC02', parcelFile);
memFC=fullfile(filePath,memFile);
mem02=load(memFC);
clear memFile memFC
memFile=strcat('mem/MSC04', parcelFile);
memFC=fullfile(filePath,memFile);
mem04=load(memFC);
clear memFile memFC
memFile=strcat('mem/MSC05', parcelFile);
memFC=fullfile(filePath,memFile);
mem05=load(memFC);
clear memFile memFC
memFile=strcat('mem/MSC07', parcelFile);
memFC=fullfile(filePath,memFile);
mem07=load(memFC);
nsamples=10;
for day=1:nsamples
    cell02{day}=mem02.parcel_corrmat(:,:,day); 
    cell04{day}=mem04.parcel_corrmat(:,:,day); 
    cell05{day}=mem05.parcel_corrmat(:,:,day);  
    cell07{day}=mem07.parcel_corrmat(:,:,day); 
end
mem_cell=[cell02,cell04,cell05,cell07];
clear cell02 cell04 cell05 cell07
%mixed
mixedFile=strcat('mixed/MSC02', parcelFile);
mixedFC=fullfile(filePath,mixedFile);
mixed02=load(mixedFC);
clear mixedFile mixedFC
mixedFile=strcat('mixed/MSC04', parcelFile);
mixedFC=fullfile(filePath,mixedFile);
mixed04=load(mixedFC);
clear mixedFile mixedFC
mixedFile=strcat('mixed/MSC05', parcelFile);
mixedFC=fullfile(filePath,mixedFile);
mixed05=load(mixedFC);
clear mixedFile mixedFC
mixedFile=strcat('mixed/MSC07', parcelFile);
mixedFC=fullfile(filePath,mixedFile);
mixed07=load(mixedFC);

for day=1:nsamples
    cell02{day}=mixed02.parcel_time{day}(logical(mixed02.tmask_all{day}),:); 
    cell04{day}=mixed04.parcel_time{day}(logical(mixed04.tmask_all{day}),:); 
    cell05{day}=mixed05.parcel_time{day}(logical(mixed05.tmask_all{day}),:); 
    cell07{day}=mixed07.parcel_time{day}(logical(mixed07.tmask_all{day}),:); 
end
mix_cell=[cell02,cell04,cell05,cell07];

clear cell02 cell04 cell05 cell07
%rest
restFile=strcat('rest/MSC02', parcelFile);
restFC=fullfile(filePath,restFile);
rest02=load(restFC);
clear restFile restFC
restFile=strcat('rest/MSC04', parcelFile);
restFC=fullfile(filePath,restFile);
rest04=load(restFC);
clear restFile restFC
restFile=strcat('rest/MSC05', parcelFile);
restFC=fullfile(filePath,restFile);
rest05=load(restFC);
clear restFile restFC
restFile=strcat('rest/MSC07', parcelFile);
restFC=fullfile(filePath,restFile);
rest07=load(restFC);

for day=1:nsamples
    cell02{day}=rest02.parcel_time{day}(logical(rest02.tmask_all{day}),:); 
    cell04{day}=rest04.parcel_time{day}(logical(rest04.tmask_all{day}),:); 
    cell05{day}=rest05.parcel_time{day}(logical(rest05.tmask_all{day}),:); 
    cell07{day}=rest07.parcel_time{day}(logical(rest07.tmask_all{day}),:); 
end
rest_cell=[cell02,cell04,cell05,cell07];

%now that you have all your time series 
aligned=hyperalign(mem_cell{:},mix_cell{:},rest_cell{:});
    
memT=aligned(1,1:40);
mixT=aligned(1, 41:80);
restT=aligned(1, 81:120);
    
    
mem02=memT(1, 1:10);
mem04=memT(1, 11:20);
mem05=memT(1, 21:30);
mem07=memT(1, 31:40);
parcel_corrmat=[];
for d=1:nsamples
    reformat=mem02{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mem/MSC02_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;
    
parcel_corrmat=[];
for d=1:nsamples
    reformat=mem04{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mem/MSC04_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;

parcel_corrmat=[];
for d=1:nsamples
    reformat=mem05{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mem/MSC05_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;
    
parcel_corrmat=[];
for d=1:nsamples
    reformat=mem07{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mem/MSC07_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
mix02=mixT(1, 1:10);
mix04=mixT(1, 11:20);
mix05=mixT(1, 21:30);
mix07=mixT(1, 31:40);
parcel_corrmat=[];
for d=1:nsamples
    reformat=mix02{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mixed/MSC02_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;
    
parcel_corrmat=[];
for d=1:nsamples
    reformat=mix04{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mixed/MSC04_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;

parcel_corrmat=[];
for d=1:nsamples
    reformat=mix05{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mixed/MSC05_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;
    
parcel_corrmat=[];
for d=1:nsamples
    reformat=mix07{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/mixed/MSC07_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
rest02=restT(1, 1:10);
rest04=restT(1, 11:20);
rest05=restT(1, 21:30);
rest07=restT(1, 31:40);
parcel_corrmat=[];
for d=1:nsamples
    reformat=rest02{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/rest/MSC02_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;
    
parcel_corrmat=[];
for d=1:nsamples
    reformat=rest04{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/rest/MSC04_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;

parcel_corrmat=[];
for d=1:nsamples
    reformat=rest05{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/rest/MSC05_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;
    
parcel_corrmat=[];
for d=1:nsamples
    reformat=rest07{1,d};
    rt=corr(reformat);
    %zt=atanh(rt);
    parcel_corrmat=cat(3, parcel_corrmat, rt);
end 
saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign/rest/MSC07_parcel_corrmat.mat')];
save(saveName, 'parcel_corrmat');
clear  parcel_corrmat rt reformat;

end 
