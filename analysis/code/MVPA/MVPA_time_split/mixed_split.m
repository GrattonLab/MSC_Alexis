function mixed_split(sub)
%load mixed timeseries
    mixedFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_timecourse.mat'];
    mixedFC=load(mixedFile);
%load tmask
    mixed_tmaskFile=['~/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mixed_pass2/condindices.mat'];
    mixedTmask=load(mixed_tmaskFile);

%after this masked time split up into pieces then run as corr
%loop through all days
    parcel_corrmat=[]
    mixsamples = size(mixedFC.parcel_time, 2);
    for i=1:mixsamples
        Semmasked_time = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllSemantic),:);
        %Glassmasked_time = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllGlass),:);
        if isempty(Semmasked_time)
            continue
        end 
        Gt1=corr(Semmasked_time);
        zGt1=atanh(Gt1);
        parcel_corrmat=cat(3, parcel_corrmat, zGt1);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/', sub, '_AllSemantic_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
end 