function timeseries_fourth(sub)
%load rest timeseries
    restFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_timecourse.mat'];
    restFC=load(restFile);
%after this masked time split up into pieces then run as corr
%rest
%loop through all days
    parcel_corrmat=[];
    restsamples = size(restFC.parcel_time, 2);
    for i=1:restsamples
        restmasked_time = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
%cut time
        
        rtimeSlice1=round(size(restmasked_time,1)/4);
        rtimeSlice2=rtimeSlice1*2;
        rtimeSlice3=rtimeSlice2*2;
        resttime1=restmasked_time(1:rtimeSlice1, :);
        resttime2=restmasked_time(rtimeSlice1:rtimeSlice2, :);
        resttime3=restmasked_time(rtimeSlice2:rtimeSlice3, :);
        resttime4=restmasked_time(rtimeSlice3:end, :);
        rt1=corr(resttime1);
        zrt1=atanh(rt1);
        rt2=corr(resttime2);
        zrt2=atanh(rt2);
        rt3=corr(resttime3);
        zrt3=atanh(rt3);
        rt4=corr(resttime4);
        zrt4=atanh(rt4);
        rt=cat(3, zrt1, zrt2, zrt3, zrt4);
        parcel_corrmat=cat(3, parcel_corrmat, rt);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/corrmats_timesplit/', sub, '_fourth_parcel_corrmat.mat')]
    save(saveName, 'parcel_corrmat')
end  