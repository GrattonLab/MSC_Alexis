function timeseries_thirds(sub)
%load mem timeseries
    memFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);
%load tmask
    mem_tmaskFile=['~/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mem_pass2/condindices.mat'];
    memTmask=load(mem_tmaskFile);
%.Allmem will be different for Mixed
%load motor timeseries
    motorFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_timecourse.mat'];
    motorFC=load(motorFile);
%load tmask
    motor_tmaskFile=['~/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_motor_pass2/condindices.mat'];
    motorTmask=load(motor_tmaskFile);
%load mixed timeseries
    mixedFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_timecourse.mat'];
    mixedFC=load(mixedFile);
%load tmask
    mixed_tmaskFile=['~/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mixed_pass2/condindices.mat'];
    mixedTmask=load(mixed_tmaskFile);
%load rest timeseries
    restFile=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_timecourse.mat'];
    restFC=load(restFile);
%after this masked time split up into pieces then run as corr
%loop through all days
    memFC_all=[]
    nsamples = size(memFC.parcel_time, 2);
    for i=1:nsamples
        masked_time = memFC.parcel_time{i}(logical(memTmask.TIndFin(i).AllMem),:);
%cut time in half
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
        t=cat(3, zt1, zt2, zt3);
        memFC_all=cat(3, memFC_all, t);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/corrmats_timesplit/', sub, '_thirds_parcel_corrmat.mat')]
    save(saveName, 'memFC_all')
%now for rest
%loop through all days
    restFC_all=[]
    restsamples = size(restFC.parcel_time, 2);
    for i=1:restsamples
        restmasked_time = restFC.parcel_time{i}(logical(restFC.tmask_all{i}),:);
%cut time in half
        resttimeSlice=round(size(restmasked_time,1)/3);
        resttime1=restmasked_time(1:resttimeSlice, :);
        resttime2=restmasked_time(resttimeSlice:resttimeSlice*2, :);
        resttime3=restmasked_time(resttimeSlice*2:end, :);
        rt1=corr(resttime1);
        zrt1=atanh(rt1);
        rt2=corr(resttime2);
        zrt2=atanh(rt2);
        rt3=corr(resttime3);
        zrt3=atanh(rt3);
        rt=cat(3, zrt1, zrt2, zrt3);
        restFC_all=cat(3, restFC_all, rt);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/corrmats_timesplit/', sub, '_thirds_parcel_corrmat.mat')]
    save(saveName, 'restFC_all')
%now for mixed
%all Glass
%loop through all days
    glassFC_all=[]
    mixsamples = size(mixedFC.parcel_time, 2);
    for i=1:mixsamples
        Glassmasked_time = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllGlass),:);
        if isempty(Glassmasked_time)
            continue
        end 
%cut time in half
        GtimeSlice=round(size(Glassmasked_time,1)/3);
        Gtime1=Glassmasked_time(1:GtimeSlice, :);
        Gtime2=Glassmasked_time(GtimeSlice:GtimeSlice*2, :);
        Gtime3=Glassmasked_time(GtimeSlice*2:end, :);
        Gt1=corr(Gtime1);
        zGt1=atanh(Gt1);
        Gt2=corr(Gtime2);
        zGt2=atanh(Gt2);
        Gt3=corr(Gtime3);
        zGt3=atanh(Gt3);
        Gt=cat(3, zGt1, zGt2, zGt3);
        glassFC_all=cat(3, glassFC_all, Gt);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/corrmats_timesplit/', sub, '_thirds_AllGlass_parcel_corrmat.mat')]
    save(saveName, 'glassFC_all')
%All semantic
%loop through all days
    semFC_all=[]
    for i=1:mixsamples
        Semmasked_time = mixedFC.parcel_time{i}(logical(mixedTmask.TIndFin(i).AllSemantic),:);
%cut time in half
        SemtimeSlice=round(size(Semmasked_time,1)/3);
        Stime1=Semmasked_time(1:SemtimeSlice, :);
        Stime2=Semmasked_time(SemtimeSlice:SemtimeSlice*2, :);
        Stime3=Semmasked_time(SemtimeSlice*2:end, :);
        St1=corr(Stime1);
        zSt1=atanh(St1);
        St2=corr(Stime2);
        zSt2=atanh(St2);
        St3=corr(Stime3);
        zSt3=atanh(St3);
        St=cat(3, zSt1, zSt2, zSt3);
        semFC_all=cat(3, semFC_all, St);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/corrmats_timesplit/', sub, '_thirds_AllSemantic_parcel_corrmat.mat')]
    save(saveName, 'semFC_all')
%now for motor
%loop through all days
    motorFC_all=[]
    motorsamples = size(motorFC.parcel_time, 2);
    for i=1:motorsamples
        motmasked_time = motorFC.parcel_time{i}(logical(motorTmask.TIndFin(i).AllMotor),:);
        if isempty(motmasked_time)
            continue
        end 
%cut time in half
        mottimeSlice=round(size(motmasked_time,1)/3);
        Mtime1=motmasked_time(1:mottimeSlice, :);
        Mtime2=motmasked_time(mottimeSlice:mottimeSlice*2, :);
        Mtime3=motmasked_time(mottimeSlice*2:end, :);
        Mt1=corr(Mtime1);
        zMt1=atanh(Mt1);
        Mt2=corr(Mtime2);
        zMt2=atanh(Mt2);
        Mt3=corr(Mtime3);
        zMt3=atanh(Mt3);
        Mt=cat(3, zMt1, zMt2, zMt3);
        motorFC_all=cat(3, motorFC_all, Mt);
    end
    saveName=[strcat('~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/corrmats_timesplit/', sub, '_thirds_parcel_corrmat.mat')]
    save(saveName, 'motorFC_all')
end 