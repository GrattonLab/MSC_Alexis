%What if we used as much data as possible
function full_DS()
    trainList={'MSC01','MSC02','MSC04','MSC05'};%,'MSC10'};
    % load the data into a struct containing all subjest task and rest
    for i = 1:length(trainList)
        %memory
        mem=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' trainList{i} '_parcel_corrmat.mat'];
        memFC=load(mem);
        memoryDB.(trainList{i})=memFC.parcel_corrmat;
        %mixed
        mix=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' trainList{i} '_parcel_corrmat.mat'];
        mixFC=load(mix);
        mixedDB.(trainList{i})=mixFC.parcel_corrmat;
        %motor
        mot=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' trainList{i} '_parcel_corrmat.mat'];
        motFC=load(mot);
        motorDB.(trainList{i})=motFC.parcel_corrmat;
        %rest
        %restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat'];
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/corrmats_timesplit/' trainList{i} '_thirds_parcel_corrmat.mat'];
        rFC=load(restFC);
        r.(trainList{i})=rFC.restFC_all
    end
    
    % train and test
    for i = 1:length(trainList)
        trainsub = trainList{i};
        testsubs = setdiff((trainList),(trainsub)); %initialize string of variables to pull from struct
        %memory
        train_memory=memoryDB.(trainsub);
        %mixed
        train_mixed=mixedDB.(trainsub);
        %motor
        train_motor=motorDB.(trainsub);
        %rest
        rest=r.(trainsub);
        
        good_mem = ~isnan(squeeze(sum(sum(train_memory,2),1)));
        good_mix = ~isnan(squeeze(sum(sum(train_mixed,2),1)));
        good_motor = ~isnan(squeeze(sum(sum(train_motor,2),1)));
        good_rest = ~isnan(squeeze(sum(sum(rest,2),1)));
        %only_good = logical(good_mem .* good_mix .* good_motor .* good_rest);
        %only_good = logical(good_mem .* good_mix .* good_motor);
        memFC_clean = train_memory(:,:,good_mem);
        motorFC_clean = train_motor(:,:,good_motor);
        mixFC_clean = train_mixed(:,:,good_mix);
        task_data=cat(3, memFC_clean, motorFC_clean, mixFC_clean); 
        good_taskAll=~isnan(squeeze(sum(sum(task_data,2),1)));
        good_taskRest_matched= logical(good_taskAll .* good_rest);
        %restFC_clean= rest(:,:, only_good);
        restFC_clean= rest(:,:, good_taskRest_matched);
        train=cat(3, task_data, restFC_clean);
        for j = 1:length(testsubs)
            testsub=testsubs{j};
            %memory
            tmem=memoryDB.(testsub);
            %mixed
            tmix=mixedDB.(testsub);
            %motor
            tmotor=motorDB.(testsub);
            %rest
            trest=r.(testsub);
            
            tgood_mem = ~isnan(squeeze(sum(sum(tmem,2),1)));
            tgood_mix = ~isnan(squeeze(sum(sum(tmix,2),1)));
            tgood_motor = ~isnan(squeeze(sum(sum(tmotor,2),1)));
            tgood_rest = ~isnan(squeeze(sum(sum(trest,2),1)));
            %tonly_good = logical(tgood_mem .* tgood_mix .* tgood_motor .* tgood_rest);
            %tonly_good = logical(tgood_mem .* tgood_mix .* tgood_motor);
            tmemFC_clean = tmem(:,:,tgood_mem);
            tmotorFC_clean = tmotor(:,:,tgood_motor);
            tmixFC_clean = tmix(:,:,tgood_mix);
            
            %trestFC_clean= trest(:,:, tonly_good);
            ttask_data=cat(3, tmemFC_clean, tmotorFC_clean, tmixFC_clean); 
            tgood_taskAll=~isnan(squeeze(sum(sum(ttask_data,2),1)));
            tgood_taskRest_matched= logical(tgood_taskAll .* tgood_rest);
            trestFC_clean= trest(:,:, tgood_taskRest_matched);
            test=cat(3, ttask_data, trestFC_clean);
            trainingLabel=size(task_data,3);
            testingLabel=size(ttask_data,3);
            results=svm_scripts_beta(train, [ones(trainingLabel,1); -ones(size(restFC_clean,3),1)],0,test,[ones(testingLabel,1); -ones(size(trestFC_clean,3),1)],0) 
            saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/', trainsub, '_test_', testsub, '_all.mat')];
            save(saveName, 'results')
        end 
    end 
end 

    
    
    