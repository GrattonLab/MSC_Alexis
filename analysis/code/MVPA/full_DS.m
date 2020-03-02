%What if we used as much data as possible
function full_DS()
    trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
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
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat'];
        rFC=load(restFC);
        r.(trainList{i})=rFC.parcel_corrmat
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
        only_good = logical(good_mem .* good_mix .* good_motor .* good_rest);
        memFC_clean = train_memory(:,:,only_good);
        motorFC_clean = train_motor(:,:,only_good);
        mixFC_clean = train_mixed(:,:,only_good);
        restFC_clean= rest(:,:, only_good);
        train=cat(3, memFC_clean, motorFC_clean, mixFC_clean, restFC_clean);
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
            tonly_good = logical(tgood_mem .* tgood_mix .* tgood_motor .* tgood_rest);
            tmemFC_clean = tmem(:,:,tonly_good);
            tmotorFC_clean = tmotor(:,:,tonly_good);
            tmixFC_clean = tmix(:,:,tonly_good);
            trestFC_clean= trest(:,:, tonly_good);
            test=cat(3, tmemFC_clean, tmotorFC_clean, tmixFC_clean, trestFC_clean);
            trainingLabel=size(memFC_clean,3)+size(motorFC_clean,3)+size(mixFC_clean,3);
            testingLabel=size(tmemFC_clean,3)+size(tmotorFC_clean,3)+size(tmixFC_clean,3);
            results=svm_scripts_beta(train, [ones(trainingLabel,1); -ones(size(restFC_clean,3),1)],0,test,[ones(testingLabel,1); -ones(size(trestFC_clean,3),1)],0) 
            saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/', trainsub, '_test_', testsub, '_all.mat')];
            save(saveName, 'results')
        end 
    end 
end 

    
    
    