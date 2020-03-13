function DS(task)
    trainList={'MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    %trainList={'MSC01','MSC02','MSC04','MSC05'}; %timesplit data
    % load the data into a struct containing all subjest task and rest
    for i = 1:length(trainList)
        %taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/corrmats_timesplit/' trainList{i} '_parcel_corrmat.mat'];
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' trainList{i} '_parcel_corrmat.mat'];
        tFC=load(taskFC);
        %t=tFC.parcel_corrmat;
        t.(trainList{i})=tFC.parcel_corrmat;
        %t.(trainList{i})=tFC.memFC_all;
        %t.(trainList{i})=tFC.glassFC_all;
        %t.(trainList{i})=tFC.semFC_all;
        %t.(trainList{i})=tFC.motorFC_all;
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat'];
        rFC=load(restFC);
        r.(trainList{i})=rFC.parcel_corrmat;
        %r.(trainList{i})=rFC.restFC_all;
        %r=rFC.parcel_corrmat;
    end
    
    % train and test
    for i = 1:length(trainList)
        trainsub = trainList{i};
        testsubs = setdiff((trainList),(trainsub)); %initialize string of variables to pull from struct
        train_task=t.(trainsub);
        rest=r.(trainsub);
        good_task = ~isnan(squeeze(sum(sum(train_task,2),1)));
        good_rest = ~isnan(squeeze(sum(sum(rest,2),1)));
        only_good = logical(good_task .* good_rest);
        taskFC_clean = train_task(:,:,only_good);
        restFC_clean= rest(:,:, only_good);
        train=cat(3, taskFC_clean, restFC_clean);
        for j = 1:length(testsubs)
            testsub=testsubs{j};
            ttask=t.(testsub);
            trest=r.(testsub);
            tgood_task = ~isnan(squeeze(sum(sum(ttask,2),1)));
            tgood_rest = ~isnan(squeeze(sum(sum(trest,2),1)));
            tonly_good = logical(tgood_task .* tgood_rest);
            ttaskFC_clean = ttask(:,:,tonly_good);
            trestFC_clean= trest(:,:, tonly_good);
            test=cat(3, ttaskFC_clean, trestFC_clean);
            results=svm_scripts_beta(train, [ones(size(taskFC_clean,3),1); -ones(size(restFC_clean,3),1)],0,test,[ones(size(ttaskFC_clean,3),1); -ones(size(trestFC_clean,3),1)],0) %to arrange in pairs options=1
            saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/timesplit_train_', trainsub, '_test_', testsub, task, '.mat')];
            save(saveName, 'results')
        end 
    end 
end 

    
    
    