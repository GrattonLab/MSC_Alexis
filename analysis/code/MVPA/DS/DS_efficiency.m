function DS_efficiency(task,  saveFolder, restdir, timesplit)
    trainList={'MSC02','MSC04','MSC05'};
    % load the data into a struct containing all subjest task and rest
    for i = 1:length(trainList)
        filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data';
        parcelFile='_parcel_corrmat.mat';
        %if using regular task data
        if nargin<4
            subFile=strcat(trainList{i}, parcelFile);
            taskFC=fullfile(filePath, task, subFile);
            t.(trainList{i})=load(taskFC).parcel_corrmat;
        end 
        if nargin>4
            %if using time split data
            filePath=strcat(filePath, task, 'corrmats_timesplit', timesplit);
            taskFC=fullfile(filePath, subFile);
            t.(trainList{i})=load(taskFC).parcel_corrmat;
            
            %setup matching rest
            filePath=strcat(filePath, 'rest', 'corrmats_timesplit', timesplit);
            restFC=fullfile(filePath, subFile);
            r.(trainList{i})=load(restFC).parcel_corrmat;   
        end 
            
        if nargin<3
            %use regular rest
            restFC=fullfile(filePath, 'rest', subFile);
            r.(trainList{i})=load(restFC).parcel_corrmat;
        end 
        if nargin>2
            %if you altered rest
            filePath=strcat(filePath, restdir, task);
            restFC=fullfile(filePath, subFile);
            r.(trainList{i})=load(restFC).parcel_corrmat;
        end 
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
            savePath='~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/';
            saveName=[strcat(savePath, saveFolder, 'train_', trainsub, '_test_', testsub, task, '.mat')];
            save(saveName, 'results')
        end 
    end 
end 

    
    
    