function DS(task)
    trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    predictList = trainList;
    
    % load the data
    for i = 1:length(trainList)
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' trainList{i} '_parcel_corrmat.mat'];
        tFC=load(taskFC);
        t.(trainList{i})=tFC.parcel_corrmat;
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat'];
        rFC=load(restFC);
        r.(trainList{i})=rFC.parcel_corrmat;
    end
    
    % train and test
    for i = 1:length(trainlist)
        trainsub = trainlist{i};
        testsubs = setdiff(set(trainList),set(testsub)); %check google for set that can operate on cells/strings
        
        for j = 1:length(testsubs)
            %svmscripts command
        
    
    predictList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    for i=1:length(trainList)
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' trainList{i} '_parcel_corrmat.mat'];
        tFC=load(taskFC);
        t=tFC.parcel_corrmat;
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat'];
        rFC=load(restFC);
        r=rFC.parcel_corrmat;
        good_task = ~isnan(squeeze(sum(sum(t,2),1)));
        good_rest = ~isnan(squeeze(sum(sum(r,2),1)));
        only_good = logical(good_task .* good_rest);
        taskFC_clean = t(:,:,only_good);
        restFC_clean= r(:,:, only_good);
        train=cat(3, taskFC_clean, restFC_clean);
        for j=1:length(predictList)
            if i==j
                continue
            else
                testFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' predictList{j} '_parcel_corrmat.mat'];
                test=load(testFC);
                tt=test.parcel_corrmat;
                test_restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' predictList{j} '_parcel_corrmat.mat'];
                test_rest=load(test_restFC);
                tr=test_rest.parcel_corrmat;
                test_good_task = ~isnan(squeeze(sum(sum(tt,2),1)));
                test_good_rest = ~isnan(squeeze(sum(sum(tr,2),1)));
                test_only_good = logical(test_good_task .* test_good_rest);
                test_taskFC_clean = tt(:,:,test_only_good);
                test_restFC_clean= tr(:,:, test_only_good);
                test=cat(3, test_taskFC_clean, test_restFC_clean);
                %different subject same task
                results=svm_scripts_beta(train, [ones(size(taskFC_clean,3),1); -ones(size(restFC_clean,3),1)],0,test,[ones(size(test_taskFC_clean,3),1); -ones(size(test_restFC_clean,3),1)],0) %to arrange in pairs options=1
                saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/train_' trainList{i} '_test_' predictList{j} task '.mat'];
                save(saveName, 'results')
            end 
        end
    end
end