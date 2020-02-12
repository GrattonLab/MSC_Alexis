 function SS(sub)
    %open all the relevant files
    %motor
    motorFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_corrmat.mat'];
    motFile=load(motorFC);
    motor=motFile.parcel_corrmat;
    %memory
    memoryFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_corrmat.mat'];
    memFile=load(memoryFC);
    mem=memFile.parcel_corrmat;
    %mixed
    mixedFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_corrmat.mat'];
    mixFile=load(mixedFC);
    mixed=mixFile.parcel_corrmat;   
    %rest
    rFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat'];
    restFile=load(rFC);
    rest=restFile.parcel_corrmat;
    trainList={mem,mixed,motor};
    trainListNames={'mem', 'mixed','motor'};
    testList={mem,mixed,motor};
    testListNames={'mem', 'mixed','motor'};
    for i=1:length(trainList)
        taskFC=trainList{i};
        restFC=rest;
        %select good days
        good_task = ~isnan(squeeze(sum(sum(taskFC,2),1)));
        good_rest = ~isnan(squeeze(sum(sum(restFC,2),1)));
        only_good = logical(good_task .* good_rest);
        taskFC_clean = taskFC(:,:,only_good);
        restFC_clean= restFC(:,:, only_good);
        train=cat(3, taskFC_clean, restFC_clean); 
        for j=1:length(testList)
            if i==j
                continue
            end
            ttaskFC=testList{i};
            %select good days
            good_test=~isnan(squeeze(sum(sum(ttaskFC,2),1)));
            only_good_test=logical(good_test);
            test=ttaskFC(:,:, only_good_test);
            %results=svm_scripts_beta(train,trainLabels(idx_rand),0,test,testLabels,0);
            results=svm_scripts_beta(train,[ones(size(taskFC_clean,3),1); -ones(size(restFC_clean,3),1)],0,test,[ones(size(test, 3),1)],0)
            %saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/results_mat/test_' testList{j} '_train_' trainList{i} sub '.mat']
            saveName=[strcat('~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/within_sub_test/results_mat/only_good/test_', testListNames{j}, '_train_', trainListNames{i}, sub, '.mat')]
            save(saveName, 'results')
        end
    end
end