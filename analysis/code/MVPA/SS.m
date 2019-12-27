function SS(sub)
    trainList={'mem','mixed','motor'}
    predictList={'mem','mixed','motor'}
    myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' %defining working directory
    for i=1:3
        %baseFileName=trainList{i} '/' sub '_parcel_corrmat.mat'
        %fullFileName = fullfile(myFolder, baseFileName);
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' trainList{i} '/' sub '_parcel_corrmat.mat']
        load(taskFC)
        t=parcel_corrmat
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat']
        load(restFC)
        r=parcel_corrmat
        train=cat(3, t, r)
        for j=1:3
            testFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' predictList{j} '/' sub '_parcel_corrmat.mat']
            load(testFC)
            test=parcel_corrmat
            idx_rand = randperm(20)
            trainLabels = [ones(10,1);-ones(10,1)]
            test_rand=randperm(10)
            testLabels= [ones(10,1)]
            %for random permutations
            results=svm_scripts_beta(train,trainLabels(idx_rand),0,test,testLabels(test_rand),0)
            %results=svm_scripts_beta(train, [ones(10,1); -ones(10,1)],0,test,[ones(10,1)],0)
            saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/test_' trainList{i} '_train_' predictList{j} sub '.mat']
            %saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/test_' trainList{i} '_train_' predictList{j} sub '.mat']
            save(saveName, 'results')
        end
    end
end