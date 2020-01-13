function DS(task)
    trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC08','MSC09','MSC10'}
    predictList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC08','MSC09','MSC10'}
    myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' %defining working directory
    for i=1:10
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' trainList{i} '_parcel_corrmat.mat']
        load(taskFC)
        t=parcel_corrmat
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat']
        load(restFC)
        r=parcel_corrmat
        train=cat(3, t, r)
        for j=1:10
            testFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' predictList{j} '_parcel_corrmat.mat']
            load(testFC)
            tt=parcel_corrmat
            test_restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' predictList{j} '_parcel_corrmat.mat']
            load(test_restFC)
            tr=parcel_corrmat
            test=cat(3, tt, tr)
            %different subject same task
            results=svm_scripts_beta(train, [ones(10,1); -ones(10,1)],0,test,[ones(10,1); -ones(10,1)],0)
            saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/train_' trainList{i} '_test_' predictList{j} task '.mat']
            save(saveName, 'results')
        end
    end
end