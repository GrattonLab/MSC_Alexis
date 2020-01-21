%Lets just see what would happen if we did a classic loocv leaving one
%subject out
function all_subs(task)
    task_df=[]
    rest_df=[]
    test_df=[]
    test_rest=[]
    trainList={'MSC02','MSC04','MSC05','MSC10'}
    testList={'MSC01','MSC03','MSC06','MSC07'}
    myFolder='~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' %defining working directory
    for i=1:4 %task data
        taskFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' trainList{i} '_parcel_corrmat.mat']
        load(taskFC)
        t=parcel_corrmat
        task_df=cat(3,task_df,t) %concatenate all task data from every subject
    end
    for i=1:4 %rest data
        restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' trainList{i} '_parcel_corrmat.mat']
        load(restFC)
        r=parcel_corrmat
        rest_df=cat(3, rest_df, r) %concatenate all rest data from every subject
    end 
    for i=1:4 %task data
        testFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' testList{i} '_parcel_corrmat.mat']
        load(testFC)
        tt=parcel_corrmat
        test_df=cat(3,test_df,tt) %concatenate all task data from every subject
    end
    for i=1:4 %rest data
        test_restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' testList{i} '_parcel_corrmat.mat']
        load(test_restFC)
        tr=parcel_corrmat
        test_rest=cat(3, test_rest, tr) %concatenate all rest data from every subject
    end 
    final_df=cat(3,task_df, rest_df)
    test_final_df=cat(3, test_df, test_rest)
    results=svm_scripts_beta(final_df, [ones(40,1); -ones(40,1)],0, test_final_df,[ones(40,1); -ones(40,1)],0)
    saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/all_subs/results_mat/train_' task '.mat']
    save(saveName, 'results')
end