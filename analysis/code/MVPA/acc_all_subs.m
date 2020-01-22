function acc_all_subs()
    taskList={'mem','mixed','motor'}
    C=cell(1,3)
    for i=1:3
        results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/all_subs/results_mat/train_' taskList{i} '.mat']
        load(results)
        acc=mean((sum(results.predictedTestLabels(1:40,:)==1)+sum(results.predictedTestLabels(41:80,:)==-1))./80)
        C{1,i}=acc
    end 
T=cell2table(C)
T.Properties.VariableNames=taskList
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/all_subs/allTask_acc.csv']
writetable(T, sname, 'WriteRowNames', true)
end