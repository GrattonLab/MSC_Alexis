function acc_betweenScore(task)
    trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC08','MSC09','MSC10'}
    predictList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC08','MSC09','MSC10'}
    %create empty cell to store
    C=cell(10,10)
    for i=1:10
        for j=1:10
            if i==j
                C{j,i}=0
            else
            %open results file
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/train_' trainList{i} '_test_' predictList{j} task '.mat']
                load(results)
            %calculate the accuracy 
                acc=mean((sum(results.predictedTestLabels(1:10,:)==1)+sum(results.predictedTestLabels(11:20,:)==-1))./20)
                C{j,i}=acc
            end
        end
end
T=cell2table(C)
T.Properties.VariableNames=trainList
T.Properties.RowNames=predictList
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/' task '_acc.csv']
writetable(T, sname, 'WriteRowNames', true)
%type sname

