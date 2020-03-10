function accScore(sub)
    trainList={'mem','mixed','motor'}
    predictList={'mem','mixed','motor'}
    %create empty cell to store
    C=cell(3,3)
    for i=1:3
        for j=1:3
            if i==j
                C{j,i}=0
            else
            %open results file
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/within_sub_test/results_mat/only_good/test_' predictList{j} '_train_' trainList{i} sub '.mat'];
            %open randomly permuted file
                %results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/results_mat/test_' predictList{j} '_train_' trainList{i} sub '.mat']
                load(results);
            %calculate the accuracy 
                acc=mean(sum(results.predictedTestLabels)./size(results.predictedTestLabels,1).*100);
                C{j,i}=acc
            end
        end
end
T=cell2table(C)
T.Properties.VariableNames=trainList
T.Properties.RowNames=predictList
%sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/' sub '_accRandom.csv']
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/within_sub_test/' sub '_onlyGood_acc.csv']
writetable(T, sname, 'WriteRowNames', true)
%type sname

