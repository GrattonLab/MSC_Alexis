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
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/regular_test/test_' trainList{i} '_train_' predictList{j} 'MSC02.mat']
                load(results)
            %calculate the accuracy 
                acc=mean((sum(results.predictedTestLabels(1:10,:)==1))./10)
                C{j,i}=acc
            end
        end
end
T=cell2table(C)
T.Properties.VariableNames=trainList
T.Properties.RowNames=predictList
writetable(T, '~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/regular_test/MSC02_acc.csv', 'WriteRowNames', true)
type '~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/regular_test/MSC02_acc.csv'