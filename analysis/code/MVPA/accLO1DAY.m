function accLO1DAY(sub)
    trainList={'mem','mixed','motor'}
    %create empty cell to store
    C=cell(3,3)
    for i=1:3
        for j=1:3
            if i==j
                C{j,i}=0
            else
            %open results file
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/results_mat/' trainList{i} sub '.mat']
                load(results)
            %calculate the accuracy 
                acc=mean((sum(results.predictedTestLabels(1,:)==1)+sum(results.predictedTestLabels(2,:)==-1))./36)
                C{j,i}=acc
            end
        end
end
T=cell2table(C)
T.Properties.VariableNames=trainList
%T.Properties.RowNames=predictList
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/LO1DAY/' sub '_acc.csv']
writetable(T, sname, 'WriteRowNames', true)
%type sname

