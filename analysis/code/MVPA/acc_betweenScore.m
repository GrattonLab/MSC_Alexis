function acc_betweenScore(task)
    %trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    %predictList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    trainList={'MSC01','MSC02','MSC04','MSC05'};
    predictList={'MSC01','MSC02','MSC04','MSC05'};
    %C=cell(8,8);
    C=cell(4,4);
    for i=1:length(trainList)
        for j=1:length(predictList)
            if i==j
                C{j,i}=0;
            else
            %open results file
                %results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/train_' trainList{i} '_test_' predictList{j} task '.mat'];
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/timesplit_train_' trainList{i} '_test_' predictList{j} task '.mat'];
                load(results);
            %calculate the accuracy 
                taskcut=size(results.predictedTestLabels,1)/2;
                aa = [results.predictedTestLabels(1:taskcut,:)==1;results.predictedTestLabels(taskcut+1:end,:)==-1];
                acc=mean(aa(:));
                C{j,i}=acc;
            end
        end
end
T=cell2table(C)
T.Properties.VariableNames=trainList
T.Properties.RowNames=predictList
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/' task '_onlyGood_timeSeries_acc.csv']
writetable(T, sname, 'WriteRowNames', true)
%type sname

