function acc_betweenScore()
    trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    predictList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
    %trainList={'MSC01','MSC02','MSC04','MSC05'};
    %predictList={'MSC01','MSC02','MSC04','MSC05'};
    C=cell(8,8);
    T=cell(8,8);
    R=cell(8,8);
    %C=cell(4,4);
    %clean_days=cell(8,8);
    %T=cell(4,4);
    %R=cell(4,4);
    for i=1:length(trainList)
        for j=1:length(predictList)
            if i==j
                C{j,i}=0;
                T{j,i}=0;
                R{j,i}=0;
            else
            %open results file
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/' trainList{i} '_test_' predictList{j} '_all.mat'];
                %results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/timesplit_train_' trainList{i} '_test_' predictList{j} task '.mat'];
                load(results);
            %calculate the accuracy
                clean_days{j,i}=size(results.predictedTestLabels,1);
                taskcut=size(results.predictedTestLabels,1)/4;
                tcut=size(results.predictedTestLabels,1)-taskcut;
                task_acc= results.predictedTestLabels(1:tcut,:)==1;
                rest_acc=results.predictedTestLabels(tcut+1:end,:)==-1;
                tacc=mean(task_acc(:));
                racc=mean(rest_acc(:));
                T{j,i}=tacc;
                R{j,i}=racc;
                aa = [results.predictedTestLabels(1:tcut,:)==1;results.predictedTestLabels(tcut+1:end,:)==-1];
                acc=mean(aa(:));
                C{j,i}=acc;
            end
        end
    end
%clean=cell2table(clean_days);
%clean.Properties.VariableNames=trainList;
%clean.Properties.RowNames=predictList;
%cleanname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/' task 'clean_days.csv']
%writetable(clean, cleanname, 'WriteRowNames', true)

t_tab=cell2table(T);
t_tab.Properties.VariableNames=trainList;
t_tab.Properties.RowNames=predictList;
tname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/onlyGood_taskOnly_all_acc.csv']
writetable(t_tab, tname, 'WriteRowNames', true)

r_tab=cell2table(R);
r_tab.Properties.VariableNames=trainList;
r_tab.Properties.RowNames=predictList;
rname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/onlyGood_restOnly_all_acc.csv']
writetable(r_tab, rname, 'WriteRowNames', true)

T=cell2table(C)
T.Properties.VariableNames=trainList
T.Properties.RowNames=predictList
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/onlyGood_all_acc.csv']
writetable(T, sname, 'WriteRowNames', true)


