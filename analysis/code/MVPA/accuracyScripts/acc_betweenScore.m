function acc_betweenScore(task)
    %trainList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07'};
    %predictList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07'};
    trainList={'MSC02','MSC04','MSC05'};
    predictList={'MSC02','MSC04','MSC05'};
    %C=cell(7,7);
    %T=cell(7,7);
    %R=cell(7,7);
    C=cell(3,3);
    %clean_days=cell(8,8);
    T=cell(3,3);
    R=cell(3,3);
    for i=1:length(trainList)
        for j=1:length(predictList)
            if i==j
                C{j,i}=0;
                T{j,i}=0;
                R{j,i}=0;
            else
            %open results file
                %results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/' trainList{i} '_test_' predictList{j} '_all.mat'];
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/results_mat/only_good/timeQuality_rest/mixedsplit_train_' trainList{i} '_test_' predictList{j} 'AllGlass.mat'];
                load(results);
            %calculate the accuracy
                %clean_days{j,i}=size(results.predictedTestLabels,1);
                taskcut=size(results.predictedTestLabels,1)/2;
                %tcut=size(results.predictedTestLabels,1)-taskcut;
                task_acc= results.predictedTestLabels(1:taskcut,:)==1;
                rest_acc=results.predictedTestLabels(taskcut+1:end,:)==-1;
                tacc=mean(task_acc(:));
                racc=mean(rest_acc(:));
                T{j,i}=tacc;
                R{j,i}=racc;
                aa = [results.predictedTestLabels(1:taskcut,:)==1;results.predictedTestLabels(taskcut+1:end,:)==-1];
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
tname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/acc_onlyGood/AllGlass' task 'Only_all_acc.csv']
writetable(t_tab, tname, 'WriteRowNames', true)

r_tab=cell2table(R);
r_tab.Properties.VariableNames=trainList;
r_tab.Properties.RowNames=predictList;
rname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/acc_onlyGood/AllGlass' task '_restOnly_all_acc.csv']
writetable(r_tab, rname, 'WriteRowNames', true)

T=cell2table(C)
T.Properties.VariableNames=trainList
T.Properties.RowNames=predictList
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/acc_onlyGood/AllGlass' task '_acc.csv']
writetable(T, sname, 'WriteRowNames', true)


