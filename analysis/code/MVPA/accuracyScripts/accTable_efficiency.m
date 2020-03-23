function accTable_efficiency(task, sublist,saveFolder)
    %get the size for your cells
    filePath='/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test'
    matFile='results_mat/only_good'
    accPath=strcat('acc_', saveFolder);
    tableSize=length(sublist)
    C=cell(tableSize,tableSize);
    T=cell(tableSize,tableSize);
    R=cell(tableSize,tableSize);
    %clean_days=cell(8,8);
    for i=1:length(sublist)
        for j=1:length(sublist)
            if i==j
                C{j,i}=0;
                T{j,i}=0;
                R{j,i}=0;
            else
            %open results file
                saveFile=strcat( 'train_', sublist{i}, '_test_', sublist{j}, task, '.mat')
                results=fullfile(filePath, matFile, saveFolder, saveFile)
                load(results);
            %calculate the accuracy
                %clean_days{j,i}=size(results.predictedTestLabels,1);
                taskcut=size(results.predictedTestLabels,1)/2;
                task_acc= results.predictedTestLabels(1:taskcut,:)==1;
                rest_acc=results.predictedTestLabels(taskcut+1:end,:)==-1;
                tacc=mean(task_acc(:));
                racc=mean(rest_acc(:));
                T{j,i}=tacc;
                R{j,i}=racc;
                aa = [task_acc;rest_acc];
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

taskOnly=cell2table(T);
taskOnly.Properties.VariableNames=sublist;
taskOnly.Properties.RowNames=sublist;
taskOnly_file=strcat(task, '_taskOnly.csv')
tname=fullfile(filePath, accPath, taskOnly_file)
writetable(taskOnly, tname, 'WriteRowNames', true)

restOnly=cell2table(R);
restOnly.Properties.VariableNames=sublist;
restOnly.Properties.RowNames=sublist;
restOnly_file=strcat(task, '_restOnly.csv')
rname=fullfile(filePath, accPath, restOnly_file)
writetable(restOnly, rname, 'WriteRowNames', true)

T=cell2table(C)
T.Properties.VariableNames=sublist
T.Properties.RowNames=sublist
task_file=strcat(task, '.csv')
sname=fullfile(filePath, accPath, task_file)
writetable(T, sname, 'WriteRowNames', true)


