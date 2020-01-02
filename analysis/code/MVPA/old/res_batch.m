function results(sub)
    trainList={'mem','mixed','motor'}
    predictList={'mem','mixed','motor'}
    myFolder='~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/' %defining working directory
    for i=1:3
        %create empty cell to store
        C=cell(3,3)
        for j=1:3
            if i==j
                C{i,j}=0
            else
            %open results file
                results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/test_' trainList{i} '_train_' predictList{j} sub '.mat']
                load(results)
            %calculate the accuracy 
                acc=mean((sum(results.predictedTestLabels(1:10,:)==1))./10)
                C{i,j}=acc
            %return acc and test train names
            %saveName=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/test_' trainList{i} '_train_' predictList{j} sub '.mat']
            %save(saveName, 'results')
            end
        end
    end
end


