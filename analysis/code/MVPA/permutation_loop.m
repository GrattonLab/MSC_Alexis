subList={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC08','MSC09','MSC10'};
trainList={'mem','mixed','motor'};
predictList={'mem','mixed','motor'};
%create empty cell to store
C=cell(3,30)
for i=1:10
    for j=1:3
        for k=1:3
                %if test is the same as train put a zero
                if j==k
                    C{j,i}=0
                else
            %open randomly permuted file
                    results=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/test_' trainList{j} '_train_' predictList{k} subList{i} '.mat']
                    load(results)
            %calculate the accuracy 
                    acc=mean((sum(results.predictedTestLabels(1:10,:)==1))./10)
                    C{j,i}=acc
                end 
            end
        end
    end 


