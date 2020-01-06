 function random(sub)
memFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_corrmat.mat']
load(memFC)
mem=parcel_corrmat
restFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat']
load(restFC)
r=parcel_corrmat
train=cat(3, mem, r)
testFC=['~/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_corrmat.mat']
load(testFC)
test=parcel_corrmat
C=[]
for i=1:1000
    %iterate through randomely permuting the labels 1000 times
    idx_rand = randperm(20)
    trainLabels = [ones(10,1);-ones(10,1)]
    test_rand=randperm(10)
    testLabels= [ones(10,1)]
    %for random permutations
    results=svm_scripts_beta(train,trainLabels(idx_rand),0,test,testLabels(test_rand),0)
    %calculate the mean
    acc=mean((sum(results.predictedTestLabels(1:10,:)==1))./10)
    %save to a table
    C=[C;{acc}];
    sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/permuted_' sub '_accRandom.csv']
    writetable(T, sname, 'WriteRowNames', true)
end 
 end 