 function x=permutation_loop(sub)
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
random_matrix=[]
for i=1:10
    %iterate through randomely permuting the labels 10 times across 10 subs
    idx_rand = randperm(20)
    trainLabels = [ones(10,1);-ones(10,1)]
    testLabels= [ones(10,1)]
    %How accurate are the scambles labels?
    t=trainLabels(idx_rand)
    rand_acc=mean((sum(t(1:10)==1)+sum(t(11:20)==-1))./20)
    random_matrix=[random_matrix;{rand_acc}];
    %for random permutations
    results=svm_scripts_beta(train,trainLabels(idx_rand),0,test,testLabels,0)
    %calculate the mean
    acc=mean((sum(results.predictedTestLabels(1:10,:)==1))./10)
    %save to a table
    C=[C;{acc}];
end
Random=cell2table(random_matrix)
rand_name=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/permutedLabel_' sub '_Random.csv']
writetable(Random, rand_name, 'WriteRowNames', true)
T=cell2table(C)
sname=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/random_permutation_test/permuted_' sub '_accRandom.csv']
writetable(T, sname, 'WriteRowNames', true)
B=cell2mat(C)
x=mean(B)
 end 