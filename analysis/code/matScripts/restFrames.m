subs={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07', 'MSC10'};
filePath='/Users/Alexis/Desktop/'
sub=[]
subID=[]
for i=1:length(subs)
%load rest timeseries
    restFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' subs{i} '_parcel_timecourse.mat'];
    restFC=load(restFile);
    rest=[];
    nsamples = size(restFC.parcel_time, 2);
    %make list of times per day
    for day=1:nsamples
        rest_time = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        resttimeSlice=round(size(rest_time,1));
        rest=cat(1, rest, resttimeSlice);
        test=subs{i};
        subID=cat(1,subID, test);
    end
    sub=cat(1,sub, rest);
    
    clear rest
end
    c=cellstr(subID)
    n=num2cell(sub)
    concat=[c,n];
    T = array2table(concat, 'VariableNames',{'sub','rest'});
    task_file=strcat('rest.csv')
    sname=fullfile(filePath, task_file)
    writetable(T, sname, 'WriteRowNames', true)
