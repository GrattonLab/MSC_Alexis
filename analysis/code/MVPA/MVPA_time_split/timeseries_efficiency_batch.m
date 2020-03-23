splits={'half', 'thirds'}
subs={'MSC01', 'MSC02','MSC04','MSC05'}
for j=1:length(splits) 
    for i=1:length(subs)
        timeseries_efficiency(subs{i}, 'mem', splits{j},'AllMem')
        timeseries_efficiency(subs{i}, 'motor', splits{j},'AllMotor')
        timeseries_efficiency(subs{i}, 'mixed', splits{j},'AllGlass')
        timeseries_efficiency(subs{i}, 'mixed', splits{j},'AllSemantic')
        timeseries_efficiency_rest(subs{i}, splits{j})
    end
end 

