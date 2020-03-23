subs={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07'}
for i=1:length(subs)
    timeseries_fourth(subs{i}, 'mem', 'half','AllMem')
end