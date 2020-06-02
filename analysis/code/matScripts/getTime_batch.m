%subs={'MSC01','MSC02','MSC04','MSC05','MSC06','MSC07'}
%subs={'MSC03','MSC10'}
%for i=1:length(subs)
%    getExtraTime(subs{i})
%end

subs={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07', 'MSC10'};
for i=1:length(subs)
    tmask_all(subs{i})
end
