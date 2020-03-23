sublist={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'}
taskList={'mem','motor','mixed'}
for i=length(taskList)
    accTable_efficiency(taskList{i}, sublist, 'DS')
end 