import nibabel as nib
import glob
import numpy as np
session=1
for task_fname in glob.glob('/projects/b1081/member_directories/aporter/Mem/*.dconn.nii'):
    data_file=nib.load(task_fname)
    task=data_file.get_data()
    first, second, third, fourth=np.array_split(task,4)
    np.savetxt(task_fname +str(session)+ 'first.csv', first, delimiter=',')
    np.savetxt(task_fname +str(session)+ 'second.csv', second, delimiter=',')
    np.savetxt(task_fname +str(session)+ 'third.csv', third, delimiter=',')
    np.savetxt(task_fname +str(session)+ 'fourth.csv', fourth, delimiter=',')
    session=session+1


for task_fname in glob.glob('/projects/b1081/member_directories/aporter/Rest/*.dconn.nii'):
    data_file=nib.load(task_fname)
    task=data_file.get_data()
    first, second, third, fourth=np.array_split(task,4)
    np.savetxt(task_fname +str(session)+ 'first.csv', first, delimiter=',')
    np.savetxt(task_fname +str(session)+ 'second.csv', second, delimiter=',')
    np.savetxt(task_fname +str(session)+ 'third.csv', third, delimiter=',')
    np.savetxt(task_fname +str(session)+ 'fourth.csv', fourth, delimiter=',')
    session=session+1
