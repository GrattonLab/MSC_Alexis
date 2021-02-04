#!/bin/bash

set wbdir	= /projects/b1081/Scripts/workbench2/bin_linux64
#try these out with dtsereies instead
#sub 1 session of mem task
set file_32k	= /projects/b1081/MSC/TaskFC/FCProc_MSC01_mem_pass2/cifti_timeseries_normalwall_native_freesurf/vc38671_LR_surf_subcort_333_32k_fsLR_smooth2.55.dtseries.nii
set template_4k	= /projects/p31240/downsampled_4k_surfs/template.4k.dtseries.nii
#I think this is the output filename
set file_4k	= /projects/p31240/downsampled_4k_surfs/mem/MSC01/vc38671_LR_surf_subcort_333_4k_fsLR_smooth2.55.dtseries.nii

set left_sphere_32k	= /projects/b1081/Scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.L.sphere.32k_fs_LR.surf.gii
set right_sphere_32k	= /projects/b1081/Scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.R.sphere.32k_fs_LR.surf.gii

set left_sphere_4k	= /projects/p31240/downsampled_4k_surfs/Sphere.4k.L.surf.gii
set right_sphere_4k	= /projects/p31240/downsampled_4k_surfs/Sphere.4k.R.surf.gii

set left_midthick_32k	= /projects/b1081/MSC/MSCdata_v1/FREESURFER_fs_LR/MSC01/7112b_fs_LR/fsaverage_LR32k/MSC01.L.midthickness.32k_fs_LR.surf.gii
set right_midthick_32k	= /projects/b1081/MSC/MSCdata_v1/FREESURFER_fs_LR/MSC01/7112b_fs_LR/fsaverage_LR32k/MSC01.R.midthickness.32k_fs_LR.surf.gii

set left_midthick_4k	= /projects/p31240/downsampled_4k_surfs/$subj.L.midthickness.4k.surf.gii
set right_midthick_4k	= /projects/p31240/downsampled_4k_surfs/$subj.R.midthickness.4k.surf.gii

$wbdir/wb_command -cifti-resample $file_32k COLUMN $template_4k COLUMN ADAP_BARY_AREA CUBIC $file_4k -surface-largest -left-spheres $left_sphere_32k $left_sphere_4k -left-area-surfs $left_midthick_32k $left_midthick_4k -right-spheres $right_sphere_32k $right_sphere_4k -right-area-surfs $right_midthick_32k $right_midthick_4k
#look into difference in not smoothing v smoothing
#$wbdir/wb_command -cifti-smoothing $file_4k 4.25 2.55 COLUMN $file_4k -left-surface $left_midthick_4k -right-surface $right_midthick_4k
