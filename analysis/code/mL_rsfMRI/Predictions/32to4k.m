

wbdir	= '/projects/b1081/Scripts/workbench2/bin_linux64'

%sub 1 session of mem task
file_32k	= '/projects/b1081/MSC/TaskFC/FCProc_MSC01_mem_pass2/cifti_timeseries_normalwall_native_freesurf/vc38671_LR_surf_subcort_333_32k_fsLR_smooth2.55.dtseries.nii'
template_4k	= '/projects/p31240/downsampled_4k_surfs/template.4k.dtseries.nii'
file_4k	= '/projects/p31240/downsampled_4k_surfs/mem/MSC01/vc38671_LR_surf_subcort_333_4k_fsLR_smooth2.55.dtseries.nii'

left_sphere_32k	= '/projects/b1081/Scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.L.sphere.32k_fs_LR.surf.gii'
right_sphere_32k	= '/projects/b1081/Scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.R.sphere.32k_fs_LR.surf.gii'

left_sphere_4k	= '/projects/p31240/downsampled_4k_surfs/Sphere.4k.L.surf.gii'
right_sphere_4k	= '/projects/p31240/downsampled_4k_surfs/Sphere.4k.R.surf.gii'

left_midthick_32k	= '/projects/b1081/MSC/MSCdata_v1/FREESURFER_fs_LR/MSC01/7112b_fs_LR/fsaverage_LR32k/MSC01.L.midthickness.32k_fs_LR.surf.gii'
right_midthick_32k	= '/projects/b1081/MSC/MSCdata_v1/FREESURFER_fs_LR/MSC01/7112b_fs_LR/fsaverage_LR32k/MSC01.R.midthickness.32k_fs_LR.surf.gii'

left_midthick_4k	= '/projects/p31240/downsampled_4k_surfs/$subj.L.midthickness.4k.surf.gii'
right_midthick_4k	= '/projects/p31240/downsampled_4k_surfs/$subj.R.midthickness.4k.surf.gii'

system([wbdir '/wb_command -cifti-resample ' file_32k ' COLUMN ' template_4k ' COLUMN ADAP_BARY_AREA CUBIC ' $file_4k '-surface-largest -left-spheres ' left_sphere_32k left_sphere_4k ' -left-area-surfs ' left_midthick_32k left_midthick_4k ' -right-spheres ' right_sphere_32k right_sphere_4k ' -right-area-surfs ' right_midthick_32k right_midthick_4k])
%$wbdir/wb_command -cifti-resample $file_32k COLUMN $template_4k COLUMN ADAP_BARY_AREA CUBIC $file_4k -surface-largest -left-spheres $left_sphere_32k $left_sphere_4k -left-area-surfs $left_midthick_32k $left_midthick_4k -right-spheres $right_sphere_32k $right_sphere_4k -right-area-surfs $right_midthick_32k $right_midthick_4k
