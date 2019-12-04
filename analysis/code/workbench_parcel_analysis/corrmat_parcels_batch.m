clear;
clear all;


load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC02_mem_corrmat.mat', 'MSC02_mem')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC05_mem_corrmat.mat', 'MSC05_mem')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC06_mem_corrmat.mat', 'MSC06_mem')

load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC02_mix_corrmat.mat', 'MSC02_mixed')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC05_mix_corrmat.mat', 'MSC05_mixed')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC06_mix_corrmat.mat', 'MSC06_mixed')

load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC02_motor_corrmat.mat', 'MSC02_motor')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC05_motor_corrmat.mat', 'MSC05_motor')
%load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC06_motor_corrmat.mat', 'MSC06_motor')

load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC02_rest_corrmat.mat', 'MSC02_rest')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC05_rest_corrmat.mat', 'MSC05_rest')
load('/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/MSC06_rest_corrmat.mat', 'MSC06_rest')


f=figure_corrmat_network_generic(MSC02_mem, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_mem.png')

f=figure_corrmat_network_generic(MSC05_mem, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_mem.png')

f=figure_corrmat_network_generic(MSC06_mem, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC06'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_mem.png')

f=figure_corrmat_network_generic(MSC02_mixed, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_mix.png')

f=figure_corrmat_network_generic(MSC05_mixed, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_mix.png')

f=figure_corrmat_network_generic(MSC06_mixed, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC06'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_mix.png')

f=figure_corrmat_network_generic(MSC02_motor, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_motor.png')

f=figure_corrmat_network_generic(MSC05_motor, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_motor.png')
%f=figure_corrmat_network_generic(MSC06_motor, atlas_parameters('Parcels','/Users/Alexis/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC06'))
%colormapeditor
%saveas(f, '/Users/Alexis/Documents/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_motor.png')

f=figure_corrmat_network_generic(MSC02_rest, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_rest.png')

f=figure_corrmat_network_generic(MSC05_rest, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_rest.png')

f=figure_corrmat_network_generic(MSC06_rest, atlas_parameters('Parcels','/Users/aporter1350/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC06'))
colormapeditor
saveas(f, '/Users/aporter1350/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_rest.png')

MSC02_memMrest=MSC02_mem-MSC02_rest
MSC05_memMrest=MSC05_mem-MSC05_rest
MSC06_memMrest=MSC06_mem-MSC06_rest

MSC02_mixMrest=MSC02_mixed-MSC02_rest
MSC05_mixMrest=MSC05_mixed-MSC05_rest
MSC06_mixMrest=MSC06_mixed-MSC06_rest

MSC02_motorMrest=MSC02_motor-MSC02_rest
MSC05_motorMrest=MSC05_motor-MSC05_rest



%task minus rest corr maps 
f=figure_corrmat_network_generic(MSC02_memMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_memMrest.png')

f=figure_corrmat_network_generic(MSC05_memMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_memMrest.png')

f=figure_corrmat_network_generic(MSC06_memMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC06'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_memMrest.png')

f=figure_corrmat_network_generic(MSC02_mixMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_mixMrest.png')

f=figure_corrmat_network_generic(MSC05_mixMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_mixMrest.png')

f=figure_corrmat_network_generic(MSC06_mixMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC06'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_mixMrest.png')

f=figure_corrmat_network_generic(MSC02_motorMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC02'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_02_motorMrest.png')

f=figure_corrmat_network_generic(MSC05_motorMrest, atlas_parameters('Parcels','/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/FC_Parcels/','MSC05'))
colormapeditor
saveas(f, '/Users/aporter1350/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_05_motorMrest.png')
%f=figure_corrmat_network_generic(MSC06_motor, atlas_parameters('Parcels','/Users/Alexis/Box/Quest_Backup/Atlases/Evan_parcellation/','MSC06'))
%colormapeditor
%saveas(f, '/Users/Alexis/Documents/MSC_Alexis/analysis/output/images/heatmap_corr/MSC_06_motor.png')

