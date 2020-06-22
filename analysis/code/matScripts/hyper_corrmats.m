function hyper_corrmats(sub)
%load all your data
filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/all_sub_hyperalign';
%filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data';
parcelFile='_parcel_corrmat.mat';
%mem
memFile=strcat('mem/', sub, parcelFile);
memFC=fullfile(filePath,memFile);
mem=load(memFC).parcel_corrmat;
%mixed
mixFile=strcat('mixed/', sub, parcelFile);
mixFC=fullfile(filePath,mixFile);
mixed=load(mixFC).parcel_corrmat;
%rest
restFile=strcat('rest/', sub, parcelFile);
restFC=fullfile(filePath,restFile);
rest=load(restFC).parcel_corrmat;





mem_avg=mean(mem,3)
mem_avg=real(mem_avg)
fig=figure_corrmat_network_generic(mem_avg, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
caxis([-1,1]);


rest_avg=mean(rest,3)
rest_avg=real(rest_avg)
fig=figure_corrmat_network_generic(rest_avg, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
%caxis([.75,1]);


%loop through each day 
for i=1:size(mem,3)
    day=mem(:,:, i);
    fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
    caxis([.75,1]);
    saveas(fig, sprintf('/Users/Alexis/Desktop/MSC_Alexis/analysis/output/all_sub_hyperalign/images/corrmat/%s_day%d_mem.png', sub,i));
end 


for i=1:size(mixed,3)
    day=mixed(:,:, i);
    fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
    caxis([-.4,1]);
    saveas(fig, sprintf('/Users/Alexis/Desktop/MSC_Alexis/analysis/output/all_sub_hyperalign/images/corrmat/%s_day%d_mixed.png', sub,i));
end 
    for i=1:size(rest,3)
       day=rest(:,:, i);
       fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
       caxis([-.4,1]);
       saveas(fig, sprintf('/Users/Alexis/Desktop/MSC_Alexis/analysis/output/all_sub_hyperalign/images/corrmat/%s_day%d_rest.png', sub,i));
    end 
end 

