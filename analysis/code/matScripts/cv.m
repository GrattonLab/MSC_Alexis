clear;
clear all;


memF=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/MSC03_parcel_corrmat.mat'];
restF=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/MSC03_parcel_corrmat.mat'];

mem_MSC06=load(memF).parcel_corrmat;
rest_MSC06=load(restF).parcel_corrmat;
%mem_MSC06(isinf(mem_MSC06)|isnan(mem_MSC06)) = 0;
%mem_MSC06=real(mem_MSC06);

%rest_MSC06(isinf(rest_MSC06)|isnan(rest_MSC06)) = 0;
%rest_MSC06=real(rest_MSC06);

parcel_corrmat=mem_MSC06-rest_MSC06;
saveName='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/MSC03_memMrest_parcel_corrmat.mat';
save(saveName, 'parcel_corrmat');

%test=mem_MSC06.parcel_corrmat(:,:,1);

%foo=figure_corrmat_network_generic(test, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'),-.4, 1);
%bar=figure_corrmat_network_generic(test, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'),-.4, 1);
%fig1=figure('Name', 'Cross Validation');