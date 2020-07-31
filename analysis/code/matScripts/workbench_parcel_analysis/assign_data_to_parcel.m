function assign_data_to_parcel(data,watershed_L,watershed_R,outputdir,outputname)
% Assign data to watersheds, LR together, TOL 05/2014
HEMS = {'L';'R'};
for h = 1:2
    if h == 1
        watershed = gifti(watershed_L);
    else
        watershed = gifti(watershed_R);
    end
    watershed = watershed.cdata;
    maskname = ['/Users/Alexis/Box/Quest_Backup/Atlases/32k_ConteAtlas_v2/medial_wall.' HEMS{h} '.32k_fs_LR.func.gii'];
    mask = gifti(maskname);
    mask = mask.cdata;
    watershed(logical(mask)) = 0;
    waternum = unique(watershed);
    waternum(waternum==0) = [];
    water_com_assigns = zeros(32492,size(data,2));
    
    for t = 1:size(data,2)
        
        for i = 1:length(waternum)
            
            water_com_assigns(watershed==waternum(i),t) = data(i,t);
        end
    end
    save(gifti(water_com_assigns),[outputdir '/' outputname '_' HEMS{h} '.func.gii'])
    data(1:length(waternum),:) =[]; %Remove rows from left hem
end
