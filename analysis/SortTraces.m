function [sortind] = SortTraces(raster)
%sort traces by order of peak activity
%raster is a matrix(cell,time)

raster = round(raster,2); %ignore insignificant digits when finding the peak

[val ind]=max(raster');
[val sortind] = sort(ind);