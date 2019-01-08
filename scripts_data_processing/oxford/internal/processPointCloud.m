% Crops and downsamples point cloud
%
% Author:
%   Zi Jian Yew <zijian.yew@comp.nus.edu.sg>
%
function [processed, transform] = processPointCloud(pcloud, retainIntensity)

if nargin == 1
    retainIntensity = true;
end

DIST_THRESH = 30;  % Crop radius

xyz = pcloud.Location;
mu = mean(xyz, 1);

xyz_processed = xyz - mu;
mask = sum(xyz_processed.^2, 2) < DIST_THRESH*DIST_THRESH;

xyz_cropped = xyz_processed(mask,:);

transform = eye(4);
transform(1:3, 4) = mu;

processed = pointCloud(xyz_cropped);

% Voxelize point cloud
processed = pcdownsample(processed, 'gridAverage', 0.2);

[normals, ~, ~] = findPointNormals(processed.Location, 9, [0,0,0], true);

processed.Normal = normals;

if retainIntensity
    intensity_cropped = pcloud.Intensity(mask);
    [D, I] = pdist2(xyz_cropped, processed.Location, 'euclidean', 'Smallest', 1);
    processed.Intensity = intensity_cropped(I);
end
