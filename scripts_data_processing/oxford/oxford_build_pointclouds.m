% Accumulates the LMS_Front line scans of the Oxford Robotcar dataset.
% 
% Outputs:
%   Each trajectory is output to an individual folder, where each .bin file
%   corresponds to a 3D accumulated point cloud. Each point cloud is
%   aligned to ENU axes, and trans
%   Each output folder also contains a metadata.txt, which indicates the
%   position of each point.
%
% Note: 
% - The generated point clouds are positioned w.r.t. ORIGIN_POSE
%
% Author:
%   Zi Jian Yew <zijian.yew@comp.nus.edu.sg>
%

%% Configurations
clear
addpath('internal');

% Set the path to the downloaded data here
FOLDER = '../../../data_raw/oxford';
% Path for processed data
DST_FOLDER = '../../../data/oxford/train';

% Origin pose: All point clouds use this point as a reference
ORIGIN_POSE = [0, 1, 0, 5735000;
               1, 0, 0, 620000;
               0, 0, -1, -109;
               0, 0, 0 1];

%% Generates point clouds

% Get list of datasets to convert
datasets = readtable('datasets_train.txt', 'ReadVariableNames', false);
datasets = datasets.Var1;


totalClouds = 0;
for d = 1 : length(datasets)

    if ~exist(fullfile(FOLDER, datasets{d}), 'dir')
        continue
    end
    
    metadataPath = fullfile(DST_FOLDER, datasets{d}, 'metadata.txt');
    if ~exist(metadataPath, 'file') || isempty(readtable(metadataPath))
        nClouds = BuildPointclouds(fullfile(FOLDER, datasets{d}), DST_FOLDER, ORIGIN_POSE);
        
        totalClouds = totalClouds + nClouds;
        fprintf('Total: %i clouds generated', totalClouds);
    else
        % Skip datasets already created
        fprintf('Skipping %s\n', datasets{d})
    end    
    
end

