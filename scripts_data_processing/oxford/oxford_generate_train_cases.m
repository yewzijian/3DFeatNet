% Generates the positive and negative test cases for training 3DFeat-net
%
% Outputs "train.txt". Each line has the format:
%   <point cloud> | <positves> | <non-negatives>
%
% positves and negatives are 0-indexed. i.e. 0 corresponds to the point
% cloud on the 1st line
%
%
% Author:
%   Zi Jian Yew <zijian.yew@comp.nus.edu.sg>
%

%% Configuration (Update DST_FOLDER, as in oxford_build_pointsclouds.m)
clear
DST_FOLDER = '../../../data/oxford/train';  % Should be the same as in oxford_build_pointsclouds.m

POSITIVE_THRESH = 5;  % positive sets must be less than 5m away
NEGATIVE_THRESH = 50  ;  % negative sets must be more than 50m away

% Train/test split
TESTSET_BOUNDS = [[-inf, inf]; [-inf, 100]];  % Reserve the bottom part for testing (X,Y)

%%
datasets = readtable('datasets_train.txt', 'ReadVariableNames', false);
datasets = datasets.Var1;

% Load all the positions
all_fnames = [];
all_xyz = [];

for d = 1 : length(datasets)
    disp(d)
    metadata = readtable(fullfile(DST_FOLDER, datasets{d}, 'metadata.txt'));
    if isempty(metadata)
        continue
    end

    fnames = strcat(datasets{d}, '/', strtrim(cellstr(num2str(metadata.Idx))), '.bin');
    xyz = [metadata.X, metadata.Y, metadata.Z];
    
    all_fnames = [all_fnames; fnames];
    all_xyz = [all_xyz; xyz];
end

%%
% Remove point clouds which are in the test region
inTest = all_xyz(:,1) > TESTSET_BOUNDS(1,1) & all_xyz(:,1) < TESTSET_BOUNDS(1,2) ...
    & all_xyz(:,2) > TESTSET_BOUNDS(2,1) & all_xyz(:,2) < TESTSET_BOUNDS(2,2);

all_fnames = all_fnames(~inTest);
all_xyz = all_xyz(~inTest, :);

%% Generates metadata containing positives and non-negatives
fid = fopen(fullfile(DST_FOLDER, 'train.txt'), 'w');
    
pairwise_dist = pdist2(all_xyz, all_xyz);

num_models = size(all_xyz, 1);

for i = 1 : num_models

    % Get models below low and high threshold
    below_high = pairwise_dist(i, :) <= NEGATIVE_THRESH;
    below_low = pairwise_dist(i, :) < POSITIVE_THRESH;

    positives = find(below_low);
    nonnegatives = find(below_high & ~below_low);

    fprintf(fid, '%s', all_fnames{i});
    fprintf(fid, '\t|');
    fprintf(fid, '\t%i', positives-1);  % Indices are saved using 0-indexing
    fprintf(fid, '\t|');
    fprintf(fid, '\t%i', nonnegatives-1);
    fprintf(fid, '\n');

    if mod(i, 100) == 0
        fprintf('Processed %i / %i models\n', i, num_models)
    end

end

fclose(fid);
