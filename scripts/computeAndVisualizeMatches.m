% Visualizes matches from descriptors
%
% i.e. no augmentations, etc
addpath('./external');

% clear;
m = 6;  % Dimensionality of raw data (all datasets are XYZNxNyNz)
DRAW_ALL_PUTATIVE = false;  % If true, will draw all inlier/outlier matches
MAX_MATCHES = 1000; % Maximum number of inlier+outlier matches to draw

DATA_FOLDER = '../example_data';
RESULT_FOLDER = '../example_data/results';
DATA_PAIRS = {'oxford_270', 'oxford_456';
              'kitti_00_001554', 'kitti_00_004534'};

FEATURE_DIM = 32;

%% Load pairs and runs matching+RANSAC
for iPair = 1 : size(DATA_PAIRS, 1)
    
    pair = DATA_PAIRS(iPair,:);
    cloud_fnames = {[fullfile(DATA_FOLDER, pair{1}), '.bin'], ...
                [fullfile(DATA_FOLDER, pair{2}), '.bin']};
    desc_fnames = {[fullfile(RESULT_FOLDER, pair{1}), '.bin'], ...
                   [fullfile(RESULT_FOLDER, pair{2}), '.bin']};
               
               
    % Load point cloud and descriptors
    fprintf('Running on frames:\n');
    fprintf('- %s\n', cloud_fnames{1});
    fprintf('- %s\n', cloud_fnames{2});

    for i = 1 : 2
        pointcloud{i} = Utils.loadPointCloud(cloud_fnames{i}, m);

        xyz_features = Utils.load_descriptors(desc_fnames{i}, sum(FEATURE_DIM+3));

        result{i}.xyz = xyz_features(:, 1:3);
        result{i}.desc = xyz_features(:, 4:end);
    end

    % Match
    [~, matches12] = pdist2(result{2}.desc, result{1}.desc, 'euclidean', 'smallest', 1);
    matches12 = [1:length(matches12); matches12]';  

    %  RANSAC
    cloud1_pts = result{1}.xyz(matches12(:,1), :);
    cloud2_pts = result{2}.xyz(matches12(:,2), :);
    [estimateRt, inlierIdx, trialCount] = ransacfitRt([cloud1_pts'; cloud2_pts'], 1.0, false);
    fprintf('Number of inliers: %i / %i (Proportion: %.3f. #RANSAC trials: %i)\n', ...
            length(inlierIdx), size(matches12, 1), ...
            length(inlierIdx)/size(matches12, 1), trialCount);

    
    % Shows result
    figure(iPair * 2 - 1); clf
    if DRAW_ALL_PUTATIVE
        Utils.pcshow_matches(pointcloud{1}, pointcloud{2}, ...
                         result{1}.xyz, result{2}.xyz, ...
                         matches12, 'inlierIdx', inlierIdx, 'k', MAX_MATCHES);
    else
        Utils.pcshow_matches(pointcloud{1}, pointcloud{2}, ...
                             result{1}.xyz, result{2}.xyz, ...
                             matches12(inlierIdx, :), 'k', MAX_MATCHES);
    end
    title('Matches')

    % Show alignment
    figure(iPair * 2); clf
    Utils.pcshow_multiple(pointcloud, {eye(4), estimateRt});
    title('Alignment')
        
end


