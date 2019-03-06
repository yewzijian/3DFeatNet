% Evaluates descriptor + detector.
% Assumes computed descriptors are stored under '../../results/oxford/f3d_d32'

addpath('./external');

clear;
m = 6;  % Dimensionality of raw data

INTERSECTION_DISTANCE_THRESH = 0.75;  % Keypoints without any points in the other model within this distance will be ignored
CORRECT_MATCH_THRESH = 1.0;

DATA_FOLDER = '../../data/oxford/test_models';
RESULT_FOLDER = '../test_results';

ALGO = 'f3d_d32'; featureDim = 32;

%% Load pairs
algoResultFolder = fullfile(RESULT_FOLDER, ALGO);
test_pairs = readtable(fullfile(DATA_FOLDER, 'groundtruths.txt'));

tic
for iPair = 1 : height(test_pairs)
    
    frames = [test_pairs.idx1(iPair), test_pairs.idx2(iPair)];
    fprintf('Running pair %i of %i, containing frames %i and %i\n', ...
        iPair, height(test_pairs), frames(1), frames(2));
    
    % Load point cloud and descriptors
    for i = 1 : 2
        r = frames(i);

        pointcloud{i} = Utils.loadPointCloud(fullfile(DATA_FOLDER, sprintf('%d.bin', r)), m);

        binfile = fullfile(algoResultFolder, sprintf('%d.bin', r));
        xyz_features = Utils.load_descriptors(binfile, sum(featureDim+3));

        result{i}.xyz = xyz_features(:, 1:3);
        result{i}.desc = xyz_features(:, 4:end);
    end
    
    % Load Groundtruth
    t_gt = [test_pairs.t_1(iPair), test_pairs.t_2(iPair), test_pairs.t_3(iPair)];
    q_gt = [test_pairs.q_1(iPair), test_pairs.q_2(iPair), test_pairs.q_3(iPair), test_pairs.q_4(iPair)];
    T_gt = [quat2rotm(q_gt) t_gt'; 0 0 0 1];
    
    % Count number of matches in the region of intersection of the two
    % point clouds
    pointcloud2_warped = Utils.apply_transform(pointcloud{2}, T_gt);
    D = pdist2(pointcloud2_warped, result{1}.xyz, 'euclidean', 'smallest', 1);
    inIntersection = D < INTERSECTION_DISTANCE_THRESH;
    
    %% Find nearest match
    [matchDist, matches12] = pdist2(result{2}.desc, result{1}.desc, 'euclidean', 'smallest', 1);
    matches12 = [1:length(matches12); matches12]';
    
    %% Evaluate matching performance
    
    % Compute number of correct matches
    pts1 = result{1}.xyz(matches12(:,1), :);
    pts2 = result{2}.xyz(matches12(:,2), :);
    pts2_transformed = Utils.apply_transform(pts2, T_gt);
    delta = sqrt(sum((pts1 - pts2_transformed).^2, 2));
    
    isCorrect = delta < CORRECT_MATCH_THRESH;
    
    % Consider only intersection (since two point clouds won't have the
    % exact same coverage)
    isCorrectMasked = isCorrect(inIntersection);
    matchDistMasked = matchDist(inIntersection);
    deltaMasked = delta(inIntersection);
    
    distTable = [matchDistMasked' isCorrectMasked];
    
    fprintf('Number of correct @ %.1f meters: %i / %i\n', ...
        CORRECT_MATCH_THRESH, nnz(isCorrectMasked), nnz(inIntersection));
    
    %%
    statistic(iPair).idx = iPair;
    statistic(iPair).num_putative = nnz(inIntersection);
    statistic(iPair).num_correct = nnz(isCorrectMasked);
    statistic(iPair).distTable = distTable;
    statistic(iPair).nearestMatchDist = deltaMasked;
    
end

%% Stores the result
if ~exist('results_oxford', 'dir')
    mkdir('results_oxford')
end
statisticTable = struct2table(statistic);
save(sprintf('results_oxford/matching_statistic-%s.mat', ALGO), 'statisticTable');