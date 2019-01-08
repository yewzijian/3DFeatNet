% Processes a single folder
function num_clouds = BuildPointclouds(folder, outFolder, origin_pose)

addpath('../common');  % for findPointNormals()

MIN_SPEED = 0.2;  % Minimum Speed, so remove instances where car was stationary
ACCUMULATE_DISTANCE = 60;  % Accumulate 60m of travelling to form one point cloud
METERS_PER_POINT_CLOUD = 10;  % Distance between point cloud
IGNORE_FIRST_N_SEC = 20;  % Ignore first n seconds (for INS to initialize)

laser_dir = fullfile(folder, 'lms_front/');
ins_file = fullfile(folder, 'gps/ins.csv');
extrinsics_dir = './oxford_extrinsics/';
[~, dataset, ~] = fileparts(folder(1:end));

laser = regexp(laser_dir, '(lms_front|lms_rear|ldmrs)', 'match');
laser = laser{end};
laser_timestamp_fname = [laser_dir '../' laser '.timestamps'];
if ~exist(laser_timestamp_fname, 'file')
    warning('Invalid dataset')
    num_clouds = 0;
    return
end
laser_timestamps = dlmread(laser_timestamp_fname);

outFolder = fullfile(outFolder, dataset);
mkdir(outFolder);

metadata_fid = fopen(fullfile(outFolder, 'metadata.txt'), 'w');
fprintf(metadata_fid, 'Idx\tDataset\tStartIdx\tEndIdx\tNumPts\tX\tY\tZ\n');

%%
fprintf('Processing dataset: %s\n', dataset);

% Load transforms between laser/ins and vehicle
laser_extrinisics = dlmread([extrinsics_dir laser '.txt']);
ins_extrinsics = dlmread([extrinsics_dir 'ins.txt']);
G_ins_laser = SE3MatrixFromComponents(ins_extrinsics) \ ...
      SE3MatrixFromComponents(laser_extrinisics);

% Load INS file
ins_data = LoadInsFile(ins_file);
ins_timestamps = ins_data{1};

jumps = detectJumps(ins_data);

% Filter out times before/after INS recording
start_timestamp = max(laser_timestamps(1, 1), ins_timestamps(1) + IGNORE_FIRST_N_SEC*1e6);
end_timestamp = min(laser_timestamps(end, 1), ins_timestamps(end));
mask = laser_timestamps(:,1) >= start_timestamp & laser_timestamps(:,1) <= end_timestamp;
mask = mask & getJumpMask(laser_timestamps, jumps);
laser_timestamps = laser_timestamps(mask, :);

% Retrieves the INS_poses
fprintf('Interpolating poses... ')
[ins_poses, ins_vel] = InterpolatePoses(ins_file, laser_timestamps(:,1)');
    G_ins_laser = SE3MatrixFromComponents(ins_extrinsics) \ ...
      SE3MatrixFromComponents(laser_extrinisics);
disp('done')

% Mask away stationary frames
ins_speed = sqrt(sum(ins_vel.^2, 2));
mask_moving = ins_speed > MIN_SPEED;
laser_timestamps = laser_timestamps(mask_moving, :);
ins_poses = ins_poses(mask_moving, :);
ins_vel = ins_vel(mask_moving, :);
numFrames = size(laser_timestamps, 1);

% Apply origin offset
ins_poses_w_offset = OffsetPoses(ins_poses, origin_pose);

ins_positions = [ins_poses{:}];
ins_positions = ins_positions(1:3, 4:4:end)';
fprintf('Running pairwise distances between INS positions... ')
distances = compute_subsequent_offsets(ins_positions, 5000);
max_distances = max(distances, [], 2);
disp('Done')

startIdx = 1;
iCloud = 0;

while true
    %% Build one point cloud at a time
    endIdx = GetEndIdx(distances, startIdx, ACCUMULATE_DISTANCE);
    if endIdx < 1
        assert(startIdx + 5000 > length(ins_positions))
        break
    end
    
    if distances(startIdx, endIdx-startIdx-1) > ACCUMULATE_DISTANCE - 5  % Ensure that the distance didn't go over threshold because of a GPS jump
        [pcloud, reflectance] = ...
            BuildSinglePointCloud(laser_dir, ...
                             laser_timestamps(startIdx:endIdx,:), ...
                             ins_poses_w_offset(startIdx:endIdx,:), G_ins_laser);
        pcloud = pointCloud(pcloud', 'Intensity', reflectance);

        % Processes (crops and centers) the point cloud
        [pcloud, transform] = processPointCloud(pcloud, false);

        % Save out the point cloud
        fname = fullfile(outFolder, sprintf('%i.bin', iCloud));
        Utils.savePointCloud(pcloud, fname)

        % Write out metadata
        fprintf(metadata_fid, '%i\t%s\t%i\t%i\t%i\t%f\t%f\t%f\n', ...
            iCloud, dataset, ...
            laser_timestamps(startIdx, 1), laser_timestamps(endIdx, 1), size(pcloud.Location, 1), ...
            transform(1, 4), transform(2, 4), transform(3,4));

        fprintf('Processed %i-%i out of %i frames \n', startIdx, endIdx, numFrames);
    end
    
    startIdx = GetEndIdx(distances, startIdx, METERS_PER_POINT_CLOUD);
    if max_distances(startIdx) < ACCUMULATE_DISTANCE && startIdx + 5000 <= length(ins_positions)
        offset = find(max_distances(startIdx:end) > ACCUMULATE_DISTANCE, 1, 'first');
        assert(~isempty(offset));
        warning(sprintf('Skipping %i frames due to accumulate distance not satisfied', offset-1))
        startIdx = startIdx + offset - 1;
    end
    
    iCloud = iCloud + 1;
    
end

num_clouds = iCloud;
fclose(metadata_fid);

end


function endIdx = GetEndIdx(distance_matrix, startIdx, distTravelled)
% Given the start idx, returns the index which corresponds to having
% travelled "distTravelled" distance.

    if startIdx >= size(distance_matrix, 1)
        endIdx = -1;
    else
        offset = find(distance_matrix(startIdx, :) > distTravelled, 1, 'first');
        if isempty(offset)
            endIdx = -1;
        else
            endIdx = startIdx + offset;
        end
    end

end

function jumps = detectJumps(ins_data)
% Detect jumps in INS positional data. Uses a simple heuristic of
% detecting consecutive positions which are more than 5m apart.

    THRESH = 5;

    ins_ned = [ins_data{6:8}];

    diff = sqrt(sum((ins_ned(2:end, :) - ins_ned(1:end-1,:)) .^2, 2));

    jumps_mask = diff > THRESH;
    jumps = ins_data{1}(jumps_mask);
end

function mask = getJumpMask(laser_timestamps, jumps)
% Returns a mask that masks out 30s before and 10s after each INS jump

    TIME_BEFORE = 30e6;
    TIME_AFTER = 10e6;

    mask = ones(size(laser_timestamps, 1), 1);
    
    for i = 1 : length(jumps)
        s = jumps(i);
        mask(laser_timestamps(:,1) > (s - TIME_BEFORE) & laser_timestamps(:,1) < (s + TIME_AFTER)) = 0;
    end

end

function distances = compute_subsequent_offsets(X, k)
% compute_subsequent_offsets - Computes distance between each INS frame and
%   the subsequent k timesteps

N = size(X, 1);  % number of instances
if nargin < 2
    k = 2500;
end

distances = zeros(N, k);
for i = 1 : N
    k1 = min(k, N - i);
    distances(i,1:k1) = sqrt(sum((X(i+1:i+k1, :) - X(i,:)) .^ 2, 2));
end

end


function [pointcloud, reflectance] = BuildSinglePointCloud(laser_dir, laser_timestamps, ins_poses, G_ins_laser)
% BuildSinglePointCloud - builds a 3-dimensional pointcloud from multiple 
%   2-dimensional LIDAR scans, given timestamps

  assert(size(laser_timestamps,1) == size(ins_poses, 1))

  if laser_dir(end) ~= '/'
    laser_dir = [laser_dir '/'];
  end
  
  n = size(laser_timestamps,1);
  pointcloud = [];
  reflectance = [];
  for i=1:n
    scan_path = [laser_dir num2str(laser_timestamps(i,1)) '.bin'];
    if ~exist(scan_path, 'file')
      continue;
    end
    scan_file = fopen(scan_path);
    scan = fread(scan_file, 'double');
    fclose(scan_file);
    
    % The scan file contains repeated tuples of three values
    scan = reshape(scan, [3 numel(scan)/3]);
    if regexp(laser_dir, '(lms_rear|lms_front)')
      % LMS tuples are of the form (x, y, R)
      reflectance = [reflectance scan(3,:)];
      scan(3,:) = zeros(1, size(scan,2));
    end

    % Transform scan to INS frame, move to the INS pose at the scan's timestamp,
    % then transform back to LIDAR frame
    scan = ins_poses{i} * G_ins_laser * [scan; ones(1, size(scan,2))];
    pointcloud = [pointcloud scan(1:3,:)];
    
  end
  
  if size(pointcloud) == 0
    error(['No valid scans found. Missing chunk ' num2str(laser_timestamps(end,2))]);
  end
  

end
