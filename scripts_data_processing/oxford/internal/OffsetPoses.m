function poses = OffsetPoses(poses, origin)

% Shifts the transformations w.r.t. to the origin pose
poses_array = horzcat(poses{:});
poses_array = origin \ poses_array;
poses_3d = reshape(poses_array, [4, 4, length(poses)]);
poses = squeeze(num2cell(poses_3d, [1,2]));

end