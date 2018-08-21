""" PointNet++ Layers, originally from Charles R. Qi, modified by Zi Jian Yew to add
    query_and_group_points() sample_points(), as well as functions. Note that certain functionality
    in tf_util are not included here as it is not used.
"""

import tensorflow as tf
import numpy as np

from models.layers import conv2d
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point


def sample_points(xyz, npoint):
    '''

    :param xyz:
    :param npoint:
    :param knn:
    :param use_xyz:
    :return: new_xyz - Cluster centers
    '''

    if npoint <= 0:
        new_xyz = tf.identity(xyz)
    else:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))

    return new_xyz


def query_and_group_points(xyz, points, new_xyz, nsample, radius, knn=False,
                           use_xyz=True, normalize_radius=True, orientations=None):

    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
        pts_cnt = nsample  # Hack. By right should make sure number of input points < nsample
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    tf.summary.histogram('pts_cnt', pts_cnt)

    # Group XYZ coordinates
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if normalize_radius:
        grouped_xyz /= radius  # Scale normalization
    # 2D-rotate via orientations if necessary
    if orientations is not None:
        cosval = tf.expand_dims(tf.cos(orientations), axis=2)
        sinval = tf.expand_dims(tf.sin(orientations), axis=2)
        grouped_xyz = tf.stack([cosval * grouped_xyz[:, :, :, 0] + sinval * grouped_xyz[:, :, :, 1],
                                -sinval * grouped_xyz[:, :, :, 0] + cosval * grouped_xyz[:, :, :, 1],
                                grouped_xyz[:, :, :, 2]], axis=3)

    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:

            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_points, idx


def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True,
                     keypoints=None, orientations=None, normalize_radius=False):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        keypoints: None or tensor with shape [None, None, 3], containing the xyz of keypoints.
                   If provided, npoint will be ignored, and iterative furthest sampling will be skipped
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor, i.e. cluster center (dim=3)
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor (dim=3+c, first 3 dimensions are normalized XYZ)
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions. This is usually the first 3 dimensions of new_points
    '''

    end_points = {}

    if keypoints is not None:
        new_xyz = keypoints
    else:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)

    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
        pts_cnt = nsample  # Hack. By right should make sure number of input points < nsample
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if normalize_radius:
        grouped_xyz /= radius

    end_points['grouped_xyz_before'] = grouped_xyz

    # 2D-rotate via orientations if necessary
    if orientations is not None:
        cosval = tf.cos(orientations)
        sinval = tf.sin(orientations)
        one = tf.ones_like(cosval)
        zero = tf.zeros_like(cosval)
        R = tf.stack([(cosval, sinval, zero), (-sinval, cosval, zero), (zero, zero, one)], axis=0)
        R = tf.transpose(R, perm=[2,3,0,1])
        grouped_xyz = tf.matmul(grouped_xyz, R)
        end_points['rotation'] = R

    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    end_points['grouped_xyz'] = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz, end_points


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = tf.shape(xyz)[0]
    nsample = xyz.get_shape()[1].value

    new_xyz = tf.zeros((batch_size, 1, 3), tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (2, 1, 1)))
    idx = tf.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
