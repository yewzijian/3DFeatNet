'''
Restores a trained 3DFeat-Net model and uses it to extract keypoints + descriptors on all point clouds in the
input folder

Author: Zi Jian Yew <zijian.yew@comp.nus.edu.sg>
'''

import argparse
import logging
import logging.config
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

from config import *
from data.datagenerator import DataGenerator
from models.net_factory import get_network
from utils import get_tensors_in_checkpoint_file

# Defaults
CKPT_PATH = './ckpt/checkpoint.ckpt'
MAX_POINTS = 30000

# Arguments
parser = argparse.ArgumentParser(description='Trains pointnet')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use (default: 0)')
# Model
parser.add_argument('--model', type=str, default='3DFeatNet',
                    help='Model to load')
# Data
parser.add_argument('--data_dim', type=int, default=6,
                    help='Input dimension for data. Note: Feat3D-Net will only use the first 3 dimensions (default: 6)')
parser.add_argument('--num_points', type=int, default=-1,
                    help='Number of points to downsample model to. (default:-1, i.e no downsample)')
parser.add_argument('--base_scale', type=float, default=2.0,
                    help='Radius for sampling clusters (default: 2.0)')
parser.add_argument('--num_samples', type=int, default=64,
                    help='Maximum number of points to consider per cluster (default: 64)')
parser.add_argument('--use_keypoints_from', default=None,
                    help='If set, keypoints will be loaded from this folder.')
parser.add_argument('--feature_dim', type=int, default=32, choices=[16, 32, 64, 128],
                    help='Feature dimension size (default: 32)')
parser.add_argument('--randomize_points', action='store_true')
# Inference
parser.add_argument('--nms_radius', type=float, default=0.5,
                    help='Radius for non-maximal suppression (default: 0.5)')
parser.add_argument('--min_response_ratio', type=float, default=1e-2,
                    help='Keypoints with response below this ratio from the max will be pruned away')
parser.add_argument('--max_keypoints', type=int, default=1024,
                    help='Maximum number of keypoints to detect')
# Input/checkpoint/output paths
parser.add_argument('--data_dir', type=str,
                    help='Directory to save results to')
parser.add_argument('--checkpoint', type=str, default=CKPT_PATH,
                    help='Checkpoint to restore from (optional)')
parser.add_argument('--output_dir', type=str,
                    help='Directory to save results to')
args = parser.parse_args()

# Create Logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def compute_descriptors():

    log_arguments()

    logger.debug('In compute_descriptors()')
    logger.info('Computed descriptors will be saved to %s', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    binFiles = [f for f in os.listdir(args.data_dir) if f.endswith('.bin')]
    data_dim = args.data_dim
    logger.info('Found %i bin files in directory: %s, each assumed to be of dim %i',
                len(binFiles), args.data_dir, data_dim)

    # Model
    param = {'NoRegress': False, 'BaseScale': args.base_scale, 'Attention': True,
             'num_clusters': -1, 'num_samples': args.num_samples, 'feature_dim': args.feature_dim}
    model = get_network(args.model)(param)

    # placeholders
    is_training = tf.placeholder(tf.bool)
    cloud_pl, _, _ = model.get_placeholders(data_dim)

    # Ops1
    xyz_op, features_op, attention_op, end_points = model.get_inference_model(cloud_pl, is_training, use_bn=USE_BN)

    with tf.Session(config=config) as sess:

        initialize_model(sess, args.checkpoint)

        num_processed = 0

        # Training data
        for iBin in range(0, len(binFiles)):

            binFile = binFiles[iBin]
            fname_no_ext = binFile[:-4]
            pointcloud = DataGenerator.load_point_cloud(os.path.join(args.data_dir, binFile), num_cols=data_dim)

            if args.randomize_points:
                permutation = np.random.choice(pointcloud.shape[0], size=pointcloud.shape[0], replace=False)
                inv_permutation = np.zeros_like(permutation)
                inv_permutation[permutation] = range(0, pointcloud.shape[0])
                pointcloud = pointcloud[permutation, :]
            else:
                inv_permutation = np.arange(0, pointcloud.shape[0], dtype=np.int64)
            if args.num_points > 0:
                pointcloud = pointcloud[:args.num_points, :]

            pointclouds = pointcloud[None, :, :]
            num_models = pointclouds.shape[0]

            if args.use_keypoints_from is None:
                # Detect features

                # Compute attention in batches due to limited memory
                xyz, attention = [], []
                for startPt in range(0, pointcloud.shape[0], MAX_POINTS):
                    endPt = min(pointcloud.shape[0], startPt + MAX_POINTS)
                    xyz_subset = pointclouds[:, startPt:endPt, :3]

                    # Compute attention over all points
                    xyz_cur, attention_cur = \
                        sess.run([xyz_op, attention_op],
                                 feed_dict={cloud_pl: pointclouds, is_training: False,
                                            end_points['keypoints']: xyz_subset})

                    xyz.append(xyz_cur)
                    attention.append(attention_cur)

                xyz = np.concatenate(xyz, axis=1)
                attention = np.concatenate(attention, axis=1)

                # # Uncomment to save out attention to file
                # with open(os.path.join(args.output_dir, '{}_attention.bin'.format(fname_no_ext)), 'wb') as f:
                #     if args.num_points > 0:
                #         xyz_attention = np.concatenate((xyz[0, :, :],
                #                                         np.expand_dims(attention[0, :], 1),), axis=1)
                #     else:
                #         xyz_attention = np.concatenate((xyz[0, inv_permutation, :],
                #                                         np.expand_dims(attention[0, inv_permutation], 1),), axis=1)
                #     xyz_attention.tofile(f)

                # Non maximal suppression to select keypoints based on attention
                xyz_nms, attention_nms, num_keypoints = nms(xyz, attention)

            else:
                # Load keypoints from file
                xyz_nms = []
                for i in range(num_models):
                    kp_fname = os.path.join(args.use_keypoints_from, '{}_kp.bin'.format(fname_no_ext))
                    xyz_nms.append(DataGenerator.load_point_cloud(kp_fname, num_cols=3))

                # Pad to make same size
                num_keypoints = [kp.shape[0] for kp in xyz_nms]
                largest_kp_count = max(num_keypoints)
                for i in range(num_models):
                    num_to_pad = largest_kp_count-xyz_nms[i].shape[0]
                    to_pad_with = np.repeat(xyz_nms[i][0,:][None, :], num_to_pad, axis=0)
                    xyz_nms[i] = np.concatenate((xyz_nms[i], to_pad_with), axis=0)
                xyz_nms = np.stack(xyz_nms, axis=0)

            # Compute features
            xyz, features = \
                sess.run([xyz_op, features_op],
                         feed_dict={cloud_pl: pointclouds, is_training: False, end_points['keypoints']: xyz_nms})

            # Save out the output
            with open(os.path.join(args.output_dir, '{}.bin'.format(fname_no_ext)), 'wb') as f:
                xyz_features = np.concatenate([xyz[0, 0:num_keypoints[0], :], features[0, 0:num_keypoints[0], :]],
                                              axis=1)
                xyz_features.tofile(f)

            num_processed += 1
            logger.info('Processed %i / %i images', num_processed, len(binFiles))


def initialize_model(sess, checkpoint, ignore_missing_vars=False, restore_exclude=None):
    logger.info('Initializing weights')

    sess.run(tf.global_variables_initializer())

    if checkpoint is not None:

        logger.info('Restoring model from {}'.format(args.checkpoint))

        model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        exclude_list = []
        if restore_exclude is not None:
            for e in restore_exclude:
                exclude_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=e)
        for e in exclude_list:
            logger.info('Excluded from model restore: %s', e.op.name)

        if ignore_missing_vars:
            checkpoint_var_names = get_tensors_in_checkpoint_file(checkpoint)
            missing = [m.op.name for m in model_var_list if m.op.name not in checkpoint_var_names and m not in exclude_list]

            for m in missing:
                logger.warning('Variable missing from checkpoint: %s', m)

            var_list = [m for m in model_var_list if m.op.name in checkpoint_var_names and m not in exclude_list]

        else:
            var_list = [m for m in model_var_list if m not in exclude_list]

        saver = tf.train.Saver(var_list)

        saver.restore(sess, checkpoint)

    logger.info('Weights initialized')


def log_arguments():
    s = '\n'.join(['    {}: {}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    s = 'Arguments:\n' + s
    logger.info(s)


def nms(xyz, attention):

    num_models = xyz.shape[0]  # Should be equals to batch size
    num_keypoints = [0] * num_models

    xyz_nms = np.zeros((num_models, args.max_keypoints, 3), xyz.dtype)
    attention_nms = np.zeros((num_models, args.max_keypoints), xyz.dtype)

    for i in range(num_models):

        nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(xyz[i, :, :])
        distances, indices = nbrs.kneighbors(xyz[i, :, :])

        knn_attention = attention[i, indices]
        outside_ball = distances > args.nms_radius
        knn_attention[outside_ball] = 0.0
        is_max = np.where(np.argmax(knn_attention, axis=1) == 0)[0]

        # Extract the top k features, filtering out weak responses
        attention_thresh = np.max(attention[i, :]) * args.min_response_ratio
        is_max_attention = [(attention[i, m], m) for m in is_max if attention[i, m] > attention_thresh]
        is_max_attention = sorted(is_max_attention, reverse=True)
        max_indices = [m[1] for m in is_max_attention]

        if len(max_indices) >= args.max_keypoints:
            max_indices = max_indices[:args.max_keypoints]
            num_keypoints[i] = len(max_indices)
        else:
            num_keypoints[i] = len(max_indices)  # Retrain original number of points
            max_indices = np.pad(max_indices, (0, args.max_keypoints - len(max_indices)), 'constant',
                                 constant_values=max_indices[0])

        xyz_nms[i, :, :] = xyz[i, max_indices, :]
        attention_nms[i, :] = attention[i, max_indices]

    return xyz_nms, attention_nms, num_keypoints


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu_string = '/gpu:{}'.format(args.gpu)

    with tf.device(gpu_string):
        compute_descriptors()
