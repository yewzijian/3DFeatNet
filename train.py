import argparse
import coloredlogs, logging
import logging.config
import numpy as np
import os
import sys
import tensorflow as tf

from models.net_factory import get_network
from config import *
from data.datagenerator import DataGenerator
from data.augment import get_augmentations_from_list
from utils import get_tensors_in_checkpoint_file

NUM_CLUSTERS = 512
UPRIGHT_AXIS = 2  # Will learn invariance along this axis
VAL_PROPORTION = 1.0

# Arguments
parser = argparse.ArgumentParser(description='Trains pointnet')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use (default: 0)')
# data
parser.add_argument('--data_dim', type=int, default=6,
                    help='Input dimension for data. Note: Feat3D-Net will only use the first 3 dimensions (default: 6)')
parser.add_argument('--data_dir', type=str, default='../data/oxford',
                    help='Path to dataset. Should contain "train" and "clusters" folders')
# Model
parser.add_argument('--model', type=str, default='3DFeatNet', help='Model to load')
parser.add_argument('--noregress', action='store_true',
                    help='If set, regression of feature orientation will not be performed')
parser.add_argument('--noattention', action='store_true',
                    help='If set, model will not learn to predict attention')
parser.add_argument('--margin', type=float, default=0.2,
                    help='Margin for triplet loss. Default=0.2')
parser.add_argument('--feature_dim', type=int, default=32, choices=[16, 32, 64, 128],
                    help='Feature dimension size')
# Data
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points to downsample model to')
parser.add_argument('--base_scale', type=float, default=2.0,
                    help='Base scale (radius) for sampling clusters. Set to around 2.0 for oxford dataset')
parser.add_argument('--num_samples', type=int, default=64,
                    help='Maximum number of points to consider per cluster (default: 64)')
parser.add_argument('--augmentation', type=str, nargs='+', default=['Jitter', 'RotateSmall', 'Shift', 'Rotate1D'],
                    choices=['Jitter', 'RotateSmall', 'Rotate1D', 'Rotate3D', 'Scale', 'Shift'],
                    help='Data augmentation settings to use during training')
# Logging
parser.add_argument('--log_dir', type=str, default='./ckpt',
                    help='Directory to save tf summaries, checkpoints, and log. Default=./ckpt')
parser.add_argument('--ignore_missing_vars', action='store_true',
                    help='Whether to crash if variables are missing')
parser.add_argument('--summary_every_n_steps', type=int, default=20,
                    help='Save to tf summary every N steps. Default=20')
parser.add_argument('--validate_every_n_steps', type=int, default=250,
                    help='Run over validation data every n steps. Default=250')
# Saver
parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint to restore from (optional)')
parser.add_argument('--checkpoint_every_n_steps', type=int, default=500,
                    help='Save a checkpoint every n steps. Default=500')
parser.add_argument('--restore_exclude', type=str, nargs='+', default=None,
                    help='To ignore from checkpoint')
# Training
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Number of epochs to train for. Default=1000')
args = parser.parse_args()

# Prepares the folder for saving checkpoints, summary, logs
log_dir = args.log_dir
checkpoint_dir = os.path.join(log_dir, 'ckpt')
os.makedirs(checkpoint_dir, exist_ok=True)

# Create Logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

fileHandler = logging.FileHandler("{0}/log.txt".format(checkpoint_dir))
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)


def log_arguments():
    logger.info('Command: %s', ' '.join(sys.argv))

    s = '\n'.join(['    {}: {}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    s = 'Arguments:\n' + s
    logger.info(s)


def train():

    log_arguments()

    # Training Data
    train_file = os.path.join(args.data_dir, 'train/train.txt')
    train_data = DataGenerator(train_file, num_cols=args.data_dim)

    logger.info('Loaded train data: %s (# instances: %i)',
                train_file, train_data.size)

    train_augmentations = get_augmentations_from_list(args.augmentation, upright_axis=UPRIGHT_AXIS)

    # Validation data validation
    val_folder = os.path.join(args.data_dir, 'clusters')
    val_groundtruths = load_validation_groundtruths(os.path.join(val_folder, 'filenames.txt'), proportion=VAL_PROPORTION)

    # Model
    param = {'NoRegress': args.noregress, 'BaseScale': args.base_scale, 'Attention': not args.noattention,
             'margin': args.margin, 'num_clusters': NUM_CLUSTERS, 'num_samples': args.num_samples,
             'feature_dim': args.feature_dim, 'freeze_scopes': None,
             }
    model = get_network(args.model)(param)

    # placeholders
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool)
    anchor_pl, positive_pl, negative_pl = model.get_placeholders(args.data_dim)

    # Ops
    xyz_op, features_op, anchor_attention_op, end_points = model.get_train_model(anchor_pl, positive_pl, negative_pl, is_training, use_bn=USE_BN)
    loss_op, end_points = model.get_loss(xyz_op, features_op, anchor_attention_op, end_points)
    train_op = model.get_train_op(loss_op, global_step=global_step)

    # Saver and summary writers
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
    train_writer, test_writer = get_summary_writers(args.log_dir)
    summary_op = tf.summary.merge_all()

    logger.info('Training Batch size: %i, validation batch size: %i', BATCH_SIZE, VAL_BATCH_SIZE)

    with tf.Session(config=config) as sess:

        initialize_model(sess, args.checkpoint,
                         ignore_missing_vars=args.ignore_missing_vars,
                         restore_exclude=args.restore_exclude)
        train_writer.add_graph(sess.graph)

        for iEpoch in range(args.num_epochs):

            logger.info('Starting epoch %i', iEpoch)

            train_data.shuffle()

            # Training data
            while True:
                anchors, positives, negatives = train_data.next_triplet(k=BATCH_SIZE,
                                                                        num_points=args.num_points,
                                                                        augmentation=train_augmentations)
                if anchors is None or anchors.shape[0] != BATCH_SIZE:
                    break

                xyz, features, train_loss, step, summary, ep, _ = \
                    sess.run([xyz_op, features_op, loss_op, global_step, summary_op, end_points, train_op],
                             feed_dict={anchor_pl: anchors, positive_pl: positives, negative_pl: negatives,
                                        is_training: True})

                if step % args.summary_every_n_steps == 0:
                    train_writer.add_summary(summary, step)
                if step % args.checkpoint_every_n_steps == 0:
                    saver.save(sess, os.path.join(checkpoint_dir, 'checkpoint.ckpt'), global_step=step)

                sys.stdout.write('\rStep {}, Loss: {}'.format(step, train_loss))

                # # Run through validation data
                if step % args.validate_every_n_steps == 0 or step == 1:

                    print()
                    # ---------------------------- TEST EVAL -----------------------
                    fp_rate = validate(sess, end_points, is_training, val_folder, val_groundtruths, args.data_dim)

                    test_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="fp_rate", simple_value=fp_rate),
                    ])
                    test_writer.add_summary(test_summary, step)
                    logger.info('Step %i. FP Rate: %f', step, fp_rate)
                    # ---------------------------- TEST EVAL End -----------------------

                    test_writer.flush()
                    train_writer.flush()

            print()


def initialize_model(sess, checkpoint, ignore_missing_vars=False, restore_exclude=None):
    """ Initialize model weights

    :param sess: tf.Session
    :param checkpoint: Checkpoint to load. Set to none if starting from scratch
    :param ignore_missing_vars: If false, will throw exception when variables are not found
                                in the checkpoint
    :param restore_exclude: Scopes to exclude from checkpoint
    :return:
    """
    logger.info('Initializing weights')

    sess.run(tf.global_variables_initializer())

    if checkpoint is not None:

        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)

        logger.info('Restoring model from {}'.format(checkpoint))

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


def get_summary_writers(log_dir):

    logger.info('Summaries will be stored in: %s', log_dir)
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
    test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

    return train_writer, test_writer


def load_validation_groundtruths(fname, proportion=1):
    groundtruths = []
    iGt = 0
    with open(fname) as fid:
        fid.readline()
        for line in fid:
            groundtruths.append((iGt, int(line.split()[-1])))
            iGt += 1

    if 0 < proportion < 1:
        skip = int(1.0/proportion)
        groundtruths = groundtruths[0::skip]

    return groundtruths


def validate(sess, end_points, is_training, val_folder, val_groundtruths, data_dim):

    if val_groundtruths is None or len(val_groundtruths) == 0:
        return 1

    positive_dist = []
    negative_dist = []
    for iTest in range(0, len(val_groundtruths), NUM_CLUSTERS):

        clouds1, clouds2 = [], []
        # We batch the validation by stacking all the validation clusters into a single point cloud,
        # while keeping them apart such that they do not overlap each other. This way NUM_CLUSTERS
        # clusters can be computed in a single pass
        for jTest in range(iTest, min(iTest + NUM_CLUSTERS, len(val_groundtruths))):
            offset = (jTest - iTest) * 100
            cluster_idx = val_groundtruths[jTest][0]

            cloud1 = DataGenerator.load_point_cloud(
                os.path.join(val_folder, '{}_0.bin'.format(cluster_idx)), data_dim)
            cloud1[:, 0] += offset
            clouds1.append(cloud1)

            cloud2 = DataGenerator.load_point_cloud(
                os.path.join(val_folder, '{}_1.bin'.format(cluster_idx)), data_dim)
            cloud2[:, 0] += offset
            clouds2.append(cloud2)

        offsets = np.arange(0, NUM_CLUSTERS * 100, 100)
        num_clusters = min(len(val_groundtruths) - iTest, NUM_CLUSTERS)
        offsets[num_clusters:] = 0
        offsets = np.pad(offsets[:, None], ((0, 0), (0, 2)), mode='constant', constant_values=0)[None, :, :]

        clouds1 = np.concatenate(clouds1, axis=0)[None, :, :]
        clouds2 = np.concatenate(clouds2, axis=0)[None, :, :]

        xyz1, features1 = \
            sess.run([end_points['output_xyz'], end_points['output_features']],
                     feed_dict={end_points['input_pointclouds']: clouds1, is_training: False,
                                end_points['keypoints']: offsets})
        xyz2, features2 = \
            sess.run([end_points['output_xyz'], end_points['output_features']],
                     feed_dict={end_points['input_pointclouds']: clouds2, is_training: False,
                                end_points['keypoints']: offsets})

        d = np.sqrt(np.sum(np.square(np.squeeze(features1 - features2)), axis=1))
        d = d[:num_clusters]

        positive_dist += [d[i] for i in range(len(d)) if val_groundtruths[iTest + i][1] == 1]
        negative_dist += [d[i] for i in range(len(d)) if val_groundtruths[iTest + i][1] == 0]

    d_at_95_recall = np.percentile(positive_dist, 95)
    num_FP = np.count_nonzero(np.array(negative_dist) < d_at_95_recall)
    num_TN = len(negative_dist) - num_FP
    fp_rate = num_FP / (num_FP + num_TN)

    return fp_rate


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    gpu_string = '/gpu:{}'.format(args.gpu)
    config.gpu_options.allow_growth = True
    with tf.device(gpu_string):
        train()
