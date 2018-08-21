import logging
import numpy as np
import random
import os
from collections import deque


class DataGenerator(object):

    def __init__(self, filename, num_cols=6):
        """ Constructor to data generator

        Args:
            filename: Path to dataset
            num_cols (int): Number of columns in binary file
        """

        self.dataset = os.path.split(os.path.split(filename)[0])[1]

        self.logger = logging.getLogger(self.__class__.__name__)

        self.paths_and_labels = []
        self.load_metadata(filename)
        self.logger.info('Loaded metadata file')

        self.num_cols = num_cols
        self.size = len(self.paths_and_labels)  # Number of data instances
        self.indices = deque(range(self.size))

        self.data = [None] * self.size

    def load_metadata(self, path):
        self.paths_and_labels = []
        with open(path) as f:
            for line in f:
                fname, positives, negatives = [l.strip() for l in line.split('|')]
                positives = [int(s) for s in positives.split()]
                nonnegatives = [int(s) for s in negatives.split()]

                self.paths_and_labels.append((fname, set(positives), set(nonnegatives)))

    def reset(self):
        """ Resets the data generator, so that it returns the first instance again.
            Either this or shuffle() should be called
        """
        self.indices = deque(list(range(len(self.data))))

    def shuffle(self):
        ''' Shuffle training data. This function should be called at the start of each epoch
        '''
        ind = list(range(len(self.data)))
        random.shuffle(ind)
        self.indices = deque(ind)

    def next_triplet(self, k=1, num_points=4096, augmentation=[]):
        """ Retrieves the next triplet(s) for training

        Args:
            k (int): Number of triplets
            num_points: Number of points to downsample pointcloud to
            augmentation: Types of augmentation to perform

        Returns:
            (anchors, positives, negatives)
        """

        anchors, positives, negatives = [], [], []

        for _ in range(k):
            try:
                i_anchor = self.indices.popleft()
                i_positive, i_negative = self.get_positive_negative(i_anchor)
            except IndexError:
                break

            anchor = self.get_point_cloud(i_anchor)
            positive = self.get_point_cloud(i_positive)
            negative = self.get_point_cloud(i_negative)

            anchor = self.process_point_cloud(anchor, num_points=num_points)
            positive = self.process_point_cloud(positive, num_points=num_points)
            negative = self.process_point_cloud(negative, num_points=num_points)

            for a in augmentation:
                anchor[:, :3], _ = a.apply(anchor[:, :3])
                positive[:, :3], _ = a.apply(positive[:, :3])
                negative[:, :3], _ = a.apply(negative[:, :3])

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        if len(anchors) != 0:
            anchors = np.stack(anchors, axis=0)
            positives = np.stack(positives, axis=0)
            negatives = np.stack(negatives, axis=0)
        else:
            anchors, positives, negatives = None, None, None

        return anchors, positives, negatives

    def get_point_cloud(self, i):
        """ Retrieves the i'th point cloud

        Args:
            i (int): Index of point cloud to retrieve

        Returns:
            cloud (np.array) point cloud containing N points, each of D dim
        """
        assert(0 <= i < len(self.data))

        cloud = DataGenerator.load_point_cloud(self.paths_and_labels[i][0], num_cols=self.num_cols)
        return cloud

    def get_positive_negative(self, anchor):
        """ Gets positive and negative indices

        Args:
            anchor (int): Index of anchor point cloud

        Returns:
            positive (int), negative (int)
        """

        _, positives, nonnegatives = self.paths_and_labels[anchor]

        positive = random.sample(positives, 1)[0]

        negative_not_found = True
        while negative_not_found:
            negative = random.sample(range(self.size), 1)[0]
            if negative not in positives and negative not in nonnegatives:
                negative_not_found = False

        return positive, negative

    def process_point_cloud(self, cloud, num_points=4096):
        """
        Crop and randomly downsamples of point cloud.
        """

        # Crop to 20m radius
        mask = np.sum(np.square(cloud[:, :3]), axis=1) <= 20 * 20
        cloud = cloud[mask, :]

        # Downsample
        if cloud.shape[0] <= num_points:
            # Add in artificial points if necessary
            self.logger.warning('Only %i out of %i required points in raw point cloud. Duplicating...', cloud.shape[0],
                                num_points)

            num_to_pad = num_points - cloud.shape[0]
            pad_points = cloud[np.random.choice(cloud.shape[0], size=num_to_pad, replace=True), :]
            cloud = np.concatenate((cloud, pad_points), axis=0)

            return cloud
        else:
            cloud = cloud[np.random.choice(cloud.shape[0], size=num_points, replace=False), :]
            return cloud

    @staticmethod
    def load_point_cloud(path, num_cols=6):
        """ Reads point cloud, in our binary/text format

        Args:
            path (str): Path to .bin or .txt file
                        (bin will be assumed to be binary, txt will be assumed to be in ascii comma-delimited)
            num_cols: Number of columns. This needs to be specified for binary files.

        Returns:
            np.array of size Nx(num_cols) containing the point cloud.
        """
        if path.endswith('bin'):
            model = np.fromfile(path, dtype=np.float32)
            model = np.reshape(model, (-1, num_cols))

        else:
            model = np.loadtxt(path, dtype=np.float32, delimiter=',')

        return model
