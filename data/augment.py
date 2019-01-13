import numpy as np


def get_augmentations_from_list(str_list, upright_axis=2):
    '''
    :param str_list: List of string indicating the augmentation type
    :param upright_axis: Set to 1 for modelnet (i.e. y-axis is vertical axis), but 2 otherwise (i.e. z-axis)
    :return:
    '''

    if str_list is None:
        return []

    augmentations = []
    if 'Rotate1D' in str_list:
        if upright_axis == 1:
            augmentations.append(RotateY())
        elif upright_axis == 2:
            augmentations.append(RotateZ())
    if 'Jitter' in str_list:
        augmentations.append(Jitter())
    if 'Scale' in str_list:
        augmentations.append(Scale())
    if 'RotateSmall' in str_list:
        augmentations.append(RotateSmall())
    if 'Shift' in str_list:
        augmentations.append(Shift())

    return augmentations


class Augmentation(object):

    def apply(self, data):
        raise NotImplementedError


class Jitter(Augmentation):
    '''
    Applies a small jitter to the position of each point
    '''

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def apply(self, data):
        assert (self.clip > 0)
        jittered_data = np.clip(self.sigma * np.random.randn(*data.shape), -1 * self.clip, self.clip)
        jittered_data += data

        return jittered_data


class Shift(Augmentation):

    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def apply(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data += shift

        return data


class RotateZ(Augmentation):
    '''
    Rotation perturbation around Z-axis.
    '''

    def apply(self, data):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        rotated_data = np.dot(data, rotation_matrix)

        return rotated_data


class RotateY(Augmentation):
    '''
    Rotation perturbation around Y-axis.
    '''

    def apply(self, data):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix)

        return rotated_data


class RotateSmall(Augmentation):
    '''
    Applies a small rotation perturbation around all axes
    '''
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def apply(self, data):
        angles = np.clip(self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))

        rotated_data = np.dot(data, R)

        return rotated_data


class Scale(Augmentation):

    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def apply(self, data, keypoints=None):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        data *= scale

        return data

