# Processing Scripts

This folder contains MATLAB processing scripts to generate the datasets used in 3DFeat-Net paper, in their respective subfolders.

* [Oxford Robotcar](#oxford-robotcar-dataset)
* [KITTI](kitti-dataset)

For the ETH dataset, we did not do much preprocessing (other than voxelgrid filtering) of the raw data. Details of the data format can be found in the [last section](#data-format).



## Oxford Robotcar dataset

### Introduction

The scripts are found in the folder "oxford".

It contains scripts for generating the training data for Oxford Robotcar dataset used in 3DFeat-Net. The test data requires manual work, so we provide it as a direct download (see [here](#test-data)).

### Prerequisites

You'll first need register for an account to download the raw Oxford dataset manually at the [Oxford Robotcar dataset](http://robotcar-dataset.robots.ox.ac.uk/) website.

For our work, we use 35 training + 5 test trajectories, with the remaining filtered out for various reasons, e.g. Bad GPS, poor quality Lidar scans due to rain, etc. The trajectories used can be found in *datasets.txt* (The first 35 are training, and last 5 are used for testing)

For each of the datasets, download (1) *LMS Front* , and (2) *GPS data*. Unzip all the download files into the same directory, so each folder should contain the gps and lms_front subfolders.

### Training data

TODO: Code to prepare training data will be uploaded at a later date.

### Test Data

We provide the test data as a direct download, which has been pairwise registered (using ICP) and manually cleaned up.

* <u>Descriptor matching</u>: The 30,000 cluster pairs for evaluating descriptor matching (Fig. 3) can be downloaded from [here](https://drive.google.com/open?id=17kZh4TMhEmC8ia3bovjtfZ6k0rbChj0I). The clusters have been cropped to 4.0m radius, but note that the **results in the paper consider a 2m radii for all descriptors**.
* <u>Detection+Feature Description, and Registration</u>: The test models can be downloaded from [here](https://drive.google.com/open?id=1GZpyHz5-XRdwoKwiRM-7i46XpsWvzIN5). They are generated from oxford_build_pointclouds above, but have been randomly rotated and downsampled to evaluate rotational equivariances and robustness to sparse point clouds.



## KITTI Dataset

### Introduction

The folder "kitti" contains the scripts to process the KITTI dataset for evaluation.

### Prerequisites

Download the odometry dataset from the [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). You'll need to download (1) velodyne laser data, (2)  ground truth poses, and (3) calibration files. Unzip the files into the same folder, and you'll end up with 2 subfolders:

1. poses (containing 00.txt, 01.txt, ..., 10.txt)
2. sequences (containing 21 subfolders each containing the velodyne data in a folder "velodyne" and calib.txt)

### Processing

Open *process_kitti_data.m*, set KITTI_FOLDER to point to the folder from the previous step, and run the script. The processed data will be stored in OUTPUT_FOLDER.



## Data Format

- Each bin file in the processed directory is a binary file containing Nx6 float32 for the N points in the point cloud: (x1 y1 z1 Nx1 Ny1, Nz1), (x2 y2 z2 Nx2 Ny2, Nz2), ...
- "groundtruths.txt" contains the transformation between each of the local point clouds and the global point cloud. See MATLAB script `[ROOT]/scripts/show_alignment.m` to understand how to interpret the transformation.