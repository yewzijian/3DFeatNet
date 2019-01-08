# Processing Scripts

This folder contains MATLAB processing scripts to generate the datasets used in 3DFeat-Net paper, in their respective subfolders.

* [Oxford Robotcar](#oxford-robotcar-dataset)
* [KITTI](kitti-dataset)
* [ETH](eth-dataset) (Only results are provided, see section for details)

Details of the data format can be found in the [last section](#data-format).



## Oxford Robotcar dataset

### Introduction

The scripts are found in the folder "oxford".

It contains scripts for generating the training data for Oxford Robotcar dataset used in 3DFeat-Net. The test data requires manual work, so we provide it as a direct download (see [here](#test-data)).

### Prerequisites

You'll first need register for an account to download the raw Oxford dataset manually at the [Oxford Robotcar dataset](http://robotcar-dataset.robots.ox.ac.uk/) website.

For our work, we use 35 training + 5 test trajectories, with the remaining filtered out for various reasons, e.g. Bad GPS, poor quality Lidar scans due to rain, etc. The trajectories used can be found in `datasets_train.txt` and `datasets_test.txt` respectively.

For each of the datasets, download (1) *LMS Front* , and (2) *GPS data*. Unzip all the download files into the same directory, so each folder should contain the gps and lms_front subfolders.

### Training data

Generating the training data comprises running two scripts:

1. `oxford_build_pointclouds.m`: This script accumulates the line scans into 3D point clouds. Before running, set the following two variables at the top of the script:

   * FOLDER: Point to the folder containing the raw data, containing the unzipped data in the previous section.

   * DST_FOLDER: Destination folder to store the unzipped data

2. `oxford_generate_test_cases.m`: This generates the positive and non-negatives training triplet (we do not store the negatives since they're too many). As before, set DST_FOLDER to the same directory as the previous step.

After running the scripts, DST_FOLDER should contain the generated point clouds. DST_FOLDER/train.txt will be in the following format:

[bin-file] | $p_1 p_2 ... p_n$ | $nn_1 nn_2 ... nn_m$

, where $p_i$ indicates the positive indices, and $nn_i​$ indicates the non-negative indices. The indices are 0-based, so 0 indicates the bin file in the first line of train.txt.

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



## ETH Dataset

For this dataset, we did not do much preprocessing (other than voxelgrid filtering) of the raw data. We instead provide a copy of our **computed keypoints and descriptors**, which can be downloaded from [here](https://drive.google.com/open?id=1hdhIJjmyf9EkSgvsfuQdeR5SvcUTNz1-).

The computed descriptors are stored in two folders according to their dataset: 1) gazebo_winter, and 2) wood_autumn. Each .bin file corresponds to keypoints+descriptors for the respective point clouds. Note however that `Hokuyo_-1.bin` contains the keypoints+descriptors for the global point cloud constructed from individual point clouds for the _other_ season, i.e. `wood_autumn\Hokuyo_-1.bin` contains the results for Wood summer point clouds.

Each .bin file is stored in binary format containing single precision floats: $(x_1, y_1, z_1, f_1^{1}, f_1^{2}, f_1^{32}), (x_2, y_2, z_2, f_2^{1}, f_2^{2}, f_2^{32}), ...$, where the $f_i^{(*)}$'s' correspond to the 32D descriptors for the $i^{th}$ keypoint.



## Point Cloud Data Format

- Each bin file in the processed directory is a binary file containing Nx6 float32 for the N points in the point cloud: $(x_1, y_1, z_1, Nx_1, Ny_1, Nz_1), (x_2, y_2, z_2, Nx_2, Ny_2, Nz_2)$, ...
- `groundtruths.txt` contains the transformation between each of the local point clouds and the global point cloud. See MATLAB script `[ROOT]/scripts/show_alignment.m` to understand how to interpret the transformation.