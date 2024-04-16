# LiDAR data embedding for machine learning

There a several different ways to embed point cloud data from LiDAR to extract interesting features. Below are some of them that could be interesting for instance segmentation of trees.

## Preprocessing

The first step that is very important for the standardization of the data is to subtract the altitude of the terrain. This can be done by computing a Digital Terrain Model (DTM). I should look for a way to do this that doesn't necessarily involve a grid to create a new standardized point cloud.

Then, ground points should also be removed to create point clouds as small as possible and quicker to process.

## Deep Learning Models

The different methods mostly come from [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9609839/).

### Projection-based methods

These methods will probably be easier to fuse with RGB data since they are already based on grids.

#### 2D convolutional neural networks

This mainly implies to create 2D grids of data from the point cloud, and then use 2D convolution layers to extract features from the 2D grids. There are different ways to create these 2D grids, the most common one being to use projections and weighted averages. But we could also try to extract features using other kinds of data obtained from LiDAR, such as the intensity or the number of returns, and we could extract not the average value but the maximum one for example.

#### 3D convolutional neural networks

Since the point cloud is in 3D, it could also be interesting to use 3D convolutions. [This paper](https://arxiv.org/abs/2004.12636) uses such convolutions to extract features from the point cloud and then fuse then with image features. Some other interesting models could be found on the [KITTI benchmark leaderboard](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

#### Multiview representation

The concept of multiview representation is simple. It boils down to making multiple 2D renders of the point cloud from different view points and angles. Then, features can be extracted from these 2D images using 2D convolution layers. [This paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html) implements this method.

#### Volumetric grid representation

This representation uses 3D occupancy grids and 3D convolution layers to extract features from it. [VoxelNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html) is an example of such a model.

### Point-based methods

These methods will probably be harder to fuse with RGB data since they are not based on grids.

#### PointNets

These architectures take point clouds as input and try to extract features from them. [PointNet++](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html) should be a great model to start looking at this.

#### (Graph) Convolutional Point Networks

These methods apply some kind of convolutions to point clouds. I could take a look at [Dynamic Graph CNN](https://dl.acm.org/doi/abs/10.1145/3326362) for example.
