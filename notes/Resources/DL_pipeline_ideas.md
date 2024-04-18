# Deep Learning Pipeline Ideas

Below are a few ideas I had or found that could help during the training of the dataset.

## Preprocessing

### Classify points

If there are methods (such as DL networks) that are able to very reliably classify point cloud data to get rid of cars, bikes and buildings, it could be interesting to use them to clean the AHN4 point clouds before feeding them into the network.

### LiDAR point cloud bias

It could maybe also be interesting to look at the bias of the point cloud and find a way to get rid of it. Maybe simply adding some noise to the positions could help.

## Data Augmentation

### Rotations and Symmetries

Applying rotations and symmetries to the data is a simple way to increase the data. We can easily create 8 data tiles from 1 with 0°, 90°, 180° and 270° rotations. Some other transformations might be possible to consider.

### Noise

Adding noise to the point cloud or to the RGB image could also help training a more robust model.

### Generative Adversarial Networks

There might be a way to use GANs to generate more data from the data we have. But this would probably be difficult since we want to use images and point clouds together. One way to still do this could be to deal only with grid data (images or voxels depending on what I choose to use), on which a GAN might be easier to use.

Also, for the generated data to be interesting, bounding boxes are necessary. Therefore, it might be hard to generate some new data as a whole chunk with several trees and other things, and it might be easier to generate individual elements and put them together.

[This paper](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.914974/full) is where I found the idea of GAN for this task.

## Data Precision Adaptation

It would probably be better to train and use the model on point clouds (and images) having consistent precisions, even if this means reducing the amount of points in some LiDAR point clouds for example.

## Postprocessing

### Watershed algorithm

It could be interesting to apply the watershed algorithm when something detected as one tree seems too large to be one tree. The idea comes from [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9652063).
