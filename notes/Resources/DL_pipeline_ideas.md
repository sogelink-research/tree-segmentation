# Deep Learning Pipeline Ideas

Below are a few ideas I had or found that could help during the training of the dataset.

## Preprocessing

### Classify points

If there are methods (such as DL networks) that are able to very reliably classify point cloud data to get rid of cars, bikes and buildings, it could be interesting to use them to clean the AHN4 point clouds before feeding them into the network.

### LiDAR point cloud bias

It could maybe also be interesting to look at the bias of the point cloud and find a way to get rid of it. Maybe simply adding some noise to the positions could help.

## Data Augmentation

See [Albumentations](https://albumentations.ai/docs/) for an image augmentation library. For now, I use image augmentations that I have selected, but it might be interesting to look at [AutoAlbument](https://albumentations.ai/docs/autoalbument/) to automatically find the right transformations using GAN methods.

### Rotations and Symmetries

Applying rotations and symmetries to the data is a simple way to increase the data. We can easily create 8 data tiles from 1 with 0°, 90°, 180° and 270° rotations. Some other transformations might be possible to consider.

### Different slicing

I don't know whether the model will handle every part of the image in the same way. It should not be a problem due to how convolution layers work, but it might be interesting to cut the full images at different positions in case any kind of problem like this occurs.

### Noise

Adding noise to the point cloud or to the RGB image could also help training a more robust model.

### Generative Adversarial Networks

There might be a way to use GANs to generate more data from the data we have. But this would probably be difficult since we want to use images and point clouds together. One way to still do this could be to deal only with grid data (images or voxels depending on what I choose to use), on which a GAN might be easier to use.

Also, for the generated data to be interesting, bounding boxes are necessary. Therefore, it might be hard to generate some new data as a whole chunk with several trees and other things, and it might be easier to generate individual elements and put them together.

[This paper](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.914974/full) is where I found the idea of GAN for this task.

## Data Precision Adaptation

It would probably be better to train and use the model on point clouds (and images) having consistent precisions, even if this means reducing the amount of points in some LiDAR point clouds for example.

## Error evaluation

### IoU

IoU is one of the common ways to tell if the model finds proper masks corresponding to bounding boxes.

### Weighting

The model will probably have issues finding trees that are below other trees. Since there can be a lot of small trees like this, it might be interesting to weight the loss corresponding to each tree with the area of the corresponding bounding box.

It might also be worth it to weight the loss function with the distance between trees, or with the density of trees in the area of the tree, to make sure the model performs well on isolated trees.

## Postprocessing

### Output format

I don't know how I would be possible to allow the model to have multiple trees to share an (x,y) coordinate. This is necessary since I aim at also detecting the trees that are below other trees.

### Watershed algorithm

It could be interesting to apply the watershed algorithm when something detected as one tree seems too large to be one tree. The idea comes from [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9652063).

## Potential things to try

### Point cloud as 3rd channel

Maybe it could be interesting to create images with 3 channels, where the two first channels come from a PCA on RGB, and the last channel comes from the point cloud height values. The image could then be directly given as input to a model?
