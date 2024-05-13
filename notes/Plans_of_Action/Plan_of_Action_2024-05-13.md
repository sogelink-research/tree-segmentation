# Plan of Action (May 13 2024)

## Last week

Last week, I did two things:

1. I finished annotating a small part fo the dataset with a lot of trees that will be interesting to look at since this is a part where the point clouds are really necessary to find all the smaller trees below the few hue trees.
2. I implemented the structure of the first model ([AMF GD YOLO v8](https://www.mdpi.com/1999-4907/15/2/293)) in [layers.py](../../src/layers.py)

## Next goals

This week, I want to train my first model. This will require:

1. To properly transfer the annotations I did last week using QGIS to Label Studio because the LiDAR point cloud and the images are not perfectly aligned.
2. To look at [Albumentations](https://albumentations.ai/docs/) to use dataset augmentation techniques to have enough data to train the model
3. To set up a common training loop (with usual optimizers, learning rate schedulers...)
