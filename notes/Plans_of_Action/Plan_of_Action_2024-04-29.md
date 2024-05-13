# Plan of Action (April 29 2024)

## Last week

Last week, I worked on the two parts that I mentioned last week to get to the first model:

- Creation of the tree dataset with using an area of 1 km $\times$ 1 km in Amsterdam
- Build the model pipeline starting with the preprocessing of the data

Even if it didn't go as fast as I wished, here is what was achieved so far:

- About 40% of the dataset is labeled. However, some of the most difficult parts of the image still need to be labeled, which might take a significant amount of time. However, I can't skip these areas as the most difficult parts are also the most interesting ones to see if we can make a good use of the LiDAR data. I hope that labeling the rest of the dataset won't take me more than one week, but I'm am unsure whether I will manage to go through the most difficult parts of the area with convincing bounding boxes.
- I have created a pipeline to pre-process images and the LiDAR data, which can be found in this [notebook](../../src/notebooks/preprocessing.ipynb). There are still a few things to modify as for the pre-processing of the point clouds, but otherwise it is almost finished.

## Next goals

As for next week, the goal is to keep working on the same two tasks and hope to finish them quickly:

- Create the tree dataset with bounding boxes,
- Implement the first ML pipeline using this dataset.
