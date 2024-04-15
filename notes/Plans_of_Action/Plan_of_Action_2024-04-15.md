# Plan of Action (April 5 2024)

## Last week

Last week, my main goal was to gather and understand the data I had, to start preprocessing it and be able to feed it into a neural network (aiming for the one from this [paper](https://www.mdpi.com/1999-4907/15/2/293)). With this in mind I did the following:

- Set up Ubuntu and other things on the **computer with a GPU** for future computations with the help from Tom.
- Create a small script to create **Canopy Height Model (CHM) images** from the point cloud (in [`src/chm.py`](../../src/chm.py)).
- Look for different ways of **processing a point cloud** to see how I could keep more information than only the CHM (see [here](../Resources/LiDAR_Embedding.md)). I think that for now I will stick to using **projection methods** with convolutional layers, and perhaps use other features from the LiDAR point cloud (number of returns, intensity, reflectance, amplitude).
- Decide to rather focus on **the Netherlands** to start to have mode consistent data (since the LiDAR data in France seems to be acquired at any season).
- Look at the dataset of the **trees in Amsterdam** (can be downloaded [here](https://maps.amsterdam.nl/open_geodata/?k=505) and visualized [here](https://maps.amsterdam.nl/bomen/)). Since there is only the position and an interval for the height, and all the trees in the private areas are missing, it will be hard to use this dataset to train the model. But there are a lot of information about the species, which is great.

## Next goals

The goals for the next week are the following:
