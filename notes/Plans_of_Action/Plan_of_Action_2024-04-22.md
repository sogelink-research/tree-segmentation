# Plan of Action (April 22 2024)

## Last week

Last week, the goal was to see how I could get a dataset containing the following data over a given surface area:

- airborne RGB images (and potentially other wave lengths)
- airborne LiDAR point clouds
- bounding boxes or masks indicating the positions and areas of the trees
- preferably in the Netherlands or in a region with similar species of trees

However, I didn't really find anything matching all the criteria. The most promising is the [NeonTreeEvaluation Benchmark](https://github.com/weecology/NeonTreeEvaluation/), which I could use since it contains airborne RGB, hyperspectral and LIDAR point clouds. However, it is data from the USA, based only on forests. See [here](../Resources/Resources.md#datasets) for the list of potentially interesting datasets that were found. However, most of them lack at least one thing (whether it's RGB images, LiDAR point clouds or the precise positions and areas of the trees).

Therefore, I chose to create my own tree dataset for the Netherlands. This has many advantages:

- We have precise and consistent LiDAR point clouds for the whole country [here](https://hub.arcgis.com/maps/esrinl-content::ahn4-download-kaartbladen-1/explore) or [here](https://geotiles.citg.tudelft.nl/) for smaller chunks
- We have RGB images with a resolution of 8 cm for the whole country [here](https://www.beeldmateriaal.nl/data-room)
- We have positions for the trees monitored by most large cities, such as [here](https://maps.amsterdam.nl/open_geodata/?k=505) for Amsterdam
- We have a very convenient tool to visualize the LiDAR point cloud [here](https://ns_hwh.fundaments.nl/hwh-ahn/AHN_POTREE/index.html)
- We also have access to some really recent and high resolution airborne images and Street View images that can be used either as data or as a support to annotate the trees

The only real drawback is the necessity for me to annotate by hand each individual tree that I want to train my model on, or that I want to use for validation. This will take some time but this will ensure to have a high-quality dataset to train the model.

## Next goals

Due to the conclusions of the previous week, the goals for the next weeks are pretty clear and straight-forward:

- Create a proper dataset of bounding boxes for the trees. I chose the area around the Geodan building as it can allow on-terrain verifications and it has quite diverse areas (buildings, a river, parks, small dense forests...). My goal is to annotate an image that is 1 km $\times$ 1 km large. This will take some time but I will then be able to use this area both as training and validation data.
- In parallel of this task, I should start implementing the paper I wanted to focus on ([here](../Resources/Resources.md#individual-tree-species-identification-for-complex-coniferous-and-broad-leaved-mixed-forests-based-on-deep-learning-combined-with-uav-lidar-data-and-rgb-images)). I will process in the following order:
    1. Extraction of images of the right size from larger images (see [here](../Resources/Data_preprocessing.md#image-size))
    2. Computation of CHM images from the LiDAR data (to start with the exact same pipeline as in the paper)
    3. Implementation of the full model from input to output with all the different layers
    4. Addition of the data augmentation methods (probably using [Albumentations](https://albumentations.ai/docs/))
    5. Then I will be able to try new things, such as other features from the point cloud, other resolutions...
