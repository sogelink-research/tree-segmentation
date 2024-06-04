# TODO

- Add full GeoJSON conversion for the data and the outputs of the model -> Make it possible to read other things than rectangles, handle intersections and conversion to rectangle for bounding boxes
- Create a better output visualization with all layers, found and ground truth bounding boxes
- Look how to add other input data channels to the model (CHM and hyper-spectral) including:
    - Data collection
    - Model adaptation and in particular information propagation
- Check that the augmentations work well during training
- Try to find how to add pixel augmentations to the CHM
- Maybe try using AutoAlbumentations to find the best augmentations
- If possible at one point, improve path manipulation
- Change the labels to a simpler and more understandable format:
    - "Tree" remains.
    - "Tree_unsure" should be removed. I could consider adding "Vegetation" or something like this for low vegetation.
    - "Tree_disappeared" becomes "Tree_LiDAR".
    - "Tree_new" becomes "Tree_RGB".
    - "Tree_replaced" is removed, and its instanced are replaced by two instances: one "Tree_disappeared" and one "Tree_new".
