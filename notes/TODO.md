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
