# Plan of Action (June 10 2024)

## Last weeks

Last weeks, I mainly focused on:

- Getting, pre-processing and including the other types of data to the pipeline (**infrared** images and **multiple layers of CHM**)
- **Cleaning** the code and improving the whole **setup** after testing it with Brian and Tom.
- Implementing **GeoJSON conversions** for inputs and outputs to allow annotation and visualization using external tools like QGIS
- **Extending the dataset** (from about 140 images to about 240 images) by annotating the east side of the area I work on
- Adding a few improvements to the pipeline:
    - Automatic separation of the dataset into **training**, **validation** and **test** sets
    - **Random data dropout** to force the model to learn how to use CHMs alone and RGB/CIR alone
    - **Better visualization** of the training process with plotting of the loss during training

## Next goals

Now that all this is set up, I am very close to being able to start comparisons between the different models and data types. I still need to:

1. Implement functions to **evaluate the results** of the model with the usual metrics
2. Define a set of **augmentations** that I will be able to use in any situation
3. Improve the training pipeline to have it to **stop when there is no more improvement** (I don't know yet how this can be decided)

Finally, it would also be nice to see whether the model that is currently used can output masks instead of bounding boxes, or what would need to change.
