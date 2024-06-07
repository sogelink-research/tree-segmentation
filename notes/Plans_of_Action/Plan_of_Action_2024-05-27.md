# Plan of Action (May 27 2024)

## Last weeks

Last weeks, I focused on set up everything that was still needed for the model to be run on a dataset:

- I created structures using PyTorch to load a tree dataset
- I implemented **normalization** for the RGB images and the CHM images
- I changed the **loss function** to use the one used by Ultralytics to train YOLOv8
- I added a few things to the training, such as **batch accumulation** to ensure better *stability* in the training.

I also spent quite some time on exploring the different image **augmentation** techniques available with *Albumentations*, and I chose a set of transformations to apply. In practice, there are two types of transformations:

- **Spatial** transformations which include rotations, translations, cropping, but also other kind of deformations. These transformations are applied in the same way to RGB and CHM to keep the coherence between the two images.
- **Pixel-based** transformations which modify the values of the pixels with different kinds of filters, such as noise, intensity change, channel dropout, color shift, etc. These transformations are for now only applied to the RGB image, but it would be great to also have a set of transformations for the CHM image.

At the end of last week, I managed to run **small training sessions** on the whole dataset. It showed that the model was able to learn to recognize trees up to a certain degree even with a dataset that is not very large. However, since I trained and evaluated on the whole dataset, the results are obviously much better than when evaluating on areas that were not seen during the training. But one promising point is that the model was sometimes capable of identifying trees on the side of the image which I didn't annotate, thinking that they would not appear in the training dataset. This shows that it was **able to generalize** at least to a certain degree.

Furthermore, I have included the part of the area which is *very dense in high and low trees* in the dataset. This part will most likely be very hard to handle for the model, but it is also one of the most interesting one. I haven't yet looked at the results on this part, but this will be one of the areas to keep an eye on.

## Next goals

As for this week, there are many things to do now that the main training functions are working. Here is a list of the different topics:

1. **Clean up** all the code to have a better control of the different parameters when testing different combinations in the future
2. Finish setting functions to be able to **download and pre-process data** easily without having to manually download the files
3. Implement functions to have a **better visualization** of the results during the **training process**
4. Implement functions to **evaluate the results** of the model with the usual metrics
5. Keep **improving the dataset**, with mainly two things:
    - Extend it by annotating new areas
    - Improve it by getting rid of a small part of the border to remove trees that are only partly visible and not annotated

When all this will be done, it will be time to start exploring new things, such as using **more layers**. This might imply to think about small modifications of the architecture to properly mix information from all kinds of data. The potential layers are the following:

- **Hyper-spectral** images
- Multiple layers of **height**, obtained by removing all the points above a certain height above the ground, and computing a CHM on the remaining points.

Finally, it would also be nice to see whether the model that is currently used can output masks instead of bounding boxes, or what would need to change.
