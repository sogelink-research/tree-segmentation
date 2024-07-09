# Plan of Action (July 8 2024)

## Last weeks

Last weeks, I mainly worked on:

- Modifying the training pipeline to:
    - **Speed up** the pre-processing that happens at the beginning of each training session with **multi-processing** for almost all steps
    - **Speed up** the loading of the data by pre-normalizing the rasters and using **more efficient raster formats**
    - Create a simple class with default arguments and the possibility to modify all the **hyperparameters**
    - Make the whole pipeline work even when using **only one of the data types** (RGB/CIR images or CHM layers)
- Improving **temporary files management** to have everything cleaned up even when the process ends abruptly
- Improving the split of files in multiple groups to be able to do **cross-validation**
- **Automatically choosing the batch size** to make it work on any machine and choose the best solution in terms of speed
- Saving more info about the training instance (initial hyperparameters, AP metrics results)
- Computing **AP metrics per class** for non-agnostic models
- **Visualizing** in a much better way the **progress** of the training session through standard output
- Starting to create a few functions to **post-process the results** of each set of parameters used to train the model

## Next goals

For this week, the main goal should be to run experiments with different set of parameters and analyze the results to find the best combinations.

Then, the next priorities are the following:

- Modify the model to output **precise masks** instead of bounding boxes. This might be very quick or painful, depending on what really needs to be changed for this to work.
- Add the possibility to **load a model and test it on any image** / any location (should be easy)
- Improve the process of **downloading the data** to make it possible to quickly test the model at any location without having the data pre-installed (potentially hard, I will need some help)
- Add the detection of species:
    - Add species in the dataset (potentially very hard depending on how precise we want to be)
    - Add the distinction to the model (easy)
- Add post-processing steps:
    - Finding the height of the tree (difficulty should be decent)
    - Finding the shape of the tree (potentially really hard)
