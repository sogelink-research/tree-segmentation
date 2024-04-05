# Plan of Action (April 5 2024)

## Introduction

### Topic of the internship

The goal of the internship is to develop a method to **segment and identify trees using remote sensing data**. The two main types of data will be RGB **images** (or other wave lengths) and **LiDAR** point clouds. The main focus of the internship will be to properly identify and separate each **individual tree**, even in dense forests. Then, we should be able to extract information about each individual tree, such as its **height**, its **shape** or its **species** (starting only with deciduous and evergreen).

### Plans of action

This document is the first travel block that I will be writing during my whole internship. The plans of action have will have two main objectives:

1. Describe what was done in the **past few days or weeks** since the last plan of action
2. Describe the **next short- and medium-term objectives** that should be focused on

Plans of action should be updated every 2 weeks. I plan to write them on Fridays to have them ready for the internship updates every Monday.

## Last few days

In the last few days, not much happened in regards to the project. The main goals were:

- Start setting everything up to be ready to start the internship properly
- Start diving into literature to find the most promising papers for the subject
- Settle on the aspects to focus on in priority

In practice, we set up my laptop with everything needed and started setting up a computer with a GPU for the future training of the models. I also found one very interesting and recent [paper](https://www.mdpi.com/1999-4907/15/2/293) which combines LiDAR point clouds and images to segment individual trees in dense forests and guess their species. However, this paper has no available implementation.

## Next goals

The goals for the next weeks are the following:

- Properly **set up the training machine** that will be used to train the models with Ubuntu, CUDA, PyTorch, etc
- Look in the **literature** for papers close the the goal of the project. The main criteria for these papers are:
    - Combining aerial LiDAR point clouds and aerial images as input
    - Having an implementation available online
    - Be focused on the specific topics of interest: individual tree segmentation and species recognition
- Find **databases** to use for the training of the models. If possible, these databases should:
    - Come from the region of interest (France)
    - Contain required data for the training of the model (positions of the trees, something to evaluate their outline like a radius or a shape, their species, their height, etc). It will probably be necessary to combine multiple datasets to have everything needed (such as the altitude for example)
- Depending on what will be found in the literature, choose a **first implementation** to focus on and apply it on the data that was found
