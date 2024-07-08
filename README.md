# Tree Segmentation

## Goal

The goal of this project is to develop a method to **segment and identify trees using remote sensing data**. The two main types of data will be RGB **images** (or other wave lengths) and **LiDAR** point clouds. The main focus of the project will be to properly identify and separate each **individual tree**, even in dense forests. Then, we should be able to extract information about each individual tree, such as its **height**, its **shape** or its **species** (starting only with deciduous and evergreen).

## Setup

### Requirements

Before running this repo, you need to make sure that you have **CUDA** installed if you want to use PyTorch with CUDA.

Also, the data that is used takes up to 20 GB of disk space, and the weights of each model take almost 150 MB. Therefore, you need to have enough disk space to run the project.

### Set up the environment

First, clone the repository with its submodules:

```bash
git clone git@github.com:sogelink-research/tree-segmentation.git --recursive
```

Then, move into the repo folder and create the conda environment to run the project with conda:

```bash
cd tree-segmentation

conda env create -n <env-name> -f environment.yml
```

## Execution

To run the training or the evaluation of a model, you simply have to modify and execute [`run.py`](src/run.py). You can select the hyperparameters you want to use and launch one or multiple training sessions.

All necessary data (RGB images, CIR images and LiDAR point clouds) will be downloaded and pre-processed the first time you need it, and stored in the [`data`](data) folder.
