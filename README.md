# Tree Segmentation

## Goal

The goal of this project is to develop a method to **segment and identify trees using remote sensing data**. The two main types of data will be RGB **images** (or other wave lengths) and **LiDAR** point clouds. The main focus of the project will be to properly identify and separate each **individual tree**, even in dense forests. Then, we should be able to extract information about each individual tree, such as its **height**, its **shape** or its **species** (starting only with deciduous and evergreen).

## Setup

### Requirements

Before running this repo, you need to make sure that you have CUDA installed if you want to use PyTorch with CUDA.

### Set up the environment

First, clone the repository with its submodules:

```bash
git clone git@github.com:sogelink-research/tree-segmentation.git --recursive
```

Then, move into the repo folder and create the conda environment to run the project with conda (or even better with mamba):

```bash
cd tree-segmentation

# With conda
conda env create -n <env-name> -f environment.yml

# With mamba (much quicker)
mamba create -n <env-name>
mamba update -n <env-name> -f environment.yml
```

## Execution

### Get the data

To download and pre-process the data, you can simply run the notebook [ML_preprocessing.ipynb](src/notebooks/ML_preprocessing.ipynb).

### Train a model

To train a model, you can simply run the notebook [ML_training.ipynb](src/notebooks/ML_training.ipynb).
