# Plan of Action (June 24 2024)

## Last weeks

Last weeks, I mainly focused on:

- Implementing **AP metrics** and specifically [*sortedAP*](https://arxiv.org/pdf/2309.04887) to evaluate the performance of the models.
- **Fixing** quite a lot of small and well-hidden errors in the code which made the outputs of the models completely off.
- Creating a `run.py` file with high-level classes to **easily run training sessions** given a few hyperparameters.
- **Adapting dataset augmentations** to any number of input channels.
- Improving the **visualization** of the progress of the training (in terms of loss and AP metrics).
- Improving all the **paths** used in the project.
- Looking at the slow parts of the training to try and speed it up (mainly data loading).

## Next goals

Because of all these small things I had to improve to get rid of the small errors in the code which were ruining the results, I don't yet have a fully functional pipeline to train the model with different hyperparameters and save everything that needs to be saved. But now that everything is fixed and properly working, I can focus on getting this up and running for very soon. My next priorities are:

- **Speeding up** the training by using more efficient data formats and normalizing the input once before the training loops.
- **Checking, downloading and preprocessing the necessary data** at the beginning of the training pipeline to allow for more flexibility.
- Creating a **simple interface** (probably just a class) that launches training with the right parameters and saves these parameters and the results.
- Implementing the use of **YOLO only** when using only RGB/CIR or LiDAR data.
- Deciding automatically when to **stop the training loop** using the loss.
