# TODO

## Dependencies

- Use pip and a requirements.txt file instead of conda env environment.yml
- Remove useless dependencies (there are tools to do this)

## Hyperparameters

- Solve the pickle saving issue
- Create a class for the hyperparameters which:
    - Saves the important parameters of the training in a file
    - Launches the training with the right parameters
- Implement the use of the YOLO backbone if only one input is used

## Speed

- Implement memmaps and normalization in a function before the initialization of the dataset
- Try to make `tree_dataset_collate_fn` quicker:
    - Filling the torch tensors instead of using stack didn't work
    - Initialize the batches before when we have `shuffle=False`?

## Others

- Save the components of sortedAP in a file
- Maybe try using AutoAlbumentations to find the best augmentations
- If possible at one point, improve path manipulation
