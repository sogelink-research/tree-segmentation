# TODO

## Dependencies

- Use pip and a requirements.txt file instead of conda env environment.yml
- Remove useless dependencies (there are tools to do this)

## Speed

- Replace the TIF files by using np.save or np.memmap
- Try to make `tree_dataset_collate_fn` quicker by filling the torch tensors instead of using stack
- Try to make `normalize` quicker

## Others

- Maybe try using AutoAlbumentations to find the best augmentations
- If possible at one point, improve path manipulation
