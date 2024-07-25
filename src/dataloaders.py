from typing import Callable, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Sampler

from box_cls import Box
from datasets import TreeDataset


# def quick_stack(tensor_list: List[torch.Tensor]):
#     new_tensor = torch.empty((len(tensor_list), *tensor_list[0].shape))
#     for i, tensor in enumerate(tensor_list):
#         new_tensor[i] = tensor
#     return new_tensor


def tree_dataset_collate_fn(
    batch: List[Dict[str, torch.Tensor | None]],
) -> Dict[str, torch.Tensor | None]:
    """Custom collate function for a Dataloader taking a TreeDataset.

    Args:
        batch (List[Dict[str,  torch.Tensor | None]]): A batch as a list of TreeDataset outputs.

    Returns:
        Dict[str, torch.Tensor | None]: The final batch returned by the DataLoader.
    """
    # Initialize lists to hold the extracted components
    rgb_images_list = []
    chm_images_list = []
    bboxes = []
    labels = []
    non_agnostic_labels = []
    indices = []
    image_indices = []

    use_rgb = True
    use_chm = True

    # Iterate through the batch
    for i, item in enumerate(batch):
        # Extract the components from the dictionary
        rgb_image = item["image_rgb"]
        chm_image = item["image_chm"]
        bbox = item["bboxes"]
        label = item["labels"]
        non_agnostic_label = item["non_agnostic_labels"]
        image_index = item["image_index"]

        if rgb_image is None:
            use_rgb = False
        if chm_image is None:
            use_chm = False

        # Append the extracted components to the lists
        rgb_images_list.append(rgb_image)
        chm_images_list.append(chm_image)
        bboxes.append(bbox)
        labels.append(label)
        non_agnostic_labels.append(non_agnostic_label)
        indices.extend([i] * bbox.shape[0])  # type: ignore
        image_indices.append(image_index)

    # Convert the lists to tensors and stack them
    if use_rgb:
        rgb_images = torch.stack(rgb_images_list)
        # rgb_images = quick_stack(rgb_images_list)
    else:
        rgb_images = None

    if use_chm:
        chm_images = torch.stack(chm_images_list)
        # chm_images = quick_stack(chm_images_list)
    else:
        chm_images = None

    bboxes = torch.cat(bboxes)
    labels = torch.cat(labels)
    non_agnostic_labels = torch.cat(non_agnostic_labels)
    indices = torch.tensor(indices)
    image_indices = torch.tensor(image_indices)

    output_batch = {
        "image_rgb": rgb_images,
        "image_chm": chm_images,
        "bboxes": bboxes,
        "labels": labels,
        "non_agnostic_labels": non_agnostic_labels,
        "indices": indices,
        "image_indices": image_indices,
    }

    return output_batch


def convert_ground_truth_from_tensors(
    gt_bboxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_non_agnostic_classes: torch.Tensor,
    gt_indices: torch.Tensor,
    image_indices: torch.Tensor,
) -> Tuple[List[List[Box]], List[List[int]], List[List[int]]]:
    """Extracts the ground truth components from the outputs of a TreeDataLoader.

    Args:
        gt_bboxes (torch.Tensor): data["bboxes"] from TreeDataLoader
        gt_classes (torch.Tensor): data["labels"] from TreeDataLoader
        gt_non_agnostic_classes (torch.Tensor): data["non_agnostic_labels"] from TreeDataLoader
        gt_indices (torch.Tensor): data["indices"] from TreeDataLoader
        image_indices (torch.Tensor): data["image_indices"] from TreeDataLoader

    Returns:
        Tuple[List[List[Box]], List[List[int]], List[List[int]]]: (gt_bboxes, gt_classes,
        gt_non_agnostic_classes) where each list inside the main list corresponds to one image.
    """
    number_images = image_indices.shape[0]

    bboxes: List[List[Box]] = [[] for _ in range(number_images)]
    classes: List[List[int]] = [[] for _ in range(number_images)]
    non_agnostic_classes: List[List[int]] = [[] for _ in range(number_images)]

    for i in range(number_images):
        mask = gt_indices == i
        bboxes[i] = list(map(Box.from_list, gt_bboxes[mask].tolist()))
        classes[i] = gt_classes[mask].tolist()
        non_agnostic_classes[i] = gt_non_agnostic_classes[mask].tolist()

    return bboxes, classes, non_agnostic_classes


def extract_ground_truth_from_dataloader(
    data: Dict[str, torch.Tensor],
) -> Tuple[List[List[Box]], List[List[int]], List[List[int]]]:
    """Extracts the ground truth components given the output of a TreeDataLoader.

    Args:
        data (Dict[str, torch.Tensor]): batch output of a TreeDataLoader.

    Returns:
        Tuple[List[List[Box]], List[List[int]], List[List[int]]]: (gt_bboxes, gt_classes,
        gt_non_agnostic_classes) where each list inside the main list corresponds to one image.
    """
    gt_bboxes: torch.Tensor = data["bboxes"]
    gt_classes: torch.Tensor = data["labels"]
    gt_non_agnostic_classes: torch.Tensor = data["non_agnostic_labels"]
    gt_indices: torch.Tensor = data["indices"]
    image_indices: torch.Tensor = data["image_indices"]

    return convert_ground_truth_from_tensors(
        gt_bboxes=gt_bboxes,
        gt_classes=gt_classes,
        gt_non_agnostic_classes=gt_non_agnostic_classes,
        gt_indices=gt_indices,
        image_indices=image_indices,
    )


class TreeDataLoader(DataLoader):
    def __init__(
        self,
        dataset: TreeDataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        self.dataset: TreeDataset = dataset
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            tree_dataset_collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )


def initialize_dataloaders(
    datasets: Dict[str, TreeDataset],
    batch_size: int,
    num_workers: int,
) -> Tuple[TreeDataLoader, TreeDataLoader, TreeDataLoader]:
    assert all(key in datasets for key in ["training", "validation", "test"])
    train_loader = TreeDataLoader(
        datasets["training"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = TreeDataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = TreeDataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
