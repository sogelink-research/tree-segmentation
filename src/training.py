import json
import os
import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
from tifffile import tifffile

import geojson
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations import pytorch as Atorch
from IPython import display
from ipywidgets import Output
from skimage import io
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.notebook import tqdm
from PIL import Image

from box_cls import Box
from data_processing import ImageData, get_coordinates_from_full_image_file_name
from geojson_conversions import merge_geojson_feature_collections, save_geojson
from layers import AMF_GD_YOLOv8
from plot import create_geojson_output, get_bounding_boxes
from utils import Folders, get_file_base_name


class InvalidPathException(Exception):
    """Custom exception raised when an invalid path is encountered."""

    def __init__(self, path: str, type: str = "any"):
        """Raise an exception for an invalid path.

        Args:
            path (str): The invalid path.
            type (str): The type of invalidity, among ["any", "file", "folder"].
        """
        if type == "any":
            message = f"The path {path} (absolute path is {os.path.abspath(path)}) is invalid."
        elif type == "file":
            message = (
                f"The path {path} (absolute path is {os.path.abspath(path)}) is not a valid file."
            )
        elif type == "folder":
            message = (
                f"The path {path} (absolute path is {os.path.abspath(path)}) is not a valid folder."
            )
        else:
            raise ValueError(f'No InvalidPathException for type "{type}" is implemented.')
        super().__init__(message)


def compute_channels_mean_and_std(
    file_or_folder_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and the standard deviation along every channel of the image(s) given as input
    (either as one file or a directory containing multiple files).

    Args:
        file_or_folder_path (str): Path to the image or the folder of images.

    Raises:
        InvalidPathException: If the input path is neither a file, nor a directory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std), two arrays of shape (c,) where c is the number
        of channels of the input image(s). They contain respectively the mean value and the standard
        deviation of all images along each channel.
    """
    if os.path.isfile(file_or_folder_path):
        file_path = file_or_folder_path
        return _compute_channels_mean_and_std_file(file_path)

    elif os.path.isdir(file_or_folder_path):
        means = []
        stds = []
        folder_path = file_or_folder_path
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            mean, std = _compute_channels_mean_and_std_file(file_path)
            means.append(mean)
            stds.append(std)
        mean_mean = np.mean(means, axis=0)
        mean_std = np.mean(stds, axis=0)
        return mean_mean, mean_std

    else:
        raise InvalidPathException(file_or_folder_path)


def _compute_channels_mean_and_std_file(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and the standard deviation along every channel of the image given as input

    Args:
        file_path (str): Path to the image.

    Raises:
        InvalidPathException: If the input path is not a file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std), two arrays of shape (c,) where c is the number
        of channels of the input image. They contain respectively the mean value and the standard
        deviation of all images along each channel.
    """
    if not os.path.isfile(file_path):
        raise InvalidPathException(file_path, "file")
    image = io.imread(file_path)
    mean = np.array(np.mean(image, axis=(0, 1))).reshape(-1)
    std = np.array(np.std(image, axis=(0, 1))).reshape(-1)
    return mean, std


def normalize_rgb(image_rgb: torch.Tensor) -> torch.Tensor:
    """Normalizes the RGB image given as input, with values computed using 2023_122000_484000_RGB_hrl.tif

    Args:
        image_rgb (torch.Tensor): The RGB image to normalize (3 channels).

    Returns:
        torch.Tensor: The normalized image, with a mean of 0 and a standard deviation of 1 along each channel.
    """
    # On the whole image
    mean_rgb = torch.tensor([78.152, 88.417, 86.365]).view(-1, 1, 1)
    std_rgb = torch.tensor([45.819, 42.492, 38.960]).view(-1, 1, 1)
    # # On the labeled parts
    # mean_rgb = torch.tensor([77.692, 89.142, 85.816]).view(-1, 1, 1)
    # std_rgb = torch.tensor([34.3328, 32.1673, 27.6653]).view(-1, 1, 1)
    return (image_rgb - mean_rgb) / std_rgb


def normalize_chm(image_chm: torch.Tensor, no_data_replacement: float = 0) -> torch.Tensor:
    """Normalizes the CHM image given as input, with values computed using the unfiltered CHM of
    2023_122000_484000_RGB_hrl.tif with a resolution of 8cm.
    The NO_DATA values, which are originally equal to -9999, are replaced by 0 before normalization.

    Args:
        image_rgb (torch.Tensor): The CHM image to normalize (1 channel).

    Returns:
        torch.Tensor: The normalized image, with a mean of 0 and a standard deviation of 1 along each channel.
    """
    image_chm = torch.where(image_chm == -9999, no_data_replacement, image_chm)
    # On the whole image
    mean_chm = 2.4113
    std_chm = 5.5642
    # # On the labeled parts
    # mean_chm = 3.0424
    # std_chm = 4.5557
    return (image_chm - mean_chm) / std_chm


def is_tif_file(file_path: str) -> bool:
    return file_path.lower().endswith((".tif", ".tiff"))


class TreeDataset(Dataset):
    """Tree dataset."""

    def __init__(
        self,
        files_paths_list: List[Dict[str, str]],
        labels_to_index: Dict[str, int],
        labels_to_color: Dict[str, Tuple[int, int, int]],
        proba_drop_rgb: float = 0.0,
        labels_transformation_drop_rgb: Dict[str, str | None] | None = None,
        proba_drop_chm: float = 0.0,
        labels_transformation_drop_chm: Dict[str, str | None] | None = None,
        dismissed_classes: List[str] = [],
        transform_spatial: Callable | None = None,
        transform_pixel: Callable | None = None,
    ) -> None:
        """A dataset holding the necessary data for training a model on bounding boxes with
        RGB and CHM data.

        Args:
            files_paths_list (List[Dict[str, str]]): list of dictionaries containing the paths to
            RGB, CHM and annotations.
            labels_to_index (Dict[str, int]): dictionary associating a label name with an index.
            labels_to_color (Dict[str, Tuple[int, int, int]]): dictionary associating a label name
            with a color.
            proba_drop_rgb (float, optional): probability to drop the RGB image and replace it by a
            tensor of zeros. Default to 0.0.
            labels_transformation_drop_rgb (Dict[str, str] | None): indicates the labels that
            change to another label if the RGB image is dropped. Is mandatory if proba_drop_rgb > 0.
            Defaults to None.
            proba_drop_chm (float, optional): probability to drop the CHM image and replace it by a
            tensor of zeros. Default to 0.0.
            labels_transformation_drop_chm (Dict[str, str] | None): indicates the labels that
            change to another label if the CHM image is dropped. Is mandatory if proba_drop_chm > 0.
            Defaults to None.
            dismissed_classes (List[str], optional): list of classes for which the bounding boxes
            are ignored. Defaults to [].
            transform_spatial (Callable | None, optional): spatial augmentations applied to CHM and
            RGB images. Defaults to None.
            transform_pixel (Callable | None, optional): pixel augmentations applied to RGB images.
            Defaults to None.
        """

        assert (
            0.0 <= proba_drop_rgb <= 1.0
        ), f"proba_drop_rgb must be between 0 and 1: {proba_drop_rgb} is wrong."
        assert (
            0.0 <= proba_drop_chm <= 1.0
        ), f"proba_drop_rgb must be between 0 and 1: {proba_drop_chm} is wrong."
        assert (
            0.0 <= proba_drop_rgb + proba_drop_chm <= 1.0
        ), f"(proba_drop_rgb + proba_drop_chm) must be between 0 and 1: {proba_drop_rgb + proba_drop_chm} is wrong."
        assert (
            proba_drop_rgb == 0.0 or labels_transformation_drop_rgb is not None
        ), "If (proba_drop_rgb > 0.0), then labels_transformation_drop_rgb must be a dictionary."
        assert (
            proba_drop_chm == 0.0 or labels_transformation_drop_chm is not None
        ), "If (proba_drop_chm > 0.0), then labels_transformation_drop_chm must be a dictionary."

        self.files_paths_list = files_paths_list
        self.bboxes: List[List[List[float]]] = []
        self.labels: List[List[int]] = []
        for i, files_dict in enumerate(files_paths_list):
            annotations_file_path = files_dict["annotations"]
            bboxes, labels = get_bounding_boxes(annotations_file_path, dismissed_classes)
            self.bboxes.append([bbox.as_list() for bbox in bboxes])
            self.labels.append([labels_to_index[label] for label in labels])

        self.labels_to_index = labels_to_index
        self.labels_to_str = {value: key for key, value in self.labels_to_index.items()}
        self.labels_to_color = labels_to_color
        self.proba_drop_rgb = proba_drop_rgb
        self.labels_transformation_drop_rgb = labels_transformation_drop_rgb
        self.proba_drop_chm = proba_drop_chm
        self.labels_transformation_drop_chm = labels_transformation_drop_chm
        self.transform_spatial = transform_spatial
        self.transform_pixel = transform_pixel
        self._init_channels_count()

    def _init_channels_count(self):
        if len(self) == 0:
            self.rgb_channels = 0
            self.chm_channels = 0
        else:
            files_paths = self.files_paths_list[0]
            rgb_path = files_paths["rgb"]
            image_rgb = self._read_rgb_image(rgb_path)
            chm_path = files_paths["chm"]
            image_chm = self._read_chm_image(chm_path)
            self.rgb_channels = image_rgb.shape[2]
            self.chm_channels = image_chm.shape[2]

    def __len__(self) -> int:
        return len(self.files_paths_list)

    def _read_rgb_image(self, image_path: str) -> np.ndarray:
        if is_tif_file(image_path):
            image = tifffile.imread(image_path)
        else:
            image = np.array(Image.open(image_path))
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        return image

    def _read_chm_image(self, image_path: str) -> np.ndarray:
        if is_tif_file(image_path):
            image = tifffile.imread(image_path)
        else:
            image = np.array(Image.open(image_path))
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        return image.astype(np.float32)

    def random_chm_rgb_drop(
        self,
        image_rgb: np.ndarray,
        image_chm: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[int]]:
        """Randomly drops the RGB or the CHM image with probabilities specified during
        initialization. The bounding boxes labels are modified accordingly.

        Args:
            image_rgb (np.ndarray): RGB image.
            image_chm (np.ndarray): CHM image
            bboxes (List[List[float]]): bounding boxes.
            labels (List[int]): class labels.

        Raises:
            TypeError: if self.labels_transformation_drop_rgb is None but the methods wants to drop
            RGB.
            TypeError: if self.labels_transformation_drop_chm is None but the methods wants to drop
            CHM.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[List[float]], List[int]]: (RGB image, CHM image,
            bounding boxes, class labels) after modification.
        """
        random_val = random.random()
        if random_val < self.proba_drop_rgb:
            # Drop RGB
            if self.labels_transformation_drop_rgb is None:
                raise TypeError("self.labels_transformation_drop_rgb shouldn't be None here.")
            image_rgb = np.zeros_like(image_rgb)
            # Replace the labels
            drop: List[int] = []
            for i, label in enumerate(labels):
                label_str = self.labels_to_str[label]
                new_label_str = self.labels_transformation_drop_rgb[label_str]
                if new_label_str is None:
                    drop.append(i)
                else:
                    new_label = self.labels_to_index[new_label_str]
                    labels[i] = new_label
            # Remove the bounding boxes to drop
            for idx in reversed(drop):
                bboxes.pop(idx)
                labels.pop(idx)

        elif random_val < self.proba_drop_rgb + self.proba_drop_chm:
            # Drop CHM
            if self.labels_transformation_drop_chm is None:
                raise TypeError("self.labels_transformation_drop_chm shouldn't be None here.")
            image_chm = np.zeros_like(image_chm)
            # Replace the labels
            drop: List[int] = []
            for i, label in enumerate(labels):
                label_str = self.labels_to_str[label]
                new_label_str = self.labels_transformation_drop_chm[label_str]
                if new_label_str is None:
                    drop.append(i)
                else:
                    new_label = self.labels_to_index[new_label_str]
                    labels[i] = new_label
            # Remove the bounding boxes to drop
            for idx in reversed(drop):
                bboxes.pop(idx)
                labels.pop(idx)
        return image_rgb, image_chm, bboxes, labels

    def get_not_normalized(self, idx: int) -> Dict[str, torch.Tensor]:
        # Read the images
        files_paths = self.files_paths_list[idx]
        rgb_path = files_paths["rgb"]
        image_rgb = self._read_rgb_image(rgb_path)
        chm_path = files_paths["chm"]
        image_chm = self._read_chm_image(chm_path)

        # Get bboxes and labels
        bboxes = self.bboxes[idx]
        labels = self.labels[idx]

        # Apply the spatial transform to the two images, bboxes and labels
        if self.transform_spatial is not None:
            transformed_spatial = self.transform_spatial(
                image=image_rgb,
                image_chm=image_chm,
                bboxes=bboxes,
                class_labels=labels,
            )
            image_rgb = transformed_spatial["image"]
            image_chm = transformed_spatial["image_chm"]
            bboxes = transformed_spatial["bboxes"]
            labels = transformed_spatial["class_labels"]

        # Apply the pixel transform the to RGB image
        if self.transform_pixel is not None:
            transformed = self.transform_pixel(image=image_rgb)
            image_rgb = transformed["image"]

        image_rgb, image_chm, bboxes, labels = self.random_chm_rgb_drop(
            image_rgb, image_chm, bboxes, labels
        )

        to_tensor = Atorch.ToTensorV2()

        sample = {
            "image_rgb": to_tensor(image=image_rgb)["image"],
            "image_chm": to_tensor(image=image_chm)["image"],
            "bboxes": torch.tensor(bboxes),
            "labels": torch.tensor(labels),
            "image_index": idx,
        }
        return sample

    def __getitem__(self, idx: int):
        sample = self.get_not_normalized(idx)
        sample["image_rgb"] = normalize_rgb(sample["image_rgb"])
        sample["image_chm"] = normalize_chm(sample["image_chm"])

        return sample

    def get_rgb_image(self, idx: int) -> np.ndarray:
        """Returns the RGB image corresponding to the index.

        Args:
            idx (int): index of the data.

        Returns:
            np.ndarray: RGB image.
        """
        files_paths = self.files_paths_list[idx]
        rgb_path = files_paths["rgb"]
        image_rgb = self._read_rgb_image(rgb_path)
        return image_rgb

    def get_full_coords_name(self, idx: int) -> str:
        """Returns the full coordinates name of the data corresponding to the image.

        Args:
            idx (int): index of the data.

        Returns:
            str: full coordinates name (with full image coordinates and cropped pixels).
        """
        full_image_name = self.get_full_image_name(idx)
        full_image_coords = get_coordinates_from_full_image_file_name(full_image_name)
        coord_name = self.get_cropped_coords_name(idx)
        return "_".join([str(full_image_coords[0]), str(full_image_coords[1]), coord_name])

    def get_cropped_coords_name(self, idx: int) -> str:
        """Return the pixel coordinates name of the data.

        Args:
            idx (int): index of the data.

        Returns:
            str: pixel coordinates name of the data.
        """
        files_paths = self.files_paths_list[idx]
        rgb_path = files_paths["rgb"]
        coord_name = get_file_base_name(rgb_path)
        return coord_name

    def get_full_image_name(self, idx: int) -> str:
        """Returns the name of the full image from which the data comes.

        Args:
            idx (int): index of the data.

        Returns:
            str: name of the full image (without any extension).
        """
        files_paths = self.files_paths_list[idx]
        rgb_path = files_paths["rgb"]
        full_image_name = os.path.basename(os.path.dirname(rgb_path))
        return full_image_name


def tree_dataset_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for a Dataloader taking a TreeDataset.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A batch as a list of TreeDataset outputs.

    Returns:
        Dict[str, torch.Tensor]: The final batch returned by the DataLoader.
    """
    # Initialize lists to hold the extracted components
    rgb_images = []
    chm_images = []
    bboxes = []
    labels = []
    indices = []
    image_indices = []

    # Iterate through the batch
    for i, item in enumerate(batch):
        # Extract the components from the dictionary
        rgb_image = item["image_rgb"]
        chm_image = item["image_chm"]
        bbox = item["bboxes"]
        label = item["labels"]
        image_index = item["image_index"]

        # Append the extracted components to the lists
        rgb_images.append(rgb_image)
        chm_images.append(chm_image)
        bboxes.append(bbox)
        labels.append(label)
        indices.extend([i] * bbox.shape[0])
        image_indices.append(image_index)

    # Convert the lists to tensors and stack them
    rgb_images = torch.stack(rgb_images, dim=0)
    chm_images = torch.stack(chm_images, dim=0)
    bboxes = torch.cat(bboxes).to(torch.float32)
    labels = torch.cat(labels)
    indices = torch.tensor(indices)
    image_indices = torch.tensor(image_indices)

    output_batch = {
        "image_rgb": rgb_images,
        "image_chm": chm_images,
        "bboxes": bboxes,
        "labels": labels,
        "indices": indices,
        "image_indices": image_indices,
    }

    return output_batch


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


class TrainingMetrics:
    def __init__(self, float_precision: int = 3) -> None:
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: defaultdict(lambda: {"epochs": [], "avgs": []}))
        self.metrics_loop = defaultdict(
            lambda: defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})
        )

        self.out = Output()
        display.display(self.out)

    def end_loop(self, epoch: int):
        for metric_name, metric_dict in self.metrics_loop.items():
            for category_name, category_dict in metric_dict.items():
                metric = self.metrics[metric_name][category_name]
                metric["epochs"].append(epoch)
                metric["avgs"].append(category_dict["avg"])
        self.metrics_loop = defaultdict(
            lambda: defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})
        )

    def update(self, category_name: str, metric_name: str, val: float, count: int = 1):
        metric = self.metrics_loop[metric_name][category_name]

        metric["val"] += val
        metric["count"] += count
        metric["avg"] = metric["val"] / metric["count"]

    def get_last(self, category_name: str, metric_name: str):
        return self.metrics_loop[metric_name][category_name]["avg"]

    def visualize(self, save_path: str | None = None):
        # Inspired from https://gitlab.com/robindar/dl-scaman_checker/-/blob/main/src/dl_scaman_checker/TP01.py
        with self.out:
            metrics_index: Dict[str, int] = {}
            categories_index: Dict[str, int] = {}
            for i, (metric_name, metric_dict) in enumerate(self.metrics.items()):
                metrics_index[metric_name] = i
                for category_name in metric_dict.keys():
                    if category_name not in categories_index.keys():
                        categories_index[category_name] = len(categories_index)

            nrows = max(len(metrics_index), 1)
            cmap = plt.get_cmap("tab10")

            categories_colors = {label: cmap(i) for i, label in enumerate(categories_index.keys())}

            fig = plt.figure(1, figsize=(6, 5 * nrows))

            for metric_name, metric_dict in self.metrics.items():
                ax = fig.add_subplot(nrows, 1, metrics_index[metric_name] + 1)
                for category_name, category_dict in metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]
                    fmt = "-" if len(epochs) > 100 else "-o"
                    ax.plot(
                        epochs,
                        values,
                        fmt,
                        color=categories_colors[category_name],
                        label=category_name,
                    )
                    ax.grid(alpha=0.5)
                    ax.set_xlabel("Epoch")
                    ax.set_yscale("log")
                    ax.set_ylabel(metric_name)
                ax.set_title(f"{metric_name}")
            plt.tight_layout()

            has_legend, _ = plt.gca().get_legend_handles_labels()
            if any(label != "" for label in has_legend):
                plt.legend()

            if save_path is not None:
                plt.savefig(save_path, dpi=200)

            plt.show()
            display.clear_output(wait=True)


def perfect_preds(
    gt_bboxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_indices: torch.Tensor,
    batch_size: int,
):
    extracted_bboxes = [[]] * batch_size
    extracted_classes = [[]] * batch_size
    for bbox_idx, image_idx in enumerate(gt_indices):
        slice_bboxes = gt_bboxes[bbox_idx]
        extracted_bboxes[image_idx].append(slice_bboxes)
        slice_classes = gt_classes[bbox_idx].long()
        extracted_classes[image_idx].append(slice_classes)
    scores = [
        20 * nn.functional.one_hot(torch.tensor(cls), num_classes=5) - 0.5
        for cls in extracted_classes
    ]
    prefect_preds = [
        torch.cat((torch.tensor(bboxes), classes), dim=1).permute((1, 0)).unsqueeze(0)
        for bboxes, classes in zip(extracted_bboxes, scores)
    ]
    perfect_preds = torch.cat(
        [
            torch.cat(
                (
                    pred,
                    torch.zeros((pred.shape[0], pred.shape[1], 8400 - pred.shape[2])).to(
                        pred.device
                    ),
                ),
                dim=2,
            )
            for pred in prefect_preds
        ]
    )
    return perfect_preds


def print_current_memory():
    if torch.cuda.is_available():
        current_memory_usage_bytes = torch.cuda.memory_allocated()
        current_memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
        print(f"Current GPU memory usage: {current_memory_usage_megabytes:.2f} MB")
    else:
        print("CUDA is not available.")


def train(
    train_loader: TreeDataLoader,
    model: AMF_GD_YOLOv8,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    running_accumulation_step: int,
    training_metrics: TrainingMetrics,
) -> int:
    model.train()
    stream = tqdm(train_loader, leave=False, desc="Training")
    for data in stream:
        image_rgb: torch.Tensor = data["image_rgb"]
        image_chm: torch.Tensor = data["image_chm"]
        gt_bboxes: torch.Tensor = data["bboxes"]
        gt_classes: torch.Tensor = data["labels"]
        gt_indices: torch.Tensor = data["indices"]

        image_rgb = image_rgb.to(device, non_blocking=True)
        image_chm = image_chm.to(device, non_blocking=True)
        gt_bboxes = gt_bboxes.to(device, non_blocking=True)
        gt_classes = gt_classes.to(device, non_blocking=True)
        gt_indices = gt_indices.to(device, non_blocking=True)

        output = model(image_rgb, image_chm)
        total_loss = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)[0]

        batch_size = image_rgb.shape[0]
        training_metrics.update("Training", "Loss", total_loss.item(), batch_size)

        total_loss.backward()

        # Gradient accumulation
        if (running_accumulation_step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_accumulation_step += 1

    return running_accumulation_step


def validate(
    val_loader: TreeDataLoader,
    model: AMF_GD_YOLOv8,
    device: torch.device,
    training_metrics: TrainingMetrics,
) -> float:
    model.eval()
    stream = tqdm(val_loader, leave=False, desc="Validation")
    with torch.no_grad():
        for data in stream:
            image_rgb: torch.Tensor = data["image_rgb"]
            image_chm: torch.Tensor = data["image_chm"]
            gt_bboxes: torch.Tensor = data["bboxes"]
            gt_classes: torch.Tensor = data["labels"]
            gt_indices: torch.Tensor = data["indices"]

            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)
            gt_classes = gt_classes.to(device, non_blocking=True)
            gt_indices = gt_indices.to(device, non_blocking=True)

            output = model(image_rgb, image_chm)
            total_loss = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)[0]

            batch_size = image_rgb.shape[0]
            training_metrics.update("Validation", "Loss", total_loss.item(), batch_size)

    return training_metrics.get_last("Validation", "Loss")


def test_save_output_image(
    model: AMF_GD_YOLOv8,
    test_loader: TreeDataLoader,
    epoch: int,
    device: torch.device,
    no_rgb: bool = False,
    no_chm: bool = False,
    save_path: str | None = None,
):
    model.eval()
    geojson_outputs: List[geojson.FeatureCollection] = []
    with torch.no_grad():
        for data in tqdm(test_loader, leave=False, desc="Exporting output"):
            image_rgb: torch.Tensor = data["image_rgb"]
            if no_rgb:
                image_rgb = torch.zeros_like(image_rgb)
            image_chm: torch.Tensor = data["image_chm"]
            if no_chm:
                image_chm = torch.zeros_like(image_chm)
            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)
            results = model.predict(image_rgb, image_chm)[2]

            idx = data["image_indices"]

            # initial_rgb = test_loader.dataset.get_rgb_image(idx)
            # full_coords_name = test_loader.dataset.get_full_coords_name(idx)
            # colors_dict = test_loader.dataset.labels_to_color

            if results.boxes is not None:
                bboxes = results.boxes.xyxy.tolist()
                labels = [results.names[cls.item()] for cls in results.boxes.cls]
                scores = results.boxes.conf.tolist()
            else:
                bboxes = []
                labels = []
                scores = []

            # Save the image if there is at least one bounding box
            # bboxes_image = create_bboxes_image(
            #     image=initial_rgb,
            #     bboxes=bboxes,
            #     labels=labels,
            #     colors_dict=colors_dict,
            #     scores=scores,
            #     color_mode="bgr",
            # )

            # output_name = f"Output_{epoch}ep_{full_coords_name}.png"
            # output_path = os.path.join(Folders.OUTPUT_DIR.value, output_name)
            # cv2.imwrite(output_path, bboxes_image)

            # Store the bounding boxes in a GeoJSON file
            full_image_name = test_loader.dataset.get_full_image_name(idx)
            cropped_coords_name = test_loader.dataset.get_cropped_coords_name(idx)
            bboxes_as_box = [Box.from_list(bbox) for bbox in bboxes]
            geojson_features = create_geojson_output(
                full_image_name, cropped_coords_name, bboxes_as_box, labels, scores
            )
            geojson_outputs.append(geojson_features)

    geojson_outputs_merged = merge_geojson_feature_collections(geojson_outputs)
    if save_path is None:
        geojson_save_name = f"{model.name}_output_{epoch}ep.geojson"
        geojson_save_path = os.path.join(Folders.OUTPUT_DIR.value, geojson_save_name)
    else:
        geojson_save_path = save_path
    save_geojson(geojson_outputs_merged, geojson_save_path)


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
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_and_validate(
    model: AMF_GD_YOLOv8,
    datasets: Dict[str, TreeDataset],
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    accumulate: int,
    device: torch.device,
    save_outputs: bool,
) -> nn.Module:

    train_loader, val_loader, test_loader = initialize_dataloaders(
        datasets, batch_size, num_workers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: 1 / np.sqrt(i + 2), last_epoch=-1
    )

    accumulation_steps = max(round(accumulate / batch_size), 1)
    running_accumulation_step = 0

    training_metrics = TrainingMetrics()

    best_model = model
    best_loss = np.inf

    if save_outputs:
        test_save_output_image(
            model=model,
            test_loader=test_loader,
            epoch=0,
            device=device,
        )
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        training_metrics.visualize()
        running_accumulation_step = train(
            train_loader,
            model,
            optimizer,
            device,
            accumulation_steps,
            running_accumulation_step,
            training_metrics,
        )
        current_loss = validate(val_loader, model, device, training_metrics)
        if save_outputs and epoch % 1 == 0:
            test_save_output_image(
                model=model,
                test_loader=test_loader,
                epoch=epoch,
                device=device,
            )
        scheduler.step()

        # Store the best model
        if current_loss < best_loss:
            best_model = model
            best_loss = current_loss

        training_metrics.end_loop(epoch)

    # Save the plot showing the evolution of the metrics
    training_metrics_name = f"{model.name}_training_metrics_plot.png"
    training_metrics_path = os.path.join(Folders.OUTPUT_DIR.value, training_metrics_name)
    training_metrics.visualize(save_path=training_metrics_path)
    return best_model


def get_all_files_iteratively(folder_path: str) -> List[str]:
    """Finds iteratively all the files below the input folder.

    Args:
        folder_path (str): folder to look into.

    Returns:
        List[str]: the list of all the files.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files


def split_files_into_lists(
    folder_path: str,
    sets_ratios: Sequence[int | float],
    sets_names: List[str],
    random_seed: int | None = None,
) -> Dict[str, List[str]]:
    """Splits files in a folder into multiple lists based on specified ratios.

    Args:
        folder_path (str): path to the folder containing the files.
        sets_ratios (List[int | float]): the proportions for each list.
        sets_names (List[str]): the keys for the dictionary
        random_seed (int | None, optional): a seed for the randomization. Defaults to None.

    Returns:
        Dict[str, List[str]]: a dictionary where each key from the input names is linked
        with a list of files.
    """
    files = get_all_files_iteratively(folder_path)
    total_ratio = sum(sets_ratios)
    ratios = [r / total_ratio for r in sets_ratios]

    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(files)
    split_indices = [0] * (len(ratios) + 1)
    split_indices[-1] = len(files)
    sum_ratios = 0.0
    for i in range(len(ratios) - 1):
        sum_ratios += ratios[i]
        split_indices[i + 1] = int(round(len(files) * (sum_ratios)))

    files_dict = {}
    for i in range(len(ratios)):
        files_dict[sets_names[i]] = files[split_indices[i] : split_indices[i + 1]]

    return files_dict


def create_and_save_splitted_datasets(
    rgb_folder_path: str,
    chm_folder_path: str,
    annotations_folder_path: str,
    sets_ratios: Sequence[int | float],
    sets_names: List[str],
    save_path: str,
    random_seed: int | None = None,
) -> None:
    files_dict = split_files_into_lists(
        folder_path=rgb_folder_path,
        sets_ratios=sets_ratios,
        sets_names=sets_names,
        random_seed=random_seed,
    )
    all_files_dict = {}
    for set_name, set_files in files_dict.items():
        all_files_dict[set_name] = []
        for rgb_file in set_files:
            full_image = os.path.join(
                Folders.FULL_IMAGES.value,
                f"{os.path.basename(os.path.dirname(rgb_file))}.tif",
            )
            image_data = ImageData(full_image)

            chm_file = rgb_file.replace(rgb_folder_path, chm_folder_path).replace(
                image_data.base_name, image_data.coord_name
            )

            annotations_file = rgb_file.replace(rgb_folder_path, annotations_folder_path).replace(
                ".tif", ".json"
            )

            new_dict = {
                "rgb": rgb_file,
                "chm": chm_file,
                "annotations": annotations_file,
            }
            all_files_dict[set_name].append(new_dict)

    with open(save_path, "w") as f:
        json.dump(all_files_dict, f)


def load_tree_datasets_from_split(
    data_split_file_path: str,
    labels_to_index: Dict[str, int],
    labels_to_color: Dict[str, Tuple[int, int, int]],
    dismissed_classes: List[str] = [],
    transform_spatial_training: Callable | None = None,
    transform_pixel_training: Callable | None = None,
    proba_drop_rgb: float = 0.0,
    labels_transformation_drop_rgb: Dict[str, str | None] | None = None,
    proba_drop_chm: float = 0.0,
    labels_transformation_drop_chm: Dict[str, str | None] | None = None,
) -> Dict[str, TreeDataset]:
    with open(data_split_file_path, "r") as f:
        data_split = json.load(f)

    tree_datasets = {}
    tree_datasets["training"] = TreeDataset(
        data_split["training"],
        labels_to_index=labels_to_index,
        labels_to_color=labels_to_color,
        proba_drop_rgb=proba_drop_rgb,
        labels_transformation_drop_rgb=labels_transformation_drop_rgb,
        proba_drop_chm=proba_drop_chm,
        labels_transformation_drop_chm=labels_transformation_drop_chm,
        dismissed_classes=dismissed_classes,
        transform_spatial=transform_spatial_training,
        transform_pixel=transform_pixel_training,
    )

    tree_datasets["validation"] = TreeDataset(
        data_split["validation"],
        labels_to_index=labels_to_index,
        labels_to_color=labels_to_color,
        proba_drop_rgb=0.0,
        labels_transformation_drop_rgb=None,
        proba_drop_chm=0.0,
        labels_transformation_drop_chm=None,
        dismissed_classes=dismissed_classes,
        transform_spatial=None,
        transform_pixel=None,
    )

    if "test" in data_split:
        test_data = data_split["test"]
    else:
        test_data = data_split["validation"]

    tree_datasets["test"] = TreeDataset(
        test_data,
        labels_to_index=labels_to_index,
        labels_to_color=labels_to_color,
        proba_drop_rgb=0.0,
        labels_transformation_drop_rgb=None,
        proba_drop_chm=0.0,
        labels_transformation_drop_chm=None,
        dismissed_classes=dismissed_classes,
        transform_spatial=None,
        transform_pixel=None,
    )

    return tree_datasets
