import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations import pytorch as Atorch
from layers import AMF_GD_YOLOv8
from plot import create_bboxes_image, get_bounding_boxes
from skimage import io
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.notebook import tqdm
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
            message = f"The path {path} (absolute path is {os.path.abspath(path)}) is not a valid file."
        elif type == "folder":
            message = f"The path {path} (absolute path is {os.path.abspath(path)}) is not a valid folder."
        else:
            raise ValueError(
                f'No InvalidPathException for type "{type}" is implemented.'
            )
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


def create_gt_bboxes_image(
    annotations_path: str,
    rgb_path: str,
    class_colors: Dict[str, Tuple[int, int, int]],
    output_path: str | None = None,
) -> np.ndarray:
    """Creates and returns an image with the ground truth bounding boxes. The image can also be
    saved directly if a path is given.

    Args:
        annotations_path (str): Path to the file containing the annotations.
        rgb_path (str): Path to the file containing the image.
        class_colors (Dict[str, Tuple[int, int, int]]): Dictionary associating a color to each
        class name.
        output_path (str | None, optional): An optional path to save the created image at.
        Defaults to None.

    Returns:
        np.ndarray: The image with the bounding boxes.
    """
    rgb_image = io.imread(rgb_path)
    bboxes, labels = get_bounding_boxes(annotations_path)
    bboxes_list = [bbox.as_list() for bbox in bboxes]
    image = create_bboxes_image(rgb_image, bboxes_list, labels, class_colors)
    if output_path is not None:
        io.imsave(output_path, image)
    return image


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


def normalize_chm(image_chm: torch.Tensor) -> torch.Tensor:
    """Normalizes the CHM image given as input, with values computed using the unfiltered CHM of
    2023_122000_484000_RGB_hrl.tif with a resolution of 8cm.
    The NO_DATA values, which are originally equal to -9999, are replaced by 0 before normalization.

    Args:
        image_rgb (torch.Tensor): The CHM image to normalize (1 channel).

    Returns:
        torch.Tensor: The normalized image, with a mean of 0 and a standard deviation of 1 along each channel.
    """
    image_chm = torch.where(image_chm == -9999, 0, image_chm)
    # On the whole image
    mean_chm = 2.4113
    std_chm = 5.5642
    # # On the labeled parts
    # mean_chm = 3.0424
    # std_chm = 4.5557
    return (image_chm - mean_chm) / std_chm


class TreeDataset(Dataset):
    """Tree dataset."""

    # TODO: Change to take a list of files instead of a folder? (to separate train and validation)

    def __init__(
        self,
        annotations_folder_path: str,
        rgb_folder_path: str,
        chm_folder_path: str,
        labels_to_index: Dict[str, int],
        labels_to_color: Dict[str, Tuple[int, int, int]],
        transform_spatial: Callable | None = None,
        transform_pixel: Callable | None = None,
    ) -> None:
        """A dataset holding the necessary data for training a model on bounding boxes with
        RGB anc CHM data.

        Args:
            annotations_folder_path (str): The path to the folder containing the annotations.
            rgb_folder_path (str): The path to the folder containing the RGB images.
            chm_folder_path (str): The path to the folder containing the CHM images.
            labels_to_index (Dict[str, int]): Dictionary associating a label name with an index.
            labels_to_color (Dict[str, Tuple[int, int, int]]): Dictionary associating a label name with a color.
            transform_spatial (Callable | None, optional): The spatial augmentations applied to CHM and RGB images. Defaults to None.
            transform_pixel (Callable | None, optional): The pixel augmentations applied to RGB images. Defaults to None.
        """
        self.annotations_list: List[str] = []
        self.bboxes: Dict[str, List[List[float]]] = {}
        self.labels: Dict[str, List[int]] = {}
        for file_name in os.listdir(annotations_folder_path):
            annotations_file_path = os.path.join(annotations_folder_path, file_name)
            base_name = get_file_base_name(annotations_file_path)
            self.annotations_list.append(base_name)
            bboxes, labels = get_bounding_boxes(annotations_file_path)
            self.bboxes[base_name] = [bbox.as_list() for bbox in bboxes]
            self.labels[base_name] = [labels_to_index[label] for label in labels]

        self.rgb_folder_path = rgb_folder_path
        self.chm_folder_path = chm_folder_path
        self.labels_to_index = labels_to_index
        self.labels_to_str = {value: key for key, value in self.labels_to_index.items()}
        self.labels_to_color = labels_to_color
        self.transform_spatial = transform_spatial
        self.transform_pixel = transform_pixel

    def __len__(self) -> int:
        return len(self.annotations_list)

    def get_unnormalized(self, idx) -> Dict[str, torch.Tensor]:
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Read the images
        base_name = self.annotations_list[idx]
        rgb_path = os.path.join(self.rgb_folder_path, f"{base_name}.tif")
        image_rgb = io.imread(rgb_path)
        chm_path = os.path.join(self.chm_folder_path, f"{base_name}.tif")
        image_chm = io.imread(chm_path)

        # Get bboxes and labels
        bboxes = self.bboxes[base_name]
        labels = self.labels[base_name]

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

        to_tensor = Atorch.ToTensorV2()

        sample = {
            "image_rgb": to_tensor(image=image_rgb)["image"],
            "image_chm": to_tensor(image=image_chm)["image"],
            "bboxes": torch.tensor(bboxes),
            "labels": torch.tensor(labels),
            "image_index": idx,
        }
        return sample

    def __getitem__(self, idx):
        sample = self.get_unnormalized(idx)
        sample["image_rgb"] = normalize_rgb(sample["image_rgb"])
        sample["image_chm"] = normalize_chm(sample["image_chm"])

        return sample

    def get_rgb_image(self, idx) -> np.ndarray:
        base_name = self.annotations_list[idx]
        rgb_path = os.path.join(self.rgb_folder_path, f"{base_name}.tif")
        image_rgb = io.imread(rgb_path)
        return image_rgb


def tree_dataset_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
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
        

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})

    def update(self, metric_name: str, val: float, count: int = 1):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += count
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                f"{metric_name}: {metric["avg"]:.{self.float_precision}f}"
                for (metric_name, metric) in self.metrics.items()
            ]
        )


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
                    torch.zeros(
                        (pred.shape[0], pred.shape[1], 8400 - pred.shape[2])
                    ).to(pred.device),
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
    epoch: int,
    device: torch.device,
    accumulation_steps: int,
    running_accumulation_step: int,
) -> int:
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader, leave=False)
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
        metric_monitor.update("Loss", total_loss.item(), batch_size)

        total_loss.backward()

        # Gradient accumulation
        if (running_accumulation_step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_accumulation_step += 1
        stream.set_description(f"Epoch: {epoch}. Train.      {metric_monitor}")

    return running_accumulation_step


def validate(
    val_loader: TreeDataLoader, model: AMF_GD_YOLOv8, epoch: int, device: torch.device
):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader, leave=False)
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
            total_loss = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)[
                0
            ]

            batch_size = image_rgb.shape[0]
            metric_monitor.update("Loss", total_loss.item(), batch_size)

            stream.set_description(f"Epoch: {epoch}. Validation. {metric_monitor}")


def test_save_output_image(
    model: AMF_GD_YOLOv8,
    test_loader: TreeDataLoader,
    epoch: int,
    device: torch.device,
    number_images: int = 1,
):
    saved_images = 0
    number_images = min(len(test_loader), number_images)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            image_rgb: torch.Tensor = data["image_rgb"]
            image_chm: torch.Tensor = data["image_chm"]
            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)
            results = model.predict(image_rgb, image_chm)[2]

            initial_rgb = test_loader.dataset.get_rgb_image(data["image_indices"])
            colors_dict = test_loader.dataset.labels_to_color
            if results.boxes is not None:
                bboxes = results.boxes.xyxy.tolist()
                labels = [results.names[cls.item()] for cls in results.boxes.cls]
                scores = results.boxes.conf.tolist()
            else:
                bboxes = []
                labels = []
                scores = []

            # Save the image if there is at least one bounding box
            if (len(bboxes) > 0):
                bboxes_image = create_bboxes_image(
                    image=initial_rgb,
                    bboxes=bboxes,
                    labels=labels,
                    colors_dict=colors_dict,
                    scores=scores,
                    color_mode="bgr",
                )

                output_name = f"Validation_bboxes_{epoch}_{saved_images}_{len(bboxes)}.png"
                output_path = os.path.join(Folders.OUTPUT_DIR.value, output_name)
                cv2.imwrite(output_path, bboxes_image)

            saved_images += 1
            if saved_images >= number_images:
                break
          
            
def train_and_validate(
    model: AMF_GD_YOLOv8,
    train_dataset: TreeDataset,
    val_dataset: TreeDataset,
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    accumulate: int,
    device: torch.device,
) -> nn.Module:
    train_loader = TreeDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = TreeDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = TreeDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: 1 / np.sqrt(i + 2), last_epoch=-1
    )

    accumulation_steps = round(accumulate / batch_size)
    running_accumulation_step = 0

    test_save_output_image(
        model=model, test_loader=test_loader, epoch=0, device=device, number_images=2
    )
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        running_accumulation_step = train(
            train_loader,
            model,
            optimizer,
            epoch,
            device,
            accumulation_steps,
            running_accumulation_step,
        )
        validate(val_loader, model, epoch, device)
        if epoch % 1 == 0:
            test_save_output_image(
                model=model,
                test_loader=test_loader,
                epoch=epoch,
                device=device,
                number_images=2,
            )
        scheduler.step()
    return model

