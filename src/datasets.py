import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import tifffile
import torch
from albumentations import pytorch as Atorch
from PIL import Image
from torch.utils.data import Dataset

from plot import get_bounding_boxes
from utils import get_coordinates_from_full_image_file_name, get_file_base_name


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


def _compute_channels_mean_std_tensor(
    image: torch.Tensor,
    per_channel: bool,
    replace_no_data: bool,
    no_data_new_value: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean and the standard deviation along every channel of the image given as input.

    Args:
        image (torch.Tensor): image tensor of shape [w, h] or [c, w, h].
        per_channel (bool): whether to compute one value for each channel or one for the whole images.
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before computations.
        no_data_new_value (float): the value replacing NO_DATA (-9999) before computations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mean, std), two tensors of shape (c,) if per_channel is
        True and (1,) otherwise. They contain respectively the mean value and the standard deviation.
    """
    if replace_no_data:
        NO_DATA = -9999
        image[image == NO_DATA] = no_data_new_value
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    dims = (0, 1) if per_channel else (0, 1, 2)
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    mean = torch.mean(image.to(dtype), dim=dims).reshape(-1)
    std = torch.std(image.to(dtype), dim=dims).reshape(-1)
    return mean, std


def _compute_channels_mean_and_std_file(
    file_path: str,
    per_channel: bool,
    replace_no_data: bool,
    no_data_new_value: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean and the standard deviation along every channel of the image given as input.

    Args:
        file_path (str): path to the image.
        per_channel (bool): whether to compute one value for each channel or one for the whole images.
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before computations.
        no_data_new_value (float): the value replacing NO_DATA (-9999) before computations.

    Raises:
        InvalidPathException: if the input path is not a file.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mean, std), two tensors of shape (c,) if per_channel is
        True and (1,) otherwise. They contain respectively the mean value and the standard deviation.
    """
    if not os.path.isfile(file_path):
        raise InvalidPathException(file_path, "file")
    if is_tif_file(file_path):
        image = torch.from_numpy(tifffile.imread(file_path))
    else:
        image = torch.from_numpy(np.array(Image.open(file_path)))
    return _compute_channels_mean_std_tensor(image, per_channel, replace_no_data, no_data_new_value)


def compute_mean_and_std(
    file_or_folder_path: str,
    per_channel: bool,
    replace_no_data: bool,
    no_data_new_value: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean and the standard deviation along every channel of the image(s) given as input
    (either as one file or a directory containing multiple files).

    Args:
        file_or_folder_path (str): path to the image or the folder of images.
        per_channel (bool): whether to compute one value for each channel or one for the whole images.
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before computations.
        no_data_new_value (float): the value replacing NO_DATA (-9999) before computations.

    Raises:
        InvalidPathException: if the input path is neither a file, nor a directory.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mean, std), two tensors of shape (c,) if per_channel is
        True and (1,) otherwise. They contain respectively the mean value and the standard deviation.
    """
    if os.path.isfile(file_or_folder_path):
        file_path = file_or_folder_path
        return _compute_channels_mean_and_std_file(
            file_path, per_channel, replace_no_data, no_data_new_value
        )

    elif os.path.isdir(file_or_folder_path):
        means = []
        stds = []
        folder_path = file_or_folder_path
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            mean, std = _compute_channels_mean_and_std_file(
                file_path, per_channel, replace_no_data, no_data_new_value
            )
            means.append(mean)
            stds.append(std)
        mean_mean = np.mean(means, axis=0)
        mean_std = np.mean(stds, axis=0)
        return torch.from_numpy(mean_mean), torch.from_numpy(mean_std)

    else:
        raise InvalidPathException(file_or_folder_path)


def normalize(
    image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    replace_no_data: bool,
    no_data_new_value: float = 0,
) -> torch.Tensor:
    """Normalizes the CHM image given as input.

    Args:
        image (torch.Tensor): the image to normalize with c channels. Must be of shape [w, h] or [c, w, h].
        mean (torch.Tensor): the mean to normalize with. Must be of shape [], [1] or [c].
        std (torch.Tensor): the standard deviation to normalize with. Must be of shape [], [1] or [c].
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before normalization.
        no_data_new_value (float): the value replacing NO_DATA (-9999) before normalization.

    Returns:
        torch.Tensor: the normalized image.
    """
    if replace_no_data:
        image = torch.where(image == -9999, no_data_new_value, image)

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    channels = image.shape[0]

    def reshape(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if len(tensor.shape) == 0:
            tensor = torch.full((channels,), tensor.item())
        elif len(tensor.shape) == 1 and tensor.shape[0] == 1:
            tensor = torch.full((channels,), tensor[0].item())
        elif len(tensor.shape) == 1 and tensor.shape[0] == channels:
            pass
        else:
            raise ValueError(
                f"Unsupported shape for `{name}`. It should be a tensor or shape [], [1] or [{channels}]"
            )
        return tensor.view(-1, 1, 1)

    mean = reshape(mean, "mean").to(image.dtype)
    std = reshape(std, "std").to(image.dtype)

    return (image - mean) / std


def denormalize(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    channels = image.shape[0]

    def reshape(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if len(tensor.shape) == 0:
            tensor = torch.full((channels,), tensor.item())
        elif len(tensor.shape) == 1 and tensor.shape[0] == 1:
            tensor = torch.full((channels,), tensor[0].item())
        elif len(tensor.shape) == 1 and tensor.shape[0] == channels:
            pass
        else:
            raise ValueError(
                f"Unsupported shape for `{name}`. It should be a tensor or shape [], [1] or [{channels}]"
            )
        return tensor.view(-1, 1, 1)

    mean = reshape(mean, "mean").to(image.dtype)
    std = reshape(std, "std").to(image.dtype)

    return std * image + mean


def is_tif_file(file_path: str) -> bool:
    return file_path.lower().endswith((".tif", ".tiff"))


class TreeDataset(Dataset):
    """Tree dataset."""

    def __init__(
        self,
        files_paths_list: List[Dict[str, str]],
        labels_to_index: Dict[str, int],
        mean_rgb: torch.Tensor,
        std_rgb: torch.Tensor,
        mean_chm: torch.Tensor,
        std_chm: torch.Tensor,
        proba_drop_rgb: float = 0.0,
        labels_transformation_drop_rgb: Dict[str, str | None] | None = None,
        proba_drop_chm: float = 0.0,
        labels_transformation_drop_chm: Dict[str, str | None] | None = None,
        dismissed_classes: List[str] = [],
        transform_spatial: A.Compose | None = None,
        transform_pixel_rgb: A.Compose | None = None,
        transform_pixel_chm: A.Compose | None = None,
        no_data_new_value: float = -5.0,
    ) -> None:
        """A dataset holding the necessary data for training a model on bounding boxes with
        RGB and CHM data.

        Args:
            files_paths_list (List[Dict[str, str]]): list of dictionaries containing the paths to
            RGB, CHM and annotations.
            labels_to_index (Dict[str, int]): dictionary associating a label name with an index.
            mean_rgb (torch.Tensor): mean used to normalize RGB images.
            std_rgb (torch.Tensor): standard deviation used to normalize RGB images.
            mean_chm (torch.Tensor): mean used to normalize CHM images.
            std_chm (torch.Tensor): standard deviation used to normalize CHM images.
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
            transform_pixel_rgb (Callable | None, optional): pixel augmentations applied to RGB images.
            Defaults to None.
            transform_pixel_chm (Callable | None, optional): pixel augmentations applied to CHM images.
            Defaults to None.
            no_data_new_value (float): value replacing NO_DATA (-9999) before normalizing images.
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

        self.proba_drop_rgb = proba_drop_rgb
        self.labels_transformation_drop_rgb = labels_transformation_drop_rgb
        self.proba_drop_chm = proba_drop_chm
        self.labels_transformation_drop_chm = labels_transformation_drop_chm

        self.transform_spatial = transform_spatial
        self.transform_pixel_rgb = transform_pixel_rgb
        self.transform_pixel_chm = transform_pixel_chm

        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_chm = mean_chm
        self.std_chm = std_chm

        self.no_data_new_value = no_data_new_value

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
        return image.astype(np.uint8)

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
        bboxes = deepcopy(self.bboxes[idx])
        labels = deepcopy(self.labels[idx])

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
        if self.transform_pixel_rgb is not None:
            transformed = self.transform_pixel_rgb(image=image_rgb)
            image_rgb = transformed["image"]

        # Apply the pixel transform the to CHM image
        if self.transform_pixel_chm is not None:
            transformed = self.transform_pixel_chm(image=image_chm)
            image_chm = transformed["image"]

        image_rgb, image_chm, bboxes, labels = self.random_chm_rgb_drop(
            image_rgb, image_chm, bboxes, labels
        )

        to_tensor = Atorch.ToTensorV2()

        sample = {
            "image_rgb": to_tensor(image=image_rgb)["image"].to(torch.float32),
            "image_chm": to_tensor(image=image_chm)["image"].to(torch.float32),
            "bboxes": torch.tensor(bboxes),
            "labels": torch.tensor(labels),
            "image_index": idx,
        }
        return sample

    def __getitem__(self, idx: int):
        sample = self.get_not_normalized(idx)
        sample["image_rgb"] = normalize(
            sample["image_rgb"], self.mean_rgb, self.std_rgb, replace_no_data=False
        )
        sample["image_chm"] = normalize(
            sample["image_chm"],
            self.mean_chm,
            self.std_chm,
            replace_no_data=True,
            no_data_new_value=self.no_data_new_value,
        )

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

    def get_chm_image(self, idx: int) -> np.ndarray:
        """Returns the CHM image corresponding to the index.

        Args:
            idx (int): index of the data.

        Returns:
            np.ndarray: CHM image.
        """
        files_paths = self.files_paths_list[idx]
        chm_path = files_paths["chm"]
        image_chm = self._read_chm_image(chm_path)
        return image_chm

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
