import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from math import ceil
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

import albumentations as A
import numpy as np
import tifffile
import torch
from albumentations import pytorch as Atorch
from PIL import Image
from torch.utils.data import Dataset

from plot import get_bounding_boxes
from utils import (
    Folders,
    get_coordinates_from_full_image_file_name,
    get_file_base_name,
    is_npy_file,
    is_tif_file,
    read_image,
    read_numpy,
    write_image,
)


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


def generate_chunks(
    shape: Tuple[int, ...], chunk_size: int, allow_modify_chunk_size: bool
) -> Tuple[List[Tuple[int, ...]], List[int]]:
    if allow_modify_chunk_size:
        chunk_size = ceil(shape[0] / max(round(shape[0] / chunk_size), 1))

    if len(shape) == 1:
        return [(i,) for i in range(0, shape[0], chunk_size)], [chunk_size]

    chunks = []
    for i in range(0, shape[0], chunk_size):
        prev_chunks, prev_chunk_sizes = generate_chunks(
            shape[1:], chunk_size, allow_modify_chunk_size
        )
        for rest in prev_chunks:
            chunks.append((i,) + rest)
        chunk_sizes = [chunk_size]
        chunk_sizes.extend(prev_chunk_sizes)
    return (chunks, chunk_sizes)


def generate_slices(
    shape: Tuple[int, ...], chunk_size: int, allow_modify_chunk_size: bool
) -> List[Tuple[slice, ...]]:
    chunks, chunk_sizes = generate_chunks(
        shape, chunk_size, allow_modify_chunk_size=allow_modify_chunk_size
    )
    slices = [
        tuple(
            slice(i, min(i + real_chunk_size, dim))
            for i, dim, real_chunk_size in zip(chunk_indices, shape, chunk_sizes)
        )
        for chunk_indices in chunks
    ]
    return slices


T = TypeVar("T")


def apply_chunk(
    method: Callable[[np.ndarray], T],
    array: np.ndarray,
    chunk_size: int = 1000,
    allow_modify_chunk_size: bool = True,
) -> List[T]:
    results: List[T] = []

    # Get slices
    slices = generate_slices(
        array.shape, chunk_size, allow_modify_chunk_size=allow_modify_chunk_size
    )

    # Compute method on all chunks
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(method, array[current_slice]) for current_slice in slices]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


def quick_where_chunk(
    array: np.ndarray,
    old_value: float,
    new_value: float,
    in_place: bool,
    chunk_size: int = 1000,
    allow_modify_chunk_size: bool = True,
) -> np.ndarray:
    def process_chunk_where(chunk: np.ndarray):
        chunk[:] = np.where(chunk == old_value, new_value, chunk)

    # Get slices
    slices = generate_slices(
        array.shape, chunk_size, allow_modify_chunk_size=allow_modify_chunk_size
    )

    if not in_place:
        array = np.copy(array)

    # Compute method on all chunks
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_chunk_where, array[current_slice]) for current_slice in slices
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()

    return array


def quick_merge_chunk(
    arrays: List[np.ndarray],
    axis: int = 2,
    memmap_save_path: Optional[str] = None,
    chunk_size: int = 1000,
    allow_modify_chunk_size: bool = True,
) -> np.ndarray:
    for i in range(len(arrays)):
        while arrays[i].ndim < axis:
            arrays[i] = arrays[i][..., np.newaxis]

    channels = sum([array.shape[2] for array in arrays])
    shape = (arrays[0].shape[0], arrays[0].shape[1], channels)
    dtye = arrays[0].dtype
    if memmap_save_path is not None:
        merged_array = np.lib.format.open_memmap(
            memmap_save_path, mode="w+", shape=shape, dtype=dtye
        )
    else:
        merged_array = np.empty(shape)

    def process_chunk_merge(array_slice: Tuple[slice, ...]):
        first_channel = 0
        for array in arrays:
            last_channel = first_channel + array.shape[axis]
            slice_object = tuple(list(array_slice) + [slice(first_channel, last_channel)])
            merged_array[slice_object] = array[array_slice]
            first_channel = last_channel

    # Get slices
    slices = generate_slices(
        merged_array.shape[:-1], chunk_size, allow_modify_chunk_size=allow_modify_chunk_size
    )

    # Compute method on all chunks
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk_merge, current_slice) for current_slice in slices]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()

    if memmap_save_path is not None:
        merged_array.flush()  # type: ignore
    return merged_array


def quick_normalize_chunk(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    replace_no_data: bool,
    in_place: bool,
    no_data_new_value: float = 0.0,
    chunk_size: int = 1000,
    allow_modify_chunk_size: bool = True,
) -> np.ndarray:
    # Preprocess the image
    if replace_no_data:
        quick_where_chunk(image, -9999, no_data_new_value, in_place=True)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]

    # Preprocess the mean and std to have the right shape
    channels = image.shape[2]

    def reshape(array: np.ndarray, name: str) -> np.ndarray:
        if len(array.shape) == 0:
            array = np.full((channels,), array.item())
        elif len(array.shape) == 1 and array.shape[0] == 1:
            array = np.full((channels,), array[0].item())
        elif len(array.shape) == 1 and array.shape[0] == channels:
            pass
        else:
            raise ValueError(
                f"Unsupported shape for `{name}`. It should be an array or shape [], [1] or [{channels}]"
            )
        return array.reshape(1, 1, -1)

    mean = reshape(mean, "mean").astype(dtype=image.dtype)
    std = reshape(std, "std").astype(dtype=image.dtype)

    def process_chunk_normalize(chunk: np.ndarray):
        chunk[:] = (chunk[:] - mean) / std

    # Get slices
    slices = generate_slices(
        image.shape, chunk_size, allow_modify_chunk_size=allow_modify_chunk_size
    )

    if not in_place:
        image = np.copy(image)

    # Compute method on all chunks
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_chunk_normalize, image[current_slice])
            for current_slice in slices
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()

    return image


def _compute_channels_mean_std_array(
    image: np.ndarray,
    per_channel: bool,
    replace_no_data: bool,
    no_data_new_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and the standard deviation along every channel of the image given as input.

    Args:
        image (np.ndarray): image array of shape (w, h) or (w, h, c).
        per_channel (bool): whether to compute one value for each channel or one for the whole images.
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before computations.
        no_data_new_value (float, optional): the value replacing NO_DATA (-9999) before computations.
        Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std), two arrays of shape (c,) if per_channel is
        True and (1,) otherwise. They contain respectively the mean value and the standard deviation.
    """
    if replace_no_data:
        quick_where_chunk(image, -9999, no_data_new_value, in_place=True)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]

    axis = (0, 1) if per_channel else (0, 1, 2)
    dtype = image.dtype if np.issubdtype(image.dtype, np.floating) else np.float32

    def compute_chunk_mean_std(image_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image_chunk = image_chunk.astype(dtype)
        return np.mean(image_chunk, axis=axis), np.std(image_chunk, axis=axis)

    results = apply_chunk(compute_chunk_mean_std, image)
    means = [result[0] for result in results]
    stds = [result[1] for result in results]

    mean = np.mean(means, axis=0).reshape(-1)
    std = np.mean(stds, axis=0).reshape(-1)
    return mean, std


def _compute_channels_mean_and_std_file(
    file_path: str,
    per_channel: bool,
    replace_no_data: bool,
    no_data_new_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and the standard deviation along every channel of the image given as input.

    Args:
        file_path (str): path to the image.
        per_channel (bool): whether to compute one value for each channel or one for the whole images.
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before computations.
        no_data_new_value (float, optional): the value replacing NO_DATA (-9999) before computations.
        Defaults to 0.0.

    Raises:
        InvalidPathException: if the input path is not a file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std), two arrays of shape (c,) if per_channel is
        True and (1,) otherwise. They contain respectively the mean value and the standard deviation.
    """
    if not os.path.isfile(file_path):
        raise InvalidPathException(file_path, "file")
    if is_tif_file(file_path):
        image = tifffile.imread(file_path)
    elif is_npy_file(file_path):
        image = read_numpy(file_path, mode="r+")
    else:
        image = np.array(Image.open(file_path))
    return _compute_channels_mean_std_array(image, per_channel, replace_no_data, no_data_new_value)


def compute_mean_and_std(
    array_or_file_or_folder_path: str | np.ndarray,
    per_channel: bool,
    replace_no_data: bool,
    no_data_new_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and the standard deviation along every channel of the image(s) given as input
    (either as one file or a directory containing multiple files).

    Args:
        array_or_file_or_folder_path (str | np.ndarray): path to the image or the folder of images or
        Torch tensor.
        per_channel (bool): whether to compute one value for each channel or one for the whole images.
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before computations.
        no_data_new_value (float, optional): the value replacing NO_DATA (-9999) before computations.
        Defaults to 0.0.

    Raises:
        InvalidPathException: if the input path is neither a file, nor a directory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std), two arrays of shape (c,) if per_channel is
        True and (1,) otherwise. They contain respectively the mean value and the standard deviation.
    """
    if isinstance(array_or_file_or_folder_path, np.ndarray):
        array = array_or_file_or_folder_path
        return _compute_channels_mean_std_array(
            array, per_channel, replace_no_data, no_data_new_value
        )

    if os.path.isfile(array_or_file_or_folder_path):
        file_path = array_or_file_or_folder_path
        return _compute_channels_mean_and_std_file(
            file_path, per_channel, replace_no_data, no_data_new_value
        )

    if os.path.isdir(array_or_file_or_folder_path):
        means = []
        stds = []
        folder_path = array_or_file_or_folder_path
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            mean, std = _compute_channels_mean_and_std_file(
                file_path, per_channel, replace_no_data, no_data_new_value
            )
            means.append(mean)
            stds.append(std)
        mean_mean = np.mean(means, axis=0)
        mean_std = np.mean(stds, axis=0)
        return mean_mean, mean_std

    raise Exception("The first argument should be a np.ndarray or a path to a file or a folder.")


def normalize(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    replace_no_data: bool,
    no_data_new_value: float = 0.0,
) -> np.ndarray:
    """Normalizes the image given as input.

    Args:
        image (np.ndarray): the image to normalize with c channels. Must be of shape (w, h) or (c, w, h).
        mean (np.ndarray): the mean to normalize with. Must be of shape (), (1) or (c).
        std (np.ndarray): the standard deviation to normalize with. Must be of shape (), (1) or (c).
        replace_no_data (bool): whether to replace the NO_DATA values, which are originally equal to -9999,
        before normalization.
        no_data_new_value (float, optional): the value replacing NO_DATA (-9999) before computations.
        Defaults to 0.0.

    Returns:
        np.ndarray: the normalized image.
    """
    if replace_no_data:
        quick_where_chunk(image, -9999, no_data_new_value, in_place=True)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]

    channels = image.shape[2]

    def reshape(array: np.ndarray, name: str) -> np.ndarray:
        if len(array.shape) == 0:
            array = np.full((channels,), array.item())
        elif len(array.shape) == 1 and array.shape[0] == 1:
            array = np.full((channels,), array[0].item())
        elif len(array.shape) == 1 and array.shape[0] == channels:
            pass
        else:
            raise ValueError(
                f"Unsupported shape for `{name}`. It should be an array or shape [], [1] or [{channels}]"
            )
        return array.reshape(1, 1, -1)

    mean = reshape(mean, "mean").astype(dtype=image.dtype)
    std = reshape(std, "std").astype(dtype=image.dtype)

    return (image - mean) / std


def normalize_file(
    image_path: str,
    output_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    chm: bool,
    no_data_new_value: float = 0.0,
) -> None:
    image = read_image(image_path, chm, mode="c")
    replace_no_data = chm
    normalized_image = normalize(
        image,
        mean=mean,
        std=std,
        replace_no_data=replace_no_data,
        no_data_new_value=no_data_new_value,
    )
    write_image(normalized_image, chm=chm, save_path=output_path)


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


class TreeDataset(Dataset):
    """Tree dataset."""

    def __init__(
        self,
        files_paths_list: List[Dict[str, str]],
        labels_to_index: Dict[str, int],
        labels_transformation_drop_rgb: Mapping[str, str | None],
        labels_transformation_drop_chm: Mapping[str, str | None],
        proba_drop_rgb: float = 0.0,
        proba_drop_chm: float = 0.0,
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
            labels_transformation_drop_rgb (Mapping[str, str | None]): indicates the
            labels that change to another label if the RGB image is dropped. Is mandatory if
            proba_drop_rgb > 0.
            labels_transformation_drop_chm (Mapping[str, str | None]): indicates the
            labels that change to another label if the CHM image is dropped. Is mandatory if
            proba_drop_chm > 0.
            proba_drop_rgb (float, optional): probability to drop the RGB image and replace it by a
            tensor of zeros. Default to 0.0.
            proba_drop_chm (float, optional): probability to drop the CHM image and replace it by a
            tensor of zeros. Default to 0.0.
            dismissed_classes (List[str], optional): list of classes for which the bounding boxes
            are ignored. Defaults to [].
            transform_spatial (Callable | None, optional): spatial augmentations applied to CHM and
            RGB images. Defaults to None.
            transform_pixel_rgb (Callable | None, optional): pixel augmentations applied to RGB
            images. Defaults to None.
            transform_pixel_chm (Callable | None, optional): pixel augmentations applied to CHM
            images. Defaults to None.
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

        self.no_data_new_value = no_data_new_value

        self._init_channels_count()

    def _init_channels_count(self):
        if len(self) == 0:
            self.rgb_channels = 0
            self.chm_channels = 0
            return

        files_paths = self.files_paths_list[0]
        if "rgb_cir" in files_paths.keys():
            rgb_path = files_paths["rgb_cir"]
            image_rgb = read_image(rgb_path, mode="c", chm=False)
            self.rgb_channels = image_rgb.shape[2]
            self.use_rgb = True
        else:
            assert self.proba_drop_rgb == 0.0, "If RGB is not used, proba_drop_rgb should be 0."
            self.rgb_channels = 0
            self.use_rgb = False

        if "chm" in files_paths.keys():
            chm_path = files_paths["chm"]
            image_chm = read_image(chm_path, mode="c", chm=True)
            self.chm_channels = image_chm.shape[2]
            self.use_chm = True
        else:
            assert self.proba_drop_chm == 0.0, "If CHM is not used, proba_drop_chm should be 0."
            self.chm_channels = 0
            self.use_chm = False

    def __len__(self) -> int:
        return len(self.files_paths_list)

    def random_chm_rgb_drop(
        self,
        image_rgb: np.ndarray | None,
        image_chm: np.ndarray | None,
        bboxes: List[List[float]],
        labels: List[int],
    ) -> Tuple[np.ndarray | None, np.ndarray | None, List[List[float]], List[int]]:
        """Randomly drops the RGB or the CHM image with probabilities specified during
        initialization. The bounding boxes labels are modified accordingly.

        Args:
            image_rgb (np.ndarray | None): RGB image.
            image_chm (np.ndarray | None): CHM image
            bboxes (List[List[float]]): bounding boxes.
            labels (List[int]): class labels.

        Raises:
            TypeError: if self.labels_transformation_drop_rgb is None but the methods wants to drop
            RGB.
            TypeError: if self.labels_transformation_drop_chm is None but the methods wants to drop
            CHM.

        Returns:
            Tuple[np.ndarray | None, np.ndarray | None, List[List[float]], List[int]]: (RGB image,
            CHM image, bounding boxes, class labels) after modification.
        """
        random_val = random.random()
        if random_val < self.proba_drop_rgb or not self.use_rgb:
            # Drop RGB
            if self.labels_transformation_drop_rgb is None:
                raise TypeError("self.labels_transformation_drop_rgb shouldn't be None here.")
            if image_rgb is not None:
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

        elif random_val < self.proba_drop_rgb + self.proba_drop_chm or not self.use_chm:
            # Drop CHM
            if self.labels_transformation_drop_chm is None:
                raise TypeError("self.labels_transformation_drop_chm shouldn't be None here.")
            if image_chm is not None:
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
        if self.use_rgb:
            rgb_path = files_paths["rgb_cir"]
            image_rgb = read_image(rgb_path, chm=False, mode="c")
        else:
            image_rgb = None

        if self.use_chm:
            chm_path = files_paths["chm"]
            image_chm = read_image(chm_path, chm=True, mode="c")
        else:
            image_chm = None

        # Get bboxes and labels
        bboxes = deepcopy(self.bboxes[idx])
        labels = deepcopy(self.labels[idx])

        # Apply the spatial transform to the two images, bboxes and labels
        input_data = {"bboxes": bboxes, "class_labels": labels}
        if image_rgb is not None:
            input_data["image_rgb"] = image_rgb
        if image_chm is not None:
            input_data["image_chm"] = image_chm

        if self.transform_spatial is not None:
            transformed_spatial = self.transform_spatial(**input_data)

            bboxes = transformed_spatial["bboxes"]
            labels = transformed_spatial["class_labels"]

            if image_rgb is not None:
                image_rgb = transformed_spatial["image_rgb"]
            if image_chm is not None:
                image_chm = transformed_spatial["image_chm"]

        # Apply the pixel transform the to RGB image
        if image_rgb is not None and self.transform_pixel_rgb is not None:
            transformed = self.transform_pixel_rgb(image=image_rgb)
            image_rgb = transformed["image"]

        # Apply the pixel transform the to CHM image
        if image_chm is not None and self.transform_pixel_chm is not None:
            transformed = self.transform_pixel_chm(image=image_chm)
            image_chm = transformed["image"]

        image_rgb, image_chm, bboxes, labels = self.random_chm_rgb_drop(
            image_rgb, image_chm, bboxes, labels
        )

        to_tensor = Atorch.ToTensorV2()
        image_rgb_tensor = (
            to_tensor(image=image_rgb)["image"].to(torch.float32) if image_rgb is not None else None
        )
        image_chm_tensor = (
            to_tensor(image=image_chm)["image"].to(torch.float32) if image_chm is not None else None
        )

        sample = {
            "image_rgb": image_rgb_tensor,
            "image_chm": image_chm_tensor,
            "bboxes": torch.tensor(bboxes).to(torch.float32),
            "labels": torch.tensor(labels),
            "image_index": idx,
        }
        return sample

    def __getitem__(self, idx: int):
        sample = self.get_not_normalized(idx)
        return sample

    def get_rgb_image(self, idx: int) -> np.ndarray | None:
        """Returns the RGB image corresponding to the index.

        Args:
            idx (int): index of the data.

        Returns:
            np.ndarray: RGB image.
        """
        files_paths = self.files_paths_list[idx]
        if self.use_rgb:
            rgb_path = files_paths["rgb_cir"]
            image_rgb = read_image(rgb_path, chm=False, mode="c")
        else:
            image_rgb = None
        return image_rgb

    def get_chm_image(self, idx: int) -> np.ndarray | None:
        """Returns the CHM image corresponding to the index.

        Args:
            idx (int): index of the data.

        Returns:
            np.ndarray: CHM image.
        """
        files_paths = self.files_paths_list[idx]
        if self.use_chm:
            chm_path = files_paths["chm"]
            image_chm = read_image(chm_path, chm=True, mode="c")
        else:
            image_chm = None
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
        rgb_path = files_paths["annotations"]
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
        rgb_path = files_paths["annotations"]
        full_image_name = os.path.basename(os.path.dirname(rgb_path))
        return full_image_name


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
    return sorted(all_files, key=lambda file: os.path.basename(file))


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
    rgb_cir_folder_path: str | None,
    chm_folder_path: str | None,
    annotations_folder_path: str,
    sets_ratios: Sequence[int | float],
    sets_names: List[str],
    save_path: str,
    random_seed: int | None = None,
) -> None:
    files_dict = split_files_into_lists(
        folder_path=annotations_folder_path,
        sets_ratios=sets_ratios,
        sets_names=sets_names,
        random_seed=random_seed,
    )
    all_files_dict = {}
    for set_name, set_files in files_dict.items():
        all_files_dict[set_name] = []
        for annotations_file in set_files:
            new_dict = {"annotations": annotations_file}

            if rgb_cir_folder_path is not None:
                rgb_cir_file = annotations_file.replace(
                    annotations_folder_path, rgb_cir_folder_path
                ).replace(".json", ".npy")
                new_dict["rgb_cir"] = rgb_cir_file
                new_dict["rgb_cir"] = rgb_cir_file

            if chm_folder_path is not None:
                chm_file = annotations_file.replace(
                    annotations_folder_path, chm_folder_path
                ).replace(".json", ".npy")
                new_dict["chm"] = chm_file

            all_files_dict[set_name].append(new_dict)

    with open(save_path, "w") as f:
        json.dump(all_files_dict, f)


def create_and_save_splitted_datasets_basis(
    annotations_folder_path: str,
    sets_ratios: Sequence[int | float],
    sets_names: List[str],
    save_path: str,
) -> None:
    files_dict = split_files_into_lists(
        folder_path=annotations_folder_path,
        sets_ratios=sets_ratios,
        sets_names=sets_names,
        random_seed=0,
    )
    all_files_dict = {}

    def keep_path_end(file_path: str) -> str:
        file_name = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        return os.path.join(parent_dir, file_name)

    for set_name, set_files in files_dict.items():
        all_files_dict[set_name] = list(map(keep_path_end, set_files))

    with open(save_path, "w") as f:
        json.dump(all_files_dict, f)


def create_and_save_dataset_splitted_datasets_from_basis(
    data_folder_path: str,
    use_rgb_cir: bool,
    use_chm: bool,
    data_split_files_path: str,
    parts_repartion: Dict[str, str],
    save_path: str,
) -> None:

    with open(data_split_files_path, "r") as f:
        data_split_files = json.load(f)

    # Put the base splits in the right datasets
    train_val_test_split = {"training": [], "validation": [], "test": []}
    for data_split_key, files in data_split_files.items():
        for dataset_key in train_val_test_split.keys():
            if data_split_key in parts_repartion[dataset_key]:
                train_val_test_split[dataset_key].extend(files)
                break

    # Create all the full paths
    all_files_dict = {}
    for dataset_key, files in train_val_test_split.items():
        all_files_dict[dataset_key] = []
        for annotations_file in files:
            annotations_file = os.path.join(data_folder_path, "annotations", annotations_file)
            new_dict = {"annotations": annotations_file}

            if use_rgb_cir:
                rgb_cir_file = annotations_file.replace("annotations", "rgb_cir").replace(
                    ".json", ".npy"
                )
                new_dict["rgb_cir"] = rgb_cir_file

            if use_chm is not None:
                chm_file = annotations_file.replace("annotations", "chm").replace(".json", ".npy")
                new_dict["chm"] = chm_file

            all_files_dict[dataset_key].append(new_dict)

    with open(save_path, "w") as f:
        json.dump(all_files_dict, f)


def load_tree_datasets_from_split(
    data_split_file_path: str,
    labels_to_index: Dict[str, int],
    transform_spatial_training: A.Compose | None,
    transform_pixel_rgb_training: A.Compose | None,
    transform_pixel_chm_training: A.Compose | None,
    labels_transformation_drop_rgb: Mapping[str, str | None],
    labels_transformation_drop_chm: Mapping[str, str | None],
    proba_drop_rgb: float = 0.0,
    proba_drop_chm: float = 0.0,
    dismissed_classes: List[str] = [],
    no_data_new_value: float = -5.0,
) -> Dict[str, TreeDataset]:
    with open(data_split_file_path, "r") as f:
        data_split = json.load(f)

    tree_datasets = {}
    tree_datasets["training"] = TreeDataset(
        data_split["training"],
        labels_to_index=labels_to_index,
        proba_drop_rgb=proba_drop_rgb,
        labels_transformation_drop_rgb=labels_transformation_drop_rgb,
        proba_drop_chm=proba_drop_chm,
        labels_transformation_drop_chm=labels_transformation_drop_chm,
        dismissed_classes=dismissed_classes,
        transform_spatial=transform_spatial_training,
        transform_pixel_rgb=transform_pixel_rgb_training,
        transform_pixel_chm=transform_pixel_chm_training,
        no_data_new_value=no_data_new_value,
    )

    tree_datasets["validation"] = TreeDataset(
        data_split["validation"],
        labels_to_index=labels_to_index,
        proba_drop_rgb=0.0,
        labels_transformation_drop_rgb=labels_transformation_drop_rgb,
        proba_drop_chm=0.0,
        labels_transformation_drop_chm=labels_transformation_drop_chm,
        dismissed_classes=dismissed_classes,
        transform_spatial=None,
        transform_pixel_rgb=None,
        transform_pixel_chm=None,
        no_data_new_value=no_data_new_value,
    )

    if "test" in data_split:
        test_data = data_split["test"]
    else:
        test_data = data_split["validation"]

    tree_datasets["test"] = TreeDataset(
        test_data,
        labels_to_index=labels_to_index,
        proba_drop_rgb=0.0,
        labels_transformation_drop_rgb=labels_transformation_drop_rgb,
        proba_drop_chm=0.0,
        labels_transformation_drop_chm=labels_transformation_drop_chm,
        dismissed_classes=dismissed_classes,
        transform_spatial=None,
        transform_pixel_rgb=None,
        transform_pixel_chm=None,
        no_data_new_value=no_data_new_value,
    )

    return tree_datasets
