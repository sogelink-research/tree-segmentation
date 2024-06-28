import json
import os
import random
import shutil
import string
import sys
import time
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile
import urllib3
from osgeo import gdal
from PIL import Image
from requests import get

from box_cls import Box
from dataset_constants import DatasetConst


gdal.UseExceptions()
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


def _absolute_path(relative_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


class Folders(Enum):
    DATA = _absolute_path("../data/")

    TEMP = _absolute_path("../data/temp/")

    ANNOTS = _absolute_path("../data/annotations/")
    FULL_ANNOTS = _absolute_path("../data/annotations/full/")
    CROPPED_ANNOTS = _absolute_path("../data/annotations/cropped/")

    IMAGES = _absolute_path("../data/images/")

    RGB_IMAGES = _absolute_path("../data/images/rgb/")
    FULL_RGB_IMAGES = _absolute_path("../data/images/rgb/full/")
    CROPPED_RGB_IMAGES = _absolute_path("../data/images/rgb/cropped/")

    CIR_IMAGES = _absolute_path("../data/images/cir/")
    FULL_CIR_IMAGES = _absolute_path("../data/images/cir/full/")
    CROPPED_CIR_IMAGES = _absolute_path("../data/images/cir/cropped/")

    LIDAR = _absolute_path("../data/lidar/")
    GEOTILES_LIDAR = _absolute_path("../data/lidar/geotiles/")
    GEOTILES_NO_OVERLAP_LIDAR = _absolute_path("../data/lidar/geotiles_no_overlap/")
    UNFILTERED_FULL_LIDAR = _absolute_path("../data/lidar/unfiltered/full/")
    UNFILTERED_CROPPED_LIDAR = _absolute_path("../data/lidar/unfiltered/cropped/")
    FILTERED_FULL_LIDAR = _absolute_path("../data/lidar/filtered/full/")
    FILTERED_CROPPED_LIDAR = _absolute_path("../data/lidar/filtered/cropped/")

    OTHERS_DIR = _absolute_path("../data/others")
    OUTPUT_DIR = _absolute_path("../data/others/model_output")

    MODELS_AMF_GD_YOLOV8 = _absolute_path("../models/amf_gd_yolov8")
    GD_CONFIGS = _absolute_path("../models/gd_configs")

    CHM = _absolute_path("../data/chm/")

    GOLD_YOLO = _absolute_path("../src/Efficient-Computing/Detection/Gold-YOLO")


def create_folder(folder_path: str) -> str:
    """Creates the folder if it doesn't exist, otherwise does nothing.

    Args:
        folder_path (str): path to the folder to create.

    Returns:
        str: the absolute path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.abspath(folder_path)


def create_all_folders() -> None:
    """Creates all the data folders if they don't already exist."""
    for folder in Folders:
        create_folder(folder.value)


def remove_folder(folder_path: str) -> bool:
    """Removes the folder if it exists, as well as its content.

    Args:
        folder_path (str): path to the folder to remove if it exists.

    Returns:
        bool: whether a folder was removed.
    """
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            return True
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            return False
    return False


def remove_all_files_but(folder_path: str, files_to_keep: List[str]) -> None:
    """Removes all the files in a folder except some of them.

    Args:
        folder_path (str): path to the folder where files should be removed.
        files_to_keep (List[str]): names (without folder path) of the files to keep.
    """
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file not in files_to_keep:
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))


def get_files_in_folders(folders_paths: List[str]) -> List[str]:
    """Returns a list containing the paths to all the files in all the folders given as input.

    Args:
        folders_paths (List[str]): list of paths to folders.

    Returns:
        List[str]: list of paths to files contained in the folders.
    """
    all_files = []
    for folder_path in folders_paths:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
    return all_files


def download_file(url: str, save_path: str, no_ssl: bool = False, verbose: bool = True) -> None:
    """Downloads a file from a URL and saves it at the given path.
    If a file already exists at this path, nothing is downloaded.

    Args:
        url (str): URL to download from.
        save_path (str): path to save the downloaded file.
        no_ssl (bool, optional): if True, the SSL certificate check is skipped. Defaults to False.
        verbose (bool, optional): whether to print messages about the behavior of the function. Defaults to True.
    """
    if os.path.exists(save_path):
        if verbose:
            print(f"Download skipped: there is already a file at '{os.path.abspath(save_path)}'.")
        return
    # Send a GET request to the URL
    if verbose:
        print(f"Downloading {url}...", end=" ", flush=True)
    if no_ssl:
        response = get(url, verify=False)
    else:
        response = get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        with open(save_path, "wb") as f:
            f.write(response.content)
        if verbose:
            print(f"Done.\nSaved at '{os.path.abspath(save_path)}'.")
    else:
        if verbose:
            print(f"Failed to download file from '{url}'. Status code: {response.status_code}")


def running_message(start_message: Optional[str] = None, end_message: Optional[str] = None):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Set default messages if not provided
            start_msg = (
                start_message if start_message is not None else f"Running {func.__name__}..."
            )
            start_msg = " -> " + start_msg
            end_msg = end_message if end_message is not None else "Done in {:.5f} seconds."

            # Print the start message
            print(start_msg, flush=True)

            # Execute the function
            start_time = time.time()
            result = func(*args, **kwargs)
            exec_time = time.time() - start_time

            # Print the end message
            print(end_msg.format(exec_time), flush=True)

            return result

        return wrapper

    return decorator


def get_file_base_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def open_json(json_file_path: str) -> Dict[Any, Any]:
    with open(json_file_path, "r") as file:
        return json.load(file)


def get_coordinates_from_full_image_file_name(file_name: str) -> Tuple[int, int]:
    splitted_name = file_name.split("_")
    return (int(splitted_name[-4]), int(splitted_name[-3]))


def get_coordinates_bbox_from_full_image_file_name(file_name: str) -> Box:
    x, y = get_coordinates_from_full_image_file_name(file_name)
    image_size = 1000
    return Box(x_min=x, y_min=y - image_size, x_max=x + image_size, y_max=y)


def get_pixels_bbox_from_full_image_file_name(file_name: str) -> Box:
    image_size = 12500
    return Box(x_min=0, y_min=0, x_max=image_size, y_max=image_size)


class ImageData:
    def __init__(self, image_path: str) -> None:
        self.path = image_path
        self._init_properties()
        self.base_name = get_file_base_name(self.path)
        self.coord_name = f"{round(self.coord_box.x_min)}_{round(self.coord_box.y_max)}"
        # self.coord_box = get_coordinates_bbox_from_full_image_file_name(self.base_name)
        # self.pixel_box = get_pixels_bbox_from_full_image_file_name(self.base_name)

    def _init_properties(self):
        ds = gdal.Open(self.path)

        # Get the geotransform and projection
        gt = ds.GetGeoTransform()

        # Get the extent of the TIF image
        x_min = gt[0]
        y_max = gt[3]
        x_max = x_min + gt[1] * ds.RasterXSize
        y_min = y_max + gt[5] * ds.RasterYSize

        self.coord_box = Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        self.width_pixel: int = ds.RasterXSize
        self.height_pixel: int = ds.RasterYSize
        self.pixel_box = Box(x_min=0, y_min=0, x_max=self.width_pixel, y_max=self.height_pixel)

        ds = None


def import_tqdm():
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm


def generate_random_name(length: int = 8):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def create_random_temp_folder(length: int = 8) -> str:
    def get_folder_name(random_name: str) -> str:
        return os.path.join(Folders.TEMP.value, random_name)

    temp_folder_path = get_folder_name(generate_random_name(length))
    while os.path.isdir(temp_folder_path):
        temp_folder_path = get_folder_name(generate_random_name(length))

    create_folder(temp_folder_path)

    return temp_folder_path


def is_tif_file(file_path: str) -> bool:
    return file_path.lower().endswith((".tif", ".tiff"))


def is_npy_file(file_path: str) -> bool:
    return file_path.lower().endswith(".npy")


def write_numpy(image: np.ndarray, save_path: str) -> None:
    mmapped_array = np.lib.format.open_memmap(
        save_path, mode="w+", shape=image.shape, dtype=image.dtype
    )
    mmapped_array[:] = image[:]
    mmapped_array.flush()


def write_image(image: np.ndarray, chm: bool, save_path: str) -> None:
    if is_tif_file(save_path):
        dtype_type = DatasetConst.CHM_DATA_TYPE.value if chm else DatasetConst.RGB_DATA_TYPE.value
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
        tifffile.imwrite(save_path, image)
    elif is_npy_file(save_path):
        dtype_type = np.float32
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
        write_numpy(image, save_path)
    else:
        im = Image.fromarray(image)
        im.save(save_path)


def convert_tif_to_memmappable(image_path, output_path):
    with tifffile.TiffFile(image_path) as tif:
        image = tif.asarray()
        tifffile.imwrite(output_path, image, dtype=image.dtype)


def read_tif(file_path: str, mode: str) -> np.ndarray:
    try:
        image = tifffile.memmap(file_path)
    except ValueError as e:
        if str(e) == "image data are not memory-mappable":
            mmaped_path = file_path.replace(".tif", "_memmap.tif")
            if not os.path.isfile(mmaped_path):
                print(f"Warning: {e}. Converting the image to a memory-mappable format.")
                convert_tif_to_memmappable(file_path, mmaped_path)
            image = tifffile.memmap(mmaped_path, mode=mode)  # type: ignore
        else:
            raise
    return image


def read_numpy(file_path: str, mode: str) -> np.ndarray:
    return np.lib.format.open_memmap(file_path, mode=mode)


def read_image(image_path: str, chm: bool, mode: str) -> np.ndarray:
    if is_tif_file(image_path):
        image = tifffile.imread(image_path)
        # image = read_tif(image_path, mode)
        dtype_type = DatasetConst.CHM_DATA_TYPE.value if chm else DatasetConst.RGB_DATA_TYPE.value
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
    elif is_npy_file(image_path):
        image = read_numpy(image_path, mode)
        dtype_type = np.float32
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
    else:
        image = np.array(Image.open(image_path))

    if len(image.shape) == 2:
        image = image[..., np.newaxis]

    return image


def get_sup_dtype_type(dtype_type: type):
    handled_dtype_types = [np.floating, np.unsignedinteger, np.signedinteger]
    for handled_dtype_type in handled_dtype_types:
        if np.issubdtype(dtype_type, handled_dtype_type):
            return handled_dtype_type
    raise TypeError(f"Type {dtype_type} is not handled.")


def are_same_numpy_dtype_type_nature(dtype_type1: type, dtype_type2: type):
    if get_sup_dtype_type(dtype_type1) == get_sup_dtype_type(dtype_type2):
        return True
    else:
        return False


def smallest_numpy_dtype_type(dtype_type1: type, dtype_type2: type) -> type:
    if not are_same_numpy_dtype_type_nature(dtype_type1, dtype_type2):
        raise TypeError(f"{dtype_type1} and {dtype_type2} are not the same kind of dtype.")
    if dtype_type1(0).itemsize > dtype_type2(0).itemsize:
        return dtype_type2
    else:
        return dtype_type1


def crop_dtype_type_precision(array: np.ndarray, dtype_type: type):
    smallest_dtype_type = smallest_numpy_dtype_type(array.dtype.type, dtype_type)
    return array.astype(smallest_dtype_type)


def crop_dtype_type_precision_image(image: np.ndarray):
    if get_sup_dtype_type(image.dtype.type) == np.floating:
        return crop_dtype_type_precision(image, np.float32)
    if get_sup_dtype_type(image.dtype.type) == np.unsignedinteger:
        return crop_dtype_type_precision(image, np.uint8)
    if get_sup_dtype_type(image.dtype.type) == np.signedinteger:
        return crop_dtype_type_precision(image, np.int16)
    raise TypeError(f"Type {image.dtype} is not handled.")
