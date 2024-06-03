import json
import os
from enum import Enum
from time import time
from typing import Any, Dict, Tuple

from requests import get
from osgeo import gdal

from box import Box


gdal.UseExceptions()


def _absolute_path(relative_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


class Folders(Enum):
    ANNOTS = _absolute_path("../data/annotations/")
    FULL_ANNOTS = _absolute_path("../data/annotations/full/")
    CROPPED_ANNOTS = _absolute_path("../data/annotations/cropped/")

    IMAGES = _absolute_path("../data/images/")
    FULL_IMAGES = _absolute_path("../data/images/full/")
    CROPPED_IMAGES = _absolute_path("../data/images/cropped/")

    LIDAR = _absolute_path("../data/lidar/")
    GEOTILES_LIDAR = _absolute_path("../data/lidar/geotiles/")
    GEOTILES_NO_OVERLAP_LIDAR = _absolute_path("../data/lidar/geotiles_no_overlap/")
    UNFILTERED_FULL_LIDAR = _absolute_path("../data/lidar/unfiltered/full/")
    UNFILTERED_CROPPED_LIDAR = _absolute_path("../data/lidar/unfiltered/cropped/")
    FILTERED_FULL_LIDAR = _absolute_path("../data/lidar/filtered/full/")
    FILTERED_CROPPED_LIDAR = _absolute_path("../data/lidar/filtered/cropped/")

    OUTPUT_DIR = _absolute_path("../data/others/model_output")

    CHM = _absolute_path("../data/chm/")


def create_folder(folder_path: str) -> str:
    """Creates the folder if it doesn't exist, otherwise does nothing.

    Args:
        folder_path (str): path of the folder to create.

    Returns:
        str: the absolute path of the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.abspath(folder_path)


def create_all_folders() -> None:
    """Creates all the data folders if they don't already exist."""
    for folder in Folders:
        create_folder(folder.value)


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
    if no_ssl:
        response = get(url, verify=False)
    else:
        response = get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        if verbose:
            print(f"Downloading {url}...", end=" ", flush=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        if verbose:
            print(f"Done.\nSaved at '{os.path.abspath(save_path)}'.")
    else:
        if verbose:
            print(f"Failed to download file from '{url}'. Status code: {response.status_code}")


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Execution of {func.__name__}({args})...")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Done in {round(execution_time, 3)} seconds")
        return result

    return wrapper


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
        # self._init_properties()
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
