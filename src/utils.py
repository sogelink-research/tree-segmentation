import json
import os
from enum import Enum
from time import time
from typing import Any, Dict

from requests import get


def _absolute_path(relative_path: str) -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), relative_path)
    )


class Folders(Enum):
    FULL_ANNOTS = _absolute_path("../data/annotations/full/")
    CROPPED_ANNOTS = _absolute_path("../data/annotations/cropped/")

    FULL_IMAGES = _absolute_path("../data/images/full/")
    CROPPED_IMAGES = _absolute_path("../data/images/cropped/")

    GEOTILES_LIDAR = _absolute_path("../data/lidar/geotiles/")
    GEOTILES_NO_OVERLAP_LIDAR = _absolute_path(
        "../data/lidar/geotiles_no_overlap/"
    )
    FULL_LIDAR = _absolute_path("../data/lidar/full/")
    CROPPED_LIDAR = _absolute_path("../data/lidar/cropped/")

    FULL_CHM = _absolute_path("../data/chm/full/")
    CROPPED_CHM = _absolute_path("../data/chm/cropped/")


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


def download_file(url: str, save_path: str) -> None:
    # Send a GET request to the URL
    response = get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        print(f"Downloading {url}...", end=" ", flush=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Done.\nSaved at '{os.path.abspath(save_path)}'.")
    else:
        print(
            f"Failed to download file from '{url}'. Status code: {response.status_code}"
        )


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
