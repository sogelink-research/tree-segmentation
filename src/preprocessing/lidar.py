import json
import os
import zipfile
from typing import List, Tuple

import laspy
import pdal
from osgeo import ogr
from shapely.geometry import box
from shapely.wkt import dumps

from box_cls import Box
from utils import Folders, ImageData, download_file


def merge_crop_las(input_las_list: List[str], output_las: str, crop_box: Box) -> None:
    bounds = f"([{crop_box.x_min},{crop_box.x_max}],[{crop_box.y_min},{crop_box.y_max}])"
    pipeline_list = []
    for index, input_las in enumerate(input_las_list):
        pipeline_list.append({"type": "readers.las", "filename": input_las, "tag": f"A{index}"})
    pipeline_list.extend(
        [
            {
                "type": "filters.merge",
                "inputs": [f"A{index}" for index in range(len(input_las_list))],
            },
            {"type": "filters.crop", "bounds": bounds},
            {"type": "writers.las", "filename": output_las},
        ]
    )
    pipeline = pdal.Pipeline(json.dumps(pipeline_list))
    pipeline.execute()


def crop_las(input_las: str, output_las: str, crop_box: Box) -> None:
    bounds = f"([{crop_box.x_min},{crop_box.x_max}],[{crop_box.y_min},{crop_box.y_max}])"
    pipeline_list = [
        {
            "type": "readers.las",
            "filename": input_las,
        },
        {"type": "filters.crop", "bounds": bounds},
        {"type": "writers.las", "filename": output_las},
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_list))
    pipeline.execute()


def remove_las_overlap_from_geotiles(input_las: str, output_las: str, overlap: int) -> None:
    with laspy.open(input_las, mode="r") as las_file:
        # Get the bounding box information from the header
        x_min = las_file.header.min[0] + overlap
        x_max = las_file.header.max[0] - overlap
        y_min = las_file.header.min[1] + overlap
        y_max = las_file.header.max[1] - overlap

    crop_box = Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    crop_las(input_las, output_las, crop_box)


def filter_classification_las(input_las: str, output_las: str) -> None:
    """Filters the given LAS/LAZ file to keep only points classified as one of:
    - unclassified,
    - ground,
    - vegetation.

    Args:
        input_las (str): Path to the LAS/LAZ file to filter.
        output_las (str): Path to the filtered LAS/LAZ.
    """
    pipeline_list = [
        {
            "type": "readers.las",
            "filename": input_las,
        },
        {
            "type": "filters.range",
            "limits": "Classification[1:5]",  # Keep only unclassified, ground and vegetation
        },
        {"type": "writers.las", "filename": output_las},
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_list))
    pipeline.execute()


def download_lidar_names_shapefile(verbose: bool = True) -> str:
    """Downloads the Shapefile describing the organization of the LiDAR GeoTiles.
    The download is skipped if the file already exists.

    Args:
        verbose (bool, optional): whether to print messages about the behavior of the function. Defaults to True.

    Returns:
        str: the path to the Shapefile.
    """
    # Download the Shapefile
    shapefile_path = os.path.join(Folders.LIDAR.value, "TOP-AHN_subunit_compat.zip")
    if not os.path.exists(shapefile_path):
        download_file(
            "https://static.fwrite.org/2022/01/TOP-AHN_subunit_compat.zip",
            shapefile_path,
            verbose,
        )
        with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
            zip_ref.extractall(
                os.path.join(os.path.dirname(shapefile_path), "TOP-AHN_subunit_compat")
            )

    return os.path.join(Folders.LIDAR.value, "TOP-AHN_subunit_compat", "TOP-AHN_subunit_compat.shp")


def get_lidar_files_from_image(
    image_data: ImageData, shapefile_path: str, overlap: int
) -> List[str]:
    """Returns the list of the names of the LiDAR files that are required to cover
    the area of the whole input image.

    Args:
        image_data (ImageData): The data of the image.
        shapefile_path (str): The path to the Shapefile describing the organization of the LiDAR GeoTiles.
        overlap (int): The overlap between the LiDAR files.

    Returns:
        List[str]: The names of the LiDAR files required to cover this image.
    """
    # Create a box geometry representing the extent of the TIF image
    bbox = box(
        image_data.coord_box.x_min + overlap,
        image_data.coord_box.y_min + overlap,
        image_data.coord_box.x_max - overlap,
        image_data.coord_box.y_max - overlap,
    )

    # Convert the Shapely geometry to an ogr.Geometry object
    bbox_ogr = ogr.CreateGeometryFromWkt(dumps(bbox))

    # Open the Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = driver.Open(shapefile_path, 0)
    layer = shp_ds.GetLayer()

    # Get the intersection between TIF image and Shapefile
    intersection_file_names = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom.Intersects(bbox_ogr):
            intersection_file_names.append(feature.GetField("AHN"))

    # Close the Shapefile
    shp_ds = None

    return intersection_file_names


def _get_geotiles_url(file_name: str) -> str:
    """Returns the URL that can be used to download the given file from GeoTiles.
    See https://geotiles.citg.tudelft.nl.

    Args:
        file_name (str): The name of the file from GeoTiles.

    Returns:
        str: The URL to download the file from GeoTiles.
    """
    return f"https://geotiles.citg.tudelft.nl/AHN4_T/{file_name}.LAZ"


def download_and_remove_overlap_geotiles(lidar_file_names: List[str], overlap: int) -> List[str]:
    """Downloads the LiDAR files from GeoTiles and removes the overlap.

    Args:
        lidar_file_names (List[str]): The names of the LiDAR files to download.
        overlap (int): The overlap between the LiDAR files.

    Returns:
        List[str]: The paths to the files without overlap.
    """

    lidar_file_paths = []
    for file_name in lidar_file_names:
        url = _get_geotiles_url(file_name)
        # Create the paths
        geotiles_with_overlap_path = os.path.join(Folders.GEOTILES_LIDAR.value, f"{file_name}.LAZ")
        geotiles_without_overlap_path = os.path.join(
            Folders.GEOTILES_NO_OVERLAP_LIDAR.value, f"{file_name}.LAZ"
        )
        lidar_file_paths.append(geotiles_without_overlap_path)
        # Download the point clouds
        download_file(url, geotiles_with_overlap_path)
        # Remove the overlap from the point clouds
        if not os.path.exists(geotiles_without_overlap_path):
            remove_las_overlap_from_geotiles(
                geotiles_with_overlap_path, geotiles_without_overlap_path, overlap
            )
    return lidar_file_paths


def create_full_lidar(lidar_file_paths: List[str], image_data: ImageData) -> str:
    """Merges and crops the given LiDAR files to the area of the given image.

    Args:
        lidar_file_paths (List[str]): the list of LiDAR files to cover the whole area of the image.
        These point clouds should have no overlap.
        image_data (ImageData): the data of the image.

    Returns:
        str: the path to the new LiDAR point cloud corresponding to the image.
    """
    # Crop the point clouds into the area of the full image
    full_lidar_path = os.path.join(
        Folders.UNFILTERED_FULL_LIDAR.value, f"{image_data.coord_name}.laz"
    )
    if not os.path.exists(full_lidar_path):
        merge_crop_las(lidar_file_paths, full_lidar_path, image_data.coord_box)
    return full_lidar_path


def filter_full_lidar(image_data: ImageData):
    """Filters the full LiDAR file to remove buildings.

    Args:
        image_data (ImageData): the data of the image.

    Returns:
        str: the path to the new filtered LiDAR point cloud corresponding to the image.
    """
    full_lidar_path = os.path.join(
        Folders.UNFILTERED_FULL_LIDAR.value, f"{image_data.coord_name}.laz"
    )
    # Filter the full point cloud to remove buildings
    full_lidar_filtered_path = os.path.join(
        Folders.FILTERED_FULL_LIDAR.value, f"{image_data.coord_name}.laz"
    )
    if not os.path.exists(full_lidar_filtered_path):
        filter_classification_las(full_lidar_path, full_lidar_filtered_path)
    return full_lidar_filtered_path
