import os
import zipfile
from typing import Tuple

from osgeo import ogr
from shapely.geometry import box
from shapely.wkt import dumps

from data_processing import get_coordinates_bbox_from_full_image_file_name
from utils import Folders, download_file, get_file_base_name


def _get_rgb_download_url(image_name_with_ext: str) -> str:
    """Returns the URL that can be used to download the given file from https://www.beeldmateriaal.nl/data-room.

    Args:
        image_name_with_ext (str): the name of the file with its TIFF extension.

    Returns:
        str: the URL to download the file.
    """
    block, parcel = _get_block_perceel_from_image_file_name(image_name_with_ext)
    parcel_int = parcel[-1]

    return f"https://ns_hwh.fundaments.nl/hwh-ortho/2023/Ortho/{parcel_int}/{block}/beelden_tif_tegels/{image_name_with_ext}"


def download_rgb_image(file_name: str, verbose: bool = True) -> None:
    """Downloads the RGB image corresponding to the given image.

    Args:
        file_name (str): the name or path of the image to download.
        verbose (bool, optional): whether to print messages about the behavior of the function.. Defaults to True.
    """
    image_name = f"{get_file_base_name(file_name)}.tif"
    image_url = _get_rgb_download_url(image_name)
    print(image_url)
    image_path = os.path.join(Folders.FULL_IMAGES.value, image_name)
    download_file(image_url, image_path, verbose)


def download_rgb_names_shapefile(verbose: bool = True) -> str:
    """Downloads the Shapefile describing the organization of the RGB images.
    The download is skipped if the file already exists.

    Args:
        verbose (bool, optional): whether to print messages about the behavior of the function. Defaults to True.

    Returns:
        str: the path to the Shapefile.
    """
    # Download the Shapefile
    shapefile_path = os.path.join(Folders.IMAGES.value, "2023_HRL_blokindeling.zip")
    if not os.path.exists(shapefile_path):
        download_file(
            "https://ns_hwh.fundaments.nl/hwh-stereo/AUX/BOUNDS/2023_HRL_blokindeling.zip",
            shapefile_path,
            no_ssl=True,
            verbose=verbose,
        )
        with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
            zip_ref.extractall(
                os.path.join(os.path.dirname(shapefile_path), "2023_HRL_blokindeling")
            )

    return os.path.join(Folders.IMAGES.value, "2023_HRL_blokindeling", "Blokindeling_2023.shp")


def _get_block_perceel_from_image_file_name(image_file_name: str) -> Tuple[str, str]:
    bbox = get_coordinates_bbox_from_full_image_file_name(image_file_name)

    # Create a box geometry representing the extent of the TIF image
    bbox_shapely = box(
        bbox.x_min,
        bbox.y_min,
        bbox.x_max,
        bbox.y_max,
    )

    # Convert the Shapely geometry to an ogr.Geometry object
    bbox_ogr = ogr.CreateGeometryFromWkt(dumps(bbox_shapely))

    # Path to your shapefile
    shapefile_path = download_rgb_names_shapefile()

    # Open the shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = driver.Open(shapefile_path, 0)
    layer = shp_ds.GetLayer()

    # Get the intersection between TIF image and Shapefile
    block = None
    parcel = None
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom.Intersects(bbox_ogr):
            block = feature.GetField("BLOK")
            parcel = feature.GetField("PERCEEL")
            break

    # Close the data source
    shp_ds.Destroy()

    if block is None or parcel is None:
        raise Exception("No corresponding block and parcel was found for this image.")

    return block, parcel
