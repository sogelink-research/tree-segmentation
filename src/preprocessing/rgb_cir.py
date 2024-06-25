import concurrent.futures
import io
import itertools
import math
import os
import zipfile
from typing import Any, Dict, List, Tuple

import geojson
import httpx
import numpy as np
import rasterio
import tifffile
from osgeo import gdal, ogr, osr
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.wkt import dumps

from box_cls import Box, BoxInt, box_pixels_to_coordinates
from geojson_conversions import get_bbox_polygon
from utils import (
    Folders,
    create_folder,
    download_file,
    get_coordinates_bbox_from_full_image_file_name,
    get_file_base_name,
    import_tqdm,
    measure_execution_time,
)


gdal.UseExceptions()
tqdm = import_tqdm()


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


def get_rgb_image_path_from_file_name(file_name: str) -> str:
    """Return the path to the RGB image corresponding to the given image.

    Args:
        file_name (str): the name or path of the image to download.

    Returns:
        str: the path to the RGB image file.
    """
    image_name = f"{get_file_base_name(file_name)}.tif"
    image_path = os.path.join(Folders.FULL_RGB_IMAGES.value, image_name)
    return image_path


def download_rgb_image_from_file_name(file_name: str, verbose: bool = True) -> str:
    """Downloads the RGB image corresponding to the given image.

    Args:
        file_name (str): the name or path of the image to download.
        verbose (bool, optional): whether to print messages about the behavior of the function.. Defaults to True.

    Returns:
        str: the path to the downloaded file.
    """
    image_name = f"{get_file_base_name(file_name)}.tif"
    image_url = _get_rgb_download_url(image_name)
    image_path = get_rgb_image_path_from_file_name(file_name)
    download_file(image_url, image_path, verbose)
    return image_path


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


def _get_block_perceel_from_polygon(
    polygon: geojson.Polygon,
) -> Tuple[List[str], List[str]]:
    polygon_ogr = ogr.CreateGeometryFromWkt(Polygon(polygon["coordinates"][0]).wkt)

    # Path to your shapefile
    shapefile_path = download_rgb_names_shapefile()

    # Open the shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = driver.Open(shapefile_path, 0)
    layer = shp_ds.GetLayer()

    # Get the intersection between TIF image and Shapefile
    blocks = []
    parcels = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom.Intersects(polygon_ogr):
            blocks.append(feature.GetField("BLOK"))
            parcels.append(feature.GetField("PERCEEL"))
            break

    # Close the data source
    shp_ds.Destroy()

    if len(blocks) == 0:
        raise Exception("No corresponding block and parcel was found for this image.")

    return blocks, parcels


def _get_rgb_name_from_box(box: Box) -> str:
    return f"2023_{round(box.x_min)}_{round(box.y_max)}_RGB_hrl.tif"


def get_rgb_images_file_names_from_polygon(polygon: geojson.Polygon) -> List[str]:
    """Returns the file names corresponding to the given GeoJSON Polygon.

    Args:
        polygon (geojson.Polygon): the polygon defining the bounding box of the area of interest.

    Returns:
        List[str]: the names of the images.
    """
    bbox = get_bbox_polygon(polygon)
    step = 1000
    images_file_names: List[str] = []
    for x in np.arange(bbox.x_min - bbox.x_min % step, bbox.x_max - bbox.x_max % step, step):
        for y in np.arange(bbox.y_min, bbox.y_max, step):
            file_name = _get_rgb_name_from_box(
                Box(round(x), round(y), round(x) + step, round(y) + step)
            )
            images_file_names.append(file_name)

    return images_file_names


def get_rgb_images_paths_from_polygon(polygon: geojson.Polygon) -> List[str]:
    """Returns the file paths corresponding to the given GeoJSON Polygon.

    Args:
        polygon (geojson.Polygon): the polygon defining the bounding box of the area of interest.

    Returns:
        List[str]: the paths to the images.
    """
    images_file_names = get_rgb_images_file_names_from_polygon(polygon)
    images_paths: List[str] = list(map(get_rgb_image_path_from_file_name, images_file_names))
    return images_paths


def download_rgb_image_from_polygon(polygon: geojson.Polygon, verbose: bool = True) -> List[str]:
    """Downloads the RGB image corresponding to the given GeoJSON Polygon.

    Args:
        polygon (geojson.Polygon): the polygon defining the bounding box of the area of interest.
        verbose (bool, optional): whether to print messages about the behavior of the function.. Defaults to True.

    Returns:
        List[str]: the paths to the images.
    """
    images_file_names = get_rgb_images_file_names_from_polygon(polygon)
    images_paths: List[str] = []
    for file_name in images_file_names:
        images_paths.append(download_rgb_image_from_file_name(file_name, verbose))
    return images_paths


@measure_execution_time
def download_cir(
    image_coords_box: Box,
    resolution: float,
    save_path: str,
    skip_if_file_exists: bool,
    verbose: bool = False,
    **kwargs,
) -> None:
    if skip_if_file_exists and os.path.isfile(save_path):
        return
    url = "https://services.arcgisonline.nl/arcgis/rest/services/Luchtfoto/Luchtfoto_CIR/MapServer/export"
    SESSION = httpx.Client()

    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(28992)

    WKT_28992 = web_mercator.ExportToWkt()
    BASE_MAP_SCALE = 3779.52

    download_resolution = resolution
    coord_size_x = image_coords_box.x_max - image_coords_box.x_min
    coord_size_y = image_coords_box.y_max - image_coords_box.y_min
    total_pixels_x = coord_size_x / download_resolution
    total_pixels_y = coord_size_y / download_resolution
    MAX_PIXELS_PER_DOWNLOAD = 4096
    pixels_per_download = min(
        math.ceil(total_pixels_x), math.ceil(total_pixels_y), MAX_PIXELS_PER_DOWNLOAD
    )  # Pixel size of each individual image that is downloaded

    image_pixels_box = BoxInt(
        x_min=0,
        y_min=0,
        x_max=math.ceil(total_pixels_x),
        y_max=math.ceil(total_pixels_y),
    )

    map_scale = BASE_MAP_SCALE * download_resolution

    def get_tile(url: str, params: Dict[Any, Any]) -> bytes | None:
        retry = 3
        while 1:
            try:
                r = SESSION.get(url, params=params, timeout=60)
                break
            except Exception:
                retry -= 1
                if not retry:
                    raise
        if r.status_code == 404:
            return None
        elif not r.content:
            return None
        r.raise_for_status()
        return r.content

    def is_empty(im: Image.Image) -> bool:
        extrema = im.convert("L").getextrema()
        return extrema == (0, 0)

    def paste_tile(
        full_image: Image.Image | None,
        tile: bytes | None,
        pixels_box: BoxInt,
        image_pixels_box: BoxInt,
    ) -> Image.Image | None:
        if tile is None:
            return full_image
        im = Image.open(io.BytesIO(tile))
        mode = "RGB" if im.mode == "RGB" else "RGBA"
        if full_image is None:
            new_image = Image.new(
                mode,
                (
                    image_pixels_box.x_max - image_pixels_box.x_min,
                    image_pixels_box.y_max - image_pixels_box.y_min,
                ),
            )
        else:
            new_image = full_image

        xy0 = (pixels_box.x_min, pixels_box.y_min)
        if mode == "RGB":
            new_image.paste(im, xy0)
        else:
            if im.mode != mode:
                im = im.convert(mode)
            if not is_empty(im):
                new_image.paste(im, xy0)
        im.close()
        return new_image

    def finish_picture(full_image: Image.Image, image_pixels_box: BoxInt) -> Image.Image:
        crop = (
            image_pixels_box.x_min,
            image_pixels_box.y_min,
            image_pixels_box.x_max,
            image_pixels_box.y_max,
        )
        new_image = full_image.crop(crop)
        if new_image.mode == "RGBA" and new_image.getextrema()[3] == (255, 255):  # type: ignore
            new_image = new_image.convert("RGB")
        full_image.close()
        return new_image

    iter_pixels_corners = list(
        itertools.product(
            range(image_pixels_box.x_min, image_pixels_box.x_max, pixels_per_download),
            range(image_pixels_box.y_min, image_pixels_box.y_max, pixels_per_download),
        )
    )
    iter_pixels_corners_all = list(
        map(
            lambda t: (
                t[0],
                t[1],
                t[0] + pixels_per_download,
                t[1] + pixels_per_download,
            ),
            iter_pixels_corners,
        )
    )
    iter_pixels_boxes = list(map(BoxInt, *zip(*iter_pixels_corners_all)))

    total_iters = len(iter_pixels_boxes)
    futures: List[concurrent.futures.Future[bytes | None]] = []
    with concurrent.futures.ThreadPoolExecutor(5) as executor:
        for pixels_box in iter_pixels_boxes:
            coords_box = box_pixels_to_coordinates(
                pixels_box,
                image_pixels_box=image_pixels_box,
                image_coordinates_box=image_coords_box,
            )
            current_bbox = (
                f"{coords_box.x_min},{coords_box.y_min},{coords_box.x_max},{coords_box.y_max}"
            )
            params = {
                "f": "image",  # Output format
                "bboxSR": "EPSG:28992",  # Coordinate system
                "bbox": current_bbox,
                "size": f"{pixels_per_download},{pixels_per_download}",  # Width and height of the output image
                "mapScale": map_scale,
                "formatOptions": "TIFF",
                "transparent": True,
            }
            futures.append(executor.submit(get_tile, url, params))
        full_image = None
        for k, (fut, pixels_box) in enumerate(zip(futures, iter_pixels_boxes), 1):
            full_image = paste_tile(full_image, fut.result(), pixels_box, image_pixels_box)
            if verbose:
                print(f"Downloaded image {str(k).zfill(len(str(total_iters)))}/{total_iters}")
    if full_image is None:
        print("Nothing was found at these coordinates.")
        return

    if verbose:
        print("Saving GeoTIFF. Please wait...")
    final_image = finish_picture(full_image, image_pixels_box)

    # real_total_pixels_x = math.ceil(coord_size_x / resolution)
    # real_total_pixels_y = math.ceil(coord_size_y / resolution)

    # img = img.resize((real_total_pixels_x, real_total_pixels_y))

    image_bands = len(final_image.getbands())
    driver = gdal.GetDriverByName("GTiff")

    if "options" not in kwargs:
        kwargs["options"] = [
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "ZLEVEL=9",
            "TILED=YES",
        ]

    try:
        g_tiff = driver.Create(
            save_path,
            final_image.size[0],
            final_image.size[1],
            image_bands,
            gdal.GDT_Byte,
            **kwargs,
        )
        g_tiff.SetGeoTransform(
            (
                image_coords_box.x_min,
                resolution,
                0,
                image_coords_box.y_max,
                0,
                -resolution,
            )
        )
        g_tiff.SetProjection(WKT_28992)
        for band in range(image_bands):
            array = np.array(final_image.getdata(band), dtype="u8")
            array = array.reshape((final_image.size[1], final_image.size[0]))
            band = g_tiff.GetRasterBand(band + 1)
            band.WriteArray(array)
        g_tiff.FlushCache()
    except Exception as e:
        raise e
    finally:
        g_tiff = None

    if verbose:
        print(f"Image saved to {save_path}")
