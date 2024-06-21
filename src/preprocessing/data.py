import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import geojson
import numpy as np
import numpy.typing as npt
import rasterio
import shapely.geometry as shp_geom
import tifffile
from osgeo import gdal
from PIL import Image

from box_cls import (
    Box,
    box_coordinates_to_pixels,
    box_crop_in_box,
    box_pixels_full_to_cropped,
    box_pixels_to_coordinates,
    intersection_ratio,
)
from dataset_constants import DatasetConst
from geojson_conversions import get_bbox_polygon
from speed_test import (
    crop_dtype_type_precision_image,
    read_memmap,
    read_numpy,
    write_memmap,
    write_numpy,
)
from utils import (
    Folders,
    ImageData,
    create_folder,
    get_coordinates_from_full_image_file_name,
    get_file_base_name,
    get_files_in_folders,
    import_tqdm,
    open_json,
    remove_all_files_but,
    remove_folder,
)


gdal.UseExceptions()
tqdm = import_tqdm()


def _old_get_image_path_from_full_annotation_path(annotations: Dict[Any, Any]) -> str:
    # Get the path to the full image
    image_path = annotations["task"]["data"]["image"].replace("/data/local-files/?d=", "/")
    image_path_tif = image_path.replace(".png", ".tif")

    # Keep the relative path
    base_path = "tree-segmentation/"
    parts = image_path_tif.split(base_path)
    image_path_tif = "../" + parts[1]
    return image_path_tif


def get_cropping_limits(
    image_path: str, tile_size: int, overlap: int
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    # Get the dimensions of the image
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Calculate the number of rows and columns needed
    # num_cols = int(np.ceil((image_width - overlap) / (tile_size - overlap)))
    # num_rows = int(np.ceil((image_height - overlap) / (tile_size - overlap)))
    num_cols = (image_width - overlap) // (tile_size - overlap)
    num_rows = (image_height - overlap) // (tile_size - overlap)

    # Get the limits of all the cropped images
    cropping_limits_x = np.array(
        [
            [
                i * (tile_size - overlap),
                (i + 1) * (tile_size - overlap) + overlap,
            ]
            for i in range(num_cols)
        ]
    )
    cropping_limits_y = np.array(
        [
            [
                j * (tile_size - overlap),
                (j + 1) * (tile_size - overlap) + overlap,
            ]
            for j in range(num_rows)
        ]
    )
    return cropping_limits_x, cropping_limits_y


class Annotation:
    def __init__(
        self,
        box: Box,
        label: str,
        id: str | None,
    ) -> None:
        self.box = box
        self.id = id
        self.label = label

    def __str__(self) -> str:
        return f"(box: {self.box}, id: {self.id}, label: {self.label})"

    def __repr__(self) -> str:
        return f"({self.box.__repr__()}, {self.id}, {self.label})"

    def __hash__(self) -> int:
        return (self.box, self.label).__hash__()


def _old_find_annots_repartition(
    cropping_limits_x: npt.NDArray[np.int_],
    cropping_limits_y: npt.NDArray[np.int_],
    annotations: Dict[Any, Any],
    visibility_threshold: float,
    dismissed_classes: List[str] | None = None,
) -> Dict[Box, List[Annotation]]:
    """Find the bounding boxes that fit in every image. The bounding boxes are taken
    if at least `visibility_threshold` of their area is within the image. The images
    are a grid defined using `cropping_limits_x` and `cropping_limits_y`. The
    bounding boxes are read from `annotations`.

    Also, the bounding boxes labeled as "Not_labeled" are used as areas that should be
    dismissed. Every image that intersects with such a bounding box is rejected.

    Args:
        cropping_limits_x (npt.NDArray[np.int_]): limits of the images on the x axis.
        cropping_limits_y (npt.NDArray[np.int_]): limits of the images on the y axis.
        annotations (Dict[Any, Any]): dictionary obtained from the json file
        containing the annotations, directly from Label Studio.
        visibility_threshold (float): threshold to select the bounding boxes.
        dismissed_classes (List[str] | None): classes to ignore. The bounding boxes of this
        class will be dismissed. Defaults to None.

    Returns:
        Dict[Box, List[Box]]: dictionary associating the bounding box of each image
        with the list of tree bounding boxes that fit in. The tree bounding boxes are
        not cropped to entirely fit in the image, and their coordinates are in the
        full image frame.
    """
    if dismissed_classes is None:
        dismissed_classes = []

    annots_repartition: Dict[Box, List[Annotation]] = {
        Box(x_limits[0], y_limits[0], x_limits[1], y_limits[1]): []
        for y_limits in cropping_limits_y
        for x_limits in cropping_limits_x
    }
    annots = annotations["result"]

    # Get the real width and height of the image
    if len(annots) == 0:
        return annots_repartition
    image_width_factor, image_height_factor = (
        annots[0]["original_width"] / 100.0,
        annots[0]["original_height"] / 100.0,
    )

    # Iterate over each bounding box
    for annot_info in tqdm(annots, leave=False, desc="Placing: Bounding boxes:"):
        annot_value = annot_info["value"]
        x_min = int(round(annot_value["x"] * image_width_factor))
        y_min = int(round(annot_value["y"] * image_height_factor))
        x_max = int(round((annot_value["x"] + annot_value["width"]) * image_width_factor))
        y_max = int(round((annot_value["y"] + annot_value["height"]) * image_height_factor))
        id = annot_info["id"]
        label = annot_value["rectanglelabels"][0]

        annot = Annotation(
            Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            id=id,
            label=label,
        )

        # If it is a box showing non labeled space, remove the image if it intersects with it
        if annot.label == "Not_labeled":
            keys_to_delete: List[Box] = []
            # Find the keys to delete
            for limit_box in annots_repartition:
                if intersection_ratio(annot.box, limit_box) > 0:
                    keys_to_delete.append(limit_box)

            # Delete the keys
            for key in keys_to_delete:
                del annots_repartition[key]
        # If it is a tree, add it to the image if it fits in
        else:
            if annot.label not in dismissed_classes:
                for limit_box in annots_repartition:
                    if intersection_ratio(annot.box, limit_box) > visibility_threshold:
                        annots_repartition[limit_box].append(deepcopy(annot))

    return annots_repartition


def find_annots_repartition(
    cropping_limits_x: npt.NDArray[np.int_],
    cropping_limits_y: npt.NDArray[np.int_],
    annotations: geojson.FeatureCollection,
    image_data: ImageData,
    visibility_threshold: float,
    dismissed_classes: List[str] | None = None,
) -> Dict[Box, List[Annotation]]:
    """Find the bounding boxes that fit in every image. The bounding boxes are taken
    if at least `visibility_threshold` of their area is within the image. The images
    are a grid defined using `cropping_limits_x` and `cropping_limits_y`. The
    bounding boxes are read from `annotations`.

    Also, the bounding boxes labeled as "Not_labeled" are used as areas that should be
    dismissed. Every image that intersects with such a bounding box is rejected.

    Args:
        cropping_limits_x (npt.NDArray[np.int_]): limits of the images on the x axis.
        cropping_limits_y (npt.NDArray[np.int_]): limits of the images on the y axis.
        annotations (geojson.FeatureCollection): features obtained from the GeoJSON file
        containing the annotations, directly from Label Studio.
        image_data (ImageData): data of the image.
        visibility_threshold (float): threshold to select the bounding boxes.
        dismissed_classes (List[str] | None): classes to ignore. The bounding boxes of this
        class will be dismissed. Defaults to None.

    Returns:
        Dict[Box, List[Box]]: dictionary associating the bounding box of each image
        with the list of tree bounding boxes that fit in. The tree bounding boxes are
        not cropped to entirely fit in the image, and their coordinates are in the
        full image frame.
    """
    if not isinstance(annotations, geojson.FeatureCollection):
        raise TypeError('"annotations" should be a FeatureCollection.')

    if dismissed_classes is None:
        dismissed_classes = []

    annots_repartition: Dict[Box, List[Annotation]] = {
        Box(x_limits[0], y_limits[0], x_limits[1], y_limits[1]): []
        for y_limits in cropping_limits_y
        for x_limits in cropping_limits_x
    }
    annots = annotations["features"]

    image_pixel_box = image_data.pixel_box
    image_coord_box = image_data.coord_box

    # Iterate over each polygon
    for annot_info in tqdm(annots, leave=False, desc="Placing: Bounding boxes:"):
        label = annot_info["properties"]["label"]
        # If it is a polygon showing non labeled space, remove the image if it intersects with it
        if label == "Not_labeled":
            removal_polygon = shp_geom.Polygon(annot_info["geometry"]["coordinates"][0])
            keys_to_delete: List[Box] = []
            # Find the keys to delete
            for limit_box in annots_repartition:
                limit_box_global = box_pixels_to_coordinates(
                    limit_box,
                    image_pixels_box=image_pixel_box,
                    image_coordinates_box=image_coord_box,
                )
                limit_polygon = shp_geom.box(*limit_box_global.as_tuple())
                if limit_polygon.intersects(removal_polygon):
                    keys_to_delete.append(limit_box)

            # Delete the keys
            for key in keys_to_delete:
                del annots_repartition[key]

        # If it is a tree, add it to the image if it fits in
        else:
            polygon = annot_info["geometry"]
            bbox = get_bbox_polygon(polygon)
            bbox_local = box_coordinates_to_pixels(
                bbox, image_coordinates_box=image_coord_box, image_pixels_box=image_pixel_box
            )
            id = annot_info["properties"]["id"]

            annot = Annotation(
                box=bbox_local,
                label=label,
                id=id,
            )
            if annot.label not in dismissed_classes:
                for limit_box in annots_repartition:
                    if intersection_ratio(annot.box, limit_box) > visibility_threshold:
                        annots_repartition[limit_box].append(deepcopy(annot))

    return annots_repartition


def crop_annots_into_limits(annots_repartition: Dict[Box, List[Annotation]]) -> None:
    """Modifies the bounding boxes repartition dictionary in place to crop
    each bounding box to the limits of its image

    Args:
        annots_repartition (Dict[Box, List[Box]]): dictionary associating the
        bounding box of each image with the list of tree bounding boxes that fit in.
    """
    for limits_box, annots in tqdm(
        annots_repartition.items(), leave=False, desc="Cropping: Bounding boxes"
    ):
        for i, annot in enumerate(annots):
            annots[i].box = box_crop_in_box(annot.box, limits_box)


def annots_coordinates_to_local(
    annots_repartition: Dict[Box, List[Annotation]],
) -> None:
    """Modifies the bounding boxes repartition dictionary in place to change the
    coordinates of each bounding box to the local coordinates of its image.

    Args:
        annots_repartition (Dict[Box, List[Box]]): dictionary associating the
        bounding box of each image with the list of tree bounding boxes that fit in.
    """
    for limits_box, annots in tqdm(
        annots_repartition.items(), leave=False, desc="To local: Bounding boxes"
    ):
        for i, annot in enumerate(annots):
            annots[i].box = box_pixels_full_to_cropped(annot.box, limits_box)


def save_annots_per_image(
    annots_repartition: Dict[Box, List[Annotation]],
    output_folder_path: str,
    full_image_path_tif: str,
    clear_if_not_empty: bool,
) -> None:
    """Saves the bounding boxes for each image as a json per image in the given folder

    Args:
        annots_repartition (Dict[Box, List[Annotation]]): dictionary associating the
        bounding box of each image with the list of tree bounding boxes that fit in.
        output_folder_path (str): path to the output folder.
        full_image_path_tif (str): the path to the full RGB image that is used.
        clear_if_not_empty (bool): whether to clear the output folder if it is not empty.
    """

    if clear_if_not_empty:
        remove_folder(output_folder_path)
    create_folder(output_folder_path)

    for image_box, annots in tqdm(
        annots_repartition.items(),
        leave=False,
        desc="Saving annotations: Bounding boxes",
    ):
        annots_dict = {
            "full_image": {
                "path": full_image_path_tif,
                "coordinates_of_cropped_image": {
                    "x_min": image_box.x_min,
                    "y_min": image_box.y_min,
                    "x_max": image_box.x_max,
                    "y_max": image_box.y_max,
                },
            },
            "width": image_box.x_max - image_box.x_min,
            "height": image_box.y_max - image_box.y_min,
            "bounding_boxes": [
                {
                    "id": annot.id,
                    "x_min": annot.box.x_min,
                    "y_min": annot.box.y_min,
                    "x_max": annot.box.x_max,
                    "y_max": annot.box.y_max,
                    "label": annot.label,
                }
                for annot in annots
            ],
        }

        annotation_output_file_name = f"{image_box.short_name()}.json"
        output_path = os.path.join(output_folder_path, annotation_output_file_name)
        with open(output_path, "w") as outfile:
            json.dump(annots_dict, outfile)


def get_image_box_from_cropped_annotations(cropped_annotations: Dict[Any, Any]) -> Box:
    """Returns the Box representing the boundaries of the image linked to the
    cropped annotations.

    Args:
        cropped_annotations (Dict[Any, Any]): _description_

    Returns:
        Box: boundaries of the image.
    """
    coords = cropped_annotations["full_image"]["coordinates_of_cropped_image"]
    return Box(
        x_min=coords["x_min"],
        y_min=coords["y_min"],
        x_max=coords["x_max"],
        y_max=coords["y_max"],
    )


def get_full_image_path_from_cropped_annotations(
    cropped_annotations: Dict[Any, Any],
) -> str:
    return cropped_annotations["full_image"]["path"]


def crop_image_from_box(image_path: str, crop_box: Box, output_path: str) -> None:
    window = (
        crop_box.x_min,
        crop_box.y_min,
        crop_box.x_max - crop_box.x_min,
        crop_box.y_max - crop_box.y_min,
    )

    gdal.Translate(output_path, image_path, srcWin=window)


def _get_cropped_image_name(cropped_annotations: Dict[Any, Any]) -> str:
    image_box = get_image_box_from_cropped_annotations(cropped_annotations)
    cropped_image_file = f"{image_box.short_name()}.tif"
    return cropped_image_file


def crop_all_rgb_and_chm_images_from_annotations_folder(
    annotations_folder_path: str,
    resolution: float,
    full_rgb_path: str,
    clear_if_not_empty: bool,
    remove_unused: bool,
):
    # Create the folders
    image_prefix = get_file_base_name(full_rgb_path)
    rgb_output_folder_path = os.path.join(Folders.CROPPED_RGB_IMAGES.value, image_prefix)
    coord1, coord2 = get_coordinates_from_full_image_file_name(image_prefix)
    chm_unfiltered_output_folder_path = os.path.join(
        Folders.CHM.value,
        f"{round(resolution*100)}cm",
        "unfiltered",
        "cropped",
        f"{coord1}_{coord2}",
    )
    chm_filtered_output_folder_path = os.path.join(
        Folders.CHM.value,
        f"{round(resolution*100)}cm",
        "filtered",
        "cropped",
        f"{coord1}_{coord2}",
    )

    if clear_if_not_empty:
        remove_folder(rgb_output_folder_path)
        remove_folder(chm_unfiltered_output_folder_path)
        remove_folder(chm_filtered_output_folder_path)

    create_folder(rgb_output_folder_path)
    create_folder(chm_unfiltered_output_folder_path)
    create_folder(chm_filtered_output_folder_path)

    full_chm_unfiltered_path = os.path.join(
        Folders.CHM.value,
        f"{round(resolution*100)}cm",
        "unfiltered",
        "full",
        f"{coord1}_{coord2}.tif",
    )
    full_chm_filtered_path = os.path.join(
        Folders.CHM.value,
        f"{round(resolution*100)}cm",
        "filtered",
        "full",
        f"{coord1}_{coord2}.tif",
    )

    files_to_keep: List[str] = []

    # Iterate over the cropped annotations
    for file_name in tqdm(
        os.listdir(annotations_folder_path),
        leave=False,
        desc="Cropping images: Cropped annotations",
    ):
        annotations_file_path = os.path.join(annotations_folder_path, file_name)
        if os.path.splitext(annotations_file_path)[1] == ".json":
            # Get the annotations
            cropped_annotations = open_json(annotations_file_path)
            output_file = _get_cropped_image_name(cropped_annotations)
            image_box = get_image_box_from_cropped_annotations(cropped_annotations)
            files_to_keep.append(output_file)

            # Create the cropped RGB image
            rgb_output_path = os.path.join(rgb_output_folder_path, output_file)
            if not os.path.exists(rgb_output_path):
                crop_image_from_box(full_rgb_path, image_box, rgb_output_path)

            # Create the cropped unfiltered CHM image
            chm_unfiltered_output_path = os.path.join(
                chm_unfiltered_output_folder_path, output_file
            )
            if not os.path.exists(chm_unfiltered_output_path):
                crop_image_from_box(full_chm_unfiltered_path, image_box, chm_unfiltered_output_path)

            # Create the cropped filtered CHM image
            chm_filtered_output_path = os.path.join(chm_filtered_output_folder_path, output_file)
            if not os.path.exists(chm_filtered_output_path):
                crop_image_from_box(full_chm_filtered_path, image_box, chm_filtered_output_path)

    if remove_unused:
        remove_all_files_but(rgb_output_folder_path, files_to_keep)
        remove_all_files_but(chm_unfiltered_output_folder_path, files_to_keep)
        remove_all_files_but(chm_filtered_output_folder_path, files_to_keep)


def crop_all_images_from_annotations_folder(
    annotations_folder_path: str,
    full_images_folders_paths: List[str],
    clear_if_not_empty: bool,
    remove_unused: bool,
) -> List[str]:
    full_images_paths = get_files_in_folders(full_images_folders_paths)

    cropped_images_folders_paths = [
        os.path.splitext(full_image_path.replace("full", "cropped"))[0]
        for full_image_path in full_images_paths
    ]

    if clear_if_not_empty:
        for folder_path in cropped_images_folders_paths:
            remove_folder(folder_path)

    for folder_path in cropped_images_folders_paths:
        create_folder(folder_path)

    files_to_keep: List[str] = []

    # Iterate over the cropped annotations
    for file_name in tqdm(
        os.listdir(annotations_folder_path),
        leave=False,
        desc="Cropping images: Cropped annotations",
    ):
        annotations_file_path = os.path.join(annotations_folder_path, file_name)
        if os.path.splitext(annotations_file_path)[1] == ".json":
            # Get the annotations
            cropped_annotations = open_json(annotations_file_path)
            output_file = _get_cropped_image_name(cropped_annotations)
            image_box = get_image_box_from_cropped_annotations(cropped_annotations)
            files_to_keep.append(output_file)

            # Create the cropped images
            for full_image_path, cropped_folder_path in zip(
                full_images_paths, cropped_images_folders_paths
            ):
                output_path = os.path.join(cropped_folder_path, output_file)
                if not os.path.exists(output_path):
                    crop_image_from_box(full_image_path, image_box, output_path)

    if remove_unused:
        for cropped_folder_path in cropped_images_folders_paths:
            remove_all_files_but(cropped_folder_path, files_to_keep)

    return cropped_images_folders_paths


def merge_tif(cropped_images_folders_paths: List[str]):
    if len(cropped_images_folders_paths) == 0:
        raise ValueError("cropped_images_folders_paths is empty.")

    folder_path_list = cropped_images_folders_paths[0].split(os.path.sep)
    output_folder_path = os.path.join(
        os.path.sep.join(folder_path_list[:-3]),
        "merged",
        os.path.sep.join(folder_path_list[-2:]),
    )
    output_memmap_folder_path = os.path.join(
        os.path.sep.join(folder_path_list[:-3]),
        "merged_memmap",
        os.path.sep.join(folder_path_list[-2:]),
    )
    output_npy_folder_path = os.path.join(
        os.path.sep.join(folder_path_list[:-3]),
        "merged_npy",
        os.path.sep.join(folder_path_list[-2:]),
    )

    create_folder(output_folder_path)
    create_folder(output_memmap_folder_path)
    create_folder(output_npy_folder_path)

    for image_name in tqdm(
        os.listdir(cropped_images_folders_paths[0]), desc="Merging TIFs", leave=False
    ):
        all_images_paths = [
            os.path.join(folder_path, image_name) for folder_path in cropped_images_folders_paths
        ]
        output_path = os.path.join(output_folder_path, image_name)

        # Load images as NumPy arrays
        with rasterio.open(all_images_paths[0]) as img:
            crs = img.crs
            transform = img.transform

        # tifffile is quicker to just open the files
        images: List[np.ndarray] = []
        for image_path in all_images_paths:
            image = tifffile.imread(image_path)
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
            images.append(image)

        # Stack images along a new axis to create a multi-channel image
        multi_channel_image = np.concatenate(images, axis=2)

        # Save the result
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=multi_channel_image.shape[0],
            width=multi_channel_image.shape[1],
            count=multi_channel_image.shape[2],
            dtype=multi_channel_image.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(multi_channel_image.shape[2]):
                dst.write(multi_channel_image[:, :, i], i + 1)

        # Save the memmap
        output_memmap_path = os.path.join(
            output_memmap_folder_path, image_name.replace(".tif", ".mmap")
        )
        multi_channel_image = crop_dtype_type_precision_image(multi_channel_image)
        write_memmap([multi_channel_image], [output_memmap_path])

        # Save the npy
        output_npy_path = os.path.join(output_npy_folder_path, image_name.replace(".tif", ".npy"))
        multi_channel_image = crop_dtype_type_precision_image(multi_channel_image)
        write_numpy([multi_channel_image], output_npy_path)


def get_channels_count(folder_path: str, chm: bool) -> int:
    data_type = DatasetConst.CHM_DATA_TYPE.value if chm else DatasetConst.RGB_DATA_TYPE.value
    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        if os.path.splitext(file)[1] == ".tif":
            image = tifffile.imread(image_path)
        elif os.path.splitext(file)[1] == ".mmap":
            image = read_memmap([image_path], [data_type])[0]
        elif os.path.splitext(file)[1] == ".npy":
            image = read_numpy(image_path)[0]
        else:
            raise Exception(f"Unsupported image format: {os.path.splitext(file)[1]}")
        if len(image.shape) == 2:
            return 1
        else:
            return image.shape[2]
    raise ValueError(f"There is no file in {folder_path}.")
