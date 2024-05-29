import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from osgeo import gdal
from PIL import Image
from tqdm.notebook import tqdm
from utils import Folders, create_folder, get_file_base_name, open_json

gdal.UseExceptions()


def get_image_path_from_full_annotation_path(annotations: Dict[Any, Any]) -> str:
    # Get the path to the full image
    image_path = annotations["task"]["data"]["image"].replace(
        "/data/local-files/?d=", "/"
    )
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
    num_cols = int(np.ceil((image_width - overlap) / (tile_size - overlap)))
    num_rows = int(np.ceil((image_height - overlap) / (tile_size - overlap)))

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


class Box:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)

    def __str__(self) -> str:
        return f"(x_min: {self.x_min}, y_min: {self.y_min}, x_max: {self.x_max}, y_max: {self.y_max})"

    def __repr__(self) -> str:
        return f"({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"

    def __hash__(self) -> int:
        return (self.x_min, self.y_min, self.x_max, self.y_max).__hash__()

    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def short_name(self) -> str:
        return f"{round(self.x_min)}_{round(self.y_min)}_{round(self.x_max)}_{round(self.y_max)}"

    def as_list(self) -> List[float]:
        """Returns the Box as a list.

        Returns:
            List[float]: [x_min, y_min, x_max, y_max]
        """
        return [self.x_min, self.y_min, self.x_max, self.y_max]


def intersection(box1: Box, box2: Box) -> float:
    if ((box1.x_min <= box2.x_max) and (box1.x_max >= box2.x_min)) and (
        (box1.y_min <= box2.y_max) and (box1.y_max >= box2.y_min)
    ):
        # Calculate the intersection coordinates
        inter_x_min = max(box1.x_min, box2.x_min)
        inter_y_min = max(box1.y_min, box2.y_min)
        inter_x_max = min(box1.x_max, box2.x_max)
        inter_y_max = min(box1.y_max, box2.y_max)

        # Compute the area of the intersection
        area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        return area
    else:
        # No intersection, return 0
        return 0


def intersection_ratio(annot: Box, limits: Box) -> float:
    """Computes (Area of the intersection of annot and limits) / (Area of annot)

    Args:
        annot (Box): the bounding box
        limits (Box): the limits in which we want to put it

    Returns:
        float: (Area of the intersection of annot and limits) / (Area of annot)
    """
    if annot.area() == 0:
        raise ValueError("The area of annot is 0.")
    return intersection(annot, limits) / annot.area()


def crop_box_in_box(to_crop: Box, limits: Box) -> Box:
    return Box(
        x_min=max(to_crop.x_min, limits.x_min),
        y_min=max(to_crop.y_min, limits.y_min),
        x_max=min(to_crop.x_max, limits.x_max),
        y_max=min(to_crop.y_max, limits.y_max),
    )


def compute_box_local_coord(to_modify: Box, frame: Box):
    return Box(
        x_min=to_modify.x_min - frame.x_min,
        y_min=to_modify.y_min - frame.y_min,
        x_max=to_modify.x_max - frame.x_min,
        y_max=to_modify.y_max - frame.y_min,
    )


class Annotation:
    def __init__(
        self,
        box: Box,
        id: str,
        label: str,
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


def find_annots_repartition(
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
        x_max = int(
            round((annot_value["x"] + annot_value["width"]) * image_width_factor)
        )
        y_max = int(
            round((annot_value["y"] + annot_value["height"]) * image_height_factor)
        )
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
            annots[i].box = crop_box_in_box(annot.box, limits_box)


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
            annots[i].box = compute_box_local_coord(annot.box, limits_box)


def save_annots_per_image(
    annots_repartition: Dict[Box, List[Annotation]],
    output_folder_path: str,
    full_image_path_tif: str,
) -> None:
    """Saves the bounding boxes for each image as a json per image in the given folder

    Args:
        annots_repartition (Dict[Box, List[Annotation]]): dictionary associating the
        bounding box of each image with the list of tree bounding boxes that fit in.
        output_folder_path (str): path to the output folder.
    """

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


def get_coordinates_from_full_image_file_name(file_name: str) -> Tuple[int, int]:
    splitted_name = file_name.split("_")
    return (int(splitted_name[-4]), int(splitted_name[-3]))


def get_bbox_from_full_image_file_name(file_name: str) -> Box:
    x, y = get_coordinates_from_full_image_file_name(file_name)
    image_size = 10000
    return Box(x_min=x, y_min=y - image_size, x_max=x + image_size, y_max=y)


def crop_all_rgb_and_chm_images_from_annotations_folder(
    annotations_folder_path: str,
    resolution: float,
    full_rgb_path: str,
):
    # Create the folders
    image_prefix = get_file_base_name(full_rgb_path)
    rgb_output_folder_path = os.path.join(Folders.CROPPED_IMAGES.value, image_prefix)
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

            # Create the cropped RGB image
            rgb_output_path = os.path.join(rgb_output_folder_path, output_file)
            if not os.path.exists(rgb_output_path):
                crop_image_from_box(full_rgb_path, image_box, rgb_output_path)

            # Create the cropped unfiltered CHM image
            chm_unfiltered_output_path = os.path.join(
                chm_unfiltered_output_folder_path, output_file
            )
            if not os.path.exists(chm_unfiltered_output_path):
                crop_image_from_box(
                    full_chm_unfiltered_path, image_box, chm_unfiltered_output_path
                )

            # Create the cropped filtered CHM image
            chm_filtered_output_path = os.path.join(
                chm_filtered_output_folder_path, output_file
            )
            if not os.path.exists(chm_filtered_output_path):
                crop_image_from_box(
                    full_chm_filtered_path, image_box, chm_filtered_output_path
                )


class ImageData:
    def __init__(self, image_path: str) -> None:
        self.path = image_path
        self._init_properties()
        self.base_name = get_file_base_name(self.path)
        self.coord_name = f"{round(self.coord_box.x_min)}_{round(self.coord_box.y_max)}"

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

        ds = None
