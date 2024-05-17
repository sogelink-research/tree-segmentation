import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from osgeo import gdal
from PIL import Image
from utils import create_folder, open_json


def get_image_path_from_full_annotation_path(
    annotations: Dict[Any, Any]
) -> str:
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
    def __init__(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
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
        max(to_crop.x_min, limits.x_min),
        max(to_crop.y_min, limits.y_min),
        min(to_crop.x_max, limits.x_max),
        min(to_crop.y_max, limits.y_max),
    )


def compute_box_local_coord(to_modify: Box, frame: Box):
    return Box(
        to_modify.x_min - frame.x_min,
        to_modify.y_min - frame.y_min,
        to_modify.x_max - frame.x_min,
        to_modify.y_max - frame.y_min,
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
) -> Dict[Box, List[Annotation]]:
    """Find the bounding boxes that fit in every image. The bounding boxes are taken
    if at least `visibility_threshold` of their area is within the image. The images
    are a grid defined using `cropping_limits_x` and `cropping_limits_y`. The
    bounding boxes are read from `annotations`.

    Also, the bounding boxes labeled as "Not_labeled" are used as areas that should be
    dismissed. Every image that intersects with such a bounding box is rejected.

    Args:
        cropping_limits_x (npt.NDArray[np.int_]): limits of the images on the x axis
        cropping_limits_y (npt.NDArray[np.int_]): limits of the images on the y axis
        annotations (Dict[Any, Any]): dictionary obtained from the json file
        containing the annotations, directly from Label Studio
        visibility_threshold (float): The threshold to select the bounding boxes

    Returns:
        Dict[Box, List[Box]]: dictionary associating the bounding box of each image
        with the list of tree bounding boxes that fit in. The tree bounding boxes are
        not cropped to entirely fit in the image, and their coordinates are in the
        full image frame.
    """
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
    for annot_info in annots:
        annot_value = annot_info["value"]
        x_min = int(round(annot_value["x"] * image_width_factor))
        y_min = int(round(annot_value["y"] * image_height_factor))
        x_max = int(
            round(
                (annot_value["x"] + annot_value["width"]) * image_width_factor
            )
        )
        y_max = int(
            round(
                (annot_value["y"] + annot_value["height"]) * image_height_factor
            )
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
            for limit_box in annots_repartition:
                if (
                    intersection_ratio(annot.box, limit_box)
                    > visibility_threshold
                ):
                    annots_repartition[limit_box].append(annot)

    return annots_repartition


def crop_annots_into_limits(
    annots_repartition: Dict[Box, List[Annotation]]
) -> None:
    """Modifies the bounding boxes repartition dictionary in place to crop
    each bounding box to the limits of its image

    Args:
        annots_repartition (Dict[Box, List[Box]]): dictionary associating the
        bounding box of each image with the list of tree bounding boxes that fit in.
    """
    for limits_box, annots in annots_repartition.items():
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
    for limits_box, annots in annots_repartition.items():
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

    for image_box, annots in annots_repartition.items():
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
        output_path = os.path.join(
            output_folder_path, annotation_output_file_name
        )
        with open(output_path, "w") as outfile:
            json.dump(annots_dict, outfile)


def get_image_box_from_cropped_annotations(
    cropped_annotations: Dict[Any, Any]
) -> Box:
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


def crop_image_from_cropped_annotations(
    cropped_annotations: Dict[Any, Any], output_folder_path: str
) -> None:
    image_box = get_image_box_from_cropped_annotations(cropped_annotations)
    full_image_path_tif = get_full_image_path_from_cropped_annotations(
        cropped_annotations
    )

    window = (
        image_box.x_min,
        image_box.y_min,
        image_box.x_max - image_box.x_min,
        image_box.y_max - image_box.y_min,
    )

    output_file = f"{image_box.short_name()}.tif"
    output_path = os.path.join(output_folder_path, output_file)

    gdal.Translate(output_path, full_image_path_tif, srcWin=window)


def crop_all_images_from_annotations_folder(
    annotations_folder_path: str, output_images_folder_path: str
):
    create_folder(output_images_folder_path)
    for file_name in os.listdir(annotations_folder_path):
        # Check if the file is a regular file (not a directory)
        annotations_file_path = os.path.join(annotations_folder_path, file_name)
        if os.path.splitext(annotations_file_path)[1] == ".json":
            annotations = open_json(annotations_file_path)
            crop_image_from_cropped_annotations(
                annotations, output_images_folder_path
            )


# @measure_execution_time
# def merge_crop_las(
#     input_las_list: list[str], output_las: str, x_limits: tuple, y_limits: tuple
# ):
#     if x_limits[0] > x_limits[1]:
#         raise Exception("You should have x_limits[0] <= x_limits[1]")
#     if y_limits[0] > y_limits[1]:
#         raise Exception("You should have y_limits[0] <= y_limits[1]")
#     bounds = f"([{x_limits[0]},{x_limits[1]}],[{y_limits[0]},{y_limits[1]}])"
#     pipeline_list = []
#     for index, input_las in enumerate(input_las_list):
#         pipeline_list.append(
#             {"type": "readers.las", "file_name": input_las, "tag": f"A{index}"}
#         )
#     pipeline_list.extend(
#         [
#             {
#                 "type": "filters.merge",
#                 "inputs": [f"A{index}" for index in range(len(input_las_list))],
#             },
#             {"type": "filters.crop", "bounds": bounds},
#             {"type": "writers.las", "file_name": output_las},
#         ]
#     )
#     pipeline = pdal.Pipeline(json.dumps(pipeline_list))
#     pipeline.execute()


# @measure_execution_time
# def crop_las(input_las: str, output_las: str, x_limits: tuple, y_limits: tuple):
#     if x_limits[0] > x_limits[1]:
#         raise Exception("You should have x_limits[0] <= x_limits[1]")
#     if y_limits[0] > y_limits[1]:
#         raise Exception("You should have y_limits[0] <= y_limits[1]")
#     bounds = f"([{x_limits[0]},{x_limits[1]}],[{y_limits[0]},{y_limits[1]}])"
#     pipeline_list = [
#         {
#             "type": "readers.las",
#             "file_name": input_las,
#         },
#         {"type": "filters.crop", "bounds": bounds},
#         {"type": "writers.las", "file_name": output_las},
#     ]
#     pipeline = pdal.Pipeline(json.dumps(pipeline_list))
#     pipeline.execute()


# def remove_las_overlap_from_geotiles(input_las: str, output_las: str):
#     overlap = 20
#     with laspy.open(input_las, mode="r") as las_file:
#         # Get the bounding box information from the header
#         min_x = las_file.header.min[0] + overlap
#         max_x = las_file.header.max[0] - overlap
#         min_y = las_file.header.min[1] + overlap
#         max_y = las_file.header.max[1] - overlap

#     crop_las(input_las, output_las, (min_x, max_x), (min_y, max_y))


# def remove_las_overlap_from_geotiles_all():
#     point_clouds_overlap_folder = "../data/point_clouds_geotiles"
#     point_clouds_no_overlap_folder = "../data/point_clouds_geotiles_no_overlap"
#     if not os.path.exists(point_clouds_no_overlap_folder):
#         os.makedirs(point_clouds_no_overlap_folder)
#     for file_name in os.listdir(point_clouds_overlap_folder):
#         # Check if the file is a regular file (not a directory)
#         overlap_file_path = os.path.join(point_clouds_overlap_folder, file_name)
#         no_overlap_file_path = os.path.join(
#             point_clouds_no_overlap_folder, file_name
#         )
#         if (os.path.isfile(overlap_file_path)) and (
#             not os.path.exists(no_overlap_file_path)
#         ):
#             remove_las_overlap_from_geotiles(
#                 overlap_file_path, no_overlap_file_path
#             )


# @measure_execution_time
# def filter_classification_las(input_las: str, output_las: str):
#     pipeline_list = [
#         {
#             "type": "readers.las",
#             "file_name": input_las,
#         },
#         {
#             "type": "filters.range",
#             "limits": "Classification[1:5]",  # Keep only unclassified, ground and vegetation
#         },
#         {"type": "writers.las", "file_name": output_las},
#     ]
#     pipeline = pdal.Pipeline(json.dumps(pipeline_list))
#     pipeline.execute()
