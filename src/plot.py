import json
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import geojson
import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from skimage import io

from box_cls import Box, box_pixels_cropped_to_full, box_pixels_to_coordinates
from geojson_conversions import bboxes_to_geojson_feature_collection, save_geojson
from utils import (
    get_coordinates_bbox_from_full_image_file_name,
    get_pixels_bbox_from_full_image_file_name,
)


Number = float | int


def get_bounding_boxes(
    annotations_path: str, dismissed_classes: List[str] = []
) -> Tuple[List[Box], List[str]]:
    """Reads an annotation file and returns all the bounding boxes positions and labels.

    Args:
        bboxes_path (str): path to the file containing the annotations.
        dismissed_classes (List[str], optional): list of classes for which the
        bounding boxes are ignored. Defaults to [].

    Returns:
        Tuple[List[Box], List[str]]: (bboxes, labels) with bboxes of size (n, 4)
        and labels of size (n)
    """
    with open(annotations_path, "r") as file:
        # Load the annotation data
        bboxes_json = json.load(file)

        # Get every bounding box
        bboxes = []
        labels = []
        for bbox in bboxes_json["bounding_boxes"]:
            if bbox["label"] not in dismissed_classes:
                bboxes.append(
                    Box(
                        x_min=bbox["x_min"],
                        y_min=bbox["y_min"],
                        x_max=bbox["x_max"],
                        y_max=bbox["y_max"],
                    )
                )
                labels.append(bbox["label"])
    return bboxes, labels


def add_bbox_to_image(
    image: np.ndarray,
    bbox: List[Number],
    color: Tuple[int, int, int] = (128, 128, 128),
    lw: int = 2,
):
    lw = 2
    margin_lw = int(np.ceil(lw / 2))
    x0, y0, x1, y1 = (
        round(bbox[0]),
        round(bbox[1]),
        round(bbox[2]),
        round(bbox[3]),
    )
    cv2.rectangle(
        image,
        (x0 - margin_lw - 1, y0 - margin_lw - 1),
        (x1 + margin_lw, y1 + margin_lw),
        color=color,
        thickness=lw,
        lineType=cv2.LINE_AA,
    )


def add_label_to_image(
    image: np.ndarray,
    bbox: List[Number],
    label: str | None = None,
    color: Tuple[int, int, int] = (128, 128, 128),
    txt_color: Tuple[int, int, int] = (255, 255, 255),
    lw: int = 2,
    thickness: int = 1,
    font_scale: float = 0.5,
):
    if label is None:
        return
    x0, y0, x1, y1 = (
        round(bbox[0]),
        round(bbox[1]),
        round(bbox[2]),
        round(bbox[3]),
    )
    (w, h), baseline = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=thickness)
    margin = round(0.5 * baseline)
    p1, p2 = [0, 0], [0, 0]
    if y0 - (h + baseline) - (lw + 1) >= 0:
        p1[1] = y0 - (h + baseline) - (lw + 1)
        p2[1] = y0 + margin - (lw + 1)
    elif y1 + (h + baseline + margin) + lw < image.shape[0]:
        p1[1] = y1 + lw
        p2[1] = y1 + (h + baseline + margin) + lw
    else:
        p1[1] = y0
        p2[1] = y0 + (h + baseline) + margin

    if x0 + (w + 2 * margin) < image.shape[1]:
        p1[0] = x0
        p2[0] = x0 + (w + 2 * margin)
    else:
        p1[0] = x1 - (w + 2 * margin)
        p2[0] = x1

    sub_img = deepcopy(image[p1[1] : p2[1], p1[0] : p2[0]])
    sub_img_2 = deepcopy(image[p1[1] : p2[1], p1[0] : p2[0]])

    if sub_img.size > 0:
        cv2.rectangle(sub_img, (0, 0), (p2[0] - p1[0], p2[1] - p1[1]), color, -1, cv2.LINE_AA)

        alpha = 0.5  # Weight for the rectangle image
        beta = 1 - alpha  # Weight for the original image
        gamma = 0  # Offset
        blended_image = cv2.addWeighted(sub_img, alpha, sub_img_2, beta, gamma)

        image[p1[1] : p2[1], p1[0] : p2[0]] = blended_image
        cv2.putText(
            img=image,
            text=label,
            org=(p1[0] + margin, p2[1] - baseline),
            fontFace=0,
            fontScale=font_scale,
            color=txt_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        print(f"{bbox = }")
        print(f"{(w, h), baseline = }")
        print(f"{sub_img.shape = }")
        print(f"{p1 = }")
        print(f"{p2 = }")


def create_bboxes_image(
    image: np.ndarray,
    bboxes: List[List[Number]],
    labels: List[str],
    colors_dict: Dict[str, Tuple[int, int, int]],
    scores: Optional[List[float] | np.ndarray] = None,
    color_mode: str = "rgb",
) -> np.ndarray:
    image_copy = deepcopy(image)
    if color_mode == "bgr":
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    # Plot each box
    full_labels: List[str] = [""] * len(bboxes)
    colors: List[Tuple[int, int, int]] = [(128, 128, 128)] * len(bboxes)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # Add score in label if score=True
        if scores is not None:
            full_labels[i] = f"{label} {round(100 * float(scores[i]), 1)}%"
        else:
            full_labels[i] = label
        colors[i] = colors_dict[label]
    for i, bbox in enumerate(bboxes):
        add_bbox_to_image(image_copy, bbox, colors[i])
    for i, bbox in enumerate(bboxes):
        add_label_to_image(image_copy, bbox, full_labels[i], colors[i])

    return image_copy


def create_bboxes_training_image(
    image_rgb: torch.Tensor,
    image_chm: torch.Tensor,
    pred_bboxes: List[Box],
    pred_labels: List[int],
    pred_scores: List[float],
    gt_bboxes: List[Box],
    gt_labels: List[int],
    labels_int_to_str: Dict[int, str],
    colors_dict: Dict[str, Tuple[int, int, int]],
    save_path: str,
) -> None:
    dimensions = image_rgb.shape[0]
    if dimensions % 3 != 0:
        raise ValueError(
            f"The number of dimensions of image_rgb should be a multiple of 3, not {dimensions}"
        )

    print(f"{image_rgb.shape = }")

    images = [
        image_rgb[idx : idx + 3].cpu().detach().numpy().transpose((1, 2, 0))
        for idx in range(0, dimensions, 3)
    ]

    print(f"{images[0].shape = }")
    dims = (0, 1)
    dtype = images[0].dtype if torch.is_floating_point(images[0]) else torch.float32
    print(f"{torch.mean(images[0].to(dtype), dim=dims).reshape(-1) = }")
    print(f"{torch.amin(images[0].to(dtype), dim=dims).reshape(-1) = }")
    print(f"{torch.amax(images[0].to(dtype), dim=dims).reshape(-1) = }")

    number_images = len(images)

    pred_bboxes_list = [bbox.as_list() for bbox in pred_bboxes]
    gt_bboxes_list = [bbox.as_list() for bbox in gt_bboxes]
    pred_labels_str = [labels_int_to_str[label] for label in pred_labels]
    gt_labels_str = [labels_int_to_str[label] for label in gt_labels]

    nrows = number_images
    ncols = 2
    figsize = (5 * ncols, 6 * nrows)
    plt.clf()
    fig = plt.figure(1, figsize=figsize)

    for index, image in enumerate(images):
        image_pred_boxes = create_bboxes_image(
            image, pred_bboxes_list, pred_labels_str, colors_dict, pred_scores, color_mode="bgr"
        )
        images_gt_boxes = create_bboxes_image(
            image, gt_bboxes_list, gt_labels_str, colors_dict, color_mode="bgr"
        )

        image_name = f"Image {index}"

        ax = fig.add_subplot(nrows, ncols, 2 * index + 1)
        ax.imshow(image_pred_boxes)
        ax.set_title(f"{image_name} predictions")
        ax.set_axis_off()

        ax = fig.add_subplot(nrows, ncols, 2 * index + 2)
        ax.imshow(images_gt_boxes)
        ax.set_title(f"{image_name} ground truth")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def create_gt_bboxes_image(
    annotations_path: str,
    rgb_path: str,
    class_colors: Dict[str, Tuple[int, int, int]],
    output_path: str | None = None,
) -> np.ndarray:
    """Creates and returns an image with the ground truth bounding boxes. The image can also be
    saved directly if a path is given.

    Args:
        annotations_path (str): Path to the file containing the annotations.
        rgb_path (str): Path to the file containing the image.
        class_colors (Dict[str, Tuple[int, int, int]]): Dictionary associating a color to each
        class name.
        output_path (str | None, optional): An optional path to save the created image at.
        Defaults to None.

    Returns:
        np.ndarray: The image with the bounding boxes.
    """
    rgb_image = io.imread(rgb_path)
    bboxes, labels = get_bounding_boxes(annotations_path)
    bboxes_list = [bbox.as_list() for bbox in bboxes]
    image = create_bboxes_image(rgb_image, bboxes_list, labels, class_colors)
    if output_path is not None:
        io.imsave(output_path, image)
    return image


def create_geojson_output(
    full_image_name: str,
    cropped_coords_name: str,
    bboxes: List[Box],
    labels: List[str],
    scores: List[float] | None,
    add_image_limits: bool = True,
    save_path: str | None = None,
) -> geojson.FeatureCollection:
    full_image_coordinates_box = get_coordinates_bbox_from_full_image_file_name(full_image_name)
    full_image_pixels_box = get_pixels_bbox_from_full_image_file_name(full_image_name)
    cropped_image_pixels_box = Box.from_short_name(cropped_coords_name)
    full_image_bboxes = [
        box_pixels_cropped_to_full(bbox, cropped_image_pixels_box) for bbox in bboxes
    ]
    geocoords_bboxes = [
        box_pixels_to_coordinates(bbox, full_image_pixels_box, full_image_coordinates_box)
        for bbox in full_image_bboxes
    ]

    if add_image_limits:
        geocoords_bboxes.append(
            box_pixels_to_coordinates(
                cropped_image_pixels_box,
                full_image_pixels_box,
                full_image_coordinates_box,
            )
        )
        labels.append("Image_limits")
        if scores is not None:
            scores.append(-1)

    feature_collection = bboxes_to_geojson_feature_collection(
        geocoords_bboxes, labels, scores=scores
    )

    if save_path is not None:
        save_geojson(feature_collection, save_path)

    return feature_collection
