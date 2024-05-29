import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from data_processing import Box

type Number = float | int


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
    image: npt.NDArray,
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
    image: npt.NDArray,
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
    (w, h), baseline = cv2.getTextSize(
        label, 0, fontScale=font_scale, thickness=thickness
    )
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
        cv2.rectangle(
            sub_img, (0, 0), (p2[0] - p1[0], p2[1] - p1[1]), color, -1, cv2.LINE_AA
        )

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
    image: npt.NDArray,
    bboxes: List[List[Number]],
    labels: List[str],
    colors_dict: Dict[str, Tuple[int, int, int]],
    scores: List[float] | np.ndarray[Any, np.dtype[np.float_]] | None = None,
    color_mode: str = "rgb",
) -> npt.NDArray:
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
