from __future__ import annotations

from typing import List, Sequence, Tuple, cast


class Box:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)

    def __str__(self) -> str:
        return (
            f"(x_min: {self.x_min}, y_min: {self.y_min}, x_max: {self.x_max}, y_max: {self.y_max})"
        )

    def __repr__(self) -> str:
        return f"({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"

    def __hash__(self) -> int:
        return (self.x_min, self.y_min, self.x_max, self.y_max).__hash__()

    @property
    def area(self) -> float:
        """Returns the area of the box.

        Returns:
            float: area of the box.
        """
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def very_short_name(self) -> str:
        """Returns the bounding box as a coordinates string similar to images names.

        Returns:
            str: f"{x_min}_{y_max}" where values are rounded to ints.
        """
        return f"{round(self.x_min)}_{round(self.y_max)}"

    def short_name(self) -> str:
        """Returns the bounding box as a short coordinates string.

        Returns:
            str: f"{x_min}_{y_min}_{x_max}_{y_max}" where values are rounded to ints.
        """
        return f"{round(self.x_min)}_{round(self.y_min)}_{round(self.x_max)}_{round(self.y_max)}"

    def as_list(self) -> List[float]:
        """Returns the Box as a list.

        Returns:
            List[float]: [x_min, y_min, x_max, y_max].
        """
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Returns the Box as a tuple.

        Returns:
            Tuple[float, float, float, float]: (x_min, y_min, x_max, y_max).
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @staticmethod
    def from_short_name(short_name: str) -> Box:
        return Box(*map(float, short_name.split("_")))

    @staticmethod
    def from_list(coords_list: Sequence[float | int]) -> Box:
        if len(coords_list) != 4:
            raise ValueError("Expects a list with 4 numbers.")
        return Box(*coords_list)


class BoxInt(Box):
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int) -> None:
        self.x_min = round(x_min)
        self.y_min = round(y_min)
        self.x_max = round(x_max)
        self.y_max = round(y_max)

    @property
    def area(self) -> int:
        return round(super().area)

    def as_list(self) -> List[int]:
        return list(map(round, super().as_list()))

    def as_tuple(self) -> Tuple[int, int, int, int]:
        int_tuple = tuple(map(round, super().as_tuple()))
        return cast(Tuple[int, int, int, int], int_tuple)

    @staticmethod
    def from_short_name(short_name: str) -> BoxInt:
        return BoxInt(*map(int, short_name.split("_")))

    @staticmethod
    def from_list(coords_list: Sequence[int]) -> BoxInt:
        return BoxInt(*coords_list)


def intersection(box1: Box, box2: Box) -> float:
    """Compute the area of the intersection of box1 and box2.

    Args:
        box1 (Box): first box to intersect.
        box2 (Box): second box to intersect.

    Returns:
        float: intersection of the two boxes.
    """
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
        annot (Box): the bounding box.
        limits (Box): the limits in which we want to put it.

    Returns:
        float: (Area of the intersection of annot and limits) / (Area of annot).
    """
    if annot.area == 0:
        raise ValueError("The area of annot is 0.")
    return intersection(annot, limits) / annot.area


def compute_iou(box1: Box, box2: Box) -> float:
    """Compute the area of the intersection over the area of the union (IoU).

    Args:
        box1 (Box): first box.
        box2 (Box): second box.

    Returns:
        float: IoU.
    """
    # Calculate the intersection coordinates
    inter_x_min = max(box1.x_min, box2.x_min)
    inter_y_min = max(box1.y_min, box2.y_min)
    inter_x_max = min(box1.x_max, box2.x_max)
    inter_y_max = min(box1.y_max, box2.y_max)

    # Compute the area of the intersection
    inter_area = max(inter_x_max - inter_x_min, 0) * max(inter_y_max - inter_y_min, 0)
    union_area = box1.area + box2.area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def box_crop_in_box(to_crop: Box, limits: Box) -> Box:
    """Crops the box so that it fits entirely in the limits.

    Args:
        to_crop (Box): box to crop.
        limits (Box): limits in the same reference system.

    Returns:
        Box: box cropped to fit in the limits.
    """
    return Box(
        x_min=min(max(to_crop.x_min, limits.x_min), limits.x_max),
        y_min=min(max(to_crop.y_min, limits.y_min), limits.y_max),
        x_max=min(max(to_crop.x_max, limits.x_min), limits.x_max),
        y_max=min(max(to_crop.y_max, limits.y_min), limits.y_max),
    )


def box_pixels_full_to_cropped(to_modify: Box, cropped_frame: Box) -> Box:
    """Translates the coordinates from the full image to the cropped image.

    Args:
        to_modify (Box): box pixels in the full image.
        frame (Box): cropped image box pixels in the full image.

    Returns:
        Box: box pixels in the cropped image.
    """
    return Box(
        x_min=to_modify.x_min - cropped_frame.x_min,
        y_min=to_modify.y_min - cropped_frame.y_min,
        x_max=to_modify.x_max - cropped_frame.x_min,
        y_max=to_modify.y_max - cropped_frame.y_min,
    )


def box_pixels_cropped_to_full(to_modify: Box, cropped_frame: Box) -> Box:
    """Translates the coordinates from the cropped image to the full image.

    Args:
        to_modify (Box): box pixels in the cropped image.
        frame (Box): cropped image box pixels in the full image.

    Returns:
        Box: box pixels in the full image.
    """
    return Box(
        x_min=to_modify.x_min + cropped_frame.x_min,
        y_min=to_modify.y_min + cropped_frame.y_min,
        x_max=to_modify.x_max + cropped_frame.x_min,
        y_max=to_modify.y_max + cropped_frame.y_min,
    )


def box_pixels_to_coordinates(box: Box, image_pixels_box: Box, image_coordinates_box: Box) -> Box:
    """Converts the coordinates of a Box from pixels in an image to geographical
    coordinates.

    Args:
        box (Box): bounding box in image pixels.
        image_pixels_box (Box): bounding box of the image in pixel (so something
        like Box(0, 0, 640, 640]).
        image_coordinates_box (Box): bounding box of the image in coordinates (so
        something like [122000, 483000, 123000, 484000]).

    Returns:
        Box: bounding box in image pixels.
    """
    x_factor = (image_coordinates_box.x_max - image_coordinates_box.x_min) / (
        image_pixels_box.x_max - image_pixels_box.x_min
    )
    y_factor = (image_coordinates_box.y_max - image_coordinates_box.y_min) / (
        image_pixels_box.y_max - image_pixels_box.y_min
    )

    new_x_min = (box.x_min - image_pixels_box.x_min) * x_factor + image_coordinates_box.x_min
    new_x_max = (box.x_max - image_pixels_box.x_min) * x_factor + image_coordinates_box.x_min

    new_y_min = (image_pixels_box.y_max - box.y_max) * y_factor + image_coordinates_box.y_min
    new_y_max = (image_pixels_box.y_max - box.y_min) * y_factor + image_coordinates_box.y_min

    new_box = Box(x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max)
    return new_box


def box_coordinates_to_pixels(box: Box, image_coordinates_box: Box, image_pixels_box: Box) -> Box:
    """Converts the coordinates of a Box from geographical coordinates to pixels
    in an image.

    Args:
        box (Box): bounding box in geographical coordinates.
        image_coordinates_box (Box): bounding box of the image in coordinates (so
        something like [122000, 483000, 123000, 484000]).
        image_pixels_box (Box): bounding box of the image in pixel (so something
        like [0, 0, 640, 640]).

    Returns:
        Box: bounding box in image pixels.
    """
    x_factor = (image_pixels_box.x_max - image_pixels_box.x_min) / (
        image_coordinates_box.x_max - image_coordinates_box.x_min
    )
    y_factor = (image_pixels_box.y_max - image_pixels_box.y_min) / (
        image_coordinates_box.y_max - image_coordinates_box.y_min
    )

    new_x_min = (box.x_min - image_coordinates_box.x_min) * x_factor + image_pixels_box.x_min
    new_x_max = (box.x_max - image_coordinates_box.x_min) * x_factor + image_pixels_box.x_min

    new_y_min = (image_coordinates_box.y_max - box.y_max) * y_factor + image_pixels_box.y_min
    new_y_max = (image_coordinates_box.y_max - box.y_min) * y_factor + image_pixels_box.y_min

    new_box = Box(x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max)
    return new_box
