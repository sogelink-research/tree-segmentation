from enum import Enum

import numpy as np


class DatasetConst(Enum):
    CLASS_NAMES = {
        0: "Tree",
        1: "Tree_low_hard",
        2: "Tree_LiDAR",
        3: "Tree_RGB",
    }

    CLASS_COLORS = {
        "Tree": (104, 201, 45),
        "Tree_low_hard": (255, 215, 158),
        "Tree_LiDAR": (158, 174, 255),
        "Tree_RGB": (251, 106, 225),
    }

    LABELS_TRANSFORMATION_DROP_RGB = {
        "Tree": "Tree_LiDAR",
        "Tree_low_hard": "Tree_LiDAR",
        "Tree_LiDAR": "Tree_LiDAR",
        "Tree_RGB": None,
    }

    LABELS_TRANSFORMATION_DROP_CHM = {
        "Tree": "Tree_RGB",
        "Tree_low_hard": "Tree_RGB",
        "Tree_LiDAR": None,
        "Tree_RGB": "Tree_RGB",
    }

    NO_DATA_NEW_VALUE = -5

    RGB_DATA_TYPE = np.uint8
    CHM_DATA_TYPE = np.float32
