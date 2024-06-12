from enum import Enum


class DatasetConst(Enum):
    CLASS_NAMES = {
        0: "Tree",
        1: "Tree_unsure",
        2: "Tree_disappeared",
        3: "Tree_replaced",
        4: "Tree_new",
    }

    CLASS_COLORS = {
        "Tree": (104, 201, 45),
        "Tree_unsure": (255, 215, 158),
        "Tree_disappeared": (158, 174, 255),
        "Tree_replaced": (255, 90, 82),
        "Tree_new": (251, 106, 225),
    }

    LABELS_TRANSFORMATION_DROP_RGB = {
        "Tree": "Tree_disappeared",
        "Tree_unsure": "Tree_disappeared",
        "Tree_disappeared": "Tree_disappeared",
        "Tree_replaced": "Tree_disappeared",
        "Tree_new": None,
    }

    LABELS_TRANSFORMATION_DROP_CHM = {
        "Tree": "Tree_new",
        "Tree_unsure": "Tree_new",
        "Tree_disappeared": None,
        "Tree_replaced": "Tree_new",
        "Tree_new": "Tree_new",
    }

    NO_DATA_NEW_VALUE = -5
