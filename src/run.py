import os

from chm import compute_chm, compute_dtm
from data_processing import Box
from lidar_preprocessing import crop_las

if __name__ == "__main__":
    x_min = 122100
    y_min = 483800
    size = 100
    crop_box = Box(x_min=x_min, y_min=y_min, x_max=x_min + size, y_max=y_min + size)
    crop_las(
        "data/lidar/unfiltered/full/122000_484000.laz",
        "data/lidar/unfiltered/full/122000_484000_cropped.laz",
        crop_box,
    )
    full_size = 1000
    size_ratio = size / full_size
    compute_dtm(
        "data/lidar/unfiltered/full/122000_484000_cropped.laz",
        width=round(12500 * size_ratio),
        height=round(12500 * size_ratio),
        resolution=0.08,
        verbose=True,
    )
    # compute_chm(
    #     "data/lidar/unfiltered/full/122000_484000_cropped.laz",
    #     "data/lidar/unfiltered/full/122000_484000_cropped_unfiltered_chm.tif",
    #     width=round(12500 * size_ratio),
    #     height=round(12500 * size_ratio),
    #     resolution=0.08,
    #     verbose=True,
    # )
