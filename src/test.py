import multiprocessing as mp
import os
import time
from typing import List, Optional

import numpy as np
import rasterio

from datasets import (
    compute_mean_and_std,
    normalize,
    quick_merge_chunk,
    quick_normalize_chunk,
)
from preprocessing.data import merge_tif
from utils import read_image, write_image


def process_chunk(start, end, shapes, axis, image_paths, chm):
    # Concatenate a chunk of images
    chunk_shape = list(shapes[0])
    chunk_shape[axis] = sum(shape[axis] for shape in shapes[start:end])
    chunk_image = np.zeros(chunk_shape, dtype=np.uint8)

    offset = 0
    for i in range(start, end):
        img = read_image(image_paths[i], chm=chm, mode="r")
        if axis == 0:
            chunk_image[offset : offset + img.shape[axis], :] = img
        elif axis == 1:
            chunk_image[:, offset : offset + img.shape[axis], :] = img
        elif axis == 2:
            chunk_image[:, :, offset : offset + img.shape[axis]] = img
        offset += img.shape[axis]

    return chunk_image


def concatenate_images(image_paths: List[str], chm: bool, output_path: str, axis: int = 2):
    start_time = time.time()

    # Determine the shape of the final image
    shapes = [read_image(img, chm=chm, mode="r").shape for img in image_paths]
    dtype = read_image(image_paths[0], chm=chm, mode="r").dtype
    total_shape = list(shapes[0])
    total_shape[axis] = sum(shape[axis] for shape in shapes)

    end_time = time.time()
    print(f"{'Loading time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    # Create an empty array for the final image
    output_shape = tuple(total_shape)
    output_image = np.lib.format.open_memmap(
        output_path, dtype=dtype, mode="w+", shape=output_shape
    )

    end_time = time.time()
    print(f"{'Open time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    # Split the task into chunks
    num_images = len(image_paths)
    num_chunks = mp.cpu_count()
    chunk_size = num_images // num_chunks
    ranges = [
        (i * chunk_size, (i + 1) * chunk_size if i < num_chunks - 1 else num_images)
        for i in range(num_chunks)
    ]

    end_time = time.time()
    print(f"{'Chunks time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    with mp.Pool(processes=num_chunks) as pool:
        results = pool.starmap(
            process_chunk, [(start, end, shapes, axis, image_paths, chm) for start, end in ranges]
        )

    end_time = time.time()
    print(f"{'Process time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    # Combine the results
    offset = 0
    for result in results:
        if axis == 0:
            output_image[offset : offset + result.shape[axis], :] = result
        elif axis == 1:
            output_image[:, offset : offset + result.shape[axis]] = result
        elif axis == 2:
            output_image[:, :, offset : offset + result.shape[axis]] = result
        offset += result.shape[axis]

    end_time = time.time()
    print(f"{'Combine time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    output_image.flush()

    end_time = time.time()
    print(f"{'Flush time':<20}: {(end_time - start_time)}")
    start_time = time.time()


def merge_tif_new(images_paths: List[str], chm: bool, output_path: str, memmap: bool) -> np.ndarray:
    if len(images_paths) == 0:
        raise ValueError("images_paths is empty.")

    available_output_types = [".tif", ".npy"]
    output_type = None if output_path is None else os.path.splitext(output_path)[1]
    if output_type not in available_output_types:
        raise ValueError(f"Only these output types are supported: {available_output_types}")

    if output_type == ".tif" and memmap:
        raise NotImplementedError("")

    start_time = time.time()

    # tifffile is quicker to just open the files
    images: List[np.ndarray] = []
    for image_path in images_paths:
        image = read_image(image_path, mode="c", chm=chm)
        images.append(image)

    end_time = time.time()
    print(f"{'Loading time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    # Stack images along a new axis to create a multi-channel image

    end_time = time.time()
    print(f"{'Concat time':<20}: {(end_time - start_time)}")
    start_time = time.time()

    if output_path is not None:
        if output_type == ".npy":
            if memmap:
                merged_array = quick_merge_chunk(images, memmap_save_path=output_path, axis=2)
            else:
                merged_array = quick_merge_chunk(images, axis=2)
                write_image(merged_array, chm=chm, save_path=output_path)
        elif output_type == ".tif":
            merged_array = quick_merge_chunk(images, axis=2)
            # Save the TIF
            with rasterio.open(images_paths[0]) as img:
                crs = img.crs
                transform = img.transform

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=merged_array.shape[0],
                width=merged_array.shape[1],
                count=merged_array.shape[2],
                dtype=merged_array.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                for i in range(merged_array.shape[2]):
                    dst.write(merged_array[:, :, i], i + 1)

    end_time = time.time()
    print(f"{'Write time':<20}: {(end_time - start_time)}")

    return merged_array


def main():
    images_paths = [
        "data/chm/-inf_1/full/122000_484000.tif",
        "data/chm/-inf_2/full/122000_484000.tif",
        "data/chm/-inf_3/full/122000_484000.tif",
        "data/chm/-inf_5/full/122000_484000.tif",
        "data/chm/-inf_7/full/122000_484000.tif",
        "data/chm/-inf_10/full/122000_484000.tif",
        "data/chm/-inf_15/full/122000_484000.tif",
        "data/chm/-inf_20/full/122000_484000.tif",
        "data/chm/-inf_inf/full/122000_484000.tif",
    ]
    chm = True
    no_data_new_value = -5
    for i, memmap in enumerate([True, False]):
        print(f" {memmap = } ".center(50, "#"))
        temp_path = f"TEST_{i}.npy"
        full_merged = merge_tif(images_paths, chm=chm, output_path=temp_path, memmap=memmap)

        start_time = time.time()

        mean, std = compute_mean_and_std(
            full_merged, per_channel=False, replace_no_data=chm, no_data_new_value=no_data_new_value
        )
        print(f"{(mean, std) = }")

        end_time = time.time()
        print(f"     {'Mean and std time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

    for i, in_place in enumerate([True, False]):
        print(f" {in_place = } ".center(50, "#"))

        start_time = time.time()

        normalized_full_image = quick_normalize_chunk(
            full_merged,
            mean,
            std,
            replace_no_data=chm,
            in_place=in_place,
            no_data_new_value=no_data_new_value,
        )

        end_time = time.time()
        print(f"     {'Normalize time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

        mean, std = compute_mean_and_std(
            normalized_full_image, per_channel=False, replace_no_data=chm
        )
        print(f"{(mean, std) = }")

        end_time = time.time()
        print(f"     {'Mean and std time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

    print(" Old normalize ".center(50, "#"))

    start_time = time.time()

    normalized_full_image = normalize(
        full_merged,
        mean,
        std,
        replace_no_data=chm,
        no_data_new_value=no_data_new_value,
    )

    end_time = time.time()
    print(f"     {'Normalize time':<20}: {end_time - start_time:.6f} seconds")
    start_time = time.time()

    mean, std = compute_mean_and_std(normalized_full_image, per_channel=False, replace_no_data=chm)
    print(f"{(mean, std) = }")

    end_time = time.time()
    print(f"     {'Mean and std time':<20}: {end_time - start_time:.6f} seconds")
    start_time = time.time()


if __name__ == "__main__":
    main()
