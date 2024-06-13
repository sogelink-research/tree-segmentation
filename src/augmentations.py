import random
from math import ceil
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import tifffile
from matplotlib import pyplot as plt


CROP_SIZE = 640


def get_transform_spatial():
    distort_steps = 30
    distort_limit = 0.2
    transform_spatial = A.Compose(
        [
            A.RandomCrop(width=CROP_SIZE, height=CROP_SIZE, p=1.0),
            A.GridDistortion(
                num_steps=distort_steps,
                distort_limit=(-distort_limit, distort_limit),
                border_mode=cv2.BORDER_CONSTANT,
                normalized=True,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.Perspective(interpolation=cv2.INTER_LINEAR, p=0.5),
        ],
    )
    return transform_spatial


def get_transform_pixel_rgb(channels: int):
    transform_pixel_rgb = A.Compose(
        [
            A.Sharpen(p=0.25),
            A.RingingOvershoot(p=0.5),
            A.RandomGamma(p=1.0),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
            A.Emboss(p=0.5),
            A.RandomBrightnessContrast(p=1.0),
            A.ChannelDropout(channel_drop_range=(1, channels - 1), p=0.5),
        ],
    )
    return transform_pixel_rgb


def get_transform_pixel_chm(channels: int):
    transform_pixel_chm = A.Compose(
        [
            (
                A.ChannelDropout(channel_drop_range=(1, channels - 1), p=0.5)
                if channels > 1
                else A.NoOp()
            ),
        ],
    )
    return transform_pixel_chm


def split_images(
    images: List[np.ndarray], images_transformed: List[np.ndarray], image_types: List[str]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    splitted_images = []
    splitted_images_transformed = []
    for image, image_transformed, image_type in zip(images, images_transformed, image_types):
        if image_type == "RGB":
            channels = image.shape[2]
            if channels % 3 != 0:
                raise Exception(
                    f"An RGB image mut have a number of channels that is a multiple of 3, but here the shape is {image.shape}."
                )
            for split_index in range(0, channels, 3):
                splitted_images.append(image[:, :, split_index : split_index + 3])
                splitted_images_transformed.append(
                    image_transformed[:, :, split_index : split_index + 3]
                )
        elif image_type == "GREY":
            channels = image.shape[2]
            for split_index in range(0, channels):
                splitted_images.append(image[:, :, split_index : split_index + 1])
                splitted_images_transformed.append(
                    image_transformed[:, :, split_index : split_index + 1]
                )
        else:
            raise Exception(f"The types must be either 'RGB' or 'CHM', not {image_type}.")

    return splitted_images, splitted_images_transformed


def apply_transform_to_all(transform: A.Compose, images: List[np.ndarray]) -> List[np.ndarray]:
    seed = np.random.randint(0, 100000)
    images_transformed = []
    for image in images:
        random.seed(seed)
        images_transformed.append(transform(image=image)["image"])
    return images_transformed


def main():
    image_rgb_cir = tifffile.imread(
        "data/images/merged/cropped/2023_122000_484000_RGB_hrl/0_0_640_640.tif"
    )
    image_chm = tifffile.imread(
        "data/chm/8cm/filtered/merged/cropped/122000_484000/0_0_640_640.tif"
    )[:, :, [0, 2, 4, 6, 8]]
    image_chm[image_chm == -9999] = -5
    print(f"{image_chm.dtype = }")
    print(f"{image_chm.min(), image_chm.max() = }")

    images = [image_rgb_cir, image_chm]
    types = ["RGB", "GREY"]

    # Transform images
    transform_spatial = get_transform_spatial()
    transform_pixel_rgb = get_transform_pixel_rgb(image_rgb_cir.shape[2])
    transform_pixel_chm = get_transform_pixel_chm(image_chm.shape[2] or 1)

    images_transformed = apply_transform_to_all(transform_spatial, images)
    images_transformed[0] = transform_pixel_rgb(image=images_transformed[0])["image"]
    images_transformed[1] = transform_pixel_chm(image=images_transformed[1])["image"]

    print(f"{images_transformed[1].dtype = }")
    print(f"{images_transformed[1].min(), images_transformed[1].max() = }")

    # Split images
    images, images_transformed = split_images(images, images_transformed, types)

    # Plot images
    fig = plt.figure(1, figsize=(19, 11))
    scale = ceil((len(images)) ** 0.5)
    nrows = scale
    ncols = 2 * scale

    for idx, (image, image_transformed) in enumerate(zip(images, images_transformed)):
        ax = fig.add_subplot(nrows, ncols, 2 * idx + 1)
        ax.imshow(image)
        ax.set_title(f"Image {idx}")
        ax.set_axis_off()

        ax = fig.add_subplot(nrows, ncols, 2 * idx + 2)
        ax.imshow(image_transformed)
        ax.set_title(f"Image {idx} transformed")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
