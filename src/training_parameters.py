import albumentations as A
import cv2


crop_size = 640
distort_steps = 30
distort_limit = 0.2

class_names = {
    0: "Tree",
    1: "Tree_low_hard",
    2: "Tree_LiDAR",
    3: "Tree_RGB",
}

class_colors = {
    "Tree": (104, 201, 45),
    "Tree_low_hard": (255, 215, 158),
    "Tree_LiDAR": (158, 174, 255),
    "Tree_RGB": (251, 106, 225),
}

class_indices = {value: key for key, value in class_names.items()}

bbox_params = A.BboxParams(
    format="pascal_voc", min_area=0, min_visibility=0.2, label_fields=["class_labels"]
)

transform_spatial = A.Compose(
    [
        A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
        # A.GridDistortion(
        #     num_steps=distort_steps,
        #     distort_limit=(-distort_limit, distort_limit),
        #     border_mode=cv2.BORDER_CONSTANT,
        #     normalized=True,
        #     p=0.5,
        # ),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        # A.Perspective(interpolation=cv2.INTER_LINEAR, p=0.25),
    ],
    bbox_params=bbox_params,
)

transform_pixel_rgb = A.Compose(
    [
        # A.Sharpen(p=0.25),
        # A.RingingOvershoot(p=0.5),
        # A.RandomGamma(p=1.0),
        # A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
        # A.FancyPCA(alpha=1.0, p=0.5),
        # A.Emboss(p=0.5),
        # A.RandomBrightnessContrast(p=1.0),
        # A.CLAHE(clip_limit=2.0, p=0.25),
        A.ChannelDropout(channel_drop_range=(1, 3), p=0.25),
    ],
)

transform_pixel_chm = A.Compose(
    [
        A.GaussNoise(var_limit=(0, 1.0), mean=0, p=0.5),
        A.ChannelDropout(channel_drop_range=(1, 6), p=0.5),
    ],
)

proba_drop_rgb = 0.333
labels_transformation_drop_rgb = {
    "Tree": "Tree_LiDAR",
    "Tree_low_hard": "Tree_LiDAR",
    "Tree_LiDAR": "Tree_LiDAR",
    "Tree_RGB": None,
}
proba_drop_chm = 0.333
labels_transformation_drop_chm = {
    "Tree": "Tree_RGB",
    "Tree_low_hard": "Tree_RGB",
    "Tree_LiDAR": None,
    "Tree_RGB": "Tree_RGB",
}
