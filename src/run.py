import multiprocessing as mp
import os
import sys
from typing import Tuple

import torch
from tqdm import tqdm

from geojson_conversions import open_geojson_feature_collection
from layers import AMF_GD_YOLOv8
from preprocessing.data import ImageData
from preprocessing.rgb_cir import download_rgb_image_from_polygon
from training import (
    compute_mean_and_std,
    compute_metrics,
    create_and_save_splitted_datasets,
    initialize_dataloaders,
    load_tree_datasets_from_split,
)
from training_parameters import (
    class_colors,
    class_indices,
    class_names,
    labels_transformation_drop_chm,
    labels_transformation_drop_rgb,
    proba_drop_chm,
    proba_drop_rgb,
    transform_pixel_chm,
    transform_pixel_rgb,
    transform_spatial,
)
from utils import Folders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"
print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

annotations_file_name = "122000_484000.geojson"

annotations_path = os.path.join(Folders.FULL_ANNOTS.value, annotations_file_name)
annotations = open_geojson_feature_collection(annotations_path)
full_image_path_tif = download_rgb_image_from_polygon(annotations["bbox"])[
    0
]  # TODO: Change this (create a new function)

resolution = 0.08

image_data = ImageData(full_image_path_tif)

annotations_folder_path = os.path.join(Folders.CROPPED_ANNOTS.value, image_data.base_name)
rgb_cir_folder_path = os.path.join(Folders.IMAGES.value, "merged", "cropped", image_data.base_name)
chm_folder_path = os.path.join(
    Folders.CHM.value,
    f"{round(resolution*100)}cm",
    "filtered",
    "merged",
    "cropped",
    image_data.coord_name,
)

sets_ratios = [3, 1, 1]
sets_names = ["training", "validation", "test"]
data_split_file_path = os.path.join(Folders.OTHERS_DIR.value, "data_split.json")
dismissed_classes = []

create_and_save_splitted_datasets(
    rgb_cir_folder_path,
    chm_folder_path,
    annotations_folder_path,
    sets_ratios,
    sets_names,
    data_split_file_path,
    random_seed=0,
)

mean_rgb, std_rgb = compute_mean_and_std(
    rgb_cir_folder_path, per_channel=True, replace_no_data=False
)
no_data_new_value = -5  # TODO: Variable to add to the Dataset!
mean_chm, std_chm = compute_mean_and_std(
    chm_folder_path, per_channel=False, replace_no_data=True, no_data_new_value=no_data_new_value
)
print(f"{mean_rgb = }")
print(f"{mean_chm = }")

datasets = load_tree_datasets_from_split(
    data_split_file_path,
    class_indices,
    class_colors,
    mean_rgb=mean_rgb,
    std_rgb=std_rgb,
    mean_chm=mean_chm,
    std_chm=std_chm,
    proba_drop_rgb=proba_drop_rgb,
    labels_transformation_drop_rgb=labels_transformation_drop_rgb,
    proba_drop_chm=proba_drop_chm,
    labels_transformation_drop_chm=labels_transformation_drop_chm,
    dismissed_classes=dismissed_classes,
    transform_spatial_training=transform_spatial,
    transform_pixel_rgb_training=transform_pixel_rgb,
    transform_pixel_chm_training=transform_pixel_chm,
)

# Training parameters

lr = 1e-2
epochs = 10

batch_size = 6
num_workers = mp.cpu_count()
accumulate = 12


def _get_name_and_path(index: int, postfix: str) -> Tuple[str, str]:
    model_name = f"trained_model_{epochs}ep_{index}_{postfix}"
    model_path = os.path.join(Folders.MODELS_AMF_GD_YOLOV8.value, f"{model_name}.pt")
    return model_name, model_path


def get_last_model_name_and_path(postfix: str) -> Tuple[str, str]:
    index = 0
    _, model_path = _get_name_and_path(index, postfix)
    if not os.path.exists(model_path):
        raise Exception("No such model exists.")
    while os.path.exists(model_path):
        index += 1
        _, model_path = _get_name_and_path(index, postfix)

    return _get_name_and_path(index - 1, postfix)


def get_model_name_and_path(postfix: str) -> Tuple[str, str]:
    index = 0
    model_name, model_path = _get_name_and_path(index, postfix)
    while os.path.exists(model_path):
        index += 1
        model_name, model_path = _get_name_and_path(index, postfix)

    return model_name, model_path


postfix = "multi_chm"
model_name, model_path = get_last_model_name_and_path(postfix)


model = AMF_GD_YOLOv8(
    datasets["training"].rgb_channels,
    datasets["training"].chm_channels,
    device=device,
    scale="n",
    class_names=class_names,
    name=model_name,
).to(device)

print(f"{datasets['training'].rgb_channels = }")
print(f"{datasets['training'].chm_channels = }")

state_dict_path = os.path.join(Folders.MODELS_AMF_GD_YOLOV8.value, f"{model_name}.pt")
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)

_, _, test_loader = initialize_dataloaders(
    datasets=datasets, batch_size=batch_size, num_workers=num_workers
)

no_rgbs = [False, False, True, True]
no_chms = [False, True, False, True]
test_names = ["all", "no_chm", "no_rgb", "no_chm_no_rgb"]

pbar = tqdm(zip(no_rgbs, no_chms, test_names), total=len(no_rgbs))
for no_rgb, no_chm, test_name in pbar:
    desc = f"RGB: {'No' if no_rgb else 'Yes'}, CHM: {'No' if no_chm else 'Yes'} "
    pbar.set_description(desc)
    pbar.refresh()
    compute_metrics(
        model,
        test_loader,
        device,
        no_rgb=no_rgb,
        no_chm=no_chm,
        save_path=os.path.join(Folders.OUTPUT_DIR.value, f"{model_name}_{test_name}.png"),
    )
