from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import geojson
import numpy as np
import tifffile
import torch

from augmentations import (
    get_transform_pixel_chm,
    get_transform_pixel_rgb,
    get_transform_spatial,
)
from dataloaders import convert_ground_truth_from_tensors, initialize_dataloaders
from dataset_constants import DatasetConst
from datasets import (
    compute_mean_and_std,
    create_and_save_splitted_datasets,
    load_tree_datasets_from_split,
    normalize,
)
from geojson_conversions import open_geojson_feature_collection
from layers import AMF_GD_YOLOv8
from metrics import AP_Metrics, AP_Metrics_List
from plot import create_bboxes_training_image
from preprocessing.chm import compute_slices_chm, get_full_chm_slice_path
from preprocessing.data import (
    annots_coordinates_to_local,
    crop_annots_into_limits,
    crop_image_array,
    find_annots_repartition,
    get_channels_count,
    get_cropping_limits,
    make_annots_agnostic,
    merge_tif,
    save_annots_per_image,
)
from preprocessing.lidar import (
    create_full_lidar,
    download_and_remove_overlap_geotiles,
    download_lidar_names_shapefile,
    filter_full_lidar,
    get_lidar_files_from_image,
)
from preprocessing.rgb_cir import download_cir, download_rgb_image_from_polygon
from training import (
    TrainingMetrics,
    TreeDataset,
    evaluate_model,
    rgb_chm_usage_legend,
    rgb_chm_usage_postfix,
    train_and_validate,
)
from utils import (
    Folders,
    ImageData,
    create_all_folders,
    create_folder,
    create_random_temp_folder,
    import_tqdm,
    remove_folder,
)


tqdm = import_tqdm()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("action", help="Action to perform. Must be one of [train, eval]", type=str)
    parser.add_argument(
        "params_path", help="Path to the file containing the training parameters", type=str
    )

    return parser


def running_message(start_message: Optional[str] = None, end_message: Optional[str] = None):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Set default messages if not provided
            start_msg = (
                start_message if start_message is not None else f"Running {func.__name__}..."
            )
            start_msg = " -> " + start_msg
            end_msg = end_message if end_message is not None else "Done."

            # Print the start message
            print(start_msg, flush=True)

            # Execute the function
            result = func(*args, **kwargs)

            # Print the end message
            print(end_msg, flush=True)

            return result

        return wrapper

    return decorator


class DatasetParams:
    def __init__(
        self,
        annotations_file_name: str,
        use_rgb: bool,
        use_cir: bool,
        use_chm: bool,
        chm_z_layers: Sequence[Tuple[float, float]],
        agnostic: bool,
        resolution: float = 0.08,
        tile_size: int = 640,
        tile_overlap: int = 0,
        class_names: Dict[int, str] = DatasetConst.CLASS_NAMES.value,
        split_random_seed: int = 0,
        no_data_new_value: float = DatasetConst.NO_DATA_NEW_VALUE.value,
        filter_lidar: bool = True,
        mean_rgb_cir: np.ndarray | None = None,
        std_rgb_cir: np.ndarray | None = None,
        mean_chm: np.ndarray | None = None,
        std_chm: np.ndarray | None = None,
    ) -> None:
        self.annotations_file_name = annotations_file_name
        self.use_rgb = use_rgb
        self.use_cir = use_cir
        self.use_chm = use_chm
        self.chm_z_layers = chm_z_layers
        self.agnostic = agnostic
        self.resolution = resolution
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.class_names = class_names
        self.split_random_seed = split_random_seed
        self.no_data_new_value = no_data_new_value
        self.filter_lidar = filter_lidar
        self._mean_rgb_cir = mean_rgb_cir
        self._std_rgb_cir = std_rgb_cir
        self._mean_chm = mean_chm
        self._std_chm = std_chm

        if not self.use_chm:
            self.chm_z_layers = None

        if self.agnostic:
            self.class_names = {0: self.class_names[0]}

        self.class_indices = {value: key for key, value in self.class_names.items()}

    def initialize(self) -> None:
        if hasattr(self, "cropped_data_folder_path") and os.path.isdir(
            self.cropped_data_folder_path
        ):
            return

        full_images_paths, annotations = self._download_data()
        self._merge_and_crop_data(full_images_paths, annotations)

        if self.cropped_rgb_cir_folder_path is None:
            self.channels_rgb = 0
        else:
            self.channels_rgb = get_channels_count(self.cropped_rgb_cir_folder_path, chm=False)
        if self.cropped_chm_folder_path is None:
            self.channels_chm = 0
        else:
            self.channels_chm = get_channels_count(self.cropped_chm_folder_path, chm=True)

    def _download_data(self):
        create_all_folders()

        # Initialize the list of full images paths
        full_images_paths: Dict[str, List[str]] = defaultdict(lambda: [])

        # Get the annotations
        annotations_path = os.path.join(Folders.FULL_ANNOTS.value, self.annotations_file_name)
        annotations = open_geojson_feature_collection(annotations_path)

        # Get full image data
        self.full_image_path_tif = download_rgb_image_from_polygon(annotations["bbox"])[0]
        self.image_data = ImageData(self.full_image_path_tif)

        if self.use_rgb:
            # Store the folder paths
            full_images_paths["rgb_cir"].append(self.full_image_path_tif)

        if self.use_chm:
            # Download and pre-process LiDAR
            shapefile_path = download_lidar_names_shapefile()
            intersection_file_names = get_lidar_files_from_image(
                self.image_data, shapefile_path, DatasetConst.GEOTILES_OVERLAP.value
            )
            intersection_file_paths = download_and_remove_overlap_geotiles(
                intersection_file_names, DatasetConst.GEOTILES_OVERLAP.value
            )

            # Create full LiDAR files and filter if necessary
            full_lidar_path = create_full_lidar(intersection_file_paths, self.image_data)
            if self.filter_lidar:
                full_lidar_path = filter_full_lidar(self.image_data)

            # Compute the CHM slices
            if self.chm_z_layers is not None:
                full_chm_slices_paths = [
                    get_full_chm_slice_path(
                        self.image_data, self.resolution, self.filter_lidar, z_limits
                    )
                    for z_limits in self.chm_z_layers
                ]
                full_chm_slices_folders_paths = list(map(os.path.dirname, full_chm_slices_paths))
                for full_slice_folder_path in full_chm_slices_folders_paths:
                    create_folder(full_slice_folder_path)
                compute_slices_chm(
                    laz_file_name=full_lidar_path,
                    output_tif_paths=full_chm_slices_paths,
                    resolution=self.resolution,
                    z_limits_list=self.chm_z_layers,
                    skip_if_file_exists=True,
                )
            else:
                raise Exception("self.chm_z_layers should not be None.")

            # Store the folder paths
            full_images_paths["chm"].extend(full_chm_slices_paths)

        if self.use_cir:
            # Download CIR images
            cir_full_image_path = os.path.join(
                Folders.FULL_CIR_IMAGES.value, f"{self.image_data.base_name}.tif"
            )
            download_cir(
                image_coords_box=self.image_data.coord_box,
                resolution=self.resolution,
                skip_if_file_exists=True,
                save_path=cir_full_image_path,
            )

            # Store the folder path
            full_images_paths["rgb_cir"].append(cir_full_image_path)

        return full_images_paths, annotations

    def _init_mean_std(
        self,
        image_file_path_or_tensor: str | np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
        chm: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if type(mean) != type(std):
            raise ValueError(
                f"You should either specify mean and std or none of them. Here we have type(mean)={type(mean)} and type(std)={type(std)}."
            )

        per_channel = not chm
        replace_no_data = chm

        # Compute mean and std if they are missing
        if not isinstance(mean, np.ndarray) or not isinstance(std, np.ndarray):
            mean, std = compute_mean_and_std(
                image_file_path_or_tensor,
                per_channel=per_channel,
                replace_no_data=replace_no_data,
            )
        return mean, std

    def _merge_and_crop_data(
        self, full_images_paths: Dict[str, List[str]], annotations: geojson.FeatureCollection
    ):
        start_p_time = time.process_time()
        start_time = time.time()
        # Merge full images
        temp_folder = create_random_temp_folder()
        if self.use_rgb or self.use_cir:
            temp_path = os.path.join(temp_folder, "rgb_cir.npy")
            full_merged_rgb_cir = merge_tif(
                full_images_paths["rgb_cir"], temp_path=temp_path, output_path=None, chm=False
            )
        if self.use_chm:
            temp_path = os.path.join(temp_folder, "chm.npy")
            full_merged_chm = merge_tif(
                full_images_paths["chm"], temp_path=temp_path, output_path=None, chm=True
            )

        end_p_time = time.process_time()
        print(f"{'Merge P time':<20}: {end_p_time - start_p_time:.6f} seconds")
        start_p_time = time.process_time()
        end_time = time.time()
        print(f"{'Merge time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

        # Compute mean and std of the full images
        if self.use_rgb or self.use_cir:
            self.mean_rgb_cir, self.std_rgb_cir = self._init_mean_std(
                full_merged_rgb_cir, self._mean_rgb_cir, self._std_rgb_cir, chm=False
            )
            self._mean_rgb_cir, self._std_rgb_cir = self.mean_rgb_cir, self.std_rgb_cir
        if self.use_chm:
            self.mean_chm, self.std_chm = self._init_mean_std(
                full_merged_chm, self._mean_chm, self._std_chm, chm=True
            )
            self._mean_chm, self._std_chm = self.mean_chm, self.std_chm

        end_p_time = time.process_time()
        print(f"{'Mean and std P time':<20}: {end_p_time - start_p_time:.6f} seconds")
        start_p_time = time.process_time()
        end_time = time.time()
        print(f"{'Mean and std time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

        # Normalize the full images
        if self.use_rgb or self.use_cir:
            full_merged_rgb_cir = normalize(
                full_merged_rgb_cir,
                mean=self.mean_rgb_cir,
                std=self.std_rgb_cir,
                replace_no_data=False,
            )
        if self.use_chm:
            full_merged_chm = normalize(
                full_merged_chm,
                mean=self.mean_chm,
                std=self.std_chm,
                replace_no_data=True,
                no_data_new_value=self.no_data_new_value,
            )

        end_p_time = time.process_time()
        print(f"{'Normalize P time':<20}: {end_p_time - start_p_time:.6f} seconds")
        start_p_time = time.process_time()
        end_time = time.time()
        print(f"{'Normalize time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

        # Create the cropped data folder path
        self.cropped_data_folder_path = create_random_temp_folder()

        # Get tiles
        cropping_limits_x, cropping_limits_y = get_cropping_limits(
            self.full_image_path_tif, self.tile_size, self.tile_overlap
        )

        # Crop annotations into tiles
        visibility_threshold = 0.2
        annots_repartition = find_annots_repartition(
            cropping_limits_x, cropping_limits_y, annotations, self.image_data, visibility_threshold
        )
        crop_annots_into_limits(annots_repartition)
        annots_coordinates_to_local(annots_repartition)
        if self.agnostic:
            make_annots_agnostic(annots_repartition, self.class_names[0])

        # Save cropped annotations
        output_image_prefix = self.image_data.base_name
        self.cropped_annotations_folder_path = os.path.join(
            self.cropped_data_folder_path, "annotations", output_image_prefix
        )
        save_annots_per_image(
            annots_repartition,
            self.cropped_annotations_folder_path,
            self.full_image_path_tif,
            clear_if_not_empty=True,
        )

        end_p_time = time.process_time()
        print(f"{'Crop annots P time':<20}: {end_p_time - start_p_time:.6f} seconds")
        start_p_time = time.process_time()
        end_time = time.time()
        print(f"{'Crop annots time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

        # Crop RGB/CIR images
        if self.use_rgb or self.use_cir:
            self.cropped_rgb_cir_folder_path = os.path.join(
                self.cropped_data_folder_path, "rgb_cir", output_image_prefix
            )
            crop_image_array(
                self.cropped_annotations_folder_path,
                full_merged_rgb_cir,
                chm=False,
                output_folder_path=self.cropped_rgb_cir_folder_path,
                clear_if_not_empty=False,
                remove_unused=True,
            )
        else:
            self.cropped_rgb_cir_folder_path = None

        # Crop CHM images
        if self.use_chm:
            self.cropped_chm_folder_path = os.path.join(
                self.cropped_data_folder_path, "chm", output_image_prefix
            )
            crop_image_array(
                self.cropped_annotations_folder_path,
                full_merged_chm,
                chm=True,
                output_folder_path=self.cropped_chm_folder_path,
                clear_if_not_empty=False,
                remove_unused=True,
            )
        else:
            self.cropped_chm_folder_path = None

        end_p_time = time.process_time()
        print(f"{'Crop images P time':<20}: {end_p_time - start_p_time:.6f} seconds")
        start_p_time = time.process_time()
        end_time = time.time()
        print(f"{'Crop images time':<20}: {end_time - start_time:.6f} seconds")
        start_time = time.time()

        remove_folder(temp_folder)

    def close(self):
        remove_folder(self.cropped_data_folder_path)


class TrainingParams:
    def __init__(
        self,
        lr: float,
        epochs: int,
        batch_size: int,
        num_workers: int,
        accumulate: int,
        no_improvement_stop_epochs: int,
        proba_drop_rgb: float,
        proba_drop_chm: float,
        transform_spatial_training: A.Compose | None = None,
        transform_pixel_rgb_training: A.Compose | None = None,
        transform_pixel_chm_training: A.Compose | None = None,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accumulate = accumulate
        self.no_improvement_stop_epochs = no_improvement_stop_epochs
        self.proba_drop_rgb = proba_drop_rgb
        self.proba_drop_chm = proba_drop_chm
        self.transform_spatial_training = transform_spatial_training
        self.transform_pixel_rgb_training = transform_pixel_rgb_training
        self.transform_pixel_chm_training = transform_pixel_chm_training

    def initialize(self) -> None:
        pass


class TrainingData:
    @running_message(start_message="Initializing the training parameters...")
    def __init__(self, dataset_params: DatasetParams, training_params: TrainingParams) -> None:
        self.dataset_params = dataset_params
        self.training_params = training_params

    def initialize(self) -> None:
        if not self.dataset_params.use_rgb and not self.dataset_params.use_cir:
            self.training_params.proba_drop_rgb = 0.0
            self.training_params.proba_drop_chm = 0.0

        if not self.dataset_params.use_chm:
            self.training_params.proba_drop_rgb = 0.0
            self.training_params.proba_drop_chm = 0.0

        self.dataset_params.initialize()
        self.training_params.initialize()
        self._split_data(self.dataset_params.split_random_seed)
        self._init_transforms()

    def _split_data(self, split_random_seed: int):
        SETS_RATIOS = [3, 1, 1]
        SETS_NAMES = ["training", "validation", "test"]
        self.data_split_file_path = os.path.join(
            self.dataset_params.cropped_data_folder_path, f"data_split_{split_random_seed}.json"
        )
        create_and_save_splitted_datasets(
            self.dataset_params.cropped_rgb_cir_folder_path,
            self.dataset_params.cropped_chm_folder_path,
            self.dataset_params.cropped_annotations_folder_path,
            SETS_RATIOS,
            SETS_NAMES,
            self.data_split_file_path,
            random_seed=split_random_seed,
        )

    def _init_transforms(self):
        if self.training_params.transform_spatial_training is None:
            self.transform_spatial_training = get_transform_spatial(
                use_rgb=self.dataset_params.use_rgb or self.dataset_params.use_cir,
                use_chm=self.dataset_params.use_chm,
            )
        else:
            self.transform_spatial_training = self.training_params.transform_spatial_training

        self.transform_pixel_rgb_training = (
            self.training_params.transform_pixel_rgb_training
            or get_transform_pixel_rgb(self.dataset_params.channels_rgb)
        )

        self.transform_pixel_chm_training = (
            self.training_params.transform_pixel_chm_training
            or get_transform_pixel_chm(self.dataset_params.channels_chm)
        )


class ModelSession:
    def __init__(
        self,
        training_data: TrainingData,
        device: torch.device,
        postfix: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.training_data = training_data
        self.device = device
        self.postfix = postfix

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = AMF_GD_YOLOv8.get_new_name(
                self.training_data.training_params.epochs, self.postfix
            )
        self.model_path = AMF_GD_YOLOv8.get_weights_path_from_name(self.model_name)
        self.best_epoch = -1

    @running_message("Creating the dataset...")
    def initialize(self) -> Tuple[Dict[str, TreeDataset], AMF_GD_YOLOv8]:
        self.training_data.initialize()
        datasets = self._load_datasets()
        model = self._load_model()
        return datasets, model

    @running_message("Loading the model...")
    def _load_model(self) -> AMF_GD_YOLOv8:
        model = AMF_GD_YOLOv8(
            self.training_data.dataset_params.channels_rgb,
            self.training_data.dataset_params.channels_chm,
            class_names=self.training_data.dataset_params.class_names,
            device=self.device,
            name=self.model_name,
            scale="n",
        )
        if os.path.isfile(self.model_path):
            print("Loading the weights...")
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Done")

        return model

    @running_message("Loading the dataset...")
    def _load_datasets(self) -> Dict[str, TreeDataset]:
        if self.training_data.dataset_params.agnostic:
            only_label = self.training_data.dataset_params.class_names[0]
            labels_transformation_drop_rgb = {only_label: only_label}
            labels_transformation_drop_chm = {only_label: only_label}
        else:
            labels_transformation_drop_rgb = DatasetConst.LABELS_TRANSFORMATION_DROP_RGB.value
            labels_transformation_drop_chm = DatasetConst.LABELS_TRANSFORMATION_DROP_CHM.value

        datasets = load_tree_datasets_from_split(
            self.training_data.data_split_file_path,
            labels_to_index=self.training_data.dataset_params.class_indices,
            proba_drop_rgb=self.training_data.training_params.proba_drop_rgb,
            labels_transformation_drop_rgb=labels_transformation_drop_rgb,
            proba_drop_chm=self.training_data.training_params.proba_drop_chm,
            labels_transformation_drop_chm=labels_transformation_drop_chm,
            dismissed_classes=[],
            transform_spatial_training=self.training_data.training_params.transform_spatial_training,
            transform_pixel_rgb_training=self.training_data.training_params.transform_pixel_rgb_training,
            transform_pixel_chm_training=self.training_data.training_params.transform_pixel_chm_training,
            no_data_new_value=DatasetConst.NO_DATA_NEW_VALUE.value,
        )
        return datasets

    @running_message("Running a training session...")
    def train(self, overwrite: bool = False):
        # Check if a model with this name already exists
        if os.path.isfile(self.model_path):
            if not overwrite:
                raise ValueError(
                    f"There is already a model at {self.model_path}. Specify overwrite=True in the train function to overwrite it."
                )

        # This line helps to avoid too much unused space on GPU
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.6,max_split_size_mb:512"
        )

        # Removes a warning
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")

        # Load data and model
        datasets, model = self.initialize()

        # Train and extract the best model
        model, self.best_epoch = train_and_validate(
            model=model,
            datasets=datasets,
            lr=self.training_data.training_params.lr,
            epochs=self.training_data.training_params.epochs,
            batch_size=self.training_data.training_params.batch_size,
            num_workers=self.training_data.training_params.num_workers,
            accumulate=self.training_data.training_params.accumulate,
            device=self.device,
            show_training_metrics=False,
            no_improvement_stop_epochs=self.training_data.training_params.no_improvement_stop_epochs,
        )

        # Save the best model
        self.save_model(model)
        self.save_params()

        self.compute_metrics()

    @running_message("Computing metrics...")
    def compute_metrics(self):
        datasets, model = self.initialize()

        train_loader, val_loader, test_loader = initialize_dataloaders(
            datasets=datasets,
            batch_size=self.training_data.training_params.batch_size,
            num_workers=self.training_data.training_params.num_workers,
        )

        model_folder_path = AMF_GD_YOLOv8.get_folder_path_from_name(model.name)

        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()

        loaders = [train_loader, val_loader, test_loader]
        loaders_postfix = ["train-set", "val-set", "test-set"]
        loaders_legend = ["Training set", "Validation set", "Test set"]
        loaders_zip = list(zip(loaders, loaders_postfix, loaders_legend))

        predictions_folder = os.path.join(model_folder_path, "predictions")
        ap_iou_folder = os.path.join(model_folder_path, "ap_iou")
        sap_conf_folder = os.path.join(model_folder_path, "sap_conf")
        for folder in [predictions_folder, ap_iou_folder, sap_conf_folder]:
            create_folder(folder)

        use_rgb_chm = [(True, True), (True, False), (False, True)]

        for loader, loader_postfix, loader_legend in tqdm(loaders_zip, desc="Datasets"):
            ap_metrics_list = AP_Metrics_List()
            for use_rgb, use_chm in tqdm(use_rgb_chm, desc="Type of input", leave=False):

                data_postfix = rgb_chm_usage_postfix(use_rgb=use_rgb, use_chm=use_chm)
                data_legend = rgb_chm_usage_legend(use_rgb=use_rgb, use_chm=use_chm)
                full_postfix = "_".join([loader_postfix, data_postfix])
                full_legend = " with ".join([loader_postfix, data_postfix])
                ap_metrics = evaluate_model(
                    model,
                    loader,
                    self.device,
                    use_rgb=use_rgb,
                    use_chm=use_chm,
                    ap_conf_thresholds=conf_thresholds,
                    output_geojson_save_path=os.path.join(
                        model_folder_path, "predictions", f"predictions_{full_postfix}.geojson"
                    ),
                )

                ap_metrics.plot_ap_iou_per_label(
                    save_path=os.path.join(
                        model_folder_path, "ap_iou", f"ap_iou_per_label_{full_postfix}.png"
                    ),
                    title=f"Sorted AP curve per class on the {full_legend}",
                )
                ap_metrics.plot_sap_conf_per_label(
                    save_path=os.path.join(
                        model_folder_path, "sap_conf", f"sap_conf_per_label_{full_postfix}.png"
                    ),
                    title=f"Sorted AP w.r.t the confidence threshold per class on the {full_legend}",
                )

                ap_metrics_list.add_ap_metrics(ap_metrics, legend=data_legend)

            ap_metrics_list.plot_ap_iou(
                save_path=os.path.join(model_folder_path, "ap_iou", f"ap_iou_{loader_postfix}.png"),
                title=f"Sorted AP curve on the {loader_legend}",
            )
            ap_metrics_list.plot_sap_conf(
                save_path=os.path.join(
                    model_folder_path, "sap_conf", f"sap_conf_{loader_postfix}.png"
                ),
                title=f"Sorted AP w.r.t the confidence threshold on the {loader_legend}",
            )

    def save_params(self):
        params_to_save = {
            "annotations_file": self.training_data.dataset_params.annotations_file_name,
            "agnostic": self.training_data.dataset_params.agnostic,
            "use_rgb": self.training_data.dataset_params.use_rgb,
            "use_cir": self.training_data.dataset_params.use_cir,
            "use_chm": self.training_data.dataset_params.use_chm,
            "chm_z_layers": self.training_data.dataset_params.chm_z_layers,
            "class_names": self.training_data.dataset_params.class_names,
            "filter_lidar": self.training_data.dataset_params.filter_lidar,
            "mean_rgb_cir": self.training_data.dataset_params._mean_rgb_cir,
            "mean_chm": self.training_data.dataset_params._mean_chm,
            "std_rgb_cir": self.training_data.dataset_params._std_rgb_cir,
            "std_chm": self.training_data.dataset_params._std_chm,
            "no_data_new_value": self.training_data.dataset_params.no_data_new_value,
            "resolution": self.training_data.dataset_params.resolution,
            "split_random_seed": self.training_data.dataset_params.split_random_seed,
            "tile_overlap": self.training_data.dataset_params.tile_overlap,
            "tile_size": self.training_data.dataset_params.tile_size,
            "accumulate": self.training_data.training_params.accumulate,
            "batch_size": self.training_data.training_params.batch_size,
            "epochs": self.training_data.training_params.epochs,
            "lr": self.training_data.training_params.lr,
            "no_improvement_stop_epochs": self.training_data.training_params.no_improvement_stop_epochs,
            "num_workers": self.training_data.training_params.num_workers,
            "proba_drop_chm": self.training_data.training_params.proba_drop_chm,
            "proba_drop_rgb": self.training_data.training_params.proba_drop_rgb,
            "best_epoch": self.best_epoch,
            "device": self.device.type,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "postfix": self.postfix,
        }
        save_path = ModelSession.get_params_path(self.model_name)
        with open(save_path, "w") as fp:
            json.dump(params_to_save, fp, cls=FullJsonEncoder, sort_keys=True, indent=4)

    def save_model(self, model: AMF_GD_YOLOv8):
        # Save the weights
        model.save_weights()

        # Save the class instance
        pickle_path = ModelSession.get_pickle_path(self.model_name)
        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

    def close(self):
        self.training_data.dataset_params.close()

    @staticmethod
    def get_params_path(model_name: str) -> str:
        model_folder_path = AMF_GD_YOLOv8.get_folder_path_from_name(model_name)
        return os.path.join(model_folder_path, "model_params.json")

    @staticmethod
    def get_pickle_path(model_name: str) -> str:
        model_folder_path = AMF_GD_YOLOv8.get_folder_path_from_name(model_name)
        return os.path.join(model_folder_path, "model_session.pkl")

    @staticmethod
    def already_exists(model_name: str) -> bool:
        pickle_path = ModelSession.get_pickle_path(model_name)
        return os.path.exists(pickle_path)

    @staticmethod
    def from_pickle(file_path: str, device: torch.device) -> ModelSession:
        with open(file_path, "rb") as f:
            model_session = pickle.load(f)
            model_session.device = device
        return model_session

    @staticmethod
    def from_name(
        model_name: str,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> ModelSession:
        if not ModelSession.already_exists(model_name):
            raise Exception(f"There is no model called {model_name}.")
        file_path = ModelSession.get_pickle_path(model_name)
        return ModelSession.from_pickle(file_path, device)


class FullJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def simple_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMF_GD_YOLOv8(
        3,
        1,
        class_names=DatasetConst.CLASS_NAMES.value,
        device=device,
        name="simple_test",
        scale="n",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: 1 / np.sqrt(i + 2), last_epoch=-1
    )
    training_metrics = TrainingMetrics(show=False)
    intervals: List[Tuple[int, int]] = [(0, 0)]

    image_rgb_path = os.path.join(
        Folders.RGB_IMAGES.value, "cropped", "2023_122000_484000_RGB_hrl", "0_0_640_640.tif"
    )
    image_chm_path = os.path.join(
        Folders.CHM.value,
        "8cm",
        "filtered",
        "-inf_inf",
        "cropped",
        "122000_484000",
        "0_0_640_640.tif",
    )
    gt_bboxes = torch.tensor([[128.8, 217.4, 271.0, 394.7]]).to(torch.float32).to(device)
    gt_classes = torch.tensor([1]).to(torch.float32).to(device)
    gt_indices = torch.tensor([0]).to(torch.float32).to(device)
    image_indices = torch.tensor([0]).to(device)

    image_rgb_initial = tifffile.imread(image_rgb_path).astype(np.uint8)
    if len(image_rgb_initial.shape) == 2:
        image_rgb_initial = image_rgb_initial[..., np.newaxis]
    image_rgb = image_rgb_initial.astype(np.float32)
    image_rgb = torch.from_numpy(image_rgb).permute((2, 0, 1)).unsqueeze(0).to(device)

    image_chm_initial = tifffile.imread(image_chm_path).astype(np.float32)
    if len(image_chm_initial.shape) == 2:
        image_chm_initial = image_chm_initial[..., np.newaxis]
    image_chm = image_chm_initial.astype(np.float32)
    image_chm = torch.from_numpy(image_chm).permute((2, 0, 1)).unsqueeze(0).to(device)

    ap_metrics_list = AP_Metrics_List()

    epochs = 10
    for epoch in range(1, epochs + 1):
        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()
        ap_metrics = AP_Metrics(conf_threshold_list=conf_thresholds)

        # training_metrics.visualize(
        #     intervals=intervals, save_paths=["Simple_Test_training_plot.png"]
        # )
        training_metrics.save_metrics(
            os.path.join(
                model.folder_path,
                "Simple_Test_metrics_values.json",
            )
        )
        output = model.forward(image_rgb, image_chm, device)
        total_loss, loss_dict = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)
        total_loss.backward()

        with torch.no_grad():
            preds = model.preds_from_output(output)
            ap_metrics.add_preds(
                model=model,
                preds=preds,
                gt_bboxes=gt_bboxes,
                gt_classes=gt_classes,
                gt_indices=gt_indices,
                image_indices=image_indices,
            )

            if epoch % 5 == 0:
                dataset_idx = 0
                if dataset_idx in image_indices.tolist():
                    batch_idx = image_indices.tolist().index(dataset_idx)

                    bboxes_per_image, confs_per_image, classes_per_image = model.predict_from_preds(
                        preds[batch_idx : batch_idx + 1],
                        iou_threshold=0.5,
                        conf_threshold=0.1,
                        number_best=40,
                    )
                    gt_bboxes_per_image, gt_classes_per_image = convert_ground_truth_from_tensors(
                        gt_bboxes=gt_bboxes,
                        gt_classes=gt_classes,
                        gt_indices=gt_indices,
                        image_indices=image_indices,
                    )

                    image_rgb_initial_torch = torch.tensor(image_rgb_initial).permute((2, 0, 1))
                    image_chm_initial_torch = torch.tensor(image_chm_initial).permute((2, 0, 1))

                    create_bboxes_training_image(
                        image_rgb=image_rgb_initial_torch,
                        image_chm=image_chm_initial_torch,
                        pred_bboxes=bboxes_per_image[0],
                        pred_labels=classes_per_image[0],
                        pred_scores=confs_per_image[0],
                        gt_bboxes=gt_bboxes_per_image[batch_idx],
                        gt_labels=gt_classes_per_image[batch_idx],
                        labels_int_to_str=model.class_names,
                        colors_dict=DatasetConst.CLASS_COLORS.value,
                        save_path=f"Data_epoch_{epoch}_train.png",
                    )

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if epoch % 10 == 0:
            ap_metrics_list.add_ap_metrics(ap_metrics=ap_metrics, legend=f"{epoch = }")

        batch_size = image_rgb.shape[0]
        training_metrics.update(
            "Training", "Total Loss", total_loss.item(), count=batch_size, y_axis="Loss"
        )
        for key, value in loss_dict.items():
            training_metrics.update("Training", key, value.item(), count=batch_size, y_axis="Loss")

        _, _, sorted_ap, conf_threshold = ap_metrics.get_best_sorted_ap()
        training_metrics.update("Training", "Best sortedAP", sorted_ap, y_axis="sortedAP")
        training_metrics.update(
            "Training", "Conf thres of sortedAP", conf_threshold, y_axis="Conf threshold"
        )

        training_metrics.end_loop(epoch)

    # training_metrics.visualize(
    #     intervals=intervals, save_paths=["Simple_Test_training_plot.png"]
    # )
    training_metrics.save_metrics(
        os.path.join(
            model.folder_path,
            "Simple_Test_metrics_values.json",
        )
    )

    ap_metrics_list.plot_ap_iou(
        save_path="ap_iou.png",
        title="Sorted AP curve",
    )
    ap_metrics_list.plot_sap_conf(
        save_path="sap_conf.png",
        title="Sorted AP w.r.t the confidence threshold",
    )
