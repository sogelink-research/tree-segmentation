from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch

from augmentations import (
    get_transform_pixel_chm,
    get_transform_pixel_rgb,
    get_transform_spatial,
)
from dataset_constants import DatasetConst
from datasets import compute_mean_and_std
from geojson_conversions import open_geojson_feature_collection
from layers import AMF_GD_YOLOv8
from metrics import plot_sorted_ap, plot_sorted_ap_confs
from preprocessing.data import get_channels_count
from preprocessing.rgb_cir import get_rgb_images_paths_from_polygon
from training import (
    TreeDataset,
    compute_all_ap_metrics,
    create_and_save_splitted_datasets,
    initialize_dataloaders,
    load_tree_datasets_from_split,
    predict_to_geojson,
    train_and_validate,
)
from utils import Folders, ImageData, import_tqdm


tqdm = import_tqdm()


RESOLUTION = 0.08


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
        class_names: Dict[int, str] = DatasetConst.CLASS_NAMES.value,
        split_random_seed: int = 0,
        no_data_new_value: float = DatasetConst.NO_DATA_NEW_VALUE.value,
    ) -> None:
        self.annotations_file_name = annotations_file_name
        self.use_rgb = use_rgb
        self.use_cir = use_cir
        self.use_chm = use_chm
        self.chm_z_layers = chm_z_layers
        self.class_names = class_names
        self.split_random_seed = split_random_seed
        self.no_data_new_value = no_data_new_value

        self.class_indices = {value: key for key, value in self.class_names.items()}

        self._init_data_paths()

    def _init_data_paths(self):
        annotations_path = os.path.join(Folders.FULL_ANNOTS.value, self.annotations_file_name)
        annotations = open_geojson_feature_collection(annotations_path)
        full_image_path_tif = get_rgb_images_paths_from_polygon(annotations["bbox"])[0]
        self.image_data = ImageData(full_image_path_tif)
        self.annotations_folder_path = os.path.join(
            Folders.CROPPED_ANNOTS.value, self.image_data.base_name
        )
        self.rgb_cir_folder_path = os.path.join(
            Folders.IMAGES.value, "merged", "cropped", self.image_data.base_name
        )
        self.chm_folder_path = os.path.join(
            Folders.CHM.value,
            f"{round(RESOLUTION*100)}cm",
            "filtered",
            "merged",
            "cropped",
            self.image_data.coord_name,
        )
        self.channels_rgb = get_channels_count(self.rgb_cir_folder_path)
        self.channels_chm = get_channels_count(self.chm_folder_path)


class TrainingParams:
    def __init__(
        self,
        lr: float,
        epochs: int,
        batch_size: int,
        num_workers: int,
        accumulate: int,
        proba_drop_rgb: float,
        proba_drop_chm: float,
        transform_spatial_training: A.Compose | None = None,
        transform_pixel_rgb_training: A.Compose | None = None,
        transform_pixel_chm_training: A.Compose | None = None,
        mean_rgb: torch.Tensor | None = None,
        std_rgb: torch.Tensor | None = None,
        mean_chm: torch.Tensor | None = None,
        std_chm: torch.Tensor | None = None,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accumulate = accumulate
        self.proba_drop_rgb = proba_drop_rgb
        self.proba_drop_chm = proba_drop_chm
        self.transform_spatial_training = transform_spatial_training
        self.transform_pixel_rgb_training = transform_pixel_rgb_training
        self.transform_pixel_chm_training = transform_pixel_chm_training
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_chm = mean_chm
        self.std_chm = std_chm


class TrainingData:
    @running_message(start_message="Initializing the training parameters...")
    def __init__(self, dataset_params: DatasetParams, training_params: TrainingParams) -> None:
        self.dataset_params = dataset_params
        self.training_params = training_params

        self._split_data(self.dataset_params.split_random_seed)
        self._init_mean_std(
            training_params.mean_rgb,
            training_params.std_rgb,
            training_params.mean_chm,
            training_params.std_chm,
        )
        self._init_transforms()

    def _split_data(self, split_random_seed: int):
        SETS_RATIOS = [3, 1, 1]
        SETS_NAMES = ["training", "validation", "test"]
        self.data_split_file_path = os.path.join(
            Folders.OTHERS_DIR.value, f"data_split_{split_random_seed}.json"
        )
        create_and_save_splitted_datasets(
            self.dataset_params.rgb_cir_folder_path,
            self.dataset_params.chm_folder_path,
            self.dataset_params.annotations_folder_path,
            SETS_RATIOS,
            SETS_NAMES,
            self.data_split_file_path,
            random_seed=split_random_seed,
        )

    def _init_mean_std(
        self,
        mean_rgb: torch.Tensor | None,
        std_rgb: torch.Tensor | None,
        mean_chm: torch.Tensor | None,
        std_chm: torch.Tensor | None,
    ):
        if type(mean_rgb) != type(std_rgb):
            raise ValueError(
                f"You should either specify mean_rgb and std_rgb or none of them. Here we have mean_rgb={mean_rgb} and std_rgb={std_rgb}"
            )
        if type(mean_chm) != type(std_chm):
            raise ValueError(
                f"You should either specify mean_chm and std_chm or none of them. Here we have mean_chm={mean_chm} and std_chm={std_chm}"
            )

        # Compute RGB mean and std if an of them is missing
        if mean_rgb is None or std_rgb is None:
            self.mean_rgb, self.std_rgb = compute_mean_and_std(
                self.dataset_params.rgb_cir_folder_path, per_channel=True, replace_no_data=False
            )
        else:
            self.mean_rgb, self.std_rgb = mean_rgb, std_rgb

        # Compute CHM mean and std if an of them is missing
        if mean_chm is None or std_chm is None:
            self.mean_chm, self.std_chm = compute_mean_and_std(
                self.dataset_params.chm_folder_path,
                per_channel=True,
                replace_no_data=True,
                no_data_new_value=self.dataset_params.no_data_new_value,
            )
        else:
            self.mean_chm = mean_chm
            self.std_chm = std_chm

    def _init_transforms(self):
        self.transform_spatial_training = (
            self.training_params.transform_spatial_training or get_transform_spatial()
        )

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
            self.model_path = AMF_GD_YOLOv8.get_model_path_from_name(self.model_name)
        else:
            self.model_name, self.model_path = AMF_GD_YOLOv8.get_new_model_name_and_path(
                self.training_data.training_params.epochs, self.postfix
            )

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

    @running_message("Loading the datasets...")
    def _load_datasets(self) -> Dict[str, TreeDataset]:
        datasets = load_tree_datasets_from_split(
            self.training_data.data_split_file_path,
            labels_to_index=self.training_data.dataset_params.class_indices,
            mean_rgb=self.training_data.mean_rgb,
            std_rgb=self.training_data.std_rgb,
            mean_chm=self.training_data.mean_chm,
            std_chm=self.training_data.std_chm,
            proba_drop_rgb=self.training_data.training_params.proba_drop_rgb,
            labels_transformation_drop_rgb=DatasetConst.LABELS_TRANSFORMATION_DROP_RGB.value,
            proba_drop_chm=self.training_data.training_params.proba_drop_chm,
            labels_transformation_drop_chm=DatasetConst.LABELS_TRANSFORMATION_DROP_CHM.value,
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
        model = self._load_model()
        datasets = self._load_datasets()

        # Train and extract the best model
        model = train_and_validate(
            model=model,
            datasets=datasets,
            lr=self.training_data.training_params.lr,
            epochs=self.training_data.training_params.epochs,
            batch_size=self.training_data.training_params.batch_size,
            num_workers=self.training_data.training_params.num_workers,
            accumulate=self.training_data.training_params.accumulate,
            device=self.device,
            show_training_metrics=False,
            model_save_path=self.model_path,
        )

        # # Save the best model
        # self._save_model(model)

        # self.compute_metrics()

    @running_message("Computing metrics...")
    def compute_metrics(self):
        # TODO: Modify this function to use the last metrics functions

        model = self._load_model()
        datasets = self._load_datasets()

        _, _, test_loader = initialize_dataloaders(
            datasets=datasets,
            batch_size=self.training_data.training_params.batch_size,
            num_workers=self.training_data.training_params.num_workers,
        )

        best_sorted_ious_list = []
        best_aps_list = []
        best_sorted_ap_list = []
        best_conf_threshold__list = []

        sorted_ap_lists = []
        conf_thresholds_list = []

        legend_list = []

        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()

        use_rgbs = [False, False, True, True]
        use_chms = [False, True, False, True]
        test_names = ["all", "use_chm", "use_rgb", "use_chm_use_rgb"]

        pbar = tqdm(zip(use_rgbs, use_chms, test_names), total=len(use_rgbs))
        for use_rgb, use_chm, test_name in pbar:
            if use_rgb:
                if use_chm:
                    legend = "No data"
                else:
                    legend = "CHM"
            else:
                if use_chm:
                    legend = "RGB"
                else:
                    legend = "RGB and CHM"
            pbar.set_description(legend)
            pbar.refresh()
            predict_to_geojson(
                model,
                test_loader,
                self.device,
                use_rgb=use_rgb,
                use_chm=use_chm,
                save_path=os.path.join(
                    Folders.OUTPUT_DIR.value, f"{self.model_name}_{test_name}.geojson"
                ),
            )
            (
                sorted_ious_list,
                aps_list,
                sorted_ap_list,
            ) = compute_all_ap_metrics(
                model,
                test_loader,
                self.device,
                conf_thresholds=conf_thresholds,
                use_rgb=use_rgb,
                use_chm=use_chm,
            )

            best_index = sorted_ap_list.index(max(sorted_ap_list))
            best_sorted_ious = sorted_ious_list[best_index]
            best_aps = aps_list[best_index]
            best_sorted_ap = sorted_ap_list[best_index]
            best_conf_threshold = conf_thresholds[best_index]

            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_conf_threshold__list.append(best_conf_threshold)

            sorted_ap_lists.append(sorted_ap_list)
            conf_thresholds_list.append(conf_thresholds)

            legend_list.append(legend)

        plot_sorted_ap(
            best_sorted_ious_list,
            best_aps_list,
            best_sorted_ap_list,
            conf_thresholds=best_conf_threshold__list,
            legend_list=legend_list,
            show=True,
            save_path=os.path.join(Folders.OUTPUT_DIR.value, f"{self.model_name}_ap_iou.png"),
        )

        plot_sorted_ap_confs(
            sorted_ap_lists=sorted_ap_lists,
            conf_thresholds_list=conf_thresholds_list,
            legend_list=legend_list,
            show=True,
            save_path=os.path.join(Folders.OUTPUT_DIR.value, f"{self.model_name}_sap_conf.png"),
        )

    @staticmethod
    def _pickle_path(model_name: str) -> str:
        return os.path.join(Folders.OUTPUT_DIR.value, f"{model_name}.pkl")

    def _save_model(self, model: AMF_GD_YOLOv8):
        # Save the weights
        state_dict = model.state_dict()
        torch.save(state_dict, self.model_path)

        # Save the class instance
        pickle_path = ModelSession._pickle_path(self.model_name)
        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file_path: str, device: torch.device) -> ModelSession:
        with open(file_path, "rb") as f:
            model_session = pickle.load(f)
            model_session.device = device
        return model_session

    @staticmethod
    def from_name(model_name: str, device: torch.device) -> ModelSession:
        file_path = ModelSession._pickle_path(model_name)
        return ModelSession.from_pickle(file_path, device)


def main():
    # Data parameters
    annotations_file_name = "122000_484000.geojson"
    z_tops: Sequence[float] = [1, 2, 3, 5, 7, 10, 15, 20, np.inf]
    z_limits_list = [(-np.inf, z_top) for z_top in z_tops]

    dataset_params = DatasetParams(
        annotations_file_name=annotations_file_name,
        use_rgb=True,
        use_cir=True,
        use_chm=True,
        chm_z_layers=z_limits_list,
    )

    # Training parameters

    lr = 1e-2
    epochs = 1000
    batch_size = 10
    num_workers = 0
    accumulate = 10

    proba_drop_rgb = 1 / 3
    proba_drop_chm = 1 / 3

    postfix = "rgb_cir_multi_chm"

    training_params = TrainingParams(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        accumulate=accumulate,
        proba_drop_rgb=proba_drop_rgb,
        proba_drop_chm=proba_drop_chm,
    )

    # Training data
    training_data = TrainingData(dataset_params=dataset_params, training_params=training_params)

    # Training session
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_session = ModelSession(training_data=training_data, device=device, postfix=postfix)

    model_session.train()


if __name__ == "__main__":
    main()
