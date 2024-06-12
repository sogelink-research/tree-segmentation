import multiprocessing as mp
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch

from dataset_constants import DatasetConst
from geojson_conversions import open_geojson_feature_collection
from layers import AMF_GD_YOLOv8
from metrics import plot_sorted_ap, plot_sorted_ap_confs
from preprocessing.rgb_cir import get_rgb_images_paths_from_polygon
from training import (
    compute_mean_and_std,
    compute_metrics,
    create_and_save_splitted_datasets,
    initialize_dataloaders,
    load_tree_datasets_from_split,
    test_save_output_image,
    train_and_validate,
)
from utils import Folders, ImageData, import_tqdm


tqdm = import_tqdm()


RESOLUTION = 0.08
CROP_SIZE = 640


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
        transform_spatial_training: A.Compose | None,
        transform_pixel_rgb_training: A.Compose | None,
        transform_pixel_chm_training: A.Compose | None,
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
    @running_message(start_message="Loading the dataset...")
    def __init__(self, dataset_params: DatasetParams, training_params: TrainingParams) -> None:
        self.dataset_params = dataset_params
        self.training_params = training_params

        annotations_path = os.path.join(
            Folders.FULL_ANNOTS.value, dataset_params.annotations_file_name
        )
        annotations = open_geojson_feature_collection(annotations_path)
        full_image_path_tif = get_rgb_images_paths_from_polygon(annotations["bbox"])[0]
        self.image_data = ImageData(full_image_path_tif)

        self._init_folders_paths()
        self._split_data(dataset_params.split_random_seed)
        self._init_mean_std(
            training_params.mean_rgb,
            training_params.std_rgb,
            training_params.mean_chm,
            training_params.std_chm,
        )
        self._init_datasets(training_params.proba_drop_rgb, training_params.proba_drop_chm)

    def _init_folders_paths(self):
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

    def _split_data(self, split_random_seed: int):
        SETS_RATIOS = [3, 1, 1]
        SETS_NAMES = ["training", "validation", "test"]
        self.data_split_file_path = os.path.join(Folders.OTHERS_DIR.value, "data_split.json")
        create_and_save_splitted_datasets(
            self.rgb_cir_folder_path,
            self.chm_folder_path,
            self.annotations_folder_path,
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
                self.rgb_cir_folder_path, per_channel=True, replace_no_data=False
            )
        else:
            self.mean_rgb, self.std_rgb = mean_rgb, std_rgb

        # Compute CHM mean and std if an of them is missing
        if mean_chm is None or std_chm is None:
            self.mean_chm, self.std_chm = compute_mean_and_std(
                self.chm_folder_path,
                per_channel=True,
                replace_no_data=True,
                no_data_new_value=self.dataset_params.no_data_new_value,
            )
        else:
            self.mean_chm = mean_chm
            self.std_chm = std_chm

    def _init_datasets(self, proba_drop_rgb: float, proba_drop_chm: float):
        self.datasets = load_tree_datasets_from_split(
            self.data_split_file_path,
            labels_to_index=self.dataset_params.class_indices,
            mean_rgb=self.mean_rgb,
            std_rgb=self.std_rgb,
            mean_chm=self.mean_chm,
            std_chm=self.std_chm,
            proba_drop_rgb=proba_drop_rgb,
            labels_transformation_drop_rgb=DatasetConst.LABELS_TRANSFORMATION_DROP_RGB.value,
            proba_drop_chm=proba_drop_chm,
            labels_transformation_drop_chm=DatasetConst.LABELS_TRANSFORMATION_DROP_CHM.value,
            dismissed_classes=[],
            transform_spatial_training=self.training_params.transform_spatial_training,
            transform_pixel_rgb_training=self.training_params.transform_pixel_rgb_training,
            transform_pixel_chm_training=self.training_params.transform_pixel_chm_training,
            no_data_new_value=DatasetConst.NO_DATA_NEW_VALUE.value,
        )


class TrainingSession:
    def __init__(self, training_data: TrainingData, device: torch.device, postfix: str) -> None:
        self.training_data = training_data
        self.device = device
        self.postfix = postfix

    @running_message("Running a training session...")
    def run(self):
        # This line helps to avoid too much unused space on GPU
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.6,max_split_size_mb:512"
        )

        self.model_name, self.model_path = AMF_GD_YOLOv8.get_new_model_name_and_path(
            self.training_data.training_params.epochs, self.postfix
        )

        self.model = AMF_GD_YOLOv8(
            self.training_data.datasets["training"].rgb_channels,
            self.training_data.datasets["training"].chm_channels,
            device=self.device,
            scale="n",
            class_names=self.training_data.dataset_params.class_names,
            name=self.model_name,
        )

        final_model = train_and_validate(
            model=self.model,
            datasets=self.training_data.datasets,
            lr=self.training_data.training_params.lr,
            epochs=self.training_data.training_params.epochs,
            batch_size=self.training_data.training_params.batch_size,
            num_workers=self.training_data.training_params.num_workers,
            accumulate=self.training_data.training_params.accumulate,
            device=self.device,
            save_outputs=False,
            show_training_metrics=False,
        )

        state_dict = final_model.state_dict()
        torch.save(state_dict, self.model_path)

        self._compute_metrics()

    def _compute_metrics(self):
        _, _, test_loader = initialize_dataloaders(
            datasets=self.training_data.datasets,
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

        no_rgbs = [False, False, True, True]
        no_chms = [False, True, False, True]
        test_names = ["all", "no_chm", "no_rgb", "no_chm_no_rgb"]

        pbar = tqdm(zip(no_rgbs, no_chms, test_names), total=len(no_rgbs))
        for no_rgb, no_chm, test_name in pbar:
            if no_rgb:
                if no_chm:
                    legend = "No data"
                else:
                    legend = "CHM"
            else:
                if no_chm:
                    legend = "RGB"
                else:
                    legend = "RGB and CHM"
            pbar.set_description(legend)
            pbar.refresh()
            test_save_output_image(
                self.model,
                test_loader,
                -1,
                self.device,
                no_rgb=no_rgb,
                no_chm=no_chm,
                save_path=os.path.join(
                    Folders.OUTPUT_DIR.value, f"{self.model_name}_{test_name}.geojson"
                ),
            )
            (
                best_sorted_ious,
                best_aps,
                best_sorted_ap,
                best_conf_threshold,
                sorted_ious_list,
                aps_list,
                sorted_ap_list_2,
            ) = compute_metrics(
                self.model,
                test_loader,
                self.device,
                conf_thresholds=conf_thresholds,
                no_rgb=no_rgb,
                no_chm=no_chm,
                save_path_ap_iou=os.path.join(
                    Folders.OUTPUT_DIR.value, f"{self.model_name}_ap_iou_{test_name}.png"
                ),
                save_path_sap_conf=os.path.join(
                    Folders.OUTPUT_DIR.value, f"{self.model_name}_sap_conf_{test_name}.png"
                ),
            )

            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_conf_threshold__list.append(best_conf_threshold)

            sorted_ap_lists.append(sorted_ap_list_2)
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
    transform_spatial = A.Compose(
        [
            A.RandomCrop(width=CROP_SIZE, height=CROP_SIZE, p=1.0),
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
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0.2, label_fields=["class_labels"]
        ),
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

    lr = 1e-2
    epochs = 2000
    batch_size = 1
    num_workers = mp.cpu_count() // 2
    accumulate = 12

    proba_drop_rgb = 1 / 3
    proba_drop_chm = 1 / 3

    training_params = TrainingParams(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        accumulate=accumulate,
        proba_drop_rgb=proba_drop_rgb,
        proba_drop_chm=proba_drop_chm,
        transform_spatial_training=transform_spatial,
        transform_pixel_rgb_training=transform_pixel_rgb,
        transform_pixel_chm_training=transform_pixel_chm,
    )

    # Training data
    training_data = TrainingData(dataset_params=dataset_params, training_params=training_params)

    # Training session
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    postfix = "rgb_cir_multi_chm"
    training_session = TrainingSession(training_data=training_data, device=device, postfix=postfix)

    training_session.run()


if __name__ == "__main__":

    # @running_message("Starting example_function", "Finished example_function")
    # def example_function():
    #     time.sleep(2)
    #     for _ in tqdm(range(3000), desc="Main loop"):
    #         for _ in tqdm(range(3000), leave=False):
    #             pass

    # example_function()
    main()
