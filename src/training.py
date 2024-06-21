import json
import os
import random
from collections import defaultdict
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import geojson
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython import display
from ipywidgets import Output
from ultralytics.utils.tal import make_anchors

from dataloaders import (
    TreeDataLoader,
    convert_ground_truth_from_tensors,
    initialize_dataloaders,
)
from dataset_constants import DatasetConst
from datasets import TreeDataset
from geojson_conversions import merge_geojson_feature_collections, save_geojson
from layers import AMF_GD_YOLOv8
from metrics import AP_Metrics
from plot import create_bboxes_training_image, create_geojson_output
from preprocessing.data import ImageData
from utils import Folders, import_tqdm


tqdm = import_tqdm()


class TrainingMetrics:
    def __init__(self, float_precision: int = 3, show: bool = True) -> None:
        self.float_precision = float_precision
        self.show = show
        self.reset()
        self.fig_num = None

    def _create_fig_num(self) -> None:
        if self.fig_num is None:
            existing_figures = plt.get_fignums()
            if existing_figures:
                self.fig_num = max(existing_figures) + 1
            else:
                self.fig_num = 1

    def reset(self):
        self.metrics = defaultdict(lambda: defaultdict(lambda: {"epochs": [], "avgs": []}))
        self.metrics_loop = defaultdict(
            lambda: defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})
        )
        self.y_axes = {}
        self.last_epoch = -1

        if self.show:
            self.out = Output()
            display.display(self.out)

    def end_loop(self, epoch: int):
        self.last_epoch = epoch
        for metric_name, metric_dict in self.metrics_loop.items():
            for category_name, category_dict in metric_dict.items():
                metric = self.metrics[metric_name][category_name]
                metric["epochs"].append(epoch)
                metric["avgs"].append(category_dict["avg"])
        self.metrics_loop = defaultdict(
            lambda: defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})
        )

    def update(
        self,
        category_name: str,
        metric_name: str,
        val: float,
        count: int = 1,
        y_axis: str | None = None,
    ):
        if y_axis is None:
            y_axis = metric_name
        self.y_axes[metric_name] = y_axis

        metric = self.metrics_loop[metric_name][category_name]

        metric["val"] += val
        metric["count"] += count
        metric["avg"] = metric["val"] / metric["count"]

    def get_last(self, category_name: str, metric_name: str):
        return self.metrics_loop[metric_name][category_name]["avg"]

    def save_metrics(self, save_path: str) -> None:
        with open(save_path, "w") as fp:
            json.dump(self.metrics, fp, sort_keys=True, indent=4)

    def visualize(
        self,
        intervals: List[Tuple[int, int]] = [(0, 0)],
        save_paths: Optional[Sequence[str | None]] = None,
    ):
        # Inspired from https://gitlab.com/robindar/dl-scaman_checker/-/blob/main/src/dl_scaman_checker/TP01.py
        if self.show is False and (save_paths is None or all(path is None for path in save_paths)):
            return

        if save_paths is None:
            save_paths = [None] * len(intervals)

        if len(save_paths) != len(intervals):
            raise ValueError("intervals and save_paths should have the same length.")

        metrics_index: Dict[str, int] = {}
        categories_index: Dict[str, int] = {}
        for i, (metric_name, metric_dict) in enumerate(self.metrics.items()):
            metrics_index[metric_name] = i
            for category_name in metric_dict.keys():
                if category_name not in categories_index.keys():
                    categories_index[category_name] = len(categories_index)

        n_metrics = len(metrics_index)
        scale = max(ceil(n_metrics**0.5), 1)
        nrows = scale
        ncols = max((n_metrics + scale - 1) // scale, 1)
        cmap = plt.get_cmap("tab10")

        categories_colors = {label: cmap(i) for i, label in enumerate(categories_index.keys())}
        legend_space = 1
        figsize = (7 * ncols, 5 * nrows + legend_space)
        legend_y_position = legend_space / figsize[1]

        for interval, save_path in zip(intervals, save_paths):
            self._create_fig_num()
            fig = plt.figure(self.fig_num, figsize=figsize)
            plt.clf()

            for metric_name, metric_dict in self.metrics.items():
                index = metrics_index[metric_name]
                ax = fig.add_subplot(nrows, ncols, index + 1)
                for category_name, category_dict in metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]

                    # Remove the epochs before from_epoch
                    start = (
                        interval[0] if interval[0] >= 0 else max(0, self.last_epoch + interval[0])
                    )
                    end = (
                        self.last_epoch + interval[1]
                        if interval[1] <= 0
                        else min(interval[1], self.last_epoch)
                    )
                    kept_indices = [i for i in range(len(epochs)) if start <= epochs[i] <= end]
                    epochs = [epochs[i] for i in kept_indices]
                    values = [values[i] for i in kept_indices]

                    epochs_length = max(epochs) - min(epochs) if len(epochs) > 0 else 0

                    fmt = "-" if epochs_length > 25 else "-o"
                    ax.plot(
                        epochs,
                        values,
                        fmt,
                        color=categories_colors[category_name],
                        # label=category_name,
                    )
                ax.grid(alpha=0.5)
                if index >= n_metrics - ncols:
                    ax.set_xlabel("Epoch")
                else:
                    ax.tick_params(
                        axis="x", which="both", bottom=False, top=False, labelbottom=False
                    )
                ax.set_ylabel(self.y_axes[metric_name])
                ax.set_title(f"{metric_name}")

            lines = [
                mlines.Line2D([], [], color=color, linestyle="-", label=label)
                for label, color in categories_colors.items()
            ]

            if len(lines) > 0:
                fig.legend(
                    handles=lines,
                    loc="upper center",
                    bbox_to_anchor=(0.5, legend_y_position),
                    ncol=len(lines),
                )

            fig.tight_layout(rect=(0.0, legend_y_position, 1.0, 1.0))

            if save_path is not None:
                plt.savefig(save_path, dpi=200)

        if self.show:
            with self.out:
                plt.show()
                display.clear_output(wait=True)


def print_current_memory():
    if torch.cuda.is_available():
        current_memory_usage_bytes = torch.cuda.memory_allocated()
        current_memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
        print(f"Current GPU memory usage: {current_memory_usage_megabytes:.2f} MB")
    else:
        print("CUDA is not available.")


def train(
    train_loader: TreeDataLoader,
    model: AMF_GD_YOLOv8,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    running_accumulation_step: int,
    training_metrics: TrainingMetrics,
    epoch: int,
    ap_interval: int = 20,
    image_preds_interval: int = 50,
) -> int:
    # AP metrics
    compute_ap = epoch % ap_interval == 0
    if compute_ap:
        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()
        ap_metrics = AP_Metrics(conf_threshold_list=conf_thresholds)

    # Predictions saved as an image
    compute_image_preds = epoch % image_preds_interval == 0

    model.train()
    stream = tqdm(train_loader, leave=False, desc="Training")
    for data in stream:
        # Get the data
        image_rgb: torch.Tensor = data["image_rgb"]
        image_chm: torch.Tensor = data["image_chm"]
        gt_bboxes: torch.Tensor = data["bboxes"]
        gt_classes: torch.Tensor = data["labels"]
        gt_indices: torch.Tensor = data["indices"]
        image_indices: torch.Tensor = data["image_indices"]

        image_rgb = image_rgb.to(device, non_blocking=True)
        image_chm = image_chm.to(device, non_blocking=True)
        gt_bboxes = gt_bboxes.to(device, non_blocking=True)
        gt_classes = gt_classes.to(device, non_blocking=True)
        gt_indices = gt_indices.to(device, non_blocking=True)
        image_indices = image_indices.to(device, non_blocking=True)

        # Compute the model output
        output = model.forward(image_rgb, image_chm)

        # Compute the AP metrics
        with torch.no_grad():
            # Model evaluations:
            if compute_ap or compute_image_preds:
                preds = model.preds_from_output(output)

            # Compute the AP metrics
            if compute_ap:
                ap_metrics.add_preds(
                    model=model,
                    preds=preds,
                    gt_bboxes=gt_bboxes,
                    gt_classes=gt_classes,
                    gt_indices=gt_indices,
                    image_indices=image_indices,
                )

            if compute_image_preds:
                dataset_idx = 0
                if dataset_idx in image_indices.tolist():
                    batch_idx = image_indices.tolist().index(dataset_idx)

                    bboxes_per_image, scores_per_image, classes_per_image = (
                        model.predict_from_preds(
                            preds[batch_idx : batch_idx + 1],
                            iou_threshold=0.5,
                            conf_threshold=0.1,
                            number_best=40,
                        )
                    )
                    gt_bboxes_per_image, gt_classes_per_image = convert_ground_truth_from_tensors(
                        gt_bboxes=gt_bboxes,
                        gt_classes=gt_classes,
                        gt_indices=gt_indices,
                        image_indices=image_indices,
                    )

                    image_rgb_initial = torch.tensor(
                        train_loader.dataset.get_rgb_image(dataset_idx)
                    ).permute((2, 0, 1))
                    image_chm_initial = torch.tensor(
                        train_loader.dataset.get_chm_image(dataset_idx)
                    ).permute((2, 0, 1))

                    create_bboxes_training_image(
                        image_rgb=image_rgb_initial,
                        image_chm=image_chm_initial,
                        pred_bboxes=bboxes_per_image[0],
                        pred_labels=classes_per_image[0],
                        pred_scores=scores_per_image[0],
                        gt_bboxes=gt_bboxes_per_image[batch_idx],
                        gt_labels=gt_classes_per_image[batch_idx],
                        labels_int_to_str=model.class_names,
                        colors_dict=DatasetConst.CLASS_COLORS.value,
                        save_path=os.path.join(model.folder_path, f"Data_epoch_{epoch}_train.png"),
                    )

        # Compute the loss
        total_loss, loss_dict = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)
        total_loss.backward()

        # Gradient accumulation
        if (running_accumulation_step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_accumulation_step += 1

        # Store the loss
        batch_size = image_rgb.shape[0]
        training_metrics.update(
            "Training", "Total Loss", total_loss.item(), count=batch_size, y_axis="Loss"
        )
        for key, value in loss_dict.items():
            training_metrics.update("Training", key, value.item(), count=batch_size, y_axis="Loss")

    if compute_ap:
        _, _, sorted_ap, conf_threshold = ap_metrics.get_best_sorted_ap()
        training_metrics.update("Training", "Best sortedAP", sorted_ap, y_axis="sortedAP")
        training_metrics.update(
            "Training", "Conf thres of sortedAP", conf_threshold, y_axis="Conf threshold"
        )

    return running_accumulation_step


def validate(
    val_loader: TreeDataLoader,
    model: AMF_GD_YOLOv8,
    device: torch.device,
    training_metrics: TrainingMetrics,
    epoch: int,
    ap_interval: int = 20,
    image_preds_interval: int = 50,
) -> float:
    # AP metrics
    compute_ap = epoch % ap_interval == 0
    if compute_ap:
        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()
        ap_metrics = AP_Metrics(conf_threshold_list=conf_thresholds)

    # Predictions saved as an image
    compute_image_preds = epoch % image_preds_interval == 0

    model.eval()
    stream = tqdm(val_loader, leave=False, desc="Validation")
    with torch.no_grad():
        for data in stream:
            # Get the data
            image_rgb: torch.Tensor = data["image_rgb"]
            image_chm: torch.Tensor = data["image_chm"]
            gt_bboxes: torch.Tensor = data["bboxes"]
            gt_classes: torch.Tensor = data["labels"]
            gt_indices: torch.Tensor = data["indices"]
            image_indices: torch.Tensor = data["image_indices"]

            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)
            gt_classes = gt_classes.to(device, non_blocking=True)
            gt_indices = gt_indices.to(device, non_blocking=True)
            image_indices = image_indices.to(device, non_blocking=True)

            # Compute the model output
            output = model.forward(image_rgb, image_chm)

            # Compute the loss
            total_loss, loss_dict = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)

            # Store the loss
            batch_size = image_rgb.shape[0]
            training_metrics.update(
                "Validation", "Total Loss", total_loss.item(), count=batch_size, y_axis="Loss"
            )
            for key, value in loss_dict.items():
                training_metrics.update(
                    "Validation", key, value.item(), count=batch_size, y_axis="Loss"
                )

            # Model evaluations:
            if compute_ap or compute_image_preds:
                preds = model.preds_from_output(output)

            # Compute the AP metrics
            if compute_ap:
                ap_metrics.add_preds(
                    model=model,
                    preds=preds,
                    gt_bboxes=gt_bboxes,
                    gt_classes=gt_classes,
                    gt_indices=gt_indices,
                    image_indices=image_indices,
                )

            # Save the predictions in an image
            if compute_image_preds:
                dataset_idx = 0
                if dataset_idx in image_indices.tolist():
                    batch_idx = image_indices.tolist().index(dataset_idx)

                    bboxes_per_image, scores_per_image, classes_per_image = (
                        model.predict_from_preds(
                            preds[batch_idx : batch_idx + 1],
                            iou_threshold=0.5,
                            conf_threshold=0.1,
                            number_best=40,
                        )
                    )
                    gt_bboxes_per_image, gt_classes_per_image = convert_ground_truth_from_tensors(
                        gt_bboxes=gt_bboxes,
                        gt_classes=gt_classes,
                        gt_indices=gt_indices,
                        image_indices=image_indices,
                    )

                    image_rgb_initial = torch.tensor(
                        val_loader.dataset.get_rgb_image(dataset_idx)
                    ).permute((2, 0, 1))
                    image_chm_initial = torch.tensor(
                        val_loader.dataset.get_chm_image(dataset_idx)
                    ).permute((2, 0, 1))

                    create_bboxes_training_image(
                        image_rgb=image_rgb_initial,
                        image_chm=image_chm_initial,
                        pred_bboxes=bboxes_per_image[0],
                        pred_labels=classes_per_image[0],
                        pred_scores=scores_per_image[0],
                        gt_bboxes=gt_bboxes_per_image[batch_idx],
                        gt_labels=gt_classes_per_image[batch_idx],
                        labels_int_to_str=model.class_names,
                        colors_dict=DatasetConst.CLASS_COLORS.value,
                        save_path=os.path.join(model.folder_path, f"Data_epoch_{epoch}_val.png"),
                    )
    if compute_ap:
        _, _, sorted_ap, conf_threshold = ap_metrics.get_best_sorted_ap()
        training_metrics.update("Validation", "Best sortedAP", sorted_ap, y_axis="sortedAP")
        training_metrics.update(
            "Validation", "Conf thres of sortedAP", conf_threshold, y_axis="Conf threshold"
        )

    return training_metrics.get_last("Validation", "Total Loss")


def rgb_chm_usage_postfix(use_rgb: bool, use_chm: bool):
    if use_rgb:
        if use_chm:
            return "RGB_CHM"
        else:
            return "RGB"
    else:
        if use_chm:
            return "CHM"
        else:
            return "no_data"


def rgb_chm_usage_legend(use_rgb: bool, use_chm: bool):
    if use_rgb:
        if use_chm:
            return "RGB and CHM"
        else:
            return "RGB"
    else:
        if use_chm:
            return "CHM"
        else:
            return "No data"


def evaluate_model(
    model: AMF_GD_YOLOv8,
    data_loader: TreeDataLoader,
    device: torch.device,
    use_rgb: bool = True,
    use_chm: bool = True,
    ap_conf_thresholds: Optional[List[float]] = None,
    output_geojson_save_path: Optional[str] = None,
) -> AP_Metrics:

    if ap_conf_thresholds is None and output_geojson_save_path is None:
        raise ValueError(
            "At least one of conf_thresholds and output_geojson_save_path should be specified."
        )

    ap_conf_thresholds = [] if ap_conf_thresholds is None else ap_conf_thresholds
    # AP metrics
    ap_metrics = AP_Metrics(conf_threshold_list=ap_conf_thresholds)

    if output_geojson_save_path is not None:
        geojson_outputs: List[geojson.FeatureCollection] = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, leave=False, desc="Evaluating the model"):
            # Get the data
            image_rgb: torch.Tensor = data["image_rgb"]
            if not use_rgb:
                image_rgb = torch.zeros_like(image_rgb)
            image_chm: torch.Tensor = data["image_chm"]
            if not use_chm:
                image_chm = torch.zeros_like(image_chm)
            gt_bboxes: torch.Tensor = data["bboxes"]
            gt_classes: torch.Tensor = data["labels"]
            gt_indices: torch.Tensor = data["indices"]
            image_indices: torch.Tensor = data["image_indices"]

            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)
            gt_classes = gt_classes.to(device, non_blocking=True)
            gt_indices = gt_indices.to(device, non_blocking=True)
            image_indices = image_indices.to(device, non_blocking=True)

            # Compute the model output
            output = model.forward(image_rgb, image_chm)
            preds = model.preds_from_output(output)

            # Compute the AP metrics
            if ap_conf_thresholds is not None:
                ap_metrics.add_preds(
                    model=model,
                    preds=preds,
                    gt_bboxes=gt_bboxes,
                    gt_classes=gt_classes,
                    gt_indices=gt_indices,
                    image_indices=image_indices,
                )

            if output_geojson_save_path is not None:
                bboxes_list, scores_list, classes_as_ints_list = model.predict_from_preds(
                    preds, number_best=40
                )
                classes_as_strs_list = [
                    [model.class_names[i] for i in classes_as_ints]
                    for classes_as_ints in classes_as_ints_list
                ]

                # Store the bounding boxes in a GeoJSON file
                for idx, bboxes, scores, classes_as_strs in zip(
                    image_indices, bboxes_list, scores_list, classes_as_strs_list
                ):
                    full_image_name = data_loader.dataset.get_full_image_name(int(idx))
                    cropped_coords_name = data_loader.dataset.get_cropped_coords_name(int(idx))
                    geojson_features = create_geojson_output(
                        full_image_name=full_image_name,
                        cropped_coords_name=cropped_coords_name,
                        bboxes=bboxes,
                        labels=classes_as_strs,
                        scores=scores,
                    )
                    geojson_outputs.append(geojson_features)

    if output_geojson_save_path is not None:
        geojson_outputs_merged = merge_geojson_feature_collections(geojson_outputs)
        save_geojson(geojson_outputs_merged, output_geojson_save_path)

    return ap_metrics


def train_and_validate(
    model: AMF_GD_YOLOv8,
    datasets: Dict[str, TreeDataset],
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    accumulate: int,
    device: torch.device,
    show_training_metrics: bool,
) -> AMF_GD_YOLOv8:

    train_loader, val_loader, test_loader = initialize_dataloaders(
        datasets, batch_size, num_workers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: 1 / np.sqrt(i + 2), last_epoch=-1
    )

    accumulation_steps = max(round(accumulate / batch_size), 1)
    running_accumulation_step = 0

    training_metrics = TrainingMetrics(show=show_training_metrics)
    intervals = [(0, 0), (10, 0), (-1000, 0), (-500, 0), (-100, 0)]
    training_metrics_path = [
        os.path.join(
            model.folder_path,
            f"training_plot_{interval[0]}_{interval[1]}.png",
        )
        for interval in intervals
    ]

    best_model = model
    best_loss = np.inf
    skip_until = 3
    temp_models_interval = 100

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        if epoch % temp_models_interval == 1:
            best_temp_epoch = -1
            best_temp_model = model
            best_temp_loss = np.inf

        # training_metrics.visualize(intervals=intervals, save_paths=training_metrics_path)
        training_metrics.save_metrics(
            os.path.join(
                model.folder_path,
                "metrics_values.json",
            )
        )

        running_accumulation_step = train(
            train_loader,
            model,
            optimizer,
            device,
            accumulation_steps,
            running_accumulation_step,
            training_metrics,
            epoch=epoch,
        )

        current_loss = validate(val_loader, model, device, training_metrics, epoch=epoch)
        scheduler.step()

        if epoch >= skip_until:
            # Store and save the best model
            if current_loss < best_loss:
                best_model = model
                best_loss = current_loss
                best_model.save_weights()

            # Store and save the best temp model
            if current_loss < best_temp_loss:
                best_temp_epoch = epoch
                best_temp_model = model
                best_temp_loss = current_loss

            if epoch % temp_models_interval == 0:
                best_temp_model.save_weights(best=False, epoch=best_temp_epoch)

        training_metrics.end_loop(epoch)

    # Save the plot showing the evolution of the metrics
    training_metrics.visualize(intervals=intervals, save_paths=training_metrics_path)
    training_metrics.save_metrics(
        os.path.join(
            model.folder_path,
            "metrics_values.json",
        )
    )
    return best_model


def get_all_files_iteratively(folder_path: str) -> List[str]:
    """Finds iteratively all the files below the input folder.

    Args:
        folder_path (str): folder to look into.

    Returns:
        List[str]: the list of all the files.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files


def split_files_into_lists(
    folder_path: str,
    sets_ratios: Sequence[int | float],
    sets_names: List[str],
    random_seed: int | None = None,
) -> Dict[str, List[str]]:
    """Splits files in a folder into multiple lists based on specified ratios.

    Args:
        folder_path (str): path to the folder containing the files.
        sets_ratios (List[int | float]): the proportions for each list.
        sets_names (List[str]): the keys for the dictionary
        random_seed (int | None, optional): a seed for the randomization. Defaults to None.

    Returns:
        Dict[str, List[str]]: a dictionary where each key from the input names is linked
        with a list of files.
    """
    files = get_all_files_iteratively(folder_path)
    total_ratio = sum(sets_ratios)
    ratios = [r / total_ratio for r in sets_ratios]

    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(files)
    split_indices = [0] * (len(ratios) + 1)
    split_indices[-1] = len(files)
    sum_ratios = 0.0
    for i in range(len(ratios) - 1):
        sum_ratios += ratios[i]
        split_indices[i + 1] = int(round(len(files) * (sum_ratios)))

    files_dict = {}
    for i in range(len(ratios)):
        files_dict[sets_names[i]] = files[split_indices[i] : split_indices[i + 1]]

    return files_dict


def create_and_save_splitted_datasets(
    rgb_folder_path: str,
    chm_folder_path: str,
    annotations_folder_path: str,
    sets_ratios: Sequence[int | float],
    sets_names: List[str],
    save_path: str,
    random_seed: int | None = None,
) -> None:
    files_dict = split_files_into_lists(
        folder_path=rgb_folder_path,
        sets_ratios=sets_ratios,
        sets_names=sets_names,
        random_seed=random_seed,
    )
    all_files_dict = {}
    for set_name, set_files in files_dict.items():
        all_files_dict[set_name] = []
        for rgb_file in set_files:
            full_image = os.path.join(
                Folders.FULL_RGB_IMAGES.value,
                f"{os.path.basename(os.path.dirname(rgb_file))}.tif",
            )
            image_data = ImageData(full_image)

            chm_file = rgb_file.replace(rgb_folder_path, chm_folder_path).replace(
                image_data.base_name, image_data.coord_name
            )

            # Handle tif, mmap and npy
            annotations_file = rgb_file.replace(rgb_folder_path, annotations_folder_path).replace(
                ".tif", ".json"
            )
            annotations_file = rgb_file.replace(rgb_folder_path, annotations_folder_path).replace(
                ".mmap", ".json"
            )
            annotations_file = rgb_file.replace(rgb_folder_path, annotations_folder_path).replace(
                ".npy", ".json"
            )

            new_dict = {
                "rgb": rgb_file,
                "chm": chm_file,
                "annotations": annotations_file,
            }
            all_files_dict[set_name].append(new_dict)

    with open(save_path, "w") as f:
        json.dump(all_files_dict, f)


def load_tree_datasets_from_split(
    data_split_file_path: str,
    labels_to_index: Dict[str, int],
    mean_rgb: torch.Tensor,
    std_rgb: torch.Tensor,
    mean_chm: torch.Tensor,
    std_chm: torch.Tensor,
    transform_spatial_training: A.Compose | None,
    transform_pixel_rgb_training: A.Compose | None,
    transform_pixel_chm_training: A.Compose | None,
    dismissed_classes: List[str] = [],
    proba_drop_rgb: float = 0.0,
    labels_transformation_drop_rgb: Dict[str, str | None] | None = None,
    proba_drop_chm: float = 0.0,
    labels_transformation_drop_chm: Dict[str, str | None] | None = None,
    no_data_new_value: float = -5.0,
) -> Dict[str, TreeDataset]:
    with open(data_split_file_path, "r") as f:
        data_split = json.load(f)

    tree_datasets = {}
    tree_datasets["training"] = TreeDataset(
        data_split["training"],
        labels_to_index=labels_to_index,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        mean_chm=mean_chm,
        std_chm=std_chm,
        proba_drop_rgb=proba_drop_rgb,
        labels_transformation_drop_rgb=labels_transformation_drop_rgb,
        proba_drop_chm=proba_drop_chm,
        labels_transformation_drop_chm=labels_transformation_drop_chm,
        dismissed_classes=dismissed_classes,
        transform_spatial=transform_spatial_training,
        transform_pixel_rgb=transform_pixel_rgb_training,
        transform_pixel_chm=transform_pixel_chm_training,
        no_data_new_value=no_data_new_value,
    )

    tree_datasets["validation"] = TreeDataset(
        data_split["validation"],
        labels_to_index=labels_to_index,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        mean_chm=mean_chm,
        std_chm=std_chm,
        proba_drop_rgb=0.0,
        labels_transformation_drop_rgb=None,
        proba_drop_chm=0.0,
        labels_transformation_drop_chm=None,
        dismissed_classes=dismissed_classes,
        transform_spatial=None,
        transform_pixel_rgb=None,
        transform_pixel_chm=None,
        no_data_new_value=no_data_new_value,
    )

    if "test" in data_split:
        test_data = data_split["test"]
    else:
        test_data = data_split["validation"]

    tree_datasets["test"] = TreeDataset(
        test_data,
        labels_to_index=labels_to_index,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        mean_chm=mean_chm,
        std_chm=std_chm,
        proba_drop_rgb=0.0,
        labels_transformation_drop_rgb=None,
        proba_drop_chm=0.0,
        labels_transformation_drop_chm=None,
        dismissed_classes=dismissed_classes,
        transform_spatial=None,
        transform_pixel_rgb=None,
        transform_pixel_chm=None,
        no_data_new_value=no_data_new_value,
    )

    return tree_datasets
