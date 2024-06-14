import json
import os
import random
from collections import defaultdict
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import geojson
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython import display
from ipywidgets import Output
from PIL import Image

from dataloaders import TreeDataLoader, extract_ground_truth_from_dataloader
from datasets import TreeDataset
from geojson_conversions import merge_geojson_feature_collections, save_geojson
from layers import AMF_GD_YOLOv8
from metrics import (
    APMetrics,
    compute_sorted_ap,
    compute_sorted_ap_confs,
    hungarian_algorithm,
    hungarian_algorithm_confs,
    plot_sorted_ap,
    plot_sorted_ap_confs,
)
from plot import create_geojson_output
from preprocessing.data import ImageData
from utils import Folders, import_tqdm


tqdm = import_tqdm()


class TrainingMetrics:
    def __init__(self, float_precision: int = 3, show: bool = True) -> None:
        self.float_precision = float_precision
        self.show = show
        self.reset()

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

        metrics_index: Dict[str, int] = {}
        categories_index: Dict[str, int] = {}
        for i, (metric_name, metric_dict) in enumerate(self.metrics.items()):
            metrics_index[metric_name] = i
            for category_name in metric_dict.keys():
                if category_name not in categories_index.keys():
                    categories_index[category_name] = len(categories_index)

        scale = max(ceil((len(metrics_index)) ** 0.5), 1)
        nrows = scale
        ncols = (len(metrics_index) + scale - 1) // scale
        cmap = plt.get_cmap("tab10")

        categories_colors = {label: cmap(i) for i, label in enumerate(categories_index.keys())}

        for interval, save_path in zip(intervals, save_paths):
            plt.clf()
            fig = plt.figure(1, figsize=(6 * ncols, 4 * nrows))

            for metric_name, metric_dict in self.metrics.items():
                ax = fig.add_subplot(nrows, ncols, metrics_index[metric_name] + 1)
                for category_name, category_dict in metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]

                    # Remove the epochs before from_epoch
                    start = interval[0] if interval[0] >= 0 else self.last_epoch + interval[0]
                    end = self.last_epoch + interval[1] if interval[1] <= 0 else interval[1]
                    kept_indices = [i for i in range(len(epochs)) if start <= epochs[i] <= end]
                    epochs = [epochs[i] for i in kept_indices]
                    values = [values[i] for i in kept_indices]

                    fmt = "-" if end - start > 25 else "-o"
                    ax.plot(
                        epochs,
                        values,
                        fmt,
                        color=categories_colors[category_name],
                        label=category_name,
                    )
                ax.grid(alpha=0.5)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(self.y_axes[metric_name])
                ax.set_title(f"{metric_name}")
            plt.tight_layout()

            has_legend, _ = plt.gca().get_legend_handles_labels()
            if any(label != "" for label in has_legend):
                plt.legend()

            if save_path is not None:
                plt.savefig(save_path, dpi=200)

        if self.show:
            with self.out:
                plt.show()
                display.clear_output(wait=True)


def perfect_preds(
    gt_bboxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_indices: torch.Tensor,
    batch_size: int,
):
    extracted_bboxes = [[]] * batch_size
    extracted_classes = [[]] * batch_size
    for bbox_idx, image_idx in enumerate(gt_indices):
        slice_bboxes = gt_bboxes[bbox_idx]
        extracted_bboxes[image_idx].append(slice_bboxes)
        slice_classes = gt_classes[bbox_idx].long()
        extracted_classes[image_idx].append(slice_classes)
    scores = [
        20 * nn.functional.one_hot(torch.tensor(cls), num_classes=5) - 0.5
        for cls in extracted_classes
    ]
    prefect_preds = [
        torch.cat((torch.tensor(bboxes), classes), dim=1).permute((1, 0)).unsqueeze(0)
        for bboxes, classes in zip(extracted_bboxes, scores)
    ]
    perfect_preds = torch.cat(
        [
            torch.cat(
                (
                    pred,
                    torch.zeros((pred.shape[0], pred.shape[1], 8400 - pred.shape[2])).to(
                        pred.device
                    ),
                ),
                dim=2,
            )
            for pred in prefect_preds
        ]
    )
    return perfect_preds


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
) -> int:
    model.train()
    stream = tqdm(train_loader, leave=False, desc="Training")
    for data in stream:
        image_rgb: torch.Tensor = data["image_rgb"]
        image_chm: torch.Tensor = data["image_chm"]
        gt_bboxes: torch.Tensor = data["bboxes"]
        gt_classes: torch.Tensor = data["labels"]
        gt_indices: torch.Tensor = data["indices"]

        image_rgb = image_rgb.to(device, non_blocking=True)
        image_chm = image_chm.to(device, non_blocking=True)
        gt_bboxes = gt_bboxes.to(device, non_blocking=True)
        gt_classes = gt_classes.to(device, non_blocking=True)
        gt_indices = gt_indices.to(device, non_blocking=True)

        output = model(image_rgb, image_chm)
        total_loss, loss_dict = model.compute_loss(output, gt_bboxes, gt_classes, gt_indices)

        batch_size = image_rgb.shape[0]
        training_metrics.update("Training", "Loss", total_loss.item(), batch_size)
        for key, value in loss_dict.items():
            training_metrics.update("Training", key, value.item(), batch_size)

        total_loss.backward()

        # Gradient accumulation
        if (running_accumulation_step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_accumulation_step += 1
        break

    return running_accumulation_step


def validate(
    val_loader: TreeDataLoader,
    model: AMF_GD_YOLOv8,
    device: torch.device,
    training_metrics: TrainingMetrics,
) -> float:
    # AP metrics
    thresholds_low = np.power(10, np.linspace(-4, -1, 10))
    thresholds_high = np.linspace(0.1, 1.0, 19)
    conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()
    ap_metrics = APMetrics(conf_thresholds=conf_thresholds)

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
            image_indices: torch.Tensor = data["indices"]

            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)
            gt_classes = gt_classes.to(device, non_blocking=True)
            gt_indices = gt_indices.to(device, non_blocking=True)
            image_indices = image_indices.to(device, non_blocking=True)

            # Compute the model output
            preds, output = model.forward_eval(image_rgb, image_chm)

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

            # Compute the AP metrics
            ap_metrics.add_preds(
                model=model,
                preds=preds,
                gt_bboxes=gt_bboxes,
                gt_classes=gt_classes,
                gt_indices=gt_indices,
                image_indices=image_indices,
            )

    _, _, sorted_ap, conf_threshold = ap_metrics.get_best_sorted_ap()
    training_metrics.update("Validation", "Best sortedAP", sorted_ap, y_axis="sortedAP")
    training_metrics.update(
        "Validation", "Conf thres of sortedAP", conf_threshold, y_axis="Conf threshold"
    )

    return training_metrics.get_last("Validation", "Loss")


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


def predict_to_geojson(
    model: AMF_GD_YOLOv8,
    test_loader: TreeDataLoader,
    device: torch.device,
    save_path: str,
    use_rgb: bool = False,
    use_chm: bool = False,
):
    model.eval()
    geojson_outputs: List[geojson.FeatureCollection] = []
    with torch.no_grad():
        for data in tqdm(test_loader, leave=False, desc="Exporting output"):
            # Get data
            image_rgb: torch.Tensor = data["image_rgb"]
            if not use_rgb:
                image_rgb = torch.zeros_like(image_rgb)
            image_chm: torch.Tensor = data["image_chm"]
            if not use_chm:
                image_chm = torch.zeros_like(image_chm)
            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)

            # Compute model output
            bboxes_list, scores_list, classes_as_ints_list = model.predict(image_rgb, image_chm)
            classes_as_strs_list = [
                [model.class_names[i] for i in classes_as_ints]
                for classes_as_ints in classes_as_ints_list
            ]

            idx_all = data["image_indices"]

            # Store the bounding boxes in a GeoJSON file
            for idx, bboxes, scores, classes_as_strs in zip(
                idx_all, bboxes_list, scores_list, classes_as_strs_list
            ):
                full_image_name = test_loader.dataset.get_full_image_name(idx)
                cropped_coords_name = test_loader.dataset.get_cropped_coords_name(idx)
                geojson_features = create_geojson_output(
                    full_image_name=full_image_name,
                    cropped_coords_name=cropped_coords_name,
                    bboxes=bboxes,
                    labels=classes_as_strs,
                    scores=scores,
                    save_path=save_path,
                )
                geojson_outputs.append(geojson_features)

    geojson_outputs_merged = merge_geojson_feature_collections(geojson_outputs)
    save_geojson(geojson_outputs_merged, save_path)


def compute_all_ap_metrics(
    model: AMF_GD_YOLOv8,
    test_loader: TreeDataLoader,
    device: torch.device,
    conf_thresholds: List[float],
    use_rgb: bool = False,
    use_chm: bool = False,
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    # For sortedAP/conf threshold plotting
    matched_pairs_conf: List[List[Tuple[Tuple[int, int], float]]] = [
        [] for _ in range(len(conf_thresholds))
    ]
    unmatched_pred_conf: List[List[int]] = [[] for _ in range(len(conf_thresholds))]
    unmatched_gt_conf: List[List[int]] = [[] for _ in range(len(conf_thresholds))]

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, leave=False, desc="Computing metrics"):
            # Get data
            image_rgb: torch.Tensor = data["image_rgb"]
            if not use_rgb:
                image_rgb = torch.zeros_like(image_rgb)
            image_chm: torch.Tensor = data["image_chm"]
            if not use_chm:
                image_chm = torch.zeros_like(image_chm)
            image_rgb = image_rgb.to(device, non_blocking=True)
            image_chm = image_chm.to(device, non_blocking=True)

            # Compute model output
            lowest_conf_threshold = min(conf_thresholds)
            bboxes_list, scores_list, classes_as_ints_list = model.predict(
                image_rgb, image_chm, conf_threshold=lowest_conf_threshold
            )

            # Get ground truth
            gt_bboxes_per_image, gt_classes_per_image = extract_ground_truth_from_dataloader(data)

            # Compute the matching
            iou_threshold = 1e-6
            for bboxes, scores, classes_as_ints, gt_bboxes, gt_classes in zip(
                bboxes_list,
                scores_list,
                classes_as_ints_list,
                gt_bboxes_per_image,
                gt_classes_per_image,
            ):
                matched_pairs_conf_temp, unmatched_pred_conf_temp, unmatched_gt_conf_temp = (
                    hungarian_algorithm_confs(
                        pred_bboxes=bboxes,
                        pred_labels=classes_as_ints,
                        pred_scores=scores,
                        gt_bboxes=gt_bboxes,
                        gt_labels=gt_classes,
                        iou_threshold=iou_threshold,
                        conf_thresholds=conf_thresholds,
                        agnostic=False,
                    )
                )
                for i in range(len(conf_thresholds)):
                    matched_pairs_conf[i].extend(matched_pairs_conf_temp[i])
                    unmatched_pred_conf[i].extend(unmatched_pred_conf_temp[i])
                    unmatched_gt_conf[i].extend(unmatched_gt_conf_temp[i])

    sorted_ious_list, aps_list, sorted_ap_list = compute_sorted_ap_confs(
        matched_pairs_conf, unmatched_pred_conf, unmatched_gt_conf
    )

    best_index = sorted_ap_list.index(max(sorted_ap_list))
    best_sorted_ious = sorted_ious_list[best_index]
    best_aps = aps_list[best_index]
    best_sorted_ap = sorted_ap_list[best_index]
    best_conf_threshold = conf_thresholds[best_index]

    return (
        sorted_ious_list,
        aps_list,
        sorted_ap_list,
    )


def initialize_dataloaders(
    datasets: Dict[str, TreeDataset],
    batch_size: int,
    num_workers: int,
) -> Tuple[TreeDataLoader, TreeDataLoader, TreeDataLoader]:
    assert all(key in datasets for key in ["training", "validation", "test"])
    train_loader = TreeDataLoader(
        datasets["training"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = TreeDataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = TreeDataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_and_validate(
    model: AMF_GD_YOLOv8,
    datasets: Dict[str, TreeDataset],
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    accumulate: int,
    device: torch.device,
    save_outputs: bool,
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
    intervals = [(0, 0), (5, 0), (-100, 0), (-25, 0)]
    training_metrics_path = [
        os.path.join(
            Folders.OUTPUT_DIR.value,
            f"{model.name}_training_metrics_plot_from_{interval[0]}_{interval[1]}.png",
        )
        for interval in intervals
    ]

    best_model = model
    best_loss = np.inf

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        training_metrics.visualize(intervals=intervals, save_paths=training_metrics_path)
        running_accumulation_step = train(
            train_loader,
            model,
            optimizer,
            device,
            accumulation_steps,
            running_accumulation_step,
            training_metrics,
        )
        current_loss = validate(val_loader, model, device, training_metrics)
        scheduler.step()

        # Store the best model
        if current_loss < best_loss:
            best_model = model
            best_loss = current_loss

        training_metrics.end_loop(epoch)

    # Save the plot showing the evolution of the metrics
    training_metrics.visualize(intervals=intervals, save_paths=training_metrics_path)
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

            annotations_file = rgb_file.replace(rgb_folder_path, annotations_folder_path).replace(
                ".tif", ".json"
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
