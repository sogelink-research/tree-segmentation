import bisect
import json
import os
import time
from collections import defaultdict
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

import geojson
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display
from ipywidgets import Output
from matplotlib.ticker import MaxNLocator

from dataloaders import TreeDataLoader, initialize_dataloaders
from datasets import TreeDataset
from geojson_conversions import merge_geojson_feature_collections, save_geojson
from layers import AMF_GD_YOLOv8
from metrics import AP_Metrics
from plot import create_geojson_output
from utils import RICH_PRINTING


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
            json.dump(self.metrics, fp, sort_keys=True)

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
        legend_space = 0.5
        figsize = (5 * ncols, 3.5 * nrows + legend_space)
        legend_y_position = legend_space / figsize[1]
        ax0 = None

        for interval, save_path in zip(intervals, save_paths):
            start = interval[0] if interval[0] >= 0 else max(0, self.last_epoch + interval[0])
            end = (
                self.last_epoch + interval[1]
                if interval[1] <= 0
                else min(interval[1], self.last_epoch)
            )
            x_margin = (end - start) * 0.05
            x_extent_min = start - x_margin
            x_extent_max = end + x_margin

            self._create_fig_num()
            fig = plt.figure(self.fig_num, figsize=figsize)
            plt.clf()

            for metric_name, metric_dict in self.metrics.items():
                # Extract the values in the interval
                cropped_metric_dict = {}
                y_extent_min = np.inf
                y_extent_max = -np.inf
                for category_name, category_dict in metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]

                    index_start = bisect.bisect_left(epochs, start)
                    index_end = bisect.bisect_right(epochs, end) - 1
                    index_start_values = bisect.bisect_left(epochs, x_extent_min)
                    if index_start_values > 0:
                        index_start_values -= 1
                    index_end_values = bisect.bisect_right(epochs, x_extent_max) - 1
                    if index_end_values < len(epochs) - 1:
                        index_end_values += 1

                    cropped_metric_dict[category_name] = {}
                    cropped_metric_dict[category_name]["epochs"] = epochs[
                        index_start_values : index_end_values + 1
                    ]
                    cropped_metric_dict[category_name]["avgs"] = values[
                        index_start_values : index_end_values + 1
                    ]

                    if index_end > index_start:
                        y_extent_min = min(y_extent_min, min(values[index_start : index_end + 1]))
                        y_extent_max = max(y_extent_max, max(values[index_start : index_end + 1]))

                # Plot the metrics
                index = metrics_index[metric_name]
                ax = fig.add_subplot(nrows, ncols, index + 1, sharex=ax0)
                if ax0 is None:
                    ax0 = ax
                for category_name, category_dict in cropped_metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]

                    # Remove the epochs before from_epoch
                    epochs_length = max(epochs) - min(epochs) if len(epochs) > 0 else 0

                    fmt = "-" if epochs_length > 15 else "-o"
                    ax.plot(
                        epochs,
                        values,
                        fmt,
                        color=categories_colors[category_name],
                        # label=category_name,
                    )

                ax.grid(alpha=0.5)
                if np.isfinite(y_extent_min) and np.isfinite(y_extent_max):
                    y_margin = (y_extent_max - y_extent_min) * 0.05
                    ax.set_ylim(y_extent_min - y_margin, y_extent_max + y_margin)

                if index >= n_metrics - ncols:
                    ax.set_xlabel("Epoch")
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                    ax.set_xlim(start - x_margin, end + x_margin)
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
        RICH_PRINTING.print(f"Current GPU memory usage: {current_memory_usage_megabytes:.2f} MB")
    else:
        RICH_PRINTING.print("CUDA is not available.")


def train(
    train_loader: TreeDataLoader,
    model: AMF_GD_YOLOv8,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    running_accumulation_step: int,
    training_metrics: TrainingMetrics,
    epoch: int,
    ap_interval: int,
) -> int:
    # AP metrics
    compute_ap = epoch % ap_interval == 0
    if compute_ap:
        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()
        ap_metrics = AP_Metrics(conf_threshold_list=conf_thresholds)

    model.train()
    stream = RICH_PRINTING.pbar(
        train_loader, len(train_loader), leave=False, description="Training"
    )
    for data in stream:
        # Get the data
        image_rgb: torch.Tensor | None = data["image_rgb"]
        image_chm: torch.Tensor | None = data["image_chm"]
        gt_bboxes: torch.Tensor = data["bboxes"]
        gt_classes: torch.Tensor = data["labels"]
        gt_indices: torch.Tensor = data["indices"]
        image_indices: torch.Tensor = data["image_indices"]

        if image_rgb is not None:
            image_rgb = image_rgb.to(device, non_blocking=True)
        if image_chm is not None:
            image_chm = image_chm.to(device, non_blocking=True)
        gt_bboxes = gt_bboxes.to(device, non_blocking=True)
        gt_classes = gt_classes.to(device, non_blocking=True)
        gt_indices = gt_indices.to(device, non_blocking=True)
        image_indices = image_indices.to(device, non_blocking=True)

        # Compute the model output
        output = model.forward(image_rgb, image_chm)

        # Compute the AP metrics
        with torch.no_grad():
            if compute_ap:
                preds = model.preds_from_output(output)
                ap_metrics.add_preds(
                    model=model,
                    preds=preds,
                    gt_bboxes=gt_bboxes,
                    gt_classes=gt_classes,
                    gt_indices=gt_indices,
                    image_indices=image_indices,
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
        batch_size = image_indices.shape[0]
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
    ap_interval: int,
) -> float:
    # AP metrics
    compute_ap = epoch % ap_interval == 0
    if compute_ap:
        thresholds_low = np.power(10, np.linspace(-4, -1, 10))
        thresholds_high = np.linspace(0.1, 1.0, 19)
        conf_thresholds = np.hstack((thresholds_low, thresholds_high)).tolist()
        ap_metrics = AP_Metrics(conf_threshold_list=conf_thresholds)

    model.eval()
    stream = RICH_PRINTING.pbar(val_loader, len(val_loader), leave=False, description="Validation")
    with torch.no_grad():
        for data in stream:
            # Get the data
            image_rgb: torch.Tensor | None = data["image_rgb"]
            image_chm: torch.Tensor | None = data["image_chm"]
            gt_bboxes: torch.Tensor = data["bboxes"]
            gt_classes: torch.Tensor = data["labels"]
            gt_indices: torch.Tensor = data["indices"]
            image_indices: torch.Tensor = data["image_indices"]

            if image_rgb is not None:
                image_rgb = image_rgb.to(device, non_blocking=True)
            if image_chm is not None:
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
            batch_size = image_indices.shape[0]
            training_metrics.update(
                "Validation", "Total Loss", total_loss.item(), count=batch_size, y_axis="Loss"
            )
            for key, value in loss_dict.items():
                training_metrics.update(
                    "Validation", key, value.item(), count=batch_size, y_axis="Loss"
                )

            # Model evaluations:
            if compute_ap:
                preds = model.preds_from_output(output)
                ap_metrics.add_preds(
                    model=model,
                    preds=preds,
                    gt_bboxes=gt_bboxes,
                    gt_classes=gt_classes,
                    gt_indices=gt_indices,
                    image_indices=image_indices,
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
            return "RGB only"
    else:
        if use_chm:
            return "CHM only"
        else:
            return "No data"


def rgb_cir_chm_usage_postfix(use_rgb: bool, use_cir: bool, use_chm: bool):
    if use_rgb:
        if use_cir:
            if use_chm:
                return "RGB_CIR_CHM"
            else:
                return "RGB_CIR"
        else:
            if use_chm:
                return "RGB_CHM"
            else:
                return "RGB"
    else:
        if use_cir:
            if use_chm:
                return "CIR_CHM"
            else:
                return "CIR"
        else:
            if use_chm:
                return "CHM"
            else:
                return "no_data"


def rgb_cir_chm_usage_legend(use_rgb: bool, use_cir: bool, use_chm: bool):
    if use_rgb:
        if use_cir:
            if use_chm:
                return "RGB, CIR and CHM"
            else:
                return "RGB and CIR"
        else:
            if use_chm:
                return "RGB and CHM"
            else:
                return "RGB only"
    else:
        if use_cir:
            if use_chm:
                return "CIR and CHM"
            else:
                return "CIR only"
        else:
            if use_chm:
                return "CHM only"
            else:
                return "No data"


def evaluate_model(
    model: AMF_GD_YOLOv8,
    data_loader: TreeDataLoader,
    device: torch.device,
    use_rgb_cir: bool = True,
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

    if not use_rgb_cir and not use_chm:
        raise Exception("You cannot use neither of RGB/CIR and CHM.")

    # Modify the dataset to only output the necessary data
    if not use_rgb_cir:
        old_use_rgb = data_loader.dataset.use_rgb_cir
        old_use_chm = data_loader.dataset.use_chm
        data_loader.dataset.use_rgb_cir = False
        data_loader.dataset.use_chm = True
    elif not use_chm:
        old_use_rgb = data_loader.dataset.use_rgb_cir
        old_use_chm = data_loader.dataset.use_chm
        data_loader.dataset.use_rgb_cir = True
        data_loader.dataset.use_chm = False

    model.eval()
    with torch.no_grad():
        for data in RICH_PRINTING.pbar(
            data_loader, len(data_loader), leave=False, description="Evaluating the model"
        ):
            # Get the data
            image_rgb: torch.Tensor | None = data["image_rgb"]
            image_chm: torch.Tensor | None = data["image_chm"]
            gt_bboxes: torch.Tensor = data["bboxes"]
            gt_classes: torch.Tensor = data["labels"]
            gt_indices: torch.Tensor = data["indices"]
            image_indices: torch.Tensor = data["image_indices"]

            if image_rgb is not None:
                image_rgb = image_rgb.to(device, non_blocking=True)
            if image_chm is not None:
                image_chm = image_chm.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)
            gt_classes = gt_classes.to(device, non_blocking=True)
            gt_indices = gt_indices.to(device, non_blocking=True)
            image_indices = image_indices.to(device, non_blocking=True)

            # Compute the model output
            output = model.forward(
                image_rgb, image_chm, use_left_temp=use_rgb_cir, use_right_temp=use_chm
            )
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

    # Set back the dataset to initial status
    if not use_rgb_cir or not use_chm:
        data_loader.dataset.use_rgb_cir = old_use_rgb
        data_loader.dataset.use_chm = old_use_chm

    return ap_metrics


@RICH_PRINTING.running_message("Finding the optimal batch size...")
def get_batch_size(
    model: AMF_GD_YOLOv8,
    device: torch.device,
    train_dataset: TreeDataset,
    accumulate: int,
    num_workers: int,
    max_batch_size: Optional[int] = None,
    num_iterations: int = 2,
) -> int:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    batch_sizes = list(set([accumulate // i for i in range(1, accumulate + 1)]))
    if max_batch_size is not None:
        batch_sizes = [batch_size for batch_size in batch_sizes if batch_size < max_batch_size]
    exec_times = [np.inf] * len(batch_sizes)

    for idx, batch_size in RICH_PRINTING.pbar(
        enumerate(batch_sizes),
        len(batch_sizes),
        leave=True,
        description="Simulate training with different epochs",
    ):
        try:
            train_loader = TreeDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            start_time = time.time()
            for _ in RICH_PRINTING.pbar(
                range(num_iterations),
                num_iterations,
                leave=False,
                description=f"Batch size = {batch_size}",
            ):
                accumulation_steps = max(round(accumulate / batch_size), 1)
                running_accumulation_step = 1

                stream = RICH_PRINTING.pbar(
                    train_loader, len(train_loader), leave=False, description="Epochs to accumulate"
                )
                for data in stream:
                    # Get the data
                    image_rgb: torch.Tensor | None = data["image_rgb"]
                    image_chm: torch.Tensor | None = data["image_chm"]
                    gt_bboxes: torch.Tensor = data["bboxes"]
                    gt_classes: torch.Tensor = data["labels"]
                    gt_indices: torch.Tensor = data["indices"]
                    image_indices: torch.Tensor = data["image_indices"]

                    if image_rgb is not None:
                        image_rgb = image_rgb.to(device, non_blocking=True)
                    if image_chm is not None:
                        image_chm = image_chm.to(device, non_blocking=True)
                    gt_bboxes = gt_bboxes.to(device, non_blocking=True)
                    gt_classes = gt_classes.to(device, non_blocking=True)
                    gt_indices = gt_indices.to(device, non_blocking=True)
                    image_indices = image_indices.to(device, non_blocking=True)

                    # Compute the model output
                    output = model.forward(image_rgb, image_chm)

                    # Compute the loss
                    total_loss, loss_dict = model.compute_loss(
                        output, gt_bboxes, gt_classes, gt_indices
                    )
                    total_loss.backward()

                    # Gradient accumulation
                    if running_accumulation_step % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        break

                    running_accumulation_step += 1

            end_time = time.time()

            total_processed = accumulation_steps * batch_size
            exec_times[idx] = (end_time - start_time) / total_processed

        except torch.cuda.OutOfMemoryError:
            RICH_PRINTING.print(f"\tOOM at batch size {batch_size}")
            break

        except Exception as e:
            raise e

    # Select best batch size
    best_index = min(enumerate(exec_times), key=lambda x: x[1])[0]
    best_batch_size = batch_sizes[best_index]

    exec_times_str = [f"{exec_time:.3f}" for exec_time in exec_times]
    exec_times_width = max(len(exec_time_str) for exec_time_str in exec_times_str)
    batch_sizes_width = max(len(str(batch_size)) for batch_size in batch_sizes)
    width = max(exec_times_width, batch_sizes_width)
    exec_times_full_str = ", ".join(map(lambda s: f"{s:>{width}}", exec_times_str))
    batch_sizes_full_str = ", ".join(map(lambda i: f"{i:>{width}}", batch_sizes))
    RICH_PRINTING.print(f"Batch sizes: {batch_sizes_full_str}")
    RICH_PRINTING.print(f"Total times: {exec_times_full_str}")

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    RICH_PRINTING.print(f"Final batch size: {best_batch_size}")

    return best_batch_size


@RICH_PRINTING.running_message("Training a model...")
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
    no_improvement_stop_epochs: int,
    ap_interval: int = 10,
) -> Tuple[AMF_GD_YOLOv8, int]:

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
    best_epoch = 0
    best_loss = np.inf
    skip_until = 3

    for epoch in RICH_PRINTING.pbar(range(1, epochs + 1), epochs, description="Epoch"):
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
            ap_interval=ap_interval,
        )

        current_loss = validate(
            val_loader, model, device, training_metrics, epoch=epoch, ap_interval=ap_interval
        )
        scheduler.step()

        if epoch >= skip_until:
            # Store and save the best model
            if current_loss < best_loss:
                best_epoch = epoch
                best_model = model
                best_loss = current_loss
                best_model.save_weights()

        training_metrics.end_loop(epoch)

        if (epoch - best_epoch) > no_improvement_stop_epochs:
            break

    # Save the plot showing the evolution of the metrics
    training_metrics.visualize(intervals=intervals, save_paths=training_metrics_path)
    training_metrics.save_metrics(
        os.path.join(
            model.folder_path,
            "metrics_values.json",
        )
    )
    return best_model, best_epoch
